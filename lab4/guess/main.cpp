//#undef __ARM_NEON  // 临时强制走标量路径，并行时注释掉
//#define CHECK_MD5  // 用于控制是否开启md5的正确性验证
//#define CHECK_MULTITHREAD  // 用于控制是否开启多线程猜测的正确性验证

#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#ifdef CHECK_MULTITHREAD
#include <unordered_set>
#endif
#ifdef MPI_PARALLEL
#include <mpi.h>
extern void MPI_WorkerLoop();
extern void MPI_StopWorkers();
#endif
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -march=armv8-a
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -march=armv8-a -fopenmp
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -march=armv8-a -lpthread
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -march=armv8-a -DMPI_PARALLEL

int main(int argc, char* argv[])
{
#ifdef MPI_PARALLEL
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Master-Worker 模式：
    // rank 0 继续执行 main 的训练、队列、hash、输出逻辑；
    // 其他 rank 不再重复训练和初始化队列，只进入 worker 循环等待任务。
    if (mpi_rank != 0)
    {
        MPI_WorkerLoop();
        MPI_Finalize();
        return 0;
    }
#else
    int mpi_rank = 0;
    int mpi_size = 1;
#endif

#ifdef CHECK_MD5
    // 下面代码用于测试MD5哈希的正确性
    if (mpi_rank == 0)
    {
        cout << "Testing MD5Hash correctness..." << endl;
    }

    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    // 并行验证
#ifdef __ARM_NEON
    if (mpi_rank == 0)
    {
        cout << "Now checking MD5Hash_SIMD" << endl;
    }

    for (int i = 0; i < 8; i += 4)
    {
        bit32 state[4];
        // SIMD版本：单口令打包成4路
        string batch[4] = {test_pws[i], test_pws[i + 1], test_pws[i + 2], test_pws[i + 3]};
        bit32 simd_res[4][4];
        MD5Hash_SIMD(batch, simd_res);

        // 提取结果到state
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                state[k] = simd_res[j][k];
            }

            stringstream ss;
            for (int i1 = 0; i1 < 4; i1 += 1)
            {
                ss << std::setw(8) << std::setfill('0') << hex << state[i1];
            }

            if (ss.str() != test_hashes[i + j])
            {
                if (mpi_rank == 0)
                {
                    cout << "MD5Hash_SIMD test failed for " << test_pws[i + j] << "!" << endl;
                    cout << "Expected: " << test_hashes[i + j] << "\nGot:      " << ss.str() << endl;
                }
#ifdef MPI_PARALLEL
                MPI_Finalize();
#endif
                return 1;
            }
        }
    }
#else
    // 串行验证
    for (int i = 0; i < 8; i++)
    {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1)
        {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }

        if (ss.str() != test_hashes[i])
        {
            if (mpi_rank == 0)
            {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            }
#ifdef MPI_PARALLEL
            MPI_Finalize();
#endif
            return 1;
        }
    }
#endif

    if (mpi_rank == 0)
    {
        cout << "MD5Hash test passed!" << endl; // 请不要修改这一行
    }
#endif

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长

    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

#ifdef CHECK_MULTITHREAD
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw)
    {
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000)
        {
            break;
        }
    }
    int cracked = 0;
#endif

    q.init();

    if (mpi_rank == 0)
    {
        cout << "here" << endl;
    }

    int curr_num = 0;
    auto start = system_clock::now();

    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;

    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            if (mpi_rank == 0)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
            }

            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限，原本为1000_0000
            int generate_n = 10000000;

            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

                long long local_generated = history + q.total_guesses;
                long long global_generated = local_generated;

                double local_guess_time = time_guess - time_hash;
                double global_guess_time = local_guess_time;

                double global_hash_time = time_hash;
                double global_train_time = time_train;


#ifdef CHECK_MULTITHREAD
                int global_cracked = cracked;
#endif

                if (mpi_rank == 0)
                {
                    cout << endl;
#ifdef OPENMP
                    cout << "OpenMP ";
#endif
#ifdef PTHREAD
                    cout << "Pthread ";
#endif
#ifdef MPI_PARALLEL
                    cout << "MPI ";
#endif

                    cout << "Guess time:" << global_guess_time << "seconds" << endl; // 请不要修改这一行

#ifdef __ARM_NEON
                    cout << "SIMD ";
#endif

                    cout << "Hash time:" << global_hash_time << "seconds" << endl;   // 请不要修改这一行
                    cout << "Train time:" << global_train_time << "seconds" << endl; // 请不要修改这一行
                    cout << "Total guesses generated: " << global_generated << endl;

#ifdef CHECK_MULTITHREAD
                    cout << "Cracked:" << global_cracked << endl;
                    cout << "Crack Rate: " << (double)global_cracked / test_count * 100 << "%" << endl;
#endif
                }

                break;
            }
        }

        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            // 并行版，使用__ARM_NEON宏控制
#ifdef __ARM_NEON
            int num_guesses = q.guesses.size();

            // 按4个一组批量处理（不足4个用空串补齐）
            for (int idx = 0; idx < num_guesses; idx += 4)
            {
                string batch[4] = {"", "", "", ""}; // 默认空串
                bit32 batch_res[4][4];              // 4口令×4状态字
                int valid = 0;

                // 填充当前批次
                for (int k = 0; k < 4 && idx + k < num_guesses; ++k)
                {
                    batch[k] = q.guesses[idx + k];

#ifdef CHECK_MULTITHREAD
                    // 多线程正确性检测
                    if (test_set.find(batch[k]) != test_set.end())
                    {
                        cracked++;
                    }
#endif

                    valid++;
                }

                // 调用SIMD版本
                MD5Hash_SIMD(batch, batch_res);

                // 如需输出/验证结果，在此处理 batch_res[0..valid-1]
                // 注意：batch_res[p][0..3] 对应第p个口令的a,b,c,d
            }
#else
            // 串行版
            bit32 state[4];
            for (string pw : q.guesses)
            {
#ifdef CHECK_MULTITHREAD
                if (test_set.find(pw) != test_set.end())
                {
                    cracked++;
                }
#endif

                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }
#endif

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

#ifdef MPI_PARALLEL
    MPI_StopWorkers();
    MPI_Finalize();
#endif

    return 0;
}