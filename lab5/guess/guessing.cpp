#include "PCFG.h"
using namespace std;
#ifdef OPENMP
#include <omp.h>
#endif
#ifdef PTHREAD
#include <pthread.h>
#define NUM_THREADS 4
#define PARALLEL_THRESHOLD 100
struct ThreadArgs
{
    int start_idx;
    int end_idx;
    string prefix;
    segment *seg_ptr;
    vector<string> local_result;
};
void *thread_generate(void *arg)
{
    ThreadArgs *args = (ThreadArgs *)arg;

    for (int i = args->start_idx; i < args->end_idx; i += 1)
    {
        string temp = args->prefix + args->seg_ptr->ordered_values[i];
        args->local_result.emplace_back(temp);
    }

    pthread_exit(NULL);
}
#endif
#ifdef MPI_PARALLEL
// 以下部分为mpi的master-worker工作模式的准备函数
#include <mpi.h>
#include <algorithm>

static const int TAG_MPI_WORK = 100;
static const int TAG_MPI_STOP = 101;
static const int TAG_MPI_PREFIX = 102;
static const int TAG_MPI_VALUE = 103;
static const int TAG_MPI_RESULT_COUNT = 104;
static const int TAG_MPI_RESULT_GUESS = 105;

static void MPI_SendString(const string &s, int dest, int tag)
{
    int len = static_cast<int>(s.size());
    MPI_Send(&len, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    if (len > 0)
    {
        MPI_Send(s.data(), len, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }
}

static string MPI_RecvString(int src, int tag)
{
    MPI_Status status;
    int len = 0;
    MPI_Recv(&len, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);

    string s;
    s.resize(len);

    if (len > 0)
    {
        MPI_Recv(&s[0], len, MPI_CHAR, src, tag, MPI_COMM_WORLD, &status);
    }

    return s;
}

// worker 进程执行这个函数：
// 不训练，不初始化优先队列，只等待 rank 0 发送 prefix + values。
void MPI_WorkerLoop()
{
    while (true)
    {
        MPI_Status status;
        int header[2] = {0, 0};

        MPI_Recv(header, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_MPI_STOP)
        {
            break;
        }

        int prefix_len = header[0];
        int value_count = header[1];

        string prefix;
        prefix.resize(prefix_len);

        if (prefix_len > 0)
        {
            MPI_Recv(&prefix[0], prefix_len, MPI_CHAR, 0, TAG_MPI_PREFIX, MPI_COMM_WORLD, &status);
        }

        vector<string> local_guesses;
        local_guesses.reserve(value_count);

        for (int i = 0; i < value_count; i += 1)
        {
            string value = MPI_RecvString(0, TAG_MPI_VALUE);
            local_guesses.emplace_back(prefix + value);
        }

        long long result_count = static_cast<long long>(local_guesses.size());
        MPI_Send(&result_count, 1, MPI_LONG_LONG, 0, TAG_MPI_RESULT_COUNT, MPI_COMM_WORLD);

        for (string &guess : local_guesses)
        {
            MPI_SendString(guess, 0, TAG_MPI_RESULT_GUESS);
        }
    }
}

// rank 0 结束前调用，通知所有 worker 退出。
void MPI_StopWorkers()
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0)
    {
        return;
    }

    int header[2] = {0, 0};

    for (int dest = 1; dest < size; dest += 1)
    {
        MPI_Send(header, 2, MPI_INT, dest, TAG_MPI_STOP, MPI_COMM_WORLD);
    }
}

// rank 0 调用：
// 把一个 PT 的最后一段 value 按 MPI rank 分块。
// rank 0 自己生成一块，其他块发给 worker 生成，再收回 guesses。
static long long MPI_MasterGenerateValues(
    vector<string> &guesses,
    const string &prefix,
    segment *seg_ptr,
    int total)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0)
    {
        return 0;
    }

    if (total <= 0)
    {
        return 0;
    }

    // 给 worker 发送任务
    for (int dest = 1; dest < size; dest += 1)
    {
        int start = static_cast<int>((long long)total * dest / size);
        int end = static_cast<int>((long long)total * (dest + 1) / size);
        int count = end - start;

        int header[2];
        header[0] = static_cast<int>(prefix.size());
        header[1] = count;

        MPI_Send(header, 2, MPI_INT, dest, TAG_MPI_WORK, MPI_COMM_WORLD);

        if (!prefix.empty())
        {
            MPI_Send(prefix.data(), header[0], MPI_CHAR, dest, TAG_MPI_PREFIX, MPI_COMM_WORLD);
        }

        for (int i = start; i < end; i += 1)
        {
            MPI_SendString(seg_ptr->ordered_values[i], dest, TAG_MPI_VALUE);
        }
    }

    long long generated = 0;

    // rank 0 自己负责第 0 块
    int start0 = 0;
    int end0 = static_cast<int>((long long)total / size);

    guesses.reserve(guesses.size() + total);

    for (int i = start0; i < end0; i += 1)
    {
        guesses.emplace_back(prefix + seg_ptr->ordered_values[i]);
        generated += 1;
    }

    // 收回 worker 生成的 guesses
    for (int src = 1; src < size; src += 1)
    {
        MPI_Status status;
        long long recv_count = 0;

        MPI_Recv(&recv_count, 1, MPI_LONG_LONG, src, TAG_MPI_RESULT_COUNT, MPI_COMM_WORLD, &status);

        for (long long i = 0; i < recv_count; i += 1)
        {
            string guess = MPI_RecvString(src, TAG_MPI_RESULT_GUESS);
            guesses.emplace_back(move(guess));
        }

        generated += recv_count;
    }

    return generated;
}
#endif
#ifdef CUDA_PARALLEL
#include "guessing_cuda.cuh"
#endif


void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
	// 此处建堆以优化队列插入
	make_heap(priority.begin(), priority.end(), PTCompare());
}

//void PriorityQueue::PopNext()
//{

//    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
//    Generate(priority.front());

//    // 然后需要根据即将出队的PT，生成一系列新的PT
//    vector<PT> new_pts = priority.front().NewPTs();
//    for (PT pt : new_pts)
//    {
//        // 计算概率
//        CalProb(pt);
//        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
//        for (auto iter = priority.begin(); iter != priority.end(); iter++)
//        {
//            // 对于非队首和队尾的特殊情况
//            if (iter != priority.end() - 1 && iter != priority.begin())
//            {
//                // 判定概率
//                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
//                {
//                    priority.emplace(iter + 1, pt);
//                    break;
//                }
//            }
//            if (iter == priority.end() - 1)
//            {
//                priority.emplace_back(pt);
//                break;
//            }
//            if (iter == priority.begin() && iter->prob < pt.prob)
//            {
//                priority.emplace(iter, pt);
//                break;
//            }
//        }
//    }

//    // 现在队首的PT善后工作已经结束，将其出队（删除）
//    priority.erase(priority.begin());
//}

// 堆版本PopNext
void PriorityQueue::PopNext()
{
    PT top_pt = priority.front();

    pop_heap(priority.begin(), priority.end(), PTCompare());
    priority.pop_back();

    Generate(top_pt);

    vector<PT> new_pts = top_pt.NewPTs();

    for (PT pt : new_pts)
    {
        CalProb(pt);
        priority.emplace_back(pt);
        push_heap(priority.begin(), priority.end(), PTCompare());
    }
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        #ifdef SERIAL
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
        #endif

        #ifdef OPENMP
        vector<vector<string>> thread_buf(4);
        #pragma omp parallel num_threads(4)
        {
            int tid = omp_get_thread_num();
            int total = pt.max_indices[0];
            thread_buf[tid].reserve((total + 3) / 4);

            #pragma omp for schedule(static)
            for (int i = 0; i < total; i += 1) {
                string guess = a->ordered_values[i];
                thread_buf[tid].emplace_back(guess);
            }
        }
        guesses.reserve(guesses.size() + pt.max_indices[0]);
        for (int t = 0; t < 4; t += 1) {
            for (string &guess : thread_buf[t]) {
                guesses.emplace_back(move(guess));
                total_guesses += 1;
            }
        }
        #endif

        #ifdef PTHREAD
		int total = pt.max_indices[0];

		if (total < PARALLEL_THRESHOLD)
		{
			for (int i = 0; i < total; i += 1)
			{
				string guess = a->ordered_values[i];
				guesses.emplace_back(guess);
				total_guesses += 1;
			}
		}
		else
		{
			pthread_t threads[NUM_THREADS];
			ThreadArgs args[NUM_THREADS];

			int chunk = (total + NUM_THREADS - 1) / NUM_THREADS;

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				args[t].start_idx = t * chunk;
				args[t].end_idx = min((t + 1) * chunk, total);
				args[t].prefix = "";
				args[t].seg_ptr = a;
			}

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				pthread_create(&threads[t], NULL, thread_generate, &args[t]);
			}

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				pthread_join(threads[t], NULL);
			}

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				for (string &guess : args[t].local_result)
				{
					guesses.emplace_back(move(guess));
					total_guesses += 1;
				}
			}
		}
		#endif

		#ifdef MPI_PARALLEL
		long long generated = MPI_MasterGenerateValues(guesses, "", a, pt.max_indices[0]);
		total_guesses += generated;
		#endif

		#ifdef CUDA_PARALLEL
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        CUDA_GenerateBatch(
            guesses,
            "",
            a,
            pt.max_indices[0]
        );
		#endif
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        #ifdef SERIAL
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
        #endif

        #ifdef OPENMP
        vector<vector<string>> thread_buf(4);
        #pragma omp parallel num_threads(4)
        {
            int tid = omp_get_thread_num();
            int total = pt.max_indices[pt.content.size() - 1];
            thread_buf[tid].reserve((total + 3) / 4);

            #pragma omp for schedule(static)
            for (int i = 0; i < total; i += 1) {
                string temp = guess + a->ordered_values[i];
                thread_buf[tid].emplace_back(temp);
            }
        }
        guesses.reserve(guesses.size() + pt.max_indices[pt.content.size() - 1]);
        for (int t = 0; t < 4; t += 1) {
            for (string &temp : thread_buf[t]) {
                guesses.emplace_back(move(temp));
                total_guesses += 1;
            }
        }
        #endif

        #ifdef PTHREAD
		int total = pt.max_indices[pt.content.size() - 1];

		if (total < PARALLEL_THRESHOLD)
		{
			for (int i = 0; i < total; i += 1)
			{
				string temp = guess + a->ordered_values[i];
				guesses.emplace_back(temp);
				total_guesses += 1;
			}
		}
		else
		{
			pthread_t threads[NUM_THREADS];
			ThreadArgs args[NUM_THREADS];

			int chunk = (total + NUM_THREADS - 1) / NUM_THREADS;

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				args[t].start_idx = t * chunk;
				args[t].end_idx = min((t + 1) * chunk, total);
				args[t].prefix = guess;
				args[t].seg_ptr = a;
			}

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				pthread_create(&threads[t], NULL, thread_generate, &args[t]);
			}

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				pthread_join(threads[t], NULL);
			}

			for (int t = 0; t < NUM_THREADS; t += 1)
			{
				for (string &temp : args[t].local_result)
				{
					guesses.emplace_back(move(temp));
					total_guesses += 1;
				}
			}
		}
		#endif

		#ifdef MPI_PARALLEL
		int last_idx = pt.content.size() - 1;
		long long generated = MPI_MasterGenerateValues(guesses, guess, a, pt.max_indices[last_idx]);
		total_guesses += generated;
		#endif

		#ifdef CUDA_PARALLEL
        string prefix;
        int seg_idx = 0;

        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];

            seg_idx += 1;

            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a;
        int last = pt.content.size() - 1;

        if (pt.content[last].type == 1)
            a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)
            a = &m.digits[m.FindDigit(pt.content[last])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[last])];

        CUDA_GenerateBatch(
            guesses,
            prefix,
            a,
            pt.max_indices[last]
        );

		#endif
    }
}