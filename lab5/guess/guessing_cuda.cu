#ifdef CUDA_PARALLEL

#include "guessing_cuda.cuh"
#include "PCFG.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

/*
 * CUDA 错误检查宏
 */
#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess)                                                  \
        {                                                                       \
            fprintf(stderr,                                                      \
                    "CUDA error at %s:%d: %s\n",                                  \
                    __FILE__,                                                    \
                    __LINE__,                                                    \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                       \
    } while (0)


/*
 * GPU kernel:
 *
 * 每个线程负责生成一个 guess:
 *
 *   output[idx] = prefix + values[idx]
 *
 * 数据布局：
 *
 *   values:
 *     第 i 个 value 位于 values + i * value_stride
 *
 *   output:
 *     第 i 个输出位于 output + i * output_stride
 */
__global__ void cuda_expand_values_kernel(
    const char *prefix,
    int prefix_len,
    const char *values,
    int value_stride,
    char *output,
    int output_stride,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total)
    {
        return;
    }

    const char *value_ptr = values + idx * value_stride;
    char *out_ptr = output + idx * output_stride;

    // 1. 拷贝 prefix
    for (int i = 0; i < prefix_len; i += 1)
    {
        out_ptr[i] = prefix[i];
    }

    // 2. 拼接 value
    int j = 0;

    while (j < value_stride)
    {
        char c = value_ptr[j];

        if (c == '\0')
        {
            break;
        }

        out_ptr[prefix_len + j] = c;
        j += 1;
    }

    // 3. 字符串结尾
    out_ptr[prefix_len + j] = '\0';
}


/*
 * CUDA_Init
 *
 * 只做最基础的设备检查。
 * 如果机器没有 CUDA GPU，会直接退出。
 */
void CUDA_Init()
{
    int device_count = 0;

    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count <= 0)
    {
        fprintf(stderr, "CUDA_Init failed: no CUDA device found.\n");
        exit(EXIT_FAILURE);
    }

    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("CUDA device: %s\n", prop.name);
}


/*
 * CUDA_Finalize
 *
 * CUDA 没有像 MPI_Finalize 那样必须调用的 finalize，
 * 这里主要做一次同步，保证 kernel 都执行完。
 */
void CUDA_Finalize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}


/*
 * CUDA_GenerateBatch
 *
 * 这个函数是 CPU 调用 GPU 的封装层。
 *
 * 和你的 MPI_MasterGenerateValues 类似：
 *   1. CPU 准备 prefix 和 ordered_values
 *   2. 拷贝到 GPU
 *   3. GPU 并行生成 prefix + value
 *   4. 拷贝回 CPU
 *   5. 追加到 guesses
 */
long long CUDA_GenerateBatch(
    vector<string> &guesses,
    const string &prefix,
    segment *seg_ptr,
    int total)
{
    if (seg_ptr == nullptr)
    {
        return 0;
    }

    int available = static_cast<int>(seg_ptr->ordered_values.size());

    if (available <= 0)
    {
        return 0;
    }

    if (total <= 0 || total > available)
    {
        total = available;
    }

    if (total <= 0)
    {
        return 0;
    }

    int prefix_len = static_cast<int>(prefix.size());

    // 计算最大 value 长度，用动态 stride，避免固定 MAX_LEN 截断
    int max_value_len = 0;

    for (int i = 0; i < total; i += 1)
    {
        int len = static_cast<int>(seg_ptr->ordered_values[i].size());

        if (len > max_value_len)
        {
            max_value_len = len;
        }
    }

    // 每个 value 至少需要 1 个 '\0'
    int value_stride = max_value_len + 1;

    // 每个输出为 prefix + value + '\0'
    int output_stride = prefix_len + max_value_len + 1;

    if (value_stride <= 0 || output_stride <= 0)
    {
        return 0;
    }

    size_t prefix_bytes = static_cast<size_t>(max(prefix_len, 1));
    size_t values_bytes = static_cast<size_t>(total) * value_stride;
    size_t output_bytes = static_cast<size_t>(total) * output_stride;

    vector<char> h_prefix(prefix_bytes, '\0');
    vector<char> h_values(values_bytes, '\0');
    vector<char> h_output(output_bytes, '\0');

    if (prefix_len > 0)
    {
        memcpy(h_prefix.data(), prefix.data(), prefix_len);
    }

    // 打包 ordered_values 到连续内存
    for (int i = 0; i < total; i += 1)
    {
        const string &value = seg_ptr->ordered_values[i];
        char *dst = h_values.data() + static_cast<size_t>(i) * value_stride;

        memcpy(dst, value.c_str(), value.size());

        dst[value.size()] = '\0';
    }

    char *d_prefix = nullptr;
    char *d_values = nullptr;
    char *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_prefix), prefix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_values), values_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes));

    CUDA_CHECK(cudaMemcpy(
        d_prefix,
        h_prefix.data(),
        prefix_bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_values,
        h_values.data(),
        values_bytes,
        cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    cuda_expand_values_kernel<<<grid_size, block_size>>>(
        d_prefix,
        prefix_len,
        d_values,
        value_stride,
        d_output,
        output_stride,
        total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        output_bytes,
        cudaMemcpyDeviceToHost));

    guesses.reserve(guesses.size() + total);

    for (int i = 0; i < total; i += 1)
    {
        const char *guess_ptr = h_output.data() + static_cast<size_t>(i) * output_stride;
        guesses.emplace_back(guess_ptr);
    }

    CUDA_CHECK(cudaFree(d_prefix));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_output));

    return static_cast<long long>(total);
}

#endif