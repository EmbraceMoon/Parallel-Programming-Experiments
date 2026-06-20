#pragma once

#include <vector>
#include <string>

// 前向声明，避免 .cuh 里重复展开 PCFG.h 的复杂内容
class segment;

#ifdef CUDA_PARALLEL

// CUDA 初始化：检查设备、设置设备
void CUDA_Init();

// CUDA 清理：同步设备
void CUDA_Finalize();

/*
 * CUDA_GenerateBatch
 *
 * 作用：
 *   批量生成 prefix + seg_ptr->ordered_values[i]
 *
 * 参数：
 *   guesses : 输出结果，直接追加到 PriorityQueue::guesses
 *   prefix  : 当前 PT 前面已经确定好的字符串
 *   seg_ptr : 指向最后一个 segment 的统计数据
 *   total   : 需要遍历的 value 数量，一般为 seg_ptr->ordered_values.size()
 *
 * 返回：
 *   实际生成的 guess 数量
 */
long long CUDA_GenerateBatch(
    std::vector<std::string> &guesses,
    const std::string &prefix,
    segment *seg_ptr,
    int total
);

#endif