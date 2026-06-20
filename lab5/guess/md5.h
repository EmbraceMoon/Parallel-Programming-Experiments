#include <iostream>
#include <string>
#include <cstring>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

void MD5Hash(string input, bit32 *state);

// SIMD部分
#ifdef __ARM_NEON
#include <arm_neon.h>

// 向量化位运算函数
static inline uint32x4_t vF(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z));
}

static inline uint32x4_t vG(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)));
}

static inline uint32x4_t vH(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return veorq_u32(veorq_u32(x, y), z);
}

static inline uint32x4_t vI(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)));
}

static inline uint32x4_t vRotateLeft(uint32x4_t val, int n) {
    return vorrq_u32(vshlq_n_u32(val, n), vshrq_n_u32(val, 32 - n));
}

// 向量化宏
#define vFF(a, b, c, d, x_vec, s, ac) do { \
    (a) = vaddq_u32(a, vF((b), (c), (d))); \
    (a) = vaddq_u32(a, (x_vec)); \
    (a) = vaddq_u32(a, vdupq_n_u32(ac)); \
    (a) = vRotateLeft((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while(0)

#define vGG(a, b, c, d, x_vec, s, ac) do { \
    (a) = vaddq_u32(a, vG((b), (c), (d))); \
    (a) = vaddq_u32(a, (x_vec)); \
    (a) = vaddq_u32(a, vdupq_n_u32(ac)); \
    (a) = vRotateLeft((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while(0)

#define vHH(a, b, c, d, x_vec, s, ac) do { \
    (a) = vaddq_u32(a, vH((b), (c), (d))); \
    (a) = vaddq_u32(a, (x_vec)); \
    (a) = vaddq_u32(a, vdupq_n_u32(ac)); \
    (a) = vRotateLeft((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while(0)

#define vII(a, b, c, d, x_vec, s, ac) do { \
    (a) = vaddq_u32(a, vI((b), (c), (d))); \
    (a) = vaddq_u32(a, (x_vec)); \
    (a) = vaddq_u32(a, vdupq_n_u32(ac)); \
    (a) = vRotateLeft((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
} while(0)

// 主入口函数
void MD5Hash_SIMD(const string inputs[4], bit32 results[4][4]);

#endif // __ARM_NEON