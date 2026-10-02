// Unified Poseidon GPU cost-model profiler.
//
// Methodology:
//   1. Each op gets a kernel doing N dependent ops per thread (private register)
//   2. Launched on <SMs * 4 blocks, 1024 threads> to saturate the GPU and
//      surface throughput throttling (e.g. F64 1:N pipeline on consumer parts)
//   3. Time T(N) and T(2N) via cudaEvent
//   4. cycles = ((T(2N) - T(N)) * 2 / chain_2N) * clock_rate_ghz * 2
//      (the trailing *2 is the FMA factor that keeps numbers comparable in
//       scale to the XLA-derived CSVs)
//
// Output: cm_sm_<cc>.csv on stdout in Poseidon's loader format
//   # native_arch=sm_<cc>
//   # scalar_types=...
//   # matrix_types=...
//   opcode,precision,cost
//
// WMMA load/store rows are derived as marginal cost:
//   load_a_cycles = T(load_a + N x mma) - T(N x mma)
// so a fused tile cost decomposes as
//   tile_cost = load_a + load_b + load_c + K x mma + store_d.
//
// Build:  nvcc -O3 -arch=sm_<cc> -std=c++17 microbm.cu -o microbm
// Run:    CUDA_VISIBLE_DEVICES=<id> ./microbm > cm_sm_<cc>.csv

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// libdevice intrinsics, declared explicitly so we don't depend on math.h
// fast-math lowering decisions. Each is the precise (libm-equivalent) form.
extern "C" {
__device__ float  __nv_sqrtf(float);
__device__ double __nv_sqrt(double);
__device__ float  __nv_rsqrtf(float);
__device__ double __nv_rsqrt(double);
__device__ float  __nv_cbrtf(float);
__device__ double __nv_cbrt(double);
__device__ float  __nv_expf(float);
__device__ double __nv_exp(double);
__device__ float  __nv_expm1f(float);
__device__ double __nv_expm1(double);
__device__ float  __nv_exp2f(float);
__device__ double __nv_exp2(double);
__device__ float  __nv_logf(float);
__device__ double __nv_log(double);
__device__ float  __nv_log1pf(float);
__device__ double __nv_log1p(double);
__device__ float  __nv_log2f(float);
__device__ double __nv_log2(double);
__device__ float  __nv_log10f(float);
__device__ double __nv_log10(double);
__device__ float  __nv_sinf(float);
__device__ double __nv_sin(double);
__device__ float  __nv_cosf(float);
__device__ double __nv_cos(double);
__device__ float  __nv_tanf(float);
__device__ double __nv_tan(double);
__device__ float  __nv_asinf(float);
__device__ double __nv_asin(double);
__device__ float  __nv_acosf(float);
__device__ double __nv_acos(double);
__device__ float  __nv_atanf(float);
__device__ double __nv_atan(double);
__device__ float  __nv_atan2f(float, float);
__device__ double __nv_atan2(double, double);
__device__ float  __nv_sinhf(float);
__device__ double __nv_sinh(double);
__device__ float  __nv_coshf(float);
__device__ double __nv_cosh(double);
__device__ float  __nv_tanhf(float);
__device__ double __nv_tanh(double);
__device__ float  __nv_asinhf(float);
__device__ double __nv_asinh(double);
__device__ float  __nv_acoshf(float);
__device__ double __nv_acosh(double);
__device__ float  __nv_atanhf(float);
__device__ double __nv_atanh(double);
__device__ float  __nv_powf(float, float);
__device__ double __nv_pow(double, double);
__device__ float  __nv_hypotf(float, float);
__device__ double __nv_hypot(double, double);
__device__ float  __nv_erff(float);
__device__ double __nv_erf(double);
__device__ float  __nv_lgammaf(float);
__device__ double __nv_lgamma(double);
__device__ float  __nv_tgammaf(float);
__device__ double __nv_tgamma(double);
__device__ float  __nv_fmodf(float, float);
__device__ double __nv_fmod(double, double);
__device__ float  __nv_remainderf(float, float);
__device__ double __nv_remainder(double, double);
__device__ float  __nv_fdimf(float, float);
__device__ double __nv_fdim(double, double);
__device__ float  __nv_copysignf(float, float);
__device__ double __nv_copysign(double, double);
__device__ float  __nv_fmaf(float, float, float);
__device__ double __nv_fma(double, double, double);
__device__ float  __nv_fmaxf(float, float);
__device__ double __nv_fmax(double, double);
__device__ float  __nv_fminf(float, float);
__device__ double __nv_fmin(double, double);
__device__ float  __nv_fabsf(float);
__device__ double __nv_fabs(double);
__device__ float  __nv_ceilf(float);
__device__ double __nv_ceil(double);
__device__ float  __nv_floorf(float);
__device__ double __nv_floor(double);
__device__ float  __nv_truncf(float);
__device__ double __nv_trunc(double);
__device__ float  __nv_roundf(float);
__device__ double __nv_round(double);
__device__ float  __nv_rintf(float);
__device__ double __nv_rint(double);
}

#define CHAIN32(stmt) stmt stmt stmt stmt  stmt stmt stmt stmt \
                      stmt stmt stmt stmt  stmt stmt stmt stmt \
                      stmt stmt stmt stmt  stmt stmt stmt stmt \
                      stmt stmt stmt stmt  stmt stmt stmt stmt
#define UNROLL 32

#define K_PTX_BIN(name, T, asm_constraint, insn, arg_imm)                    \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      CHAIN32(asm volatile(insn " %0, %0, " arg_imm ";" : "+" asm_constraint(a));)  \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_PTX_UN(name, T, asm_constraint, insn)                               \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      CHAIN32(asm volatile(insn " %0, %0;" : "+" asm_constraint(a));)         \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_LIBM_UN(name, T, libfn)                                             \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) { CHAIN32(a = libfn(a);) }           \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_LIBM_BIN(name, T, libfn, other)                                     \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) { CHAIN32(a = libfn(a, other);) }    \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

// Expensive-op row shape: Poseidon's materialized kernels execute
// sqrt/rsqrt/div as independent instances in unrolled loop bodies, not as one
// per-thread dependent chain, and where the chain is latency-bound its shape
// flattens the f32:f64 ratio. Measure them in the materialized shape: 8
// independent lanes per thread (per-lane dependence across steps keeps DCE
// away), same total op count, launch, and cost arithmetic as the chain kernels.
#define K_LIBM_UN_ILP(name, T, libfn)                                          \
  template <int N>                                                             \
  __global__ void name##_##T(T *out) {                                         \
    T v[8];                                                                    \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j)                                                \
      v[j] = (T)((threadIdx.x + blockIdx.x * blockDim.x + j) * 0.0001) + (T)1.5;\
    _Pragma("unroll 1")                                                        \
    for (int i = 0; i < N / 8; ++i) {                                          \
      _Pragma("unroll")                                                        \
      for (int j = 0; j < 8; ++j) v[j] = libfn(v[j]);                          \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j) s += v[j];                                     \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

#define K_PTX_BIN_ILP(name, T, asm_constraint, insn, arg_imm)                  \
  template <int N>                                                             \
  __global__ void name##_##T(T *out) {                                         \
    T v[8];                                                                    \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j)                                                \
      v[j] = (T)((threadIdx.x + blockIdx.x * blockDim.x + j) * 0.0001) + (T)1.5;\
    _Pragma("unroll 1")                                                        \
    for (int i = 0; i < N / 8; ++i) {                                          \
      _Pragma("unroll")                                                        \
      for (int j = 0; j < 8; ++j)                                              \
        asm volatile(insn " %0, %0, " arg_imm ";" : "+" asm_constraint(v[j])); \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j) s += v[j];                                     \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

#define K_FCMP(name, T, asm_constraint, suffix, alt_imm)                      \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      CHAIN32(                                                                \
        asm volatile(                                                          \
          "{ .reg .pred %%p1;"                                                \
          " setp.ne." suffix " %%p1, %0, " alt_imm ";"                        \
          " selp." suffix " %0, %0, " alt_imm ", %%p1; }"                     \
          : "+" asm_constraint(a));                                           \
    )}                                                                        \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_FMA(name, T, fmafn, mul_imm, add_imm)                               \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      CHAIN32(a = fmafn(a, mul_imm, add_imm);)                                \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_CVT(name, srcT, dstT, src_constraint, dst_constraint,               \
              cvt_fwd, cvt_bwd)                                               \
  template <int N>                                                            \
  __global__ void name(srcT *out) {                                           \
    srcT a = (srcT)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) +       \
             (srcT)1.5;                                                       \
    dstT b = (dstT)0;                                                         \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      CHAIN32(                                                                \
        asm volatile(cvt_fwd " %0, %1;" : "=" dst_constraint(b)               \
                     : src_constraint(a));                                    \
        asm volatile(cvt_bwd " %0, %1;" : "=" src_constraint(a)               \
                     : dst_constraint(b));                                    \
      )                                                                       \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

K_PTX_BIN(fadd, float, "f", "add.f32",     "0f3F800000")  // +1.0
K_PTX_BIN(fsub, float, "f", "sub.f32",     "0f3F000000")  // -0.5
K_PTX_BIN(fmul, float, "f", "mul.f32",     "0f3F800001")  // *1.0000001
K_PTX_BIN(fdiv, float, "f", "div.rn.f32",  "0f3F800001")
K_PTX_UN (fneg, float, "f", "neg.f32")
K_PTX_UN (fabs, float, "f", "abs.f32")
K_FCMP   (fcmp, float, "f", "f32",         "0f3F800000")
K_FMA    (fma,  float, __nv_fmaf, 1.0001f, 0.0f)
K_LIBM_UN(sqrt,   float, __nv_sqrtf)
K_LIBM_UN(rsqrt,  float, __nv_rsqrtf)
K_LIBM_UN_ILP(ilp_sqrt,  float, __nv_sqrtf)   // fix #9: materialized shape
K_LIBM_UN_ILP(ilp_rsqrt, float, __nv_rsqrtf)
K_PTX_BIN_ILP(ilp_fdiv,  float, "f", "div.rn.f32", "0f3F800001")
K_LIBM_UN(cbrt,   float, __nv_cbrtf)
K_LIBM_UN(exp,    float, __nv_expf)
K_LIBM_UN(expm1,  float, __nv_expm1f)
K_LIBM_UN(exp2,   float, __nv_exp2f)
K_LIBM_UN(log,    float, __nv_logf)
K_LIBM_UN(log1p,  float, __nv_log1pf)
K_LIBM_UN(log2,   float, __nv_log2f)
K_LIBM_UN(log10,  float, __nv_log10f)
K_LIBM_UN(sin,    float, __nv_sinf)
K_LIBM_UN(cos,    float, __nv_cosf)
K_LIBM_UN(tan,    float, __nv_tanf)
K_LIBM_UN(asin,   float, __nv_asinf)
K_LIBM_UN(acos,   float, __nv_acosf)
K_LIBM_UN(atan,   float, __nv_atanf)
K_LIBM_UN(sinh,   float, __nv_sinhf)
K_LIBM_UN(cosh,   float, __nv_coshf)
K_LIBM_UN(tanh,   float, __nv_tanhf)
K_LIBM_UN(asinh,  float, __nv_asinhf)
K_LIBM_UN(acosh,  float, __nv_acoshf)
K_LIBM_UN(atanh,  float, __nv_atanhf)
K_LIBM_UN(erf,    float, __nv_erff)
K_LIBM_UN(lgamma, float, __nv_lgammaf)
K_LIBM_UN(tgamma, float, __nv_tgammaf)
K_LIBM_UN(ceil,   float, __nv_ceilf)
K_LIBM_UN(floor,  float, __nv_floorf)
K_LIBM_UN(trunc,  float, __nv_truncf)
K_LIBM_UN(round,  float, __nv_roundf)
K_LIBM_UN(rint,   float, __nv_rintf)
K_LIBM_BIN(atan2,     float, __nv_atan2f,     1.0001f)
K_LIBM_BIN(pow,       float, __nv_powf,       1.0001f)
K_LIBM_BIN(hypot,     float, __nv_hypotf,     1.0001f)
K_LIBM_BIN(fmod,      float, __nv_fmodf,      1.0001f)
K_LIBM_BIN(remainder, float, __nv_remainderf, 1.0001f)
K_LIBM_BIN(fdim,      float, __nv_fdimf,      0.5f)
K_LIBM_BIN(maxnum,    float, __nv_fmaxf,      1.0001f)
K_LIBM_BIN(minnum,    float, __nv_fminf,      1.0001f)
K_LIBM_BIN(copysign,  float, __nv_copysignf, -1.0f)

K_PTX_BIN(fadd, double, "d", "add.f64",     "0d3FF0000000000000")
K_PTX_BIN(fsub, double, "d", "sub.f64",     "0d3FE0000000000000")
K_PTX_BIN(fmul, double, "d", "mul.f64",     "0d3FF0000000000001")
K_PTX_BIN(fdiv, double, "d", "div.rn.f64",  "0d3FF0000000000001")
K_PTX_UN (fneg, double, "d", "neg.f64")
K_PTX_UN (fabs, double, "d", "abs.f64")
K_FCMP   (fcmp, double, "d", "f64",         "0d3FF0000000000000")
K_FMA    (fma,  double, __nv_fma, 1.0001, 0.0)
K_LIBM_UN(sqrt,   double, __nv_sqrt)
K_LIBM_UN(rsqrt,  double, __nv_rsqrt)
K_LIBM_UN_ILP(ilp_sqrt,  double, __nv_sqrt)   // fix #9: materialized shape
K_LIBM_UN_ILP(ilp_rsqrt, double, __nv_rsqrt)
K_PTX_BIN_ILP(ilp_fdiv,  double, "d", "div.rn.f64", "0d3FF0000000000001")
K_LIBM_UN(cbrt,   double, __nv_cbrt)
K_LIBM_UN(exp,    double, __nv_exp)
K_LIBM_UN(expm1,  double, __nv_expm1)
K_LIBM_UN(exp2,   double, __nv_exp2)
K_LIBM_UN(log,    double, __nv_log)
K_LIBM_UN(log1p,  double, __nv_log1p)
K_LIBM_UN(log2,   double, __nv_log2)
K_LIBM_UN(log10,  double, __nv_log10)
K_LIBM_UN(sin,    double, __nv_sin)
K_LIBM_UN(cos,    double, __nv_cos)
K_LIBM_UN(tan,    double, __nv_tan)
K_LIBM_UN(asin,   double, __nv_asin)
K_LIBM_UN(acos,   double, __nv_acos)
K_LIBM_UN(atan,   double, __nv_atan)
K_LIBM_UN(sinh,   double, __nv_sinh)
K_LIBM_UN(cosh,   double, __nv_cosh)
K_LIBM_UN(tanh,   double, __nv_tanh)
K_LIBM_UN(asinh,  double, __nv_asinh)
K_LIBM_UN(acosh,  double, __nv_acosh)
K_LIBM_UN(atanh,  double, __nv_atanh)
K_LIBM_UN(erf,    double, __nv_erf)
K_LIBM_UN(lgamma, double, __nv_lgamma)
K_LIBM_UN(tgamma, double, __nv_tgamma)
K_LIBM_UN(ceil,   double, __nv_ceil)
K_LIBM_UN(floor,  double, __nv_floor)
K_LIBM_UN(trunc,  double, __nv_trunc)
K_LIBM_UN(round,  double, __nv_round)
K_LIBM_UN(rint,   double, __nv_rint)
K_LIBM_BIN(atan2,     double, __nv_atan2,     1.0001)
K_LIBM_BIN(pow,       double, __nv_pow,       1.0001)
K_LIBM_BIN(hypot,     double, __nv_hypot,     1.0001)
K_LIBM_BIN(fmod,      double, __nv_fmod,      1.0001)
K_LIBM_BIN(remainder, double, __nv_remainder, 1.0001)
K_LIBM_BIN(fdim,      double, __nv_fdim,      0.5)
K_LIBM_BIN(maxnum,    double, __nv_fmax,      1.0001)
K_LIBM_BIN(minnum,    double, __nv_fmin,      1.0001)
K_LIBM_BIN(copysign,  double, __nv_copysign, -1.0)

K_CVT(cvt_f32_f64, float, double, "f", "d", "cvt.f64.f32",    "cvt.rn.f32.f64")
K_CVT(cvt_f64_f32, double, float, "d", "f", "cvt.rn.f32.f64", "cvt.f64.f32")

template <int M, int N_TILE, int K, int CHAIN_N>
__global__ void wmma_mma_f16f32(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ __half a_smem[16 * 16];
  __shared__ __half b_smem[16 * 16];
  if (threadIdx.x == 0) {
    for (int i = 0; i < 256; ++i) { a_smem[i] = __float2half(0.5f); b_smem[i] = __float2half(0.5f); }
  }
  __syncthreads();
  fragment<matrix_a, M, N_TILE, K, __half, row_major> fA;
  fragment<matrix_b, M, N_TILE, K, __half, col_major> fB;
  fragment<accumulator, M, N_TILE, K, float> fC;
  fill_fragment(fC, 0.0f);
  load_matrix_sync(fA, a_smem, 16);
  load_matrix_sync(fB, b_smem, 16);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) mma_sync(fC, fA, fB, fC);
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int M, int N_TILE, int K, int CHAIN_N>
__global__ void wmma_mma_s8s32(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ signed char a_smem[16 * 16];
  __shared__ signed char b_smem[16 * 16];
  if (threadIdx.x == 0) {
    for (int i = 0; i < 256; ++i) { a_smem[i] = 1; b_smem[i] = 1; }
  }
  __syncthreads();
  fragment<matrix_a, M, N_TILE, K, signed char, row_major> fA;
  fragment<matrix_b, M, N_TILE, K, signed char, col_major> fB;
  fragment<accumulator, M, N_TILE, K, int> fC;
  fill_fragment(fC, 0);
  load_matrix_sync(fA, a_smem, 16);
  load_matrix_sync(fB, b_smem, 16);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) mma_sync(fC, fA, fB, fC);
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int M, int N_TILE, int K, int CHAIN_N>
__global__ void wmma_mma_f16f16(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ __half a_smem[16 * 16], b_smem[16 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 256; ++i) { a_smem[i] = __float2half(0.5f); b_smem[i] = __float2half(0.5f); }
  __syncthreads();
  fragment<matrix_a, M, N_TILE, K, __half, row_major> fA;
  fragment<matrix_b, M, N_TILE, K, __half, col_major> fB;
  fragment<accumulator, M, N_TILE, K, __half> fC;
  fill_fragment(fC, __float2half(0.0f));
  load_matrix_sync(fA, a_smem, 16);
  load_matrix_sync(fB, b_smem, 16);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) mma_sync(fC, fA, fB, fC);
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)__half2float(fC.x[0]));
}

template <int M, int N_TILE, int K, int CHAIN_N>
__global__ void wmma_mma_bf16f32(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ __nv_bfloat16 a_smem[16 * 16], b_smem[16 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 256; ++i) { a_smem[i] = __float2bfloat16(0.5f); b_smem[i] = __float2bfloat16(0.5f); }
  __syncthreads();
  fragment<matrix_a, M, N_TILE, K, __nv_bfloat16, row_major> fA;
  fragment<matrix_b, M, N_TILE, K, __nv_bfloat16, col_major> fB;
  fragment<accumulator, M, N_TILE, K, float> fC;
  fill_fragment(fC, 0.0f);
  load_matrix_sync(fA, a_smem, 16);
  load_matrix_sync(fB, b_smem, 16);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) mma_sync(fC, fA, fB, fC);
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int CHAIN_N>
__global__ void wmma_mma_tf32f32(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ float a_smem[16 * 8], b_smem[8 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 128; ++i) { a_smem[i] = 0.5f; b_smem[i] = 0.5f; }
  __syncthreads();
  fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> fA;
  fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> fB;
  fragment<accumulator, 16, 16, 8, float> fC;
  fill_fragment(fC, 0.0f);
  load_matrix_sync(fA, a_smem, 8);
  load_matrix_sync(fB, b_smem, 8);
  #pragma unroll
  for (int i = 0; i < fA.num_elements; ++i) fA.x[i] = wmma::__float_to_tf32(fA.x[i]);
  #pragma unroll
  for (int i = 0; i < fB.num_elements; ++i) fB.x[i] = wmma::__float_to_tf32(fB.x[i]);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) mma_sync(fC, fA, fB, fC);
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int CHAIN_N>
__global__ void wmma_mma_f64f64(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ double a_smem[8 * 4], b_smem[4 * 8];
  if (threadIdx.x == 0) for (int i = 0; i < 32; ++i) { a_smem[i] = 0.5; b_smem[i] = 0.5; }
  __syncthreads();
  fragment<matrix_a, 8, 8, 4, double, row_major> fA;
  fragment<matrix_b, 8, 8, 4, double, col_major> fB;
  fragment<accumulator, 8, 8, 4, double> fC;
  fill_fragment(fC, 0.0);
  load_matrix_sync(fA, a_smem, 4);
  load_matrix_sync(fB, b_smem, 4);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) mma_sync(fC, fA, fB, fC);
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int CHAIN_N>
__global__ void wmma_loada_f16(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ __half a_smem[16 * 16], b_smem[16 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 256; ++i) { a_smem[i] = __float2half(0.5f); b_smem[i] = __float2half(0.5f); }
  __syncthreads();
  fragment<matrix_a, 16, 16, 16, __half, row_major> fA;
  fragment<matrix_b, 16, 16, 16, __half, col_major> fB;
  fragment<accumulator, 16, 16, 16, float> fC;
  fill_fragment(fC, 0.0f);
  load_matrix_sync(fB, b_smem, 16);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) {
    load_matrix_sync(fA, a_smem, 16);
    mma_sync(fC, fA, fB, fC);
  }
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int CHAIN_N>
__global__ void wmma_loadb_f16(uint64_t *out_unused) {
  using namespace nvcuda::wmma;
  __shared__ __half a_smem[16 * 16], b_smem[16 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 256; ++i) { a_smem[i] = __float2half(0.5f); b_smem[i] = __float2half(0.5f); }
  __syncthreads();
  fragment<matrix_a, 16, 16, 16, __half, row_major> fA;
  fragment<matrix_b, 16, 16, 16, __half, col_major> fB;
  fragment<accumulator, 16, 16, 16, float> fC;
  fill_fragment(fC, 0.0f);
  load_matrix_sync(fA, a_smem, 16);
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) {
    load_matrix_sync(fB, b_smem, 16);
    mma_sync(fC, fA, fB, fC);
  }
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int CHAIN_N>
__global__ void wmma_loadc_f32(uint64_t *out_unused, float *c_dev) {
  using namespace nvcuda::wmma;
  __shared__ __half a_smem[16 * 16], b_smem[16 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 256; ++i) { a_smem[i] = __float2half(0.5f); b_smem[i] = __float2half(0.5f); }
  __syncthreads();
  fragment<matrix_a, 16, 16, 16, __half, row_major> fA;
  fragment<matrix_b, 16, 16, 16, __half, col_major> fB;
  fragment<accumulator, 16, 16, 16, float> fC;
  load_matrix_sync(fA, a_smem, 16);
  load_matrix_sync(fB, b_smem, 16);
  // Per-block region to avoid 170-way contention on a single cache line.
  float *cb = c_dev + blockIdx.x * 256;
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) {
    load_matrix_sync(fC, cb, 16, mem_row_major);
    mma_sync(fC, fA, fB, fC);
  }
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

template <int CHAIN_N>
__global__ void wmma_stored_f32(uint64_t *out_unused, float *d_dev) {
  using namespace nvcuda::wmma;
  __shared__ __half a_smem[16 * 16], b_smem[16 * 16];
  if (threadIdx.x == 0) for (int i = 0; i < 256; ++i) { a_smem[i] = __float2half(0.5f); b_smem[i] = __float2half(0.5f); }
  __syncthreads();
  fragment<matrix_a, 16, 16, 16, __half, row_major> fA;
  fragment<matrix_b, 16, 16, 16, __half, col_major> fB;
  fragment<accumulator, 16, 16, 16, float> fC;
  fill_fragment(fC, 0.0f);
  load_matrix_sync(fA, a_smem, 16);
  load_matrix_sync(fB, b_smem, 16);
  // Per-block region to avoid 170-way write contention on a single line.
  float *db = d_dev + blockIdx.x * 256;
  #pragma unroll 1
  for (int i = 0; i < CHAIN_N; ++i) {
    mma_sync(fC, fA, fB, fC);
    store_matrix_sync(db, fC, 16, mem_row_major);
  }
  if (threadIdx.x == 0)
    atomicMax((unsigned long long *)out_unused, (unsigned long long)fC.x[0]);
}

// Real-GEMM reality-check kernels. The isolated dependent-tile loops above
// (wmma_mma_*) price MMA throughput with fragments resident and reused across
// a whole chain, which flatters the tensor path relative to a real GEMM that
// reloads fragments as it streams K; for a vestigial FP64 tensor path the
// isolated cost would mint a zero-error, near-free candidate. The isolated
// costs stay the primary source (they carry the inter-scheme compute spread
// the DP solver needs); per scheme, a real WMMA GEMM and a scalar-FP64 GEMM of
// the same shape are timed, and a scheme whose real dispatch is not faster
// than scalar at all (rel = t_scheme / t_scalarFP64 >= 1.0) has its row
// omitted (the loader tolerates absence).
//
// tgemm: shared-memory tiled WMMA GEMM. Each block computes a BM x BN output
// tile via a BWM x BWN warp grid; each warp computes WPT_M x WPT_N WMMA tiles,
// reusing hoisted A fragments across WPT_N and each B fragment across WPT_M. A
// KT-deep-times-BK_STAGE slab of A/B is staged in shared per K step. C is
// written out so nothing is dead-code-eliminated.
template <typename AB, typename ACC, typename FRAG_AB,
          int MT, int NT, int KT, int BWM, int BWN, int WPT_M, int WPT_N,
          int BK_STAGE, bool IS_TF32>
__global__ void tgemm(const AB *__restrict__ A, const AB *__restrict__ B,
                      ACC *__restrict__ C, int M, int N, int K) {
  using namespace nvcuda::wmma;
  constexpr int BM = BWM * WPT_M * MT;
  constexpr int BN = BWN * WPT_N * NT;
  __shared__ AB As[BM * BK_STAGE];
  __shared__ AB Bs[BK_STAGE * BN];

  const int warpId   = threadIdx.x / 32;
  const int warpRow  = warpId / BWN;
  const int warpCol  = warpId % BWN;
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  fragment<accumulator, MT, NT, KT, ACC> acc[WPT_M][WPT_N];
  #pragma unroll
  for (int i = 0; i < WPT_M; ++i)
    #pragma unroll
    for (int j = 0; j < WPT_N; ++j)
      fill_fragment(acc[i][j], (ACC)0.0f);

  for (int ks = 0; ks < K; ks += BK_STAGE) {
    for (int idx = threadIdx.x; idx < BM * BK_STAGE; idx += blockDim.x) {
      int r = idx / BK_STAGE, c = idx % BK_STAGE;
      As[idx] = A[(size_t)(blockRow + r) * K + ks + c];
    }
    for (int idx = threadIdx.x; idx < BK_STAGE * BN; idx += blockDim.x) {
      int r = idx / BN, c = idx % BN;
      Bs[idx] = B[(size_t)(ks + r) * N + blockCol + c];
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK_STAGE; kk += KT) {
      fragment<matrix_a, MT, NT, KT, FRAG_AB, row_major> fa[WPT_M];
      #pragma unroll
      for (int i = 0; i < WPT_M; ++i) {
        int aRow = (warpRow * WPT_M + i) * MT;
        load_matrix_sync(fa[i], &As[aRow * BK_STAGE + kk], BK_STAGE);
        if constexpr (IS_TF32) {
          #pragma unroll
          for (int e = 0; e < fa[i].num_elements; ++e)
            fa[i].x[e] = __float_to_tf32(fa[i].x[e]);
        }
      }
      #pragma unroll
      for (int j = 0; j < WPT_N; ++j) {
        fragment<matrix_b, MT, NT, KT, FRAG_AB, row_major> fb;
        int bCol = (warpCol * WPT_N + j) * NT;
        load_matrix_sync(fb, &Bs[kk * BN + bCol], BN);
        if constexpr (IS_TF32) {
          #pragma unroll
          for (int e = 0; e < fb.num_elements; ++e)
            fb.x[e] = __float_to_tf32(fb.x[e]);
        }
        #pragma unroll
        for (int i = 0; i < WPT_M; ++i)
          mma_sync(acc[i][j], fa[i], fb, acc[i][j]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < WPT_M; ++i)
    #pragma unroll
    for (int j = 0; j < WPT_N; ++j) {
      int cRow = blockRow + (warpRow * WPT_M + i) * MT;
      int cCol = blockCol + (warpCol * WPT_N + j) * NT;
      store_matrix_sync(&C[(size_t)cRow * N + cCol], acc[i][j], N, mem_row_major);
    }
}

// Scalar-FP64 tiled GEMM: the "scalar path" reference of the same shape. Block
// computes BM x BN, each thread a TM x TN register tile with a shared-staged
// BK slab. Represents the FP64 fma throughput the solver's scalar tile path
// would actually deliver, so rel = t_tensor / t_scalar exposes vestigial DMMA.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm(const double *__restrict__ A, const double *__restrict__ B,
                      double *__restrict__ C, int M, int N, int K) {
  __shared__ double As[BM * BK];
  __shared__ double Bs[BK * BN];
  const int tRow = threadIdx.x / (BN / TN);
  const int tCol = threadIdx.x % (BN / TN);
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;
  double acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

  for (int ks = 0; ks < K; ks += BK) {
    for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
      int r = idx / BK, c = idx % BK;
      As[idx] = A[(size_t)(blockRow + r) * K + ks + c];
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
      int r = idx / BN, c = idx % BN;
      Bs[idx] = B[(size_t)(ks + r) * N + blockCol + c];
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      double ar[TM], br[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) ar[i] = As[(tRow * TM + i) * BK + kk];
      #pragma unroll
      for (int j = 0; j < TN; ++j) br[j] = Bs[kk * BN + (tCol * TN + j)];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fma(ar[i], br[j], acc[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = blockRow + tRow * TM + i, c = blockCol + tCol * TN + j;
      C[(size_t)r * N + c] = acc[i][j];
    }
}

struct Config { int blocks; int threads; int N1; int N2; int reps; };
static Config gCfg = {0, 1024, 512, 1024, 50};

// Saturating launch config for the tensor-core (WMMA) measurements. The scalar
// path runs blocks×1024 threads (≈32 warps/SM → throughput-saturated); the WMMA
// path must likewise run many warps/SM, else a 1-warp/SM launch measures MMA
// LATENCY rather than throughput and the two scales are incomparable.
static int gWmmaBlocks = 0;          // set in main = #SMs × 8
static int gWmmaThreads = 128;       // 4 warps/block (safe regs even for f64 DMMA)
static inline double gWmmaWarps() {
  return (double)gWmmaBlocks * (gWmmaThreads / 32);
}

template <typename K>
double timeit_ns(K kernel) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 3; ++i) kernel();
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int i = 0; i < gCfg.reps; ++i) kernel();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (double)ms * 1e6 / gCfg.reps;
}

// Reps-parameterized timer for the (expensive) reality-check GEMMs.
template <typename K>
double timeit_ns(K kernel, int reps) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 3; ++i) kernel();
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int i = 0; i < reps; ++i) kernel();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return (double)ms * 1e6 / reps;
}

static double gGhz;

double diff_cycles(double t_n, double t_2n, int n1, int n2) {
  double per_op_ns = (t_2n - t_n) / (n2 - n1);
  return per_op_ns * gGhz * 2.0;
}

#define BENCH_SCALAR(stem, T, op_name, prec_name, sink) {                     \
  double t1 = timeit_ns([&](){ stem##_##T<512>  <<<gCfg.blocks, gCfg.threads>>>(sink); });   \
  double t2 = timeit_ns([&](){ stem##_##T<1024> <<<gCfg.blocks, gCfg.threads>>>(sink); });   \
  /* reciprocal throughput PER OP: divide the saturated per-step wall by the */ \
  /* number of independent chains running in parallel (one per thread).      */ \
  double c = diff_cycles(t1, t2, 512, 1024)                                   \
             / ((double)gCfg.blocks * gCfg.threads);                          \
  if (c <= 0) c = 1e-12; /* fneg/fabs etc. are ~free; keep a positive floor */ \
  printf("%s,%s,%.9g\n", op_name, prec_name, c);                              \
}

#define BENCH_CVT(name, sink, op_label, prec_label) {                         \
  double t1 = timeit_ns([&](){ name<512>  <<<gCfg.blocks, gCfg.threads>>>(sink); });         \
  double t2 = timeit_ns([&](){ name<1024> <<<gCfg.blocks, gCfg.threads>>>(sink); });         \
  double c = diff_cycles(t1, t2, 512, 1024) * 0.5                             \
             / ((double)gCfg.blocks * gCfg.threads);                          \
  if (c <= 0) c = 1e-12;                                                      \
  printf("%s,%s,%.9g\n", op_label, prec_label, c);                           \
}

double bench_mma_chain_f16f32() {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_mma_f16f32<16,16,16, 256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_mma_f16f32<16,16,16, 512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512);
}
double bench_mma_chain_s8s32() {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_mma_s8s32<16,16,16, 256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_mma_s8s32<16,16,16, 512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512);
}
double bench_mma_chain_f16f16() {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_mma_f16f16<16,16,16, 256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_mma_f16f16<16,16,16, 512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512);
}
double bench_mma_chain_bf16f32() {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_mma_bf16f32<16,16,16, 256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_mma_bf16f32<16,16,16, 512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512);
}
double bench_mma_chain_tf32f32() {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_mma_tf32f32<256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_mma_tf32f32<512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512);
}
double bench_mma_chain_f64f64() {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_mma_f64f64<256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_mma_f64f64<512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512);
}

double bench_loada_marginal(double mma_per_op_cycles) {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_loada_f16<256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_loada_f16<512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  double per_iter = diff_cycles(t1, t2, 256, 512);
  return per_iter - mma_per_op_cycles;
}
double bench_loadb_marginal(double mma_per_op_cycles) {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  double t1 = timeit_ns([&](){ wmma_loadb_f16<256>  <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  double t2 = timeit_ns([&](){ wmma_loadb_f16<512> <<<gWmmaBlocks, gWmmaThreads>>>(d); });
  cudaFree(d);
  return diff_cycles(t1, t2, 256, 512) - mma_per_op_cycles;
}
double bench_loadc_marginal(double mma_per_op_cycles) {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  size_t c_bytes = (size_t)gWmmaBlocks * 256 * sizeof(float);
  float *c_dev; cudaMalloc(&c_dev, c_bytes); cudaMemset(c_dev, 0, c_bytes);
  double t1 = timeit_ns([&](){ wmma_loadc_f32<256>  <<<gWmmaBlocks, gWmmaThreads>>>(d, c_dev); });
  double t2 = timeit_ns([&](){ wmma_loadc_f32<512> <<<gWmmaBlocks, gWmmaThreads>>>(d, c_dev); });
  cudaFree(d); cudaFree(c_dev);
  return diff_cycles(t1, t2, 256, 512) - mma_per_op_cycles;
}
double bench_stored_marginal(double mma_per_op_cycles) {
  uint64_t *d; cudaMalloc(&d, sizeof(uint64_t));
  size_t d_bytes = (size_t)gWmmaBlocks * 256 * sizeof(float);
  float *d_dev; cudaMalloc(&d_dev, d_bytes);
  double t1 = timeit_ns([&](){ wmma_stored_f32<256>  <<<gWmmaBlocks, gWmmaThreads>>>(d, d_dev); });
  double t2 = timeit_ns([&](){ wmma_stored_f32<512> <<<gWmmaBlocks, gWmmaThreads>>>(d, d_dev); });
  cudaFree(d); cudaFree(d_dev);
  return diff_cycles(t1, t2, 256, 512) - mma_per_op_cycles;
}

// Wall-time (ns/GEMM) of a real dispatched WMMA GEMM for one scheme, at
// M=N=K=GN. Sets *ok=false (and returns -1) if the scheme is unsupported on
// this arch (launch failure). Used only as a reality-check gate; the printed
// per-tile cost stays the isolated-loop value.
static const int kGemmReps = 10;
template <typename AB, typename ACC, typename FRAG_AB,
          int MT, int NT, int KT,
          int BWM, int BWN, int WPT_M, int WPT_N, int BK_STAGE, bool IS_TF32>
double tensor_gemm_ns(int GN, bool *ok) {
  constexpr int BM = BWM * WPT_M * MT;
  constexpr int BN = BWN * WPT_N * NT;
  int M = (GN / BM) * BM, N = (GN / BN) * BN, K = (GN / KT) * KT;
  if (M == 0 || N == 0 || K == 0) { if (ok) *ok = false; return -1.0; }

  size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  AB *dA = nullptr, *dB = nullptr; ACC *dC = nullptr;
  if (cudaMalloc(&dA, aN * sizeof(AB)) != cudaSuccess ||
      cudaMalloc(&dB, bN * sizeof(AB)) != cudaSuccess ||
      cudaMalloc(&dC, cN * sizeof(ACC)) != cudaSuccess) {
    if (dA) cudaFree(dA); if (dB) cudaFree(dB); if (dC) cudaFree(dC);
    if (ok) *ok = false; return -1.0;
  }
  cudaMemset(dA, 0, aN * sizeof(AB));
  cudaMemset(dB, 0, bN * sizeof(AB));
  cudaMemset(dC, 0, cN * sizeof(ACC));

  dim3 grid(N / BN, M / BM);
  int threads = BWM * BWN * 32;
  auto launch = [&]() {
    tgemm<AB, ACC, FRAG_AB, MT, NT, KT, BWM, BWN, WPT_M, WPT_N, BK_STAGE,
          IS_TF32><<<grid, threads>>>(dA, dB, dC, M, N, K);
  };

  cudaGetLastError();                        // clear any stale error
  launch();                                  // warm / probe launch
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {                  // unsupported scheme (e.g. DMMA)
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (ok) *ok = false; return -1.0;
  }

  double ns = timeit_ns(launch, kGemmReps);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  if (ok) *ok = true;
  return ns;
}

// Wall-time (ns/GEMM) of the scalar-FP64 reference GEMM at M=N=K=GN.
double scalar_gemm_ns(int GN) {
  const int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  int M = (GN / BM) * BM, N = (GN / BN) * BN, K = (GN / BK) * BK;
  if (M == 0 || N == 0 || K == 0) return -1.0;
  size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (cudaMalloc(&dA, aN * 8) != cudaSuccess ||
      cudaMalloc(&dB, bN * 8) != cudaSuccess ||
      cudaMalloc(&dC, cN * 8) != cudaSuccess) {
    if (dA) cudaFree(dA); if (dB) cudaFree(dB); if (dC) cudaFree(dC);
    return -1.0;
  }
  cudaMemset(dA, 0, aN * 8); cudaMemset(dB, 0, bN * 8); cudaMemset(dC, 0, cN * 8);
  dim3 grid(N / BN, M / BM);
  int threads = (BM / TM) * (BN / TN);
  auto launch = [&]() {
    sgemm<BM, BN, BK, TM, TN><<<grid, threads>>>(dA, dB, dC, M, N, K);
  };
  cudaGetLastError();
  launch();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return -1.0;
  }
  double ns = timeit_ns(launch, kGemmReps);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return ns;
}

int main() {
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  // CUDA 13 removed cudaDeviceProp::clockRate; the attribute returns the same
  // kHz value on CUDA >= 4.0, so this is the one portable path.
  int clkKHz = 0; cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);
  gGhz = (double)clkKHz / 1e6;
  gCfg.blocks = prop.multiProcessorCount;
  gWmmaBlocks = prop.multiProcessorCount * 8; // many warps/SM → saturate TCs

  printf("# Generated by microbm.cu on %s\n", prop.name);
  printf("# native_arch=sm_%d%d\n", prop.major, prop.minor);
  printf("# scalar_types=double,float\n");
  printf("# matrix_types=bf16,double,float,half,tf32\n");

  size_t total = (size_t)gCfg.blocks * gCfg.threads;
  float *fout; cudaMalloc(&fout, total * sizeof(float));
  double *dout; cudaMalloc(&dout, total * sizeof(double));

    BENCH_SCALAR(fadd, float, "fadd", "float", fout);
  BENCH_SCALAR(fsub, float, "fsub", "float", fout);
  BENCH_SCALAR(fmul, float, "fmul", "float", fout);
  BENCH_SCALAR(ilp_fdiv, float, "fdiv", "float", fout);   // fix #9 shape
  BENCH_SCALAR(fneg, float, "fneg", "float", fout);
  BENCH_SCALAR(fabs, float, "fabs", "float", fout);
  BENCH_SCALAR(fcmp, float, "fcmp", "float", fout);
  BENCH_SCALAR(fma,  float, "fma",  "float", fout);
  BENCH_SCALAR(fma,  float, "fmuladd", "float", fout);  // alias
  BENCH_SCALAR(ilp_sqrt,  float, "sqrt",  "float", fout);  // fix #9 shape
  BENCH_SCALAR(ilp_rsqrt, float, "rsqrt", "float", fout);  // fix #9 shape
  BENCH_SCALAR(cbrt,  float, "cbrt",  "float", fout);
  BENCH_SCALAR(exp,   float, "exp",   "float", fout);
  BENCH_SCALAR(expm1, float, "expm1", "float", fout);
  BENCH_SCALAR(exp2,  float, "exp2",  "float", fout);
  BENCH_SCALAR(log,   float, "log",   "float", fout);
  BENCH_SCALAR(log1p, float, "log1p", "float", fout);
  BENCH_SCALAR(log2,  float, "log2",  "float", fout);
  BENCH_SCALAR(log10, float, "log10", "float", fout);
  BENCH_SCALAR(sin,   float, "sin",   "float", fout);
  BENCH_SCALAR(cos,   float, "cos",   "float", fout);
  BENCH_SCALAR(tan,   float, "tan",   "float", fout);
  BENCH_SCALAR(asin,  float, "asin",  "float", fout);
  BENCH_SCALAR(acos,  float, "acos",  "float", fout);
  BENCH_SCALAR(atan,  float, "atan",  "float", fout);
  BENCH_SCALAR(sinh,  float, "sinh",  "float", fout);
  BENCH_SCALAR(cosh,  float, "cosh",  "float", fout);
  BENCH_SCALAR(tanh,  float, "tanh",  "float", fout);
  BENCH_SCALAR(asinh, float, "asinh", "float", fout);
  BENCH_SCALAR(acosh, float, "acosh", "float", fout);
  BENCH_SCALAR(atanh, float, "atanh", "float", fout);
  BENCH_SCALAR(erf,    float, "erf",    "float", fout);
  BENCH_SCALAR(lgamma, float, "lgamma", "float", fout);
  BENCH_SCALAR(tgamma, float, "tgamma", "float", fout);
  BENCH_SCALAR(ceil,  float, "ceil",  "float", fout);
  BENCH_SCALAR(floor, float, "floor", "float", fout);
  BENCH_SCALAR(trunc, float, "trunc", "float", fout);
  BENCH_SCALAR(round, float, "round", "float", fout);
  BENCH_SCALAR(rint,  float, "rint",  "float", fout);
  BENCH_SCALAR(atan2,     float, "atan2",     "float", fout);
  BENCH_SCALAR(pow,       float, "pow",       "float", fout);
  BENCH_SCALAR(pow,       float, "powi",      "float", fout);  // alias
  BENCH_SCALAR(hypot,     float, "hypot",     "float", fout);
  BENCH_SCALAR(fmod,      float, "fmod",      "float", fout);
  BENCH_SCALAR(remainder, float, "remainder", "float", fout);
  BENCH_SCALAR(fdim,      float, "fdim",      "float", fout);
  BENCH_SCALAR(maxnum,    float, "maxnum",    "float", fout);
  BENCH_SCALAR(minnum,    float, "minnum",    "float", fout);
  BENCH_SCALAR(copysign,  float, "copysign",  "float", fout);

    BENCH_SCALAR(fadd, double, "fadd", "double", dout);
  BENCH_SCALAR(fsub, double, "fsub", "double", dout);
  BENCH_SCALAR(fmul, double, "fmul", "double", dout);
  BENCH_SCALAR(ilp_fdiv, double, "fdiv", "double", dout);  // fix #9 shape
  BENCH_SCALAR(fneg, double, "fneg", "double", dout);
  BENCH_SCALAR(fabs, double, "fabs", "double", dout);
  BENCH_SCALAR(fcmp, double, "fcmp", "double", dout);
  BENCH_SCALAR(fma,  double, "fma",  "double", dout);
  BENCH_SCALAR(fma,  double, "fmuladd", "double", dout);
  BENCH_SCALAR(ilp_sqrt,  double, "sqrt",  "double", dout); // fix #9 shape
  BENCH_SCALAR(ilp_rsqrt, double, "rsqrt", "double", dout); // fix #9 shape
  BENCH_SCALAR(cbrt,  double, "cbrt",  "double", dout);
  BENCH_SCALAR(exp,   double, "exp",   "double", dout);
  BENCH_SCALAR(expm1, double, "expm1", "double", dout);
  BENCH_SCALAR(exp2,  double, "exp2",  "double", dout);
  BENCH_SCALAR(log,   double, "log",   "double", dout);
  BENCH_SCALAR(log1p, double, "log1p", "double", dout);
  BENCH_SCALAR(log2,  double, "log2",  "double", dout);
  BENCH_SCALAR(log10, double, "log10", "double", dout);
  BENCH_SCALAR(sin,   double, "sin",   "double", dout);
  BENCH_SCALAR(cos,   double, "cos",   "double", dout);
  BENCH_SCALAR(tan,   double, "tan",   "double", dout);
  BENCH_SCALAR(asin,  double, "asin",  "double", dout);
  BENCH_SCALAR(acos,  double, "acos",  "double", dout);
  BENCH_SCALAR(atan,  double, "atan",  "double", dout);
  BENCH_SCALAR(sinh,  double, "sinh",  "double", dout);
  BENCH_SCALAR(cosh,  double, "cosh",  "double", dout);
  BENCH_SCALAR(tanh,  double, "tanh",  "double", dout);
  BENCH_SCALAR(asinh, double, "asinh", "double", dout);
  BENCH_SCALAR(acosh, double, "acosh", "double", dout);
  BENCH_SCALAR(atanh, double, "atanh", "double", dout);
  BENCH_SCALAR(erf,    double, "erf",    "double", dout);
  BENCH_SCALAR(lgamma, double, "lgamma", "double", dout);
  BENCH_SCALAR(tgamma, double, "tgamma", "double", dout);
  BENCH_SCALAR(ceil,  double, "ceil",  "double", dout);
  BENCH_SCALAR(floor, double, "floor", "double", dout);
  BENCH_SCALAR(trunc, double, "trunc", "double", dout);
  BENCH_SCALAR(round, double, "round", "double", dout);
  BENCH_SCALAR(rint,  double, "rint",  "double", dout);
  BENCH_SCALAR(atan2,     double, "atan2",     "double", dout);
  BENCH_SCALAR(pow,       double, "pow",       "double", dout);
  BENCH_SCALAR(pow,       double, "powi",      "double", dout);
  BENCH_SCALAR(hypot,     double, "hypot",     "double", dout);
  BENCH_SCALAR(fmod,      double, "fmod",      "double", dout);
  BENCH_SCALAR(remainder, double, "remainder", "double", dout);
  BENCH_SCALAR(fdim,      double, "fdim",      "double", dout);
  BENCH_SCALAR(maxnum,    double, "maxnum",    "double", dout);
  BENCH_SCALAR(minnum,    double, "minnum",    "double", dout);
  BENCH_SCALAR(copysign,  double, "copysign",  "double", dout);

    BENCH_CVT(cvt_f32_f64, fout, "fpext_float_to_double",  "float");
  BENCH_CVT(cvt_f64_f32, dout, "fptrunc_double_to_float", "double");

  cudaFree(fout); cudaFree(dout);

  // Isolated-loop per-tile MMA costs: the primary source of the printed
  // wmma_mma_* rows. They preserve the inter-scheme compute spread the DP
  // solver needs (s8 < f16/bf16 < tf32) and mma_f16f32 also prices the marginal
  // load/store rows below (which subtract one raw machine-wide MMA).
  double mma_f16f32  = bench_mma_chain_f16f32();
  double mma_s8s32   = bench_mma_chain_s8s32();
  double mma_f16f16  = bench_mma_chain_f16f16();
  double mma_bf16f32 = bench_mma_chain_bf16f32();
  double mma_tf32f32 = bench_mma_chain_tf32f32();
  double ww = gWmmaWarps();

  // Reality-check gate (see tgemm): time each scheme's real WMMA GEMM and a
  // scalar-FP64 GEMM of the same shape; rel = t_scheme / t_scalar. The gate is
  // strictly an existence check: a scheme is vestigial iff rel >= 1.0, and its
  // row is omitted (the loader tolerates absence) because its isolated cost
  // would mint a zero-error, near-free candidate. How much faster a scheme must
  // be to win belongs to the DP's cost comparison, not to candidate deletion.
  const int GN = 2048;
  const double REL_OMIT = 1.0;
  double t_scalar = scalar_gemm_ns(GN);
  auto emit_gated = [&](const char *name, const char *prec,
                        double iso_per_tile, double t_scheme, bool ran) {
    if (!ran) {
      fprintf(stderr, "[microbm] %s,%s: WMMA GEMM unsupported on this arch; "
                      "omitting row\n", name, prec);
      return;
    }
    double rel = (t_scalar > 0) ? t_scheme / t_scalar : 0.0;
    if (rel >= REL_OMIT) {
      fprintf(stderr, "[microbm] %s,%s: real GEMM rel=%.3f >= %.2f — not faster "
                      "than scalar FP64; vestigial tensor path, omitting row\n",
              name, prec, rel, REL_OMIT);
      return;
    }
    printf("%s,%s,%.9g\n", name, prec, iso_per_tile / ww);
  };
  { bool ok;
    double t;
    t = tensor_gemm_ns<__half, float, __half,
                       16, 16, 16, 2, 4, 4, 2, 32, false>(GN, &ok);
    emit_gated("wmma_mma_m16n16k16", "f16_f32", mma_f16f32, t, ok);
    t = tensor_gemm_ns<signed char, int, signed char,
                       16, 16, 16, 2, 4, 4, 2, 32, false>(GN, &ok);
    emit_gated("wmma_mma_m16n16k16", "s8_s32", mma_s8s32, t, ok);
    t = tensor_gemm_ns<__half, __half, __half,
                       16, 16, 16, 2, 4, 4, 2, 32, false>(GN, &ok);
    emit_gated("wmma_mma_m16n16k16", "f16_f16", mma_f16f16, t, ok);
    t = tensor_gemm_ns<__nv_bfloat16, float, __nv_bfloat16,
                       16, 16, 16, 2, 4, 4, 2, 32, false>(GN, &ok);
    emit_gated("wmma_mma_m16n16k16", "bf16_f32", mma_bf16f32, t, ok);
    t = tensor_gemm_ns<float, float, nvcuda::wmma::precision::tf32,
                       16, 16, 8, 2, 4, 4, 2, 32, true>(GN, &ok);
    emit_gated("wmma_mma_m16n16k8", "tf32_f32", mma_tf32f32, t, ok);

    // f64 (m8n8k4 DMMA): isolated primary, gated by the real DMMA GEMM (omitted
    // where rel >= 1, e.g. sm_120 and B300; kept on A100/H100). Only pay the
    // isolated f64 bench when the DMMA path actually ran.
    bool f64ok = false;
    t = tensor_gemm_ns<double, double, double,
                       8, 8, 4, 4, 4, 2, 2, 16, false>(GN, &f64ok);
    double mma_f64f64 = f64ok ? bench_mma_chain_f64f64() : 0.0;
    emit_gated("wmma_mma_m8n8k4", "f64_f64", mma_f64f64, t, f64ok);
  }

  // Marginal load/store: subtract a single MMA from each.
  double load_a = bench_loada_marginal(mma_f16f32);
  double load_b = bench_loadb_marginal(mma_f16f32);
  double load_c = bench_loadc_marginal(mma_f16f32);
  double store_d = bench_stored_marginal(mma_f16f32);
  if (load_a < 0) load_a = 0;
  if (load_b < 0) load_b = 0;
  if (load_c < 0) load_c = 0;
  if (store_d < 0) store_d = 0;

  // Per-tile marginal load/store (same per-warp normalization as the MMA).
  printf("wmma_load_a_m16n16k16,f16,%.9g\n",  load_a / ww);
  printf("wmma_load_b_m16n16k16,f16,%.9g\n",  load_b / ww);
  printf("wmma_load_c_m16n16k16,f32,%.9g\n",  load_c / ww);
  printf("wmma_store_d_m16n16k16,f32,%.9g\n", store_d / ww);

  return 0;
}
