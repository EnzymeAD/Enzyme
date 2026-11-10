// Poseidon GPU cost-model profiler for AMD GPUs: the per-op rows microbm.cu
// measures on NVIDIA, under the same names, in the same unit, by the same
// protocol.
//
// Methodology (microbm.cu's):
//   1. Each op gets a kernel doing N dependent ops per thread (private register)
//   2. Launched on <CUs * 2 blocks, 1024 threads>: 32 waves per CU, the wave
//      slots of its four SIMDs, so the loop is throughput-bound, not
//      latency-bound
//   3. Time T(N) and T(2N) via hipEvent
//   4. cycles = ((T(2N) - T(N)) * 2 / chain_2N) * clock_rate_ghz * 2
//
// Each loop body is the instruction its row names. An op the ISA has as one
// instruction is written as that instruction in amdgcn inline assembly, so the
// compiler can neither fold, contract nor reassociate the chain. An op the
// backend lowers to a sequence (division, round, the math library) is the IEEE
// lowering a HIP device compile emits, behind an empty asm the value passes
// through, so no identity of the op (round(round(x)) = round(x)) can shorten
// the chain. The math library is ROCm's ocml, as libdevice is on NVIDIA.
//
// The tensor-core tile rows are not written: in-kernel raises are NVPTX-only,
// so no compile for this device reads them.
//
// Output: cm_gfx<NNN>.csv rows on stdout in Poseidon's loader format
//   # native_arch=gfx<NNN>
//   # scalar_types=...
//   # matrix_types=...
//   opcode,precision,cost
//
// Build:  clang++ -x hip --offload-arch=gfx942 -O3 -std=c++17 microbm_hip.hip \
//           -lamdhip64 -o microbm_hip
// Run:    HIP_VISIBLE_DEVICES=<id> ./microbm_hip > cm_gfx942.csv
//         ./microbm_hip --device    prints "<name>\t<gcnArchName>" and exits

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>

// ocml entry points, declared explicitly as microbm.cu declares libdevice's:
// the HIP headers map some of these names to builtins instead. Each is the
// precise form.
extern "C" {
__device__ float  __ocml_sqrt_f32(float);
__device__ double __ocml_sqrt_f64(double);
__device__ float  __ocml_rsqrt_f32(float);
__device__ double __ocml_rsqrt_f64(double);
__device__ float  __ocml_cbrt_f32(float);
__device__ double __ocml_cbrt_f64(double);
__device__ float  __ocml_exp_f32(float);
__device__ double __ocml_exp_f64(double);
__device__ float  __ocml_expm1_f32(float);
__device__ double __ocml_expm1_f64(double);
__device__ float  __ocml_exp2_f32(float);
__device__ double __ocml_exp2_f64(double);
__device__ float  __ocml_log_f32(float);
__device__ double __ocml_log_f64(double);
__device__ float  __ocml_log1p_f32(float);
__device__ double __ocml_log1p_f64(double);
__device__ float  __ocml_log2_f32(float);
__device__ double __ocml_log2_f64(double);
__device__ float  __ocml_log10_f32(float);
__device__ double __ocml_log10_f64(double);
__device__ float  __ocml_sin_f32(float);
__device__ double __ocml_sin_f64(double);
__device__ float  __ocml_cos_f32(float);
__device__ double __ocml_cos_f64(double);
__device__ float  __ocml_tan_f32(float);
__device__ double __ocml_tan_f64(double);
__device__ float  __ocml_asin_f32(float);
__device__ double __ocml_asin_f64(double);
__device__ float  __ocml_acos_f32(float);
__device__ double __ocml_acos_f64(double);
__device__ float  __ocml_atan_f32(float);
__device__ double __ocml_atan_f64(double);
__device__ float  __ocml_atan2_f32(float, float);
__device__ double __ocml_atan2_f64(double, double);
__device__ float  __ocml_sinh_f32(float);
__device__ double __ocml_sinh_f64(double);
__device__ float  __ocml_cosh_f32(float);
__device__ double __ocml_cosh_f64(double);
__device__ float  __ocml_tanh_f32(float);
__device__ double __ocml_tanh_f64(double);
__device__ float  __ocml_asinh_f32(float);
__device__ double __ocml_asinh_f64(double);
__device__ float  __ocml_acosh_f32(float);
__device__ double __ocml_acosh_f64(double);
__device__ float  __ocml_atanh_f32(float);
__device__ double __ocml_atanh_f64(double);
__device__ float  __ocml_pow_f32(float, float);
__device__ double __ocml_pow_f64(double, double);
__device__ float  __ocml_hypot_f32(float, float);
__device__ double __ocml_hypot_f64(double, double);
__device__ float  __ocml_erf_f32(float);
__device__ double __ocml_erf_f64(double);
__device__ float  __ocml_lgamma_f32(float);
__device__ double __ocml_lgamma_f64(double);
__device__ float  __ocml_tgamma_f32(float);
__device__ double __ocml_tgamma_f64(double);
__device__ float  __ocml_fmod_f32(float, float);
__device__ double __ocml_fmod_f64(double, double);
__device__ float  __ocml_remainder_f32(float, float);
__device__ double __ocml_remainder_f64(double, double);
__device__ float  __ocml_fdim_f32(float, float);
__device__ double __ocml_fdim_f64(double, double);
__device__ float  __ocml_round_f32(float);
__device__ double __ocml_round_f64(double);
}

#define CHAIN32(stmt) stmt stmt stmt stmt  stmt stmt stmt stmt \
                      stmt stmt stmt stmt  stmt stmt stmt stmt \
                      stmt stmt stmt stmt  stmt stmt stmt stmt \
                      stmt stmt stmt stmt  stmt stmt stmt stmt
#define UNROLL 32
// The same, as one asm statement: the backend pads every asm statement it
// cannot see into with a wait state, which a dependent chain of ordinary VALU
// instructions does not need, so the chain is handed over whole.
#define ASM32(insn) CHAIN32(insn "\n\t")

// The value passes through an empty asm: no instruction, and nothing known
// about it afterwards.
#define OPAQUE(v) asm volatile("" : "+v"(v))

// One instruction per op on the chain value %0. Its constants are registers
// (%1, %2): the VOP3 encoding the f64 ops use takes no literal on gfx9.
#define K_ASM(name, T, insn, c1, c2)                                          \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    T x = (T)(c1), y = (T)(c2);                                               \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      asm volatile(ASM32(insn) : "+v"(a) : "v"(x), "v"(y));                   \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

// The sign-bit ops act on the 32-bit word that holds the sign: the value
// itself in f32, the high word of an f64, which is all the backend touches.
#define K_ASM_SIGN(name, T, insn, m1, m2)                                     \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    uint64_t bits = 0;                                                        \
    memcpy(&bits, &a, sizeof(T));                                             \
    uint32_t w = (uint32_t)(bits >> (8 * sizeof(T) - 32));                    \
    uint32_t x = (m1), y = (m2);                                              \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      asm volatile(ASM32(insn) : "+v"(w) : "v"(x), "v"(y));                   \
    }                                                                         \
    bits = (bits & ~(0xffffffffull << (8 * sizeof(T) - 32))) |                \
           ((uint64_t)w << (8 * sizeof(T) - 32));                             \
    memcpy(&a, &bits, sizeof(T));                                             \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_LIBM_UN(name, T, libfn)                                             \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) { CHAIN32(OPAQUE(a); a = libfn(a);) } \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_LIBM_BIN(name, T, libfn, other)                                     \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    T b = (T)(other);                                                         \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      CHAIN32(OPAQUE(a); OPAQUE(b); a = libfn(a, b);)                         \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

// Expensive-op row shape (microbm.cu's fix #9): sqrt/rsqrt/div measured as 8
// independent lanes per thread, the shape Poseidon's materialized kernels
// execute them in, with the same total op count, launch and cost arithmetic.
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
      for (int j = 0; j < 8; ++j) {                                            \
        OPAQUE(v[j]);                                                          \
        v[j] = libfn(v[j]);                                                    \
      }                                                                        \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j) s += v[j];                                     \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

#define K_FDIV_ILP(name, T, divisor)                                           \
  template <int N>                                                             \
  __global__ void name##_##T(T *out) {                                         \
    T v[8];                                                                    \
    T d = (T)(divisor);                                                        \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j)                                                \
      v[j] = (T)((threadIdx.x + blockIdx.x * blockDim.x + j) * 0.0001) + (T)1.5;\
    _Pragma("unroll 1")                                                        \
    for (int i = 0; i < N / 8; ++i) {                                          \
      OPAQUE(d);                                                               \
      _Pragma("unroll")                                                        \
      for (int j = 0; j < 8; ++j) v[j] = v[j] / d;                             \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll")                                                          \
    for (int j = 0; j < 8; ++j) s += v[j];                                     \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

// A regime split's condition and its select: fcmp une + select, lowered to
// v_cmp + v_cndmask (one per 32-bit half).
#define K_FCMP(name, T, alt)                                                  \
  template <int N>                                                            \
  __global__ void name##_##T(T *out) {                                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;     \
    T c = (T)(alt);                                                           \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      OPAQUE(c);                                                              \
      CHAIN32(a = (a != c) ? a : c;)                                          \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

#define K_CVT(name, srcT, dstT, cvt_fwd, cvt_bwd)                             \
  template <int N>                                                            \
  __global__ void name(srcT *out) {                                           \
    srcT a = (srcT)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) +       \
             (srcT)1.5;                                                       \
    dstT b = (dstT)0;                                                         \
    _Pragma("unroll 1")                                                       \
    for (int i = 0; i < N / UNROLL; ++i) {                                    \
      asm volatile(ASM32(cvt_fwd " %1, %0\n\t" cvt_bwd " %0, %1")             \
                   : "+v"(a), "+v"(b));                                       \
    }                                                                         \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                           \
  }

K_ASM(fadd, float, "v_add_f32 %0, %0, %1",    1.0f, 0.0f)
K_ASM(fsub, float, "v_sub_f32 %0, %0, %1",    0.5f, 0.0f)
K_ASM(fmul, float, "v_mul_f32 %0, %0, %1",    1.0000001f, 0.0f)
K_FDIV_ILP(ilp_fdiv, float, 1.0000001f)
K_ASM_SIGN(fneg, float, "v_xor_b32 %0, %1, %0", 0x80000000u, 0u)
K_ASM_SIGN(fabs, float, "v_and_b32 %0, %1, %0", 0x7fffffffu, 0u)
K_FCMP   (fcmp, float, 1.0f)
K_ASM(fma,  float, "v_fma_f32 %0, %0, %1, %2", 1.0001f, 0.0f)
K_LIBM_UN_ILP(ilp_sqrt,  float, __ocml_sqrt_f32)
K_LIBM_UN_ILP(ilp_rsqrt, float, __ocml_rsqrt_f32)
K_LIBM_UN(cbrt,   float, __ocml_cbrt_f32)
K_LIBM_UN(exp,    float, __ocml_exp_f32)
K_LIBM_UN(expm1,  float, __ocml_expm1_f32)
K_LIBM_UN(exp2,   float, __ocml_exp2_f32)
K_LIBM_UN(log,    float, __ocml_log_f32)
K_LIBM_UN(log1p,  float, __ocml_log1p_f32)
K_LIBM_UN(log2,   float, __ocml_log2_f32)
K_LIBM_UN(log10,  float, __ocml_log10_f32)
K_LIBM_UN(sin,    float, __ocml_sin_f32)
K_LIBM_UN(cos,    float, __ocml_cos_f32)
K_LIBM_UN(tan,    float, __ocml_tan_f32)
K_LIBM_UN(asin,   float, __ocml_asin_f32)
K_LIBM_UN(acos,   float, __ocml_acos_f32)
K_LIBM_UN(atan,   float, __ocml_atan_f32)
K_LIBM_UN(sinh,   float, __ocml_sinh_f32)
K_LIBM_UN(cosh,   float, __ocml_cosh_f32)
K_LIBM_UN(tanh,   float, __ocml_tanh_f32)
K_LIBM_UN(asinh,  float, __ocml_asinh_f32)
K_LIBM_UN(acosh,  float, __ocml_acosh_f32)
K_LIBM_UN(atanh,  float, __ocml_atanh_f32)
K_LIBM_UN(erf,    float, __ocml_erf_f32)
K_LIBM_UN(lgamma, float, __ocml_lgamma_f32)
K_LIBM_UN(tgamma, float, __ocml_tgamma_f32)
K_ASM(ceil,  float, "v_ceil_f32 %0, %0",  0.0f, 0.0f)
K_ASM(floor, float, "v_floor_f32 %0, %0", 0.0f, 0.0f)
K_ASM(trunc, float, "v_trunc_f32 %0, %0", 0.0f, 0.0f)
K_LIBM_UN(round,  float, __ocml_round_f32)
K_ASM(rint,  float, "v_rndne_f32 %0, %0", 0.0f, 0.0f)
K_LIBM_BIN(atan2,     float, __ocml_atan2_f32,     1.0001f)
K_LIBM_BIN(pow,       float, __ocml_pow_f32,       1.0001f)
K_LIBM_BIN(hypot,     float, __ocml_hypot_f32,     1.0001f)
K_LIBM_BIN(fmod,      float, __ocml_fmod_f32,      1.0001f)
K_LIBM_BIN(remainder, float, __ocml_remainder_f32, 1.0001f)
K_LIBM_BIN(fdim,      float, __ocml_fdim_f32,      0.5f)
K_ASM(maxnum, float, "v_max_f32 %0, %0, %1", 1.0f, 0.0f)
K_ASM(minnum, float, "v_min_f32 %0, %0, %1", 1.0f, 0.0f)
// copysign(a, -1.0): v_bfi_b32 takes the magnitude bits from a, the sign from
// the second operand.
K_ASM_SIGN(copysign, float, "v_bfi_b32 %0, %1, %0, %2", 0x7fffffffu,
           0xbf800000u)

K_ASM(fadd, double, "v_add_f64 %0, %0, %1",    1.0, 0.0)
K_ASM(fsub, double, "v_add_f64 %0, %0, -%1",   0.5, 0.0)
K_ASM(fmul, double, "v_mul_f64 %0, %0, %1",    1.0000000000000002, 0.0)
K_FDIV_ILP(ilp_fdiv, double, 1.0000000000000002)
K_ASM_SIGN(fneg, double, "v_xor_b32 %0, %1, %0", 0x80000000u, 0u)
K_ASM_SIGN(fabs, double, "v_and_b32 %0, %1, %0", 0x7fffffffu, 0u)
K_FCMP   (fcmp, double, 1.0)
K_ASM(fma,  double, "v_fma_f64 %0, %0, %1, %2", 1.0001, 0.0)
K_LIBM_UN_ILP(ilp_sqrt,  double, __ocml_sqrt_f64)
K_LIBM_UN_ILP(ilp_rsqrt, double, __ocml_rsqrt_f64)
K_LIBM_UN(cbrt,   double, __ocml_cbrt_f64)
K_LIBM_UN(exp,    double, __ocml_exp_f64)
K_LIBM_UN(expm1,  double, __ocml_expm1_f64)
K_LIBM_UN(exp2,   double, __ocml_exp2_f64)
K_LIBM_UN(log,    double, __ocml_log_f64)
K_LIBM_UN(log1p,  double, __ocml_log1p_f64)
K_LIBM_UN(log2,   double, __ocml_log2_f64)
K_LIBM_UN(log10,  double, __ocml_log10_f64)
K_LIBM_UN(sin,    double, __ocml_sin_f64)
K_LIBM_UN(cos,    double, __ocml_cos_f64)
K_LIBM_UN(tan,    double, __ocml_tan_f64)
K_LIBM_UN(asin,   double, __ocml_asin_f64)
K_LIBM_UN(acos,   double, __ocml_acos_f64)
K_LIBM_UN(atan,   double, __ocml_atan_f64)
K_LIBM_UN(sinh,   double, __ocml_sinh_f64)
K_LIBM_UN(cosh,   double, __ocml_cosh_f64)
K_LIBM_UN(tanh,   double, __ocml_tanh_f64)
K_LIBM_UN(asinh,  double, __ocml_asinh_f64)
K_LIBM_UN(acosh,  double, __ocml_acosh_f64)
K_LIBM_UN(atanh,  double, __ocml_atanh_f64)
K_LIBM_UN(erf,    double, __ocml_erf_f64)
K_LIBM_UN(lgamma, double, __ocml_lgamma_f64)
K_LIBM_UN(tgamma, double, __ocml_tgamma_f64)
K_ASM(ceil,  double, "v_ceil_f64 %0, %0",  0.0, 0.0)
K_ASM(floor, double, "v_floor_f64 %0, %0", 0.0, 0.0)
K_ASM(trunc, double, "v_trunc_f64 %0, %0", 0.0, 0.0)
K_LIBM_UN(round,  double, __ocml_round_f64)
K_ASM(rint,  double, "v_rndne_f64 %0, %0", 0.0, 0.0)
K_LIBM_BIN(atan2,     double, __ocml_atan2_f64,     1.0001)
K_LIBM_BIN(pow,       double, __ocml_pow_f64,       1.0001)
K_LIBM_BIN(hypot,     double, __ocml_hypot_f64,     1.0001)
K_LIBM_BIN(fmod,      double, __ocml_fmod_f64,      1.0001)
K_LIBM_BIN(remainder, double, __ocml_remainder_f64, 1.0001)
K_LIBM_BIN(fdim,      double, __ocml_fdim_f64,      0.5)
K_ASM(maxnum, double, "v_max_f64 %0, %0, %1", 1.0, 0.0)
K_ASM(minnum, double, "v_min_f64 %0, %0, %1", 1.0, 0.0)
K_ASM_SIGN(copysign, double, "v_bfi_b32 %0, %1, %0, %2", 0x7fffffffu,
           0xbff00000u)

K_CVT(cvt_f32_f64, float, double, "v_cvt_f64_f32", "v_cvt_f32_f64")
K_CVT(cvt_f64_f32, double, float, "v_cvt_f32_f64", "v_cvt_f64_f32")

struct Config { int blocks; int threads; int N1; int N2; int reps; };
static Config gCfg = {0, 1024, 512, 1024, 50};

// MI300A's clock settles slowly after a change of load: a cheap chain timed
// for about a millisecond right after expensive ones varied by up to 31%
// across runs (fadd and fneg f64, the first cheap rows after the f32 libm
// rows). Each kernel therefore runs for kWarmMs before it is timed, a batch
// lasts at least kBatchMs, and the time is the median of kBatches batches.
constexpr int kBatches = 7;
constexpr double kWarmMs = 20.0, kBatchMs = 5.0;

template <typename K>
double timeit_ns(K kernel) {
  hipEvent_t e0, e1;
  (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
  auto batch = [&](int reps) {
    (void)hipEventRecord(e0);
    for (int i = 0; i < reps; ++i) kernel();
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms; (void)hipEventElapsedTime(&ms, e0, e1);
    return (double)ms;
  };
  int reps = gCfg.reps;
  double ms = batch(reps);
  for (double warm = ms; warm < kWarmMs; warm += ms)
    ms = batch(reps);
  if (ms < kBatchMs)
    reps = (int)std::ceil(reps * kBatchMs / std::max(ms, 1e-3));
  double t[kBatches];
  for (int b = 0; b < kBatches; ++b)
    t[b] = batch(reps) * 1e6 / reps;
  (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
  std::sort(t, t + kBatches);
  return t[kBatches / 2];
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
  if (c <= 0) c = 1e-12; /* keep a positive floor, as microbm.cu does */      \
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

int main(int argc, char **argv) {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    fprintf(stderr, "[microbm_hip] no HIP device\n");
    return 1;
  }
  // gcnArchName carries the target features ("gfx942:sramecc+:xnack-"); the
  // processor alone names the cost model.
  char arch[sizeof(prop.gcnArchName)];
  strncpy(arch, prop.gcnArchName, sizeof(arch));
  arch[sizeof(arch) - 1] = 0;
  if (char *colon = strchr(arch, ':')) *colon = 0;
  if (argc > 1 && !strcmp(argv[1], "--device")) {
    printf("%s\t%s\n", prop.name, arch);
    return 0;
  }

  int clkKHz = 0;
  (void)hipDeviceGetAttribute(&clkKHz, hipDeviceAttributeClockRate, 0);
  gGhz = (double)clkKHz / 1e6;
  gCfg.blocks = prop.multiProcessorCount * 2;

  printf("# Generated by microbm_hip.hip on %s\n", prop.name);
  printf("# native_arch=%s\n", arch);
  printf("# scalar_types=double,float\n");
  printf("# matrix_types=bf16,double,float,half\n");

  size_t total = (size_t)gCfg.blocks * gCfg.threads;
  float *fout; (void)hipMalloc(&fout, total * sizeof(float));
  double *dout; (void)hipMalloc(&dout, total * sizeof(double));

  // Bring the GPU to its sustained clock before the first row.
  for (int i = 0; i < 2000; ++i)
    fma_float<1024><<<gCfg.blocks, gCfg.threads>>>(fout);
  (void)hipDeviceSynchronize();

  BENCH_SCALAR(fadd, float, "fadd", "float", fout);
  BENCH_SCALAR(fsub, float, "fsub", "float", fout);
  BENCH_SCALAR(fmul, float, "fmul", "float", fout);
  BENCH_SCALAR(ilp_fdiv, float, "fdiv", "float", fout);
  BENCH_SCALAR(fneg, float, "fneg", "float", fout);
  BENCH_SCALAR(fabs, float, "fabs", "float", fout);
  BENCH_SCALAR(fcmp, float, "fcmp", "float", fout);
  BENCH_SCALAR(fma,  float, "fma",  "float", fout);
  BENCH_SCALAR(fma,  float, "fmuladd", "float", fout);  // alias
  BENCH_SCALAR(ilp_sqrt,  float, "sqrt",  "float", fout);
  BENCH_SCALAR(ilp_rsqrt, float, "rsqrt", "float", fout);
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
  BENCH_SCALAR(ilp_fdiv, double, "fdiv", "double", dout);
  BENCH_SCALAR(fneg, double, "fneg", "double", dout);
  BENCH_SCALAR(fabs, double, "fabs", "double", dout);
  BENCH_SCALAR(fcmp, double, "fcmp", "double", dout);
  BENCH_SCALAR(fma,  double, "fma",  "double", dout);
  BENCH_SCALAR(fma,  double, "fmuladd", "double", dout);
  BENCH_SCALAR(ilp_sqrt,  double, "sqrt",  "double", dout);
  BENCH_SCALAR(ilp_rsqrt, double, "rsqrt", "double", dout);
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

  (void)hipFree(fout); (void)hipFree(dout);
  return 0;
}
