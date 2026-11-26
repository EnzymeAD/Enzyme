//===- ops_gpu.cu - per-operation costs on NVIDIA and AMD GPUs ------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The per-op rows of a GPU cost model, measured on the GPU it runs on: built
// as CUDA (poseidon-calibrate-ops-cuda) and as HIP
// (poseidon-calibrate-ops-hip).
//
// Each op gets a kernel doing N dependent ops per thread in a register,
// launched on enough threads to saturate the device and timed at N and 2N:
//   cycles = (T(2N) - T(N)) / N * clock_rate_ghz * 2 / threads
// (the trailing *2 is an FMA factor the CSV's unit includes).
//
// Each loop body is the instruction its row names. On NVIDIA that is PTX, or
// libdevice for the math library. On AMD an op the ISA has as one instruction
// is that instruction in amdgcn inline assembly, so the compiler can neither
// fold, contract nor reassociate the chain; an op the backend lowers to a
// sequence (division, round, the math library, which is ROCm's ocml) is the
// IEEE lowering a HIP device compile emits, behind an empty asm the value
// passes through, so no identity of the op (round(round(x)) = round(x)) can
// shorten the chain.
//
// On NVIDIA the tensor-core tile rows follow; in-kernel raises are NVPTX-only,
// so no compile for an AMD GPU reads them.
//
// Rows go to stdout in Poseidon's loader format. With --device the tool prints
// "<name>\t<arch>" of device 0 and exits.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/DeviceApi.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifndef __HIP__
#include <mma.h>
#endif

// The math library entry points, declared explicitly so that no math.h
// fast-math lowering applies and no header maps a name to a builtin. Each is
// the precise form.
#ifdef __HIP__
#define MATHF(fn) __ocml_##fn##_f32
#define MATHD(fn) __ocml_##fn##_f64
#else
#define MATHF(fn) __nv_##fn##f
#define MATHD(fn) __nv_##fn
#endif

#define LIBM_UNARY(X)                                                          \
  X(cbrt);                                                                     \
  X(exp);                                                                      \
  X(expm1);                                                                    \
  X(exp2);                                                                     \
  X(log);                                                                      \
  X(log1p);                                                                    \
  X(log2);                                                                     \
  X(log10);                                                                    \
  X(sin);                                                                      \
  X(cos);                                                                      \
  X(tan);                                                                      \
  X(asin);                                                                     \
  X(acos);                                                                     \
  X(atan);                                                                     \
  X(sinh);                                                                     \
  X(cosh);                                                                     \
  X(tanh);                                                                     \
  X(asinh);                                                                    \
  X(acosh);                                                                    \
  X(atanh);                                                                    \
  X(erf);                                                                      \
  X(lgamma);                                                                   \
  X(tgamma);
// The rounding ops AMD has no single instruction for.
#ifdef __HIP__
#define VENDOR_LIBM_UNARY(X) X(round);
#else
#define VENDOR_LIBM_UNARY(X)                                                   \
  X(ceil);                                                                     \
  X(floor);                                                                    \
  X(trunc);                                                                    \
  X(round);                                                                    \
  X(rint);
#endif
// (op, second operand).
#define LIBM_BINARY(X)                                                         \
  X(atan2, 1.0001);                                                            \
  X(pow, 1.0001);                                                              \
  X(hypot, 1.0001);                                                            \
  X(fmod, 1.0001);                                                             \
  X(remainder, 1.0001);                                                        \
  X(fdim, 0.5);

#define DECLARE_UNARY(fn)                                                      \
  __device__ float MATHF(fn)(float);                                           \
  __device__ double MATHD(fn)(double)
#define DECLARE_BINARY(fn, other)                                              \
  __device__ float MATHF(fn)(float, float);                                    \
  __device__ double MATHD(fn)(double, double)
extern "C" {
LIBM_UNARY(DECLARE_UNARY)
VENDOR_LIBM_UNARY(DECLARE_UNARY)
DECLARE_UNARY(sqrt);
LIBM_BINARY(DECLARE_BINARY)
#ifndef __HIP__
DECLARE_BINARY(copysign, 0);
__device__ float __nv_fmaf(float, float, float);
__device__ double __nv_fma(double, double, double);
#endif
}

#define CHAIN32(stmt)                                                          \
  stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt   \
      stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt stmt    \
          stmt stmt stmt
#define UNROLL 32

// On AMD the value passes through an empty asm: no instruction, and nothing
// known about it afterwards. NVIDIA's libdevice calls need no such barrier.
#ifdef __HIP__
#define OPAQUE(v) asm volatile("" : "+v"(v))
#else
#define OPAQUE(v)
#endif

#define K_LIBM_UN(name, T, libfn)                                              \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(OPAQUE(a); a = libfn(a);)                                        \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_LIBM_BIN(name, T, libfn, other)                                      \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    T b = (T)(other);                                                          \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(OPAQUE(a); OPAQUE(b); a = libfn(a, b);)                          \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

// Expensive-op row shape: Poseidon's materialized kernels execute sqrt and
// div as independent instances in unrolled loop bodies, and a latency-bound
// chain flattens the f32:f64 ratio. They are measured as 8 independent lanes
// per thread, with the same total op count, launch and cost arithmetic.
#define K_LIBM_UN_ILP(name, T, libfn)                                          \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T v[8];                                                                    \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) v[j] =                       \
        (T)((threadIdx.x + blockIdx.x * blockDim.x + j) * 0.0001) + (T)1.5;    \
    _Pragma("unroll 1") for (int i = 0; i < N / 8; ++i) {                      \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) {                          \
        OPAQUE(v[j]);                                                          \
        v[j] = libfn(v[j]);                                                    \
      }                                                                        \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) s += v[j];                   \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

#define DEFINE_UNARY(fn)                                                       \
  K_LIBM_UN(fn, float, MATHF(fn)) K_LIBM_UN(fn, double, MATHD(fn))
#define DEFINE_BINARY(fn, other)                                               \
  K_LIBM_BIN(fn, float, MATHF(fn), other)                                      \
  K_LIBM_BIN(fn, double, MATHD(fn), other)
LIBM_UNARY(DEFINE_UNARY)
VENDOR_LIBM_UNARY(DEFINE_UNARY)
LIBM_BINARY(DEFINE_BINARY)
K_LIBM_UN_ILP(ilp_sqrt, float, MATHF(sqrt))
K_LIBM_UN_ILP(ilp_sqrt, double, MATHD(sqrt))

#ifdef __HIP__
// The same, as one asm statement: the backend pads every asm statement it
// cannot see into with a wait state, which a dependent chain of ordinary VALU
// instructions does not need, so the chain is handed over whole.
#define ASM32(insn) CHAIN32(insn "\n\t")
// One instruction per op on the chain value %0. Its constants are registers
// (%1, %2): the VOP3 encoding the f64 ops use takes no literal on gfx9.
#define K_ASM(name, T, insn, c1, c2)                                           \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    T x = (T)(c1), y = (T)(c2);                                                \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      asm volatile(ASM32(insn) : "+v"(a) : "v"(x), "v"(y));                    \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

// The sign-bit ops act on the 32-bit word that holds the sign: the value
// itself in f32, the high word of an f64, which is all the backend touches.
#define K_ASM_SIGN(name, T, insn, m1, m2)                                      \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    uint64_t bits = 0;                                                         \
    memcpy(&bits, &a, sizeof(T));                                              \
    uint32_t w = (uint32_t)(bits >> (8 * sizeof(T) - 32));                     \
    uint32_t x = (m1), y = (m2);                                               \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      asm volatile(ASM32(insn) : "+v"(w) : "v"(x), "v"(y));                    \
    }                                                                          \
    bits = (bits & ~(0xffffffffull << (8 * sizeof(T) - 32))) |                 \
           ((uint64_t)w << (8 * sizeof(T) - 32));                              \
    memcpy(&a, &bits, sizeof(T));                                              \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_FDIV_ILP(name, T, divisor)                                           \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T v[8];                                                                    \
    T d = (T)(divisor);                                                        \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) v[j] =                       \
        (T)((threadIdx.x + blockIdx.x * blockDim.x + j) * 0.0001) + (T)1.5;    \
    _Pragma("unroll 1") for (int i = 0; i < N / 8; ++i) {                      \
      OPAQUE(d);                                                               \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) v[j] = v[j] / d;           \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) s += v[j];                   \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

// A regime split's condition and its select: fcmp une + select, lowered to
// v_cmp + v_cndmask (one per 32-bit half).
#define K_FCMP(name, T, alt)                                                   \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    T c = (T)(alt);                                                            \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      OPAQUE(c);                                                               \
      CHAIN32(a = (a != c) ? a : c;)                                           \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_CVT(name, srcT, dstT, cvt_fwd, cvt_bwd)                              \
  template <int N> __global__ void name(srcT *out) {                           \
    srcT a =                                                                   \
        (srcT)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (srcT)1.5;  \
    dstT b = (dstT)0;                                                          \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      asm volatile(ASM32(cvt_fwd " %1, %0\n\t" cvt_bwd " %0, %1")              \
                   : "+v"(a), "+v"(b));                                        \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

K_ASM(fadd, float, "v_add_f32 %0, %0, %1", 1.0f, 0.0f)
K_ASM(fsub, float, "v_sub_f32 %0, %0, %1", 0.5f, 0.0f)
K_ASM(fmul, float, "v_mul_f32 %0, %0, %1", 1.0000001f, 0.0f)
K_FDIV_ILP(ilp_fdiv, float, 1.0000001f)
K_ASM_SIGN(fneg, float, "v_xor_b32 %0, %1, %0", 0x80000000u, 0u)
K_ASM_SIGN(fabs, float, "v_and_b32 %0, %1, %0", 0x7fffffffu, 0u)
K_FCMP(fcmp, float, 1.0f)
K_ASM(fma, float, "v_fma_f32 %0, %0, %1, %2", 1.0001f, 0.0f)
K_ASM(ceil, float, "v_ceil_f32 %0, %0", 0.0f, 0.0f)
K_ASM(floor, float, "v_floor_f32 %0, %0", 0.0f, 0.0f)
K_ASM(trunc, float, "v_trunc_f32 %0, %0", 0.0f, 0.0f)
K_ASM(rint, float, "v_rndne_f32 %0, %0", 0.0f, 0.0f)
K_ASM(maxnum, float, "v_max_f32 %0, %0, %1", 1.0f, 0.0f)
K_ASM(minnum, float, "v_min_f32 %0, %0, %1", 1.0f, 0.0f)
// copysign(a, -1.0): v_bfi_b32 takes the magnitude bits from a, the sign from
// the second operand.
K_ASM_SIGN(copysign, float, "v_bfi_b32 %0, %1, %0, %2", 0x7fffffffu,
           0xbf800000u)

K_ASM(fadd, double, "v_add_f64 %0, %0, %1", 1.0, 0.0)
K_ASM(fsub, double, "v_add_f64 %0, %0, -%1", 0.5, 0.0)
K_ASM(fmul, double, "v_mul_f64 %0, %0, %1", 1.0000000000000002, 0.0)
K_FDIV_ILP(ilp_fdiv, double, 1.0000000000000002)
K_ASM_SIGN(fneg, double, "v_xor_b32 %0, %1, %0", 0x80000000u, 0u)
K_ASM_SIGN(fabs, double, "v_and_b32 %0, %1, %0", 0x7fffffffu, 0u)
K_FCMP(fcmp, double, 1.0)
K_ASM(fma, double, "v_fma_f64 %0, %0, %1, %2", 1.0001, 0.0)
K_ASM(ceil, double, "v_ceil_f64 %0, %0", 0.0, 0.0)
K_ASM(floor, double, "v_floor_f64 %0, %0", 0.0, 0.0)
K_ASM(trunc, double, "v_trunc_f64 %0, %0", 0.0, 0.0)
K_ASM(rint, double, "v_rndne_f64 %0, %0", 0.0, 0.0)
K_ASM(maxnum, double, "v_max_f64 %0, %0, %1", 1.0, 0.0)
K_ASM(minnum, double, "v_min_f64 %0, %0, %1", 1.0, 0.0)
K_ASM_SIGN(copysign, double, "v_bfi_b32 %0, %1, %0, %2", 0x7fffffffu,
           0xbff00000u)

K_CVT(cvt_f32_f64, float, double, "v_cvt_f64_f32", "v_cvt_f32_f64")
K_CVT(cvt_f64_f32, double, float, "v_cvt_f32_f64", "v_cvt_f64_f32")
#else
#define K_PTX_BIN(name, T, asm_constraint, insn, arg_imm)                      \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(asm volatile(insn " %0, %0, " arg_imm ";"                        \
                           : "+" asm_constraint(a));)                          \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_PTX_UN(name, T, asm_constraint, insn)                                \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(asm volatile(insn " %0, %0;" : "+" asm_constraint(a));)          \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_PTX_BIN_ILP(name, T, asm_constraint, insn, arg_imm)                  \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T v[8];                                                                    \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) v[j] =                       \
        (T)((threadIdx.x + blockIdx.x * blockDim.x + j) * 0.0001) + (T)1.5;    \
    _Pragma("unroll 1") for (int i = 0; i < N / 8; ++i) {                      \
      _Pragma("unroll") for (int j = 0; j < 8; ++j) asm volatile(              \
          insn " %0, %0, " arg_imm ";" : "+" asm_constraint(v[j]));            \
    }                                                                          \
    T s = (T)0;                                                                \
    _Pragma("unroll") for (int j = 0; j < 8; ++j) s += v[j];                   \
    out[threadIdx.x + blockIdx.x * blockDim.x] = s;                            \
  }

#define K_FCMP(name, T, asm_constraint, suffix, alt_imm)                       \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(asm volatile("{ .reg .pred %%p1;"                                \
                           " setp.ne." suffix " %%p1, %0, " alt_imm ";"        \
                           " selp." suffix " %0, %0, " alt_imm ", %%p1; }"     \
                           : "+" asm_constraint(a));)                          \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_FMA(name, T, fmafn, mul_imm, add_imm)                                \
  template <int N> __global__ void name##_##T(T *out) {                        \
    T a = (T)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (T)1.5;      \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(a = fmafn(a, mul_imm, add_imm);)                                 \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

#define K_CVT(name, srcT, dstT, src_constraint, dst_constraint, cvt_fwd,       \
              cvt_bwd)                                                         \
  template <int N> __global__ void name(srcT *out) {                           \
    srcT a =                                                                   \
        (srcT)((threadIdx.x + blockIdx.x * blockDim.x) * 0.0001) + (srcT)1.5;  \
    dstT b = (dstT)0;                                                          \
    _Pragma("unroll 1") for (int i = 0; i < N / UNROLL; ++i) {                 \
      CHAIN32(asm volatile(cvt_fwd " %0, %1;"                                  \
                           : "=" dst_constraint(b)                             \
                           : src_constraint(a));                               \
              asm volatile(cvt_bwd " %0, %1;"                                  \
                           : "=" src_constraint(a)                             \
                           : dst_constraint(b));)                              \
    }                                                                          \
    out[threadIdx.x + blockIdx.x * blockDim.x] = a;                            \
  }

K_PTX_BIN(fadd, float, "f", "add.f32", "0f3F800000") // +1.0
K_PTX_BIN(fsub, float, "f", "sub.f32", "0f3F000000") // -0.5
K_PTX_BIN(fmul, float, "f", "mul.f32", "0f3F800001") // *1.0000001
K_PTX_BIN_ILP(ilp_fdiv, float, "f", "div.rn.f32", "0f3F800001")
K_PTX_UN(fneg, float, "f", "neg.f32")
K_PTX_UN(fabs, float, "f", "abs.f32")
K_FCMP(fcmp, float, "f", "f32", "0f3F800000")
K_FMA(fma, float, __nv_fmaf, 1.0001f, 0.0f)
K_PTX_BIN(maxnum, float, "f", "max.f32", "0f3F800000")
K_PTX_BIN(minnum, float, "f", "min.f32", "0f3F800000")
K_LIBM_BIN(copysign, float, __nv_copysignf, -1.0f)

K_PTX_BIN(fadd, double, "d", "add.f64", "0d3FF0000000000000")
K_PTX_BIN(fsub, double, "d", "sub.f64", "0d3FE0000000000000")
K_PTX_BIN(fmul, double, "d", "mul.f64", "0d3FF0000000000001")
K_PTX_BIN_ILP(ilp_fdiv, double, "d", "div.rn.f64", "0d3FF0000000000001")
K_PTX_UN(fneg, double, "d", "neg.f64")
K_PTX_UN(fabs, double, "d", "abs.f64")
K_FCMP(fcmp, double, "d", "f64", "0d3FF0000000000000")
K_FMA(fma, double, __nv_fma, 1.0001, 0.0)
K_PTX_BIN(maxnum, double, "d", "max.f64", "0d3FF0000000000000")
K_PTX_BIN(minnum, double, "d", "min.f64", "0d3FF0000000000000")
K_LIBM_BIN(copysign, double, __nv_copysign, -1.0)

K_CVT(cvt_f32_f64, float, double, "f", "d", "cvt.f64.f32", "cvt.rn.f32.f64")
K_CVT(cvt_f64_f32, double, float, "d", "f", "cvt.rn.f32.f64", "cvt.f64.f32")
#endif

struct Config {
  int blocks;
  int threads;
  int reps;
};
static Config gCfg = {0, 1024, 50};

#ifdef __HIP__
// An AMD GPU's clock settles slowly after a change of load, so each kernel
// runs for kWarmMs before it is timed, a batch lasts at least kBatchMs, and
// the time is the median of kBatches batches of at least `reps` launches.
constexpr int kBatches = 7;
constexpr double kWarmMs = 20.0, kBatchMs = 5.0;

template <typename K> double timeit_ns(K kernel, int reps = gCfg.reps) {
  hipEvent_t e0, e1;
  (void)hipEventCreate(&e0);
  (void)hipEventCreate(&e1);
  auto batch = [&](int reps) {
    (void)hipEventRecord(e0);
    for (int i = 0; i < reps; ++i)
      kernel();
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms;
    (void)hipEventElapsedTime(&ms, e0, e1);
    return (double)ms;
  };
  double ms = batch(reps);
  for (double warm = ms; warm < kWarmMs; warm += ms)
    ms = batch(reps);
  if (ms < kBatchMs)
    reps = (int)std::ceil(reps * kBatchMs / std::max(ms, 1e-3));
  double t[kBatches];
  for (int b = 0; b < kBatches; ++b)
    t[b] = batch(reps) * 1e6 / reps;
  (void)hipEventDestroy(e0);
  (void)hipEventDestroy(e1);
  std::sort(t, t + kBatches);
  return t[kBatches / 2];
}
#else
// The mean of `reps` launches after three untimed ones.
template <typename K> double timeit_ns(K kernel, int reps = gCfg.reps) {
  cudaEvent_t e0, e1;
  (void)cudaEventCreate(&e0);
  (void)cudaEventCreate(&e1);
  for (int i = 0; i < 3; ++i)
    kernel();
  (void)cudaDeviceSynchronize();
  (void)cudaEventRecord(e0);
  for (int i = 0; i < reps; ++i)
    kernel();
  (void)cudaEventRecord(e1);
  (void)cudaEventSynchronize(e1);
  float ms;
  (void)cudaEventElapsedTime(&ms, e0, e1);
  (void)cudaEventDestroy(e0);
  (void)cudaEventDestroy(e1);
  return (double)ms * 1e6 / reps;
}
#endif

static double gGhz;

double diff_cycles(double t_n, double t_2n, int n1, int n2) {
  double per_op_ns = (t_2n - t_n) / (n2 - n1);
  return per_op_ns * gGhz * 2.0;
}

#define BENCH_SCALAR(stem, T, op_name, prec_name, sink)                        \
  {                                                                            \
    double t1 = timeit_ns(                                                     \
        [&]() { stem##_##T<512><<<gCfg.blocks, gCfg.threads>>>(sink); });      \
    double t2 = timeit_ns(                                                     \
        [&]() { stem##_##T<1024><<<gCfg.blocks, gCfg.threads>>>(sink); });     \
    /* reciprocal throughput per op: divide the saturated per-step wall */     \
    /* by the number of independent chains running in parallel (one per */     \
    /* thread). */                                                             \
    double c =                                                                 \
        diff_cycles(t1, t2, 512, 1024) / ((double)gCfg.blocks * gCfg.threads); \
    if (c <= 0)                                                                \
      c = 1e-12; /* fneg/fabs etc. are ~free; keep a positive floor */         \
    printf("%s,%s,%.9g\n", op_name, prec_name, c);                             \
  }

#define BENCH_CVT(name, sink, op_label, prec_label)                            \
  {                                                                            \
    double t1 =                                                                \
        timeit_ns([&]() { name<512><<<gCfg.blocks, gCfg.threads>>>(sink); });  \
    double t2 =                                                                \
        timeit_ns([&]() { name<1024><<<gCfg.blocks, gCfg.threads>>>(sink); }); \
    double c = diff_cycles(t1, t2, 512, 1024) * 0.5 /                          \
               ((double)gCfg.blocks * gCfg.threads);                           \
    if (c <= 0)                                                                \
      c = 1e-12;                                                               \
    printf("%s,%s,%.9g\n", op_label, prec_label, c);                           \
  }

// (kernel stem, row) of each per-op row, in the order they print; fmuladd is
// priced as fma and powi as pow.
#define PER_OP_ROWS(X)                                                         \
  X(fadd, fadd);                                                               \
  X(fsub, fsub);                                                               \
  X(fmul, fmul);                                                               \
  X(ilp_fdiv, fdiv);                                                           \
  X(fneg, fneg);                                                               \
  X(fabs, fabs);                                                               \
  X(fcmp, fcmp);                                                               \
  X(fma, fma);                                                                 \
  X(fma, fmuladd);                                                             \
  X(ilp_sqrt, sqrt);                                                           \
  X(cbrt, cbrt);                                                               \
  X(exp, exp);                                                                 \
  X(expm1, expm1);                                                             \
  X(exp2, exp2);                                                               \
  X(log, log);                                                                 \
  X(log1p, log1p);                                                             \
  X(log2, log2);                                                               \
  X(log10, log10);                                                             \
  X(sin, sin);                                                                 \
  X(cos, cos);                                                                 \
  X(tan, tan);                                                                 \
  X(asin, asin);                                                               \
  X(acos, acos);                                                               \
  X(atan, atan);                                                               \
  X(sinh, sinh);                                                               \
  X(cosh, cosh);                                                               \
  X(tanh, tanh);                                                               \
  X(asinh, asinh);                                                             \
  X(acosh, acosh);                                                             \
  X(atanh, atanh);                                                             \
  X(erf, erf);                                                                 \
  X(lgamma, lgamma);                                                           \
  X(tgamma, tgamma);                                                           \
  X(ceil, ceil);                                                               \
  X(floor, floor);                                                             \
  X(trunc, trunc);                                                             \
  X(round, round);                                                             \
  X(rint, rint);                                                               \
  X(atan2, atan2);                                                             \
  X(pow, pow);                                                                 \
  X(pow, powi);                                                                \
  X(hypot, hypot);                                                             \
  X(fmod, fmod);                                                               \
  X(remainder, remainder);                                                     \
  X(fdim, fdim);                                                               \
  X(maxnum, maxnum);                                                           \
  X(minnum, minnum);                                                           \
  X(copysign, copysign);

#ifndef __HIP__
// The tensor-core tile rows. Per scheme, a real tiled WMMA GEMM and a
// scalar-FP64 tiled GEMM of the same shape are timed; the row is written only
// when the scheme is faster, and its value is the ratio of the two times. A
// compile reads only whether the row exists.
//
// tgemm: each block computes a BM x BN output tile with a BWM x BWN warp grid,
// each warp WPT_M x WPT_N WMMA tiles from a BK_STAGE-deep slab of A and B
// staged in shared memory. C is written so nothing is dead.
template <typename AB, typename ACC, typename FRAG_AB, int MT, int NT, int KT,
          int BWM, int BWN, int WPT_M, int WPT_N, int BK_STAGE, bool IS_TF32>
__global__ void tgemm(const AB *__restrict__ A, const AB *__restrict__ B,
                      ACC *__restrict__ C, int M, int N, int K) {
  using namespace nvcuda::wmma;
  constexpr int BM = BWM * WPT_M * MT;
  constexpr int BN = BWN * WPT_N * NT;
  __shared__ AB As[BM * BK_STAGE];
  __shared__ AB Bs[BK_STAGE * BN];

  const int warpId = threadIdx.x / 32;
  const int warpRow = warpId / BWN;
  const int warpCol = warpId % BWN;
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
      store_matrix_sync(&C[(size_t)cRow * N + cCol], acc[i][j], N,
                        mem_row_major);
    }
}

// Scalar-FP64 tiled GEMM: the "scalar path" reference of the same shape. Block
// computes BM x BN, each thread a TM x TN register tile with a shared-staged
// BK slab, so rel = t_tensor / t_scalar exposes a tensor path that is slower
// than scalar FP64.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm(const double *__restrict__ A,
                      const double *__restrict__ B, double *__restrict__ C,
                      int M, int N, int K) {
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
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0;

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
      for (int i = 0; i < TM; ++i)
        ar[i] = As[(tRow * TM + i) * BK + kk];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        br[j] = Bs[kk * BN + (tCol * TN + j)];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fma(ar[i], br[j], acc[i][j]);
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

static const int kGemmReps = 10;

// Wall time (ns per GEMM) of one scheme's tiled WMMA GEMM at M = N = K = GN;
// *ok is false when the scheme does not run on this GPU.
template <typename AB, typename ACC, typename FRAG_AB, int MT, int NT, int KT,
          int BWM, int BWN, int WPT_M, int WPT_N, int BK_STAGE, bool IS_TF32>
double tensor_gemm_ns(int GN, bool *ok) {
  constexpr int BM = BWM * WPT_M * MT;
  constexpr int BN = BWN * WPT_N * NT;
  int M = (GN / BM) * BM, N = (GN / BN) * BN, K = (GN / KT) * KT;
  if (M == 0 || N == 0 || K == 0) {
    if (ok)
      *ok = false;
    return -1.0;
  }

  size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  AB *dA = nullptr, *dB = nullptr;
  ACC *dC = nullptr;
  if (cudaMalloc(&dA, aN * sizeof(AB)) != cudaSuccess ||
      cudaMalloc(&dB, bN * sizeof(AB)) != cudaSuccess ||
      cudaMalloc(&dC, cN * sizeof(ACC)) != cudaSuccess) {
    if (dA)
      cudaFree(dA);
    if (dB)
      cudaFree(dB);
    if (dC)
      cudaFree(dC);
    if (ok)
      *ok = false;
    return -1.0;
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

  cudaGetLastError(); // clear any stale error
  launch();           // warm / probe launch
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { // unsupported scheme (e.g. DMMA)
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    if (ok)
      *ok = false;
    return -1.0;
  }

  double ns = timeit_ns(launch, kGemmReps);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  if (ok)
    *ok = true;
  return ns;
}

// Wall-time (ns/GEMM) of the scalar-FP64 reference GEMM at M=N=K=GN.
double scalar_gemm_ns(int GN) {
  const int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  int M = (GN / BM) * BM, N = (GN / BN) * BN, K = (GN / BK) * BK;
  if (M == 0 || N == 0 || K == 0)
    return -1.0;
  size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (cudaMalloc(&dA, aN * 8) != cudaSuccess ||
      cudaMalloc(&dB, bN * 8) != cudaSuccess ||
      cudaMalloc(&dC, cN * 8) != cudaSuccess) {
    if (dA)
      cudaFree(dA);
    if (dB)
      cudaFree(dB);
    if (dC)
      cudaFree(dC);
    return -1.0;
  }
  cudaMemset(dA, 0, aN * 8);
  cudaMemset(dB, 0, bN * 8);
  cudaMemset(dC, 0, cN * 8);
  dim3 grid(N / BN, M / BM);
  int threads = (BM / TM) * (BN / TN);
  auto launch = [&]() {
    sgemm<BM, BN, BK, TM, TN><<<grid, threads>>>(dA, dB, dC, M, N, K);
  };
  cudaGetLastError();
  launch();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return -1.0;
  }
  double ns = timeit_ns(launch, kGemmReps);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return ns;
}

static void tileRows() {
  const int GN = 2048;
  double t_scalar = scalar_gemm_ns(GN);
  if (t_scalar <= 0) {
    fprintf(
        stderr,
        "poseidon-calibrate-ops: the scalar-FP64 reference GEMM did not run; "
        "no tile row is written\n");
    return;
  }
  auto emit = [&](const char *name, const char *prec, double t_scheme,
                  bool ran) {
    if (!ran) {
      fprintf(
          stderr,
          "poseidon-calibrate-ops: %s,%s: WMMA GEMM unsupported on this arch; "
          "omitting row\n",
          name, prec);
      return;
    }
    double rel = t_scheme / t_scalar;
    if (rel >= 1.0) {
      fprintf(stderr,
              "poseidon-calibrate-ops: %s,%s: real GEMM rel=%.3f is not faster "
              "than scalar "
              "FP64; omitting row\n",
              name, prec, rel);
      return;
    }
    printf("%s,%s,%.9g\n", name, prec, rel);
  };
  bool ok;
  double t;
  t = tensor_gemm_ns<__half, float, __half, 16, 16, 16, 2, 4, 4, 2, 32, false>(
      GN, &ok);
  emit("wmma_mma_m16n16k16", "f16_f32", t, ok);
  t = tensor_gemm_ns<signed char, int, signed char, 16, 16, 16, 2, 4, 4, 2, 32,
                     false>(GN, &ok);
  emit("wmma_mma_m16n16k16", "s8_s32", t, ok);
  t = tensor_gemm_ns<__half, __half, __half, 16, 16, 16, 2, 4, 4, 2, 32, false>(
      GN, &ok);
  emit("wmma_mma_m16n16k16", "f16_f16", t, ok);
  t = tensor_gemm_ns<__nv_bfloat16, float, __nv_bfloat16, 16, 16, 16, 2, 4, 4,
                     2, 32, false>(GN, &ok);
  emit("wmma_mma_m16n16k16", "bf16_f32", t, ok);
  t = tensor_gemm_ns<float, float, nvcuda::wmma::precision::tf32, 16, 16, 8, 2,
                     4, 4, 2, 32, true>(GN, &ok);
  emit("wmma_mma_m16n16k8", "tf32_f32", t, ok);
  t = tensor_gemm_ns<double, double, double, 8, 8, 4, 4, 4, 2, 2, 16, false>(
      GN, &ok);
  emit("wmma_mma_m8n8k4", "f64_f64", t, ok);
}
#endif

// The name and the processor of a device, as its cost model is filed under.
static void deviceArch(const gpuDeviceProp_t &prop, char *arch, size_t size) {
#ifdef __HIP__
  // gcnArchName carries the target features ("gfx942:sramecc+:xnack-"); the
  // processor alone names the cost model.
  snprintf(arch, size, "%s", prop.gcnArchName);
  if (char *colon = strchr(arch, ':'))
    *colon = 0;
#else
  snprintf(arch, size, "sm_%d%d", prop.major, prop.minor);
#endif
}

/// Whether `e` says that this vendor's runtime sees no device at all, as
/// against a device it could not initialize.
static bool isNoDevice(gpuError_t e) {
#ifdef __HIP__
  return e == hipErrorNoDevice || e == hipErrorInvalidDevice ||
         e == hipErrorInsufficientDriver;
#else
  return e == cudaErrorNoDevice || e == cudaErrorInvalidDevice ||
         e == cudaErrorInsufficientDriver;
#endif
}

/// The exit status of a run that finds no device of this vendor, after which
/// poseidon-calibrate tries the other vendor's microbenchmark; a device the
/// runtime could not initialize exits with 1.
constexpr int kExitNoDevice = 2;

int main(int argc, char **argv) {
  gpuDeviceProp_t prop;
  if (gpuError_t e = gpuGetDeviceProperties(&prop, 0); e != gpuSuccess) {
    fprintf(stderr, "%s\n", gpuGetErrorString(e));
    return isNoDevice(e) ? kExitNoDevice : 1;
  }
  char arch[sizeof(prop.name)];
  deviceArch(prop, arch, sizeof(arch));
  if (argc > 1 && !strcmp(argv[1], "--device")) {
    printf("%s\t%s\n", prop.name, arch);
    return 0;
  }

  // From CUDA 13 on, gpuDeviceProp_t carries no clock rate.
  int clkKHz = 0;
  (void)gpuDeviceGetAttribute(&clkKHz, gpuDevAttrClockRate, 0);
  gGhz = (double)clkKHz / 1e6;
  // 1024 threads are 32 warps of an SM, but only 16 waves of a CU, whose four
  // SIMDs hold 32: an AMD GPU takes two blocks per CU to be throughput-bound.
#ifdef __HIP__
  gCfg.blocks = prop.multiProcessorCount * 2;
#else
  gCfg.blocks = prop.multiProcessorCount;
#endif

  printf("# Generated by ops_gpu.cu on %s\n", prop.name);
  printf("# native_arch=%s\n", arch);
  printf("# scalar_types=double,float\n");

  size_t total = (size_t)gCfg.blocks * gCfg.threads;
  float *fout;
  (void)gpuMalloc(&fout, total * sizeof(float));
  double *dout;
  (void)gpuMalloc(&dout, total * sizeof(double));

#ifdef __HIP__
  // Bring the GPU to its sustained clock before the first row.
  for (int i = 0; i < 2000; ++i)
    fma_float<1024><<<gCfg.blocks, gCfg.threads>>>(fout);
  (void)hipDeviceSynchronize();
#endif

#define BENCH_FLOAT(stem, row) BENCH_SCALAR(stem, float, #row, "float", fout)
#define BENCH_DOUBLE(stem, row) BENCH_SCALAR(stem, double, #row, "double", dout)
  PER_OP_ROWS(BENCH_FLOAT)
  PER_OP_ROWS(BENCH_DOUBLE)
  BENCH_CVT(cvt_f32_f64, fout, "fpext_float_to_double", "float");
  BENCH_CVT(cvt_f64_f32, dout, "fptrunc_double_to_float", "double");

  (void)gpuFree(fout);
  (void)gpuFree(dout);
#ifndef __HIP__
  tileRows();
#endif
  return 0;
}
