//===- ozaki_rt.cu - Ozaki-II GEMM runtime --------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ozaki_rt.cu: Poseidon host-dispatch runtime for Ozaki Scheme II.
//
// Provenance: this file is an independent implementation of the published
// Ozaki Scheme II algorithm (K. Ozaki, Y. Uchino, T. Imamura,
// arXiv:2504.08009), fast mode (Cauchy-Schwarz operand scaling). No code is
// derived from the reference GEMMul8 library (MIT licensed). The paper leaves
// the scaling constants and the CRT working precision to the implementation;
// for those this file follows GEMMul8 v3.1.1's fast mode (the authors'
// implementation): the same moduli, the same shift formula and rounding, the
// same truncation, residue representatives and CRT constants, each computed
// here from its defining formula, so that the same operands give the same
// result bit for bit. Residues are computed by GEMMul8's byte-weight method
// (ozaki_residue.h). The pipeline layout (one strided-batched INT8 GEMM,
// layout-aware rectangular arm) is original to this implementation.
//
// GEMMul8 (https://github.com/RIKEN-RCCS/GEMMul8): Copyright (c) 2025- RIKEN
// R-CCS, MIT License. Its numerical conventions are followed with
// acknowledgment; upstream GEMMul8 can replace this runtime at link time
// (same entry points via a shim) without changing any result.
//
// Host entry points for an FP64 GEMM: row/column scaling shifts -> num_moduli
// int8 residue planes -> one strided-batched INT8 tensor-core GEMM ->
// double-double CRT. The modulus table and the scaling constant are shared
// with the compiler through OzakiII.h; the CRT constants are recomputed per nm
// at runtime. The residue and CRT kernels are instantiated per modulus count
// (kOzakiIIMinModuli..14), with the moduli and their byte weights as
// immediates.
//
// Operand scaling (the published fast mode). For each row a of A and each
// column b of B, s = ||a||_2^2 is accumulated in FP64 with every operation
// rounded upward, and the shift is
//   sft = floor(log2P(nm) - log2(s)/2),  log2P(nm) = rd(log2(P-1)/2 - 1/2),
// evaluated in float with directed rounding plus a 2^-23 allowance for the
// hardware log2 (poz_shiftOf). The scaled operand is trunc(2^sft * a), so
// ||a'||_2 <= 2^log2P <= sqrt((P-1)/2) and, by Cauchy-Schwarz, every entry of
// the integer product lies in the symmetric CRT window [-(P-1)/2, (P-1)/2].
// The sum-of-squares reduction order is fixed per operand layout (contiguous
// or strided vector, matching the reference's), so the shifts are
// reproducible. A row whose sum of squares overflows (|a| beyond ~1e154) or
// underflows to zero is outside the fast mode's range, as in the reference.
//
// The per-modulus products are one cublasGemmStridedBatchedEx(CUBLAS_OP_T,
// CUBLAS_OP_N, CUDA_R_8I -> CUDA_R_32I, CUBLAS_COMPUTE_32I) call. Residues are
// laid out for its TN form: A residues row-major (column-major view A^T) and
// B residues transposed (column-major view B), so the batched product is the
// column-major C^T = B^T*A^T, i.e. the row-major C the CRT pass reads. Because
// the accumulation is INT32 and |sum_k r_a*r_b| <= K*128*128 < 2^31 for
// K < 2^17, the result is bit-identical to any other correct INT8 GEMM engine.
//
// Semantics: C = alpha*(A@B) + beta*C. __poseidon_ozaki_dgemm handles the
// square, row-major, leading-dim-N, no-transpose case;
// __poseidon_ozaki_dgemm_ex handles general M x Ncols x K layout-aware shapes
// (rectangular arm, or the square pipeline after padding). Scratch buffers
// (sized for the max 14 moduli) and the per-nm constants are cached across
// calls. num_moduli == 0 selects native cuBLAS DGEMM.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#ifdef __HIP__
#include "../hip_as_cuda.h"
// Every shuffle below is a step of a 32-lane tree over threadIdx.x of a 32-wide
// block row (poz_warpSumRU). Width 32 keeps it in its row's half of a 64-lane
// wavefront, where a source lane past the row returns the caller's own value as
// on a warp, so both halves sum in the CUDA order and give the CUDA shifts.
#define __shfl_down_sync(mask, var, delta) __shfl_down(var, delta, 32)
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "OzakiII.h"
#include "ozaki_residue.h"

#define POZ_CK(x)                                                              \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess)                                                     \
      printf("poseidon_ozaki_rt CUDA %s:%d %s\n", __FILE__, __LINE__,          \
             cudaGetErrorString(e_));                                          \
  } while (0)

#define POZ_CB(x)                                                              \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                           \
      printf("poseidon_ozaki_rt cuBLAS %s:%d status %d\n", __FILE__, __LINE__, \
             (int)s_);                                                         \
  } while (0)

namespace {
constexpr int POZ_MAXNM = (int)kOzakiIIMaxModuli;
// Runtime-configured constants (uploaded per num_moduli).
__constant__ double poz_QHi[POZ_MAXNM];
__constant__ double poz_QLo[POZ_MAXNM];
__constant__ double poz_PHi;
__constant__ double poz_PLo;
__constant__ double poz_InvP;
__constant__ float poz_log2P;

// Sum-of-squares reduction geometry (fixed: it decides the shifts' bits).
constexpr int POZ_NRM_THREADS = 256;  // contiguous vector: threads per vector
constexpr int POZ_NRM_SPLIT_K = 1024; // strided vector: chunk K at or above
constexpr int POZ_NRM_KCHUNK = 256;   // strided vector: chunk length

// Lazily created, process-lifetime cuBLAS handle; the stream is set on every
// dispatch.
cublasHandle_t poz_cublasHandle() {
  static cublasHandle_t h = nullptr;
  if (!h)
    POZ_CB(cublasCreate(&h));
  return h;
}

// f(std::integral_constant<int, i>) for i = 0..NM-1, so that modulus i and its
// byte weights are compile-time constants in f.
template <class F, int... I>
__device__ __forceinline__ void
poz_forEachModulus(F &&f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>()), ...);
}
template <int NM, class F>
__device__ __forceinline__ void poz_forModuli(F &&f) {
  poz_forEachModulus(f, std::make_integer_sequence<int, NM>());
}

// ---------------------------------------------------------------------------
// Scaling shifts.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double poz_warpSumRU(double s) {
  s = __dadd_ru(s, __shfl_down_sync(0xffffffffu, s, 16));
  s = __dadd_ru(s, __shfl_down_sync(0xffffffffu, s, 8));
  s = __dadd_ru(s, __shfl_down_sync(0xffffffffu, s, 4));
  s = __dadd_ru(s, __shfl_down_sync(0xffffffffu, s, 2));
  s = __dadd_ru(s, __shfl_down_sync(0xffffffffu, s, 1));
  return s;
}

// floor(log2P - log2(s)/2) for s = ||a||_2^2 (an upper bound). Each rounding
// is directed so the shift can only come out smaller: log2P is rounded down,
// the mantissa of s up, and 2^-23 covers the error of the hardware log2.
__device__ __forceinline__ short poz_shiftOf(double s) {
  if (s == 0.0)
    return 0;
  const int e = ilogb(s);
  const float m = __double2float_ru(scalbn(s, -e));
  const float mu0 = __fsub_rd(poz_log2P, (float)e * 0.5f);
  const float mu1 = __fadd_ru(__log2f(m) * 0.5f, 0x1p-23f);
  return (short)__float2int_rd(__fsub_rd(mu0, mu1));
}

// Contiguous vectors (vector r is src[r*ld + k], k < K): one block per vector,
// thread t folds k = t, t+256, ..., then a warp tree and a tree over the
// eight warp sums.
__global__ void poz_sftContig(const double *src, size_t ld, int K, short *sft) {
  __shared__ double sh[POZ_NRM_THREADS / 32];
  const double *v = src + (size_t)blockIdx.x * ld;
  double s = 0.0;
  for (int k = threadIdx.x; k < K; k += POZ_NRM_THREADS)
    s = __fma_ru(v[k], v[k], s);
  s = poz_warpSumRU(s);
  if ((threadIdx.x & 31) == 0)
    sh[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = (threadIdx.x < POZ_NRM_THREADS / 32) ? sh[threadIdx.x] : 0.0;
    s = poz_warpSumRU(s);
    if (threadIdx.x == 0)
      sft[blockIdx.x] = poz_shiftOf(s);
  }
}

// Strided vectors (vector r is src[k*ld + r]), 32 vectors per 32x32 block:
// thread (x, y) folds k = k0+y, k0+y+32, ... < k1 of vector 32*blockIdx.x + x;
// a shared-memory transpose puts a vector's 32 phases in one warp.
__device__ __forceinline__ double poz_stridedTile(const double *src, size_t ld,
                                                  int R, int k0, int k1) {
  __shared__ double sh[32][33];
  const int r = blockIdx.x * 32 + threadIdx.x;
  double s = 0.0;
  if (r < R)
    for (int k = k0 + (int)threadIdx.y; k < k1; k += 32) {
      const double a = src[(size_t)k * ld + r];
      s = __fma_ru(a, a, s);
    }
  sh[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  return poz_warpSumRU(sh[threadIdx.x][threadIdx.y]);
}

// K < POZ_NRM_SPLIT_K: the whole vector in one tile.
__global__ void poz_sftStrided(const double *src, size_t ld, int R, int K,
                               short *sft) {
  const double s = poz_stridedTile(src, ld, R, 0, K);
  const int r = blockIdx.x * 32 + threadIdx.y;
  if (threadIdx.x == 0 && r < R)
    sft[r] = poz_shiftOf(s);
}

// K >= POZ_NRM_SPLIT_K: one partial per (vector, 256-long chunk) ...
__global__ void poz_sumStridedPart(const double *src, size_t ld, int R, int K,
                                   double *part) {
  const int k0 = blockIdx.y * POZ_NRM_KCHUNK;
  const double s = poz_stridedTile(src, ld, R, k0, min(K, k0 + POZ_NRM_KCHUNK));
  const int r = blockIdx.x * 32 + threadIdx.y;
  if (threadIdx.x == 0 && r < R)
    part[(size_t)r * gridDim.y + blockIdx.y] = s;
}

// ... then lane x of a vector's warp folds chunks x, x+32, ... and a warp tree
// finishes it. Block (32, 8): one warp per vector.
__global__ void poz_sftStridedReduce(const double *part, int R, int nChunk,
                                     short *sft) {
  const int r = blockIdx.x * 8 + threadIdx.y;
  if (r >= R)
    return; // warp-uniform
  double s = 0.0;
  for (int c = threadIdx.x; c < nChunk; c += 32)
    s = __dadd_ru(s, part[(size_t)r * nChunk + c]);
  s = poz_warpSumRU(s);
  if (threadIdx.x == 0)
    sft[r] = poz_shiftOf(s);
}

// Shifts of the R vectors (length K) of one operand. `part` holds
// R*ceil(K/256) doubles for the strided K >= 1024 case.
void poz_shifts(const double *src, size_t ld, int R, int K, bool contiguous,
                short *sft, double *part, cudaStream_t stream) {
  if (contiguous) {
    poz_sftContig<<<R, POZ_NRM_THREADS, 0, stream>>>(src, ld, K, sft);
    return;
  }
  const int gx = (R + 31) / 32;
  if (K < POZ_NRM_SPLIT_K) {
    poz_sftStrided<<<gx, dim3(32, 32), 0, stream>>>(src, ld, R, K, sft);
    return;
  }
  const int nChunk = (K + POZ_NRM_KCHUNK - 1) / POZ_NRM_KCHUNK;
  poz_sumStridedPart<<<dim3(gx, nChunk), dim3(32, 32), 0, stream>>>(src, ld, R,
                                                                    K, part);
  poz_sftStridedReduce<<<(R + 7) / 8, dim3(32, 8), 0, stream>>>(part, R, nChunk,
                                                                sft);
}

// Scaled operand entry: trunc(2^sft * x), exact (scalbn only moves the
// exponent and |result| < 2^55). scalbn(x, sft), not x * scalbn(1.0, sft): sft
// can exceed 1023 for a row whose norm is tiny, where scalbn(1.0, sft) is inf.
__device__ __forceinline__ long poz_quant(double x, int sft) {
  return __double2ll_rz(scalbn(x, sft));
}

// The residues of v modulo the first NM moduli, one per residue plane. v is 0
// outside the operand, and the residues of 0 are 0.
template <int NM>
__device__ __forceinline__ void poz_storeResidues(int8_t *dst, size_t plane,
                                                  size_t off, long v) {
  poz_forModuli<NM>([&](auto i) {
    constexpr int I = decltype(i)::value;
    constexpr PozResidueModulus m = pozResidueModulus(I);
    dst[(size_t)I * plane + off] = poz_modI8(v, m);
  });
}

// Both arms quantize each operand, per modulus, into a row-major (R x Kp) int8
// plane whose row is the operand's outer index (A: the output row, B: the
// output column) and whose column is the contraction index k. Read
// column-major with leading dimension Kp these are A^T and B, so the batched
// TN INT8 GEMM yields the row-major C. Per operand only the source layout
// differs: contiguous in k ("direct") or in the outer index ("transposed":
// aColMajor for A, !bColMajor for B, and the square arm's row-major B).
//
// Residue fill into the NM (Rp x Kp) planes. `transposed` selects the source
// addressing:
//   direct     (transposed==0): src(r,k) = src[r*ld + k]   (contiguous in k)
//   transposed (transposed==1): src(r,k) = src[k*ld + r]   (contiguous in r)
// In the transposed case the scaled integers are staged in shared and written
// back with the tile transposed, so both the source read and the int8 store
// stay coalesced. Every destination element with r >= R or k >= K is written
// as an explicit zero: the k pad is inside the contraction range the batched
// GEMM is issued over, so it must not carry garbage, and the r pad keeps the
// residues of the (never-read) alignment rows well defined.
template <int NM>
__global__ void poz_fill(const double *src, const short *sft, int8_t *dst,
                         int R, int K, int ld, int transposed, int Rp, int Kp) {
  __shared__ long sh[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const size_t plane = (size_t)Rp * Kp;
  if (!transposed) {
    const int kBase = blockIdx.x * 32, rBase = blockIdx.y * 32;
    const int k = kBase + tx;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = rBase + ty + 8 * j;
      if (r >= Rp || k >= Kp)
        continue;
      const bool inb = (r < R && k < K);
      const long v = inb ? poz_quant(src[(size_t)r * ld + k], sft[r]) : 0L;
      poz_storeResidues<NM>(dst, plane, (size_t)r * Kp + k, v);
    }
  } else {
    const int rBase = blockIdx.x * 32, kBase = blockIdx.y * 32;
    {
      const int r = rBase + tx;
      const short s = (r < R) ? sft[r] : (short)0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int kl = ty + 8 * j, k = kBase + kl;
        sh[kl][tx] =
            (r < R && k < K) ? poz_quant(src[(size_t)k * ld + r], s) : 0L;
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int rl = ty + 8 * j, r = rBase + rl, k = kBase + tx;
      if (r >= Rp || k >= Kp)
        continue;
      poz_storeResidues<NM>(dst, plane, (size_t)r * Kp + k, sh[tx][rl]);
    }
  }
}

// CRT recombine of the four residue vectors in `raw` planes for one output
// quad. x accumulates the high parts of q_i = (P/p_i)*((P/p_i)^-1 mod p_i)
// error-free (each is a multiple of one 2^G and the sum stays below 2^(G+53)),
// y the low parts; the quotient round(x/P) is taken from x, then
// x + y - quot*P is formed with P as a double-double. With nm <= 6 P and the
// q_i are exact doubles and the CRT is single FP64.
template <int NM>
__device__ __forceinline__ void poz_crtQuad(const int32_t *Cr, size_t off,
                                            size_t plane, double *out) {
  double x[4], y[4];
  poz_forModuli<NM>([&](auto i) {
    constexpr int I = decltype(i)::value;
    constexpr PozResidueModulus m = pozResidueModulus(I);
    const int4 raw =
        *reinterpret_cast<const int4 *>(Cr + (size_t)I * plane + off);
    const int *rawv = reinterpret_cast<const int *>(&raw);
    const double qhi = poz_QHi[I], qlo = poz_QLo[I];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const double d = (double)poz_modI32(rawv[j], m);
      if constexpr (I == 0) {
        x[j] = qhi * d;
        y[j] = qlo * d;
      } else {
        x[j] = fma(qhi, d, x[j]);
        y[j] = fma(qlo, d, y[j]);
      }
    }
  });
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const double quot = rint(poz_InvP * x[j]);
    if constexpr (NM <= (int)kOzakiIIPIsDouble)
      out[j] = fma(poz_PHi, quot, x[j]);
    else
      out[j] = fma(poz_PLo, quot, fma(poz_PHi, quot, x[j]) + y[j]);
  }
}

// C = alpha*AB + beta*C; beta == 0 never reads C.
__device__ __forceinline__ double poz_axpby(double alpha, double ab,
                                            double beta, double c) {
  return (beta == 0.0) ? alpha * ab : fma(beta, c, alpha * ab);
}

// CRT recombine over the M x Ncols useful region only. Four columns per thread
// as one int4 per residue plane: four independent accumulator pairs keep loads
// in flight. ldCr is a multiple of 16 and each thread's c0 is a multiple of 4,
// so every load is 16-byte aligned and stays inside the padded plane. Block is
// (8, 32): 8 threads x 4 columns cover 32 columns, 32 threads cover 32 rows.
//   row-major C   : the natural (r, c) mapping is already coalesced, write out.
//   column-major C: consecutive c are ldc apart, so the tile is transposed
//                   through shared and written with consecutive r in a warp.
//                   The beta!=0 read of C happens in that same transposed
//                   phase, so it is coalesced too.
template <int NM>
__global__ void poz_crt(const int32_t *Cr, const short *sftA, const short *sftB,
                        double *C, int M, int Ncols, int Mp, int ldCr, int ldc,
                        int cColMajor, double alpha, double beta) {
  __shared__ double sh[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y; // blockDim = (8, 32)
  const int cBase = blockIdx.x * 32, rBase = blockIdx.y * 32;
  const int r = rBase + ty, c0 = cBase + 4 * tx;
  double prod[4] = {0, 0, 0, 0};
  const bool live = (r < M && c0 < Ncols);
  if (live) {
    double crtv[4];
    poz_crtQuad<NM>(Cr, (size_t)r * ldCr + c0, (size_t)Mp * ldCr, crtv);
    const short sa = sftA[r];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int c = c0 + j;
      if (c >= Ncols)
        break;
      prod[j] = scalbn(crtv[j], -(sa + sftB[c]));
    }
  }
  if (!cColMajor) {
    if (!live)
      return;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int c = c0 + j;
      if (c >= Ncols)
        return;
      const size_t o = (size_t)r * ldc + c;
      C[o] = poz_axpby(alpha, prod[j], beta, C[o]);
    }
    return;
  }
  // Transposed write-out: stage (row, col) in shared, then re-linearize the
  // block so a warp walks 32 consecutive rows of one column. tid&31 is the row,
  // so the 32 lanes of a warp write 32 consecutive doubles of C and read C back
  // (beta != 0) the same way.
#pragma unroll
  for (int j = 0; j < 4; j++)
    sh[ty][4 * tx + j] = live ? prod[j] : 0.0;
  __syncthreads();
  const int tid = tx + 8 * ty;
  const int rl = tid & 31, rw = rBase + rl;
  if (rw >= M)
    return;
  for (int cl = tid >> 5; cl < 32; cl += 8) {
    const int c = cBase + cl;
    if (c >= Ncols)
      break;
    const size_t o = (size_t)c * ldc + rw;
    C[o] = poz_axpby(alpha, sh[rl][cl], beta, C[o]);
  }
}

// Host-side per-nm constants.
static long pozModInv(long a, long m) {
  long o_r = ((a % m) + m) % m, r = m, o_s = 1, s = 0;
  while (r) {
    long q = o_r / r, t;
    t = o_r - q * r;
    o_r = r;
    r = t;
    t = o_s - q * s;
    o_s = s;
    s = t;
  }
  return ((o_s % m) + m) % m;
}
static int pozBitLen(__int128 v) {
  int n = 0;
  while (v) {
    ++n;
    v >>= 1;
  }
  return n;
}
// 1/P rounded to nearest-even, by exact long division (P < 2^112, so the
// remainder stays inside __int128).
static double pozRecipRN(__int128 P) {
  __int128 r = 1;
  int e = 0;
  while (r < P) {
    r <<= 1;
    ++e;
  }                         // 2^e / P in [1, 2)
  unsigned long long q = 0; // floor(2^(e+53) / P), 54 bits
  for (int i = 0; i < 54; i++) {
    q <<= 1;
    if (r >= P) {
      r -= P;
      q |= 1;
    }
    r <<= 1;
  }
  const bool rnd = q & 1, sticky = r != 0;
  q >>= 1;
  if (rnd && (sticky || (q & 1)))
    ++q;
  return std::ldexp((double)q, -(e + 52));
}
struct PozConsts {
  int nm = 0;
  double qhi[POZ_MAXNM], qlo[POZ_MAXNM], pHi, pLo, invP;
  float log2P = 0.0f;
};
static void pozComputeConsts(int nm, PozConsts &c) {
  c.nm = nm;
  __int128 P = 1;
  int rho = 0; // sum of floor(p_i/2): bound on sum |residue_i|
  for (int i = 0; i < nm; i++) {
    P *= kOzakiIIModuli[i];
    rho += kOzakiIIModuli[i] / 2;
  }
  // -P as a round-to-nearest double-double.
  double pHiPos = (double)P;
  c.pHi = -pHiPos;
  c.pLo = -(double)(P - (__int128)pHiPos);
  c.invP = pozRecipRN(P);
  c.log2P = kOzakiIILog2P[nm];
  // High parts of q_i keep 53 - ceil(log2 rho) bits below bitlen(P), all at
  // one granularity 2^G, so sum_i qhi_i*r_i (|r_i| <= floor(p_i/2)) is exact.
  int ceilLog2Rho = pozBitLen((__int128)(rho - 1));
  int G = pozBitLen(P) - (53 - ceilLog2Rho);
  for (int i = 0; i < nm; i++) {
    long p = kOzakiIIModuli[i];
    __int128 Mi = P / p;
    long yi = pozModInv((long)(Mi % p), p);
    __int128 e = (Mi * (__int128)yi) % P;
    if (nm <= (int)kOzakiIIPIsDouble) {
      c.qhi[i] = (double)e; // exact: e < P < 2^48
      c.qlo[i] = 0.0;
    } else {
      __int128 hiI = (e >> G) << G;
      c.qhi[i] = (double)hiI;
      c.qlo[i] = (double)(e - hiI);
    }
  }
}

struct PozScratch {
  int Np = 0;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  // Strided-vector partial sums of squares: Np*ceil(Np/256) doubles.
  double *part = nullptr;
  // Padded square F64 operand/result buffers for the square arm of _ex: an
  // M x Ncols x K source is repacked into As/Bs (S x S, S = pad of
  // max(M,Ncols,K)) and the result cropped back out of Cs.
  double *As = nullptr, *Bs = nullptr, *Cs = nullptr;
};
PozScratch g_scratch;
PozConsts g_consts;
int g_uploadedNm = 0;

// INT8 tensor-core GEMM leading dimensions must be multiples of 4; the residue
// planes are held at Np = roundup(N, 16), which also keeps every per-modulus
// plane base 16-byte aligned.
static inline int pozPad(int N) { return ((N + 15) / 16) * 16; }
static inline size_t pozChunks(int K) {
  return (size_t)((K + POZ_NRM_KCHUNK - 1) / POZ_NRM_KCHUNK);
}

void poz_ensure(int N) {
  int Np = pozPad(N);
  if (g_scratch.Np == Np)
    return;
  if (g_scratch.Ar) {
    cudaFree(g_scratch.Ar);
    cudaFree(g_scratch.Br);
    cudaFree(g_scratch.Cr);
    cudaFree(g_scratch.sA);
    cudaFree(g_scratch.sB);
    cudaFree(g_scratch.part);
    cudaFree(g_scratch.As);
    cudaFree(g_scratch.Bs);
    cudaFree(g_scratch.Cs);
  }
  size_t plane = (size_t)Np * Np;
  POZ_CK(cudaMalloc(&g_scratch.Ar, POZ_MAXNM * plane));
  POZ_CK(cudaMalloc(&g_scratch.Br, POZ_MAXNM * plane));
  POZ_CK(cudaMalloc(&g_scratch.Cr, POZ_MAXNM * plane * 4));
  POZ_CK(cudaMalloc(&g_scratch.sA, Np * 2));
  POZ_CK(cudaMalloc(&g_scratch.sB, Np * 2));
  POZ_CK(
      cudaMalloc(&g_scratch.part, (size_t)Np * pozChunks(Np) * sizeof(double)));
  POZ_CK(cudaMalloc(&g_scratch.As, plane * sizeof(double)));
  POZ_CK(cudaMalloc(&g_scratch.Bs, plane * sizeof(double)));
  POZ_CK(cudaMalloc(&g_scratch.Cs, plane * sizeof(double)));
  g_scratch.Np = Np;
}
void poz_uploadConsts(int nm) {
  if (g_uploadedNm == nm)
    return;
  pozComputeConsts(nm, g_consts);
  POZ_CK(cudaMemcpyToSymbol(poz_QHi, g_consts.qhi, nm * sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_QLo, g_consts.qlo, nm * sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_PHi, &g_consts.pHi, sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_PLo, &g_consts.pLo, sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_InvP, &g_consts.invP, sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_log2P, &g_consts.log2P, sizeof(float)));
  g_uploadedNm = nm;
}

// Repack a rows x cols (layout-aware) source into the top-left of an S x S
// row-major buffer: dst[r*S+c] = src[r,c]. The padded region (r>=rows ||
// c>=cols) is left as the caller's prior memset (0). colMajor selects the
// source stride convention (transposed vs row-major).
__global__ void poz_pack(double *dst, const double *src, int rows, int cols,
                         int ld, int colMajor, int S) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols)
    return;
  dst[(size_t)r * S + c] =
      colMajor ? src[(size_t)c * ld + r] : src[(size_t)r * ld + c];
}

// Crop the M x Ncols top-left of an S x S row-major result back into a
// layout-aware destination, applying C = alpha*Cs + beta*C only over the
// useful region.
__global__ void poz_crop(double *C, const double *Cs, int M, int Ncols, int ldc,
                         int cColMajor, int S, double alpha, double beta) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= Ncols)
    return;
  size_t dst = cColMajor ? (size_t)j * ldc + i : (size_t)i * ldc + j;
  C[dst] = poz_axpby(alpha, Cs[(size_t)i * S + j], beta, C[dst]);
}

// NM moduli (kOzakiIIMinModuli..14) select the Ozaki-II precision; the per-nm
// constants are computed and uploaded here (cached). Square N x N x N, NN,
// row-major pipeline: C = alpha*(A@B)+beta*C.
template <int NM>
static void poz_run_square(double *C, const double *A, const double *B, int N,
                           double alpha, double beta, cudaStream_t stream) {
  // s32 residue-GEMM accumulator: |sum_k r_a*r_b| <= K*128*128 = K*2^14 must
  // stay under 2^31, i.e. K < 131072.
  if (N >= 131072) { // 2^31 / 2^14, written out: 1<<31 overflows a signed int
    printf("poseidon_ozaki_rt: FATAL K=%d >= 131072: the s32 residue-GEMM "
           "accumulator would overflow; needs an intermediate reduction.\n",
           N);
    abort();
  }
  poz_ensure(N);
  poz_uploadConsts(NM);
  int Np = g_scratch.Np;
  size_t plane = (size_t)Np * Np;
  dim3 tile(32, 8), gTile((Np + 31) / 32, (Np + 31) / 32);
  // Rows of A are contiguous, columns of B strided (row-major operands).
  poz_shifts(A, N, N, N, /*contiguous=*/true, g_scratch.sA, g_scratch.part,
             stream);
  poz_shifts(B, N, N, N, /*contiguous=*/false, g_scratch.sB, g_scratch.part,
             stream);
  poz_fill<NM><<<gTile, tile, 0, stream>>>(A, g_scratch.sA, g_scratch.Ar, N, N,
                                           N, /*transposed=*/0, Np, Np);
  poz_fill<NM><<<gTile, tile, 0, stream>>>(B, g_scratch.sB, g_scratch.Br, N, N,
                                           N, /*transposed=*/1, Np, Np);
  // One strided-batched INT8 tensor-core GEMM over all NM residue planes.
  // Column-major C^T(Np x Np) = op(Br)^T * op(Ar) = B^T*A^T, so the int32
  // result read row-major is the row-major product.
  cublasHandle_t h = poz_cublasHandle();
  POZ_CB(cublasSetStream(h, stream));
  const int32_t i_one = 1, i_zero = 0;
  POZ_CB(cublasGemmStridedBatchedEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, Np, Np, Np, &i_one, g_scratch.Br, CUDA_R_8I,
      Np, (long long)plane, g_scratch.Ar, CUDA_R_8I, Np, (long long)plane,
      &i_zero, g_scratch.Cr, CUDA_R_32I, Np, (long long)plane, NM,
      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
  poz_crt<NM><<<dim3((N + 31) / 32, (N + 31) / 32), dim3(8, 32), 0, stream>>>(
      g_scratch.Cr, g_scratch.sA, g_scratch.sB, C, N, N, Np, Np, N,
      /*cColMajor=*/0, alpha, beta);
}

// Rectangular arm. The square pipeline pads every operand to
// S = roundup(max(M,Ncols,K),16), so a skinny product pays
// padWaste = S^3/(M*Ncols*K) times the useful work. This arm sizes every
// buffer to the real extents (rounded up to 16 for the INT8 leading dimension)
// and repacks no operand. Shifts depend only on the vector and its layout, so
// this arm gives the square arm's bits on an exactly square dispatch.

// Runtime knobs for the rectangular arm. Parsed once; an unparseable value
// aborts rather than silently selecting the other behaviour.
//   POSEIDON_OZAKI_PATH           auto (default) | rect | square
//   POSEIDON_OZAKI_RECT_MIN_WASTE padWaste above which "auto" picks rect
//   POSEIDON_OZAKI_RECT_CAP_MB    residue-scratch budget driving the column
//                                 chunk (default 512 MB)
//   POSEIDON_OZAKI_RECT_CHUNK     explicit column chunk, overrides the budget
static long poz_envLong(const char *name, long dflt) {
  const char *e = getenv(name);
  if (!e || !*e)
    return dflt;
  char *end = nullptr;
  long v = strtol(e, &end, 10);
  if (end == e || (end && *end) || v < 0) {
    printf("poseidon_ozaki_rt: FATAL %s='%s' is not a non-negative integer\n",
           name, e);
    abort();
  }
  return v;
}
static double poz_envDouble(const char *name, double dflt) {
  const char *e = getenv(name);
  if (!e || !*e)
    return dflt;
  char *end = nullptr;
  double v = strtod(e, &end);
  if (end == e || (end && *end) || !(v > 0.0)) {
    printf("poseidon_ozaki_rt: FATAL %s='%s' is not a positive number\n", name,
           e);
    abort();
  }
  return v;
}
int g_pathMode = -1; // 0 auto, 1 rect, 2 square
int poz_pathMode() {
  if (g_pathMode < 0) {
    const char *e = getenv("POSEIDON_OZAKI_PATH");
    if (!e || !*e || !strcmp(e, "auto"))
      g_pathMode = 0;
    else if (!strcmp(e, "rect"))
      g_pathMode = 1;
    else if (!strcmp(e, "square"))
      g_pathMode = 2;
    else {
      printf("poseidon_ozaki_rt: FATAL unknown POSEIDON_OZAKI_PATH='%s' "
             "(expected 'auto', 'rect' or 'square')\n",
             e);
      abort();
    }
  }
  return g_pathMode;
}
// Threshold on padWaste = S^3/(M*Ncols*K). The 16-column alignment rounds a
// square shape up to at most (S/N)^3 <= 1.05 for N >= 320, so every square _ex
// call stays on the square arm.
static double poz_rectMinWaste() {
  static double v = poz_envDouble("POSEIDON_OZAKI_RECT_MIN_WASTE", 1.05);
  return v;
}
static size_t poz_rectCapBytes() {
  static size_t v = (size_t)poz_envLong("POSEIDON_OZAKI_RECT_CAP_MB", 512)
                    << 20;
  return v;
}

// Column chunking bounds the residue scratch, POZ_MAXNM*(Kp + 4*Mp) bytes per
// column plus the shift and its partial sums. It is numerically inert: A's
// shift is a per-row property of the whole operand and B's a per-column
// property of one column, so every chunking returns the same bits.
long g_rectChunk = -1; // <0 = not yet read from the environment
static int poz_rectChunk(int Mp, int Kp, int Ncols) {
  if (g_rectChunk < 0)
    g_rectChunk = poz_envLong("POSEIDON_OZAKI_RECT_CHUNK", 0);
  const long forced = g_rectChunk;
  if (forced > 0)
    return (int)std::min<long>(forced, Ncols);
  double perCol = (double)POZ_MAXNM * ((double)Kp + 4.0 * (double)Mp) +
                  8.0 * (double)pozChunks(Kp) + 2.0;
  long c = (long)((double)poz_rectCapBytes() / perCol);
  c &= ~127L; // keep every chunk but the last a multiple of 128 columns
  if (c < 128)
    c = 128;
  if (c > Ncols)
    c = Ncols;
  return (int)c;
}

struct PozRectScratch {
  int Mp = 0, Kp = 0, Ncp = 0;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  double *part = nullptr; // max(Mp, Ncp)*ceil(Kp/256) partial sums of squares
};
PozRectScratch g_rect;

// Grow-only: an alternating shape sequence (the last chunk of a dispatch is
// short) must not free and re-allocate on every call.
static void poz_ensureRect(int Mp, int Kp, int Ncp) {
  if (g_rect.Mp >= Mp && g_rect.Kp >= Kp && g_rect.Ncp >= Ncp)
    return;
  Mp = std::max(Mp, g_rect.Mp);
  Kp = std::max(Kp, g_rect.Kp);
  Ncp = std::max(Ncp, g_rect.Ncp);
  if (g_rect.Ar) {
    cudaFree(g_rect.Ar);
    cudaFree(g_rect.Br);
    cudaFree(g_rect.Cr);
    cudaFree(g_rect.sA);
    cudaFree(g_rect.sB);
    cudaFree(g_rect.part);
    g_rect = PozRectScratch();
  }
  POZ_CK(cudaMalloc(&g_rect.Ar, (size_t)POZ_MAXNM * Mp * Kp));
  POZ_CK(cudaMalloc(&g_rect.Br, (size_t)POZ_MAXNM * Ncp * Kp));
  POZ_CK(
      cudaMalloc(&g_rect.Cr, (size_t)POZ_MAXNM * Ncp * Mp * sizeof(int32_t)));
  POZ_CK(cudaMalloc(&g_rect.sA, (size_t)Mp * sizeof(short)));
  POZ_CK(cudaMalloc(&g_rect.sB, (size_t)Ncp * sizeof(short)));
  POZ_CK(cudaMalloc(&g_rect.part, (size_t)std::max(Mp, Ncp) * pozChunks(Kp) *
                                      sizeof(double)));
  g_rect.Mp = Mp;
  g_rect.Kp = Kp;
  g_rect.Ncp = Ncp;
}

// Rectangular Ozaki-II: C(M x Ncols) = alpha*(A@B) + beta*C, A(M x K),
// B(K x Ncols), each operand row- or column-major per the {a,b,c}ColMajor
// flags. Columns are chunked internally; A is scaled and quantized once for
// the whole dispatch because its residues are a property of A alone.
template <int NM>
static void poz_run_rect(double *C, const double *A, const double *B, int M,
                         int Ncols, int K, int lda, int ldb, int ldc,
                         int aColMajor, int bColMajor, int cColMajor,
                         double alpha, double beta, cudaStream_t stream) {
  // s32 residue-GEMM accumulator: |sum_k r_a*r_b| <= K*128*128 must stay under
  // 2^31, here on the unpadded K.
  if (K >= 131072) {
    printf("poseidon_ozaki_rt: FATAL K=%d >= 131072: the s32 residue-GEMM "
           "accumulator would overflow; needs an intermediate reduction.\n",
           K);
    abort();
  }
  poz_uploadConsts(NM);
  const int Kp = pozPad(K), Mp = pozPad(M);
  const int chunk = poz_rectChunk(Mp, Kp, Ncols);
  poz_ensureRect(Mp, Kp, pozPad(chunk));
  const dim3 tile(32, 8);

  // A, once for the whole dispatch.
  const int tr = aColMajor ? 1 : 0; // col-major A is contiguous in m, not k
  poz_shifts(A, lda, M, K, /*contiguous=*/!tr, g_rect.sA, g_rect.part, stream);
  const dim3 g = tr ? dim3((Mp + 31) / 32, (Kp + 31) / 32)
                    : dim3((Kp + 31) / 32, (Mp + 31) / 32);
  poz_fill<NM>
      <<<g, tile, 0, stream>>>(A, g_rect.sA, g_rect.Ar, M, K, lda, tr, Mp, Kp);

  cublasHandle_t h = poz_cublasHandle();
  POZ_CB(cublasSetStream(h, stream));
  const int32_t i_one = 1, i_zero = 0;

  for (int j0 = 0; j0 < Ncols; j0 += chunk) {
    const int nc = std::min(chunk, Ncols - j0);
    const int ncp = pozPad(nc);
    // Chunking is a pointer offset under either storage order.
    const double *Bc = bColMajor ? B + (size_t)j0 * ldb : B + j0;
    double *Cc = cColMajor ? C + (size_t)j0 * ldc : C + j0;
    const int trB = bColMajor ? 0 : 1; // row-major B is contiguous in c, not k

    poz_shifts(Bc, ldb, nc, K, /*contiguous=*/!trB, g_rect.sB, g_rect.part,
               stream);
    const dim3 gB = trB ? dim3((ncp + 31) / 32, (Kp + 31) / 32)
                        : dim3((Kp + 31) / 32, (ncp + 31) / 32);
    poz_fill<NM><<<gB, tile, 0, stream>>>(Bc, g_rect.sB, g_rect.Br, nc, K, ldb,
                                          trB, ncp, Kp);
    // Column-major C^T(ncp x Mp) = op_T(Br) * op_N(Ar) = B^T*A^T, so the int32
    // result read row-major with row stride ncp is the row-major product.
    POZ_CB(cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, ncp, Mp, Kp, &i_one, g_rect.Br, CUDA_R_8I,
        Kp, (long long)ncp * Kp, g_rect.Ar, CUDA_R_8I, Kp, (long long)Mp * Kp,
        &i_zero, g_rect.Cr, CUDA_R_32I, ncp, (long long)ncp * Mp, NM,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    poz_crt<NM>
        <<<dim3((nc + 31) / 32, (M + 31) / 32), dim3(8, 32), 0, stream>>>(
            g_rect.Cr, g_rect.sA, g_rect.sB, Cc, M, nc, Mp, ncp, ldc, cColMajor,
            alpha, beta);
  }
}

// General layout-aware native DGEMM: C(M x Ncols) = alpha*(A@B) + beta*C with
// A(M x K), B(K x Ncols), each column-major (X[c*ld + r]) or row-major
// (X[r*ld + c]). A row-major rows x cols matrix with row stride ld is the
// column-major cols x rows transpose with the same ld, so every layout
// combination maps to one cublasDgemm: for row-major C, the column-major
// C^T = op(B)*op(A) with the ops flipped.
static void poz_native_dgemm(double *C, const double *A, const double *B, int M,
                             int Ncols, int K, int lda, int ldb, int ldc,
                             int aColMajor, int bColMajor, int cColMajor,
                             double alpha, double beta, cudaStream_t stream) {
  cublasHandle_t h = poz_cublasHandle();
  if (!h)
    return; // creation failure already reported by POZ_CB
  POZ_CB(cublasSetStream(h, stream));
  if (cColMajor) {
    POZ_CB(cublasDgemm(h, aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T,
                       bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T, M, Ncols, K,
                       &alpha, A, lda, B, ldb, &beta, C, ldc));
  } else {
    POZ_CB(cublasDgemm(h, bColMajor ? CUBLAS_OP_T : CUBLAS_OP_N,
                       aColMajor ? CUBLAS_OP_T : CUBLAS_OP_N, Ncols, M, K,
                       &alpha, B, ldb, A, lda, &beta, C, ldc));
  }
}

// Normalize num_moduli: 0 = native DGEMM; out of range (< 0 or > 14) = 14;
// 1..kOzakiIIMinModuli-1 has no published scaling constant and is refused.
int poz_checkNm(int num_moduli) {
  if (num_moduli < 0 || num_moduli > POZ_MAXNM)
    return POZ_MAXNM;
  if (num_moduli > 0 && num_moduli < (int)kOzakiIIMinModuli) {
    printf("poseidon_ozaki_rt: FATAL num_moduli=%d: Ozaki-II is defined here "
           "for %u..%d moduli\n",
           num_moduli, kOzakiIIMinModuli, POZ_MAXNM);
    abort();
  }
  return num_moduli;
}

// f(std::integral_constant<int, nm>) for a modulus count nm in
// kOzakiIIMinModuli..14, so that f launches the kernels instantiated for nm.
// Any other count has none and aborts.
template <class F, int... I>
void poz_withModuliOf(int nm, F &&f, std::integer_sequence<int, I...>) {
  constexpr int lo = (int)kOzakiIIMinModuli;
  const bool ran =
      ((nm == lo + I && (f(std::integral_constant<int, lo + I>()), true)) ||
       ...);
  if (!ran) {
    printf("poseidon_ozaki_rt: FATAL num_moduli=%d: kernels exist for %u..%d "
           "moduli\n",
           nm, kOzakiIIMinModuli, POZ_MAXNM);
    abort();
  }
}
template <class F> void poz_withModuli(int nm, F &&f) {
  poz_withModuliOf(
      nm, f,
      std::make_integer_sequence<int,
                                 POZ_MAXNM - (int)kOzakiIIMinModuli + 1>());
}
} // namespace

// Select the _ex shape arm in-process: 0 = auto (shape-driven, default),
// 1 = force rectangular, 2 = force square-padded. Overrides
// POSEIDON_OZAKI_PATH. A test and diagnostic entry.
extern "C" void __poseidon_ozaki_set_path(int mode) {
  if (mode < 0 || mode > 2) {
    printf("poseidon_ozaki_rt: FATAL __poseidon_ozaki_set_path(%d): expected 0 "
           "(auto), 1 (rect) or 2 (square)\n",
           mode);
    abort();
  }
  g_pathMode = mode;
}

// Override the rectangular arm's internal column chunk (0 = size it from the
// scratch budget). A test and diagnostic entry.
extern "C" void __poseidon_ozaki_set_rect_chunk(int cols) {
  if (cols < 0) {
    printf("poseidon_ozaki_rt: FATAL __poseidon_ozaki_set_rect_chunk(%d): "
           "expected a non-negative column count\n",
           cols);
    abort();
  }
  g_rectChunk = cols;
}

// Square entry: row-major, leading dim N, no transpose.
extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli) {
  if (N <= 0)
    return;
  num_moduli = poz_checkNm(num_moduli);
  if (transA || transB || (lda && lda != N) || (ldb && ldb != N) ||
      (ldc && ldc != N)) {
    printf("poseidon_ozaki_rt: unsupported GEMM shape (transA=%d transB=%d "
           "lda=%d ldb=%d ldc=%d N=%d)\n",
           transA, transB, lda, ldb, ldc, N);
    return;
  }
  if (num_moduli == 0) {
    // Row-major square NN GEMM == column-major C^T = alpha*B^T*A^T + beta*C^T.
    poz_native_dgemm(C, A, B, N, N, N, N, N, N, /*aColMajor=*/0,
                     /*bColMajor=*/0, /*cColMajor=*/0, alpha, beta, stream);
    return;
  }
  poz_withModuli(num_moduli, [&](auto nm) {
    poz_run_square<decltype(nm)::value>(C, A, B, N, alpha, beta, stream);
  });
}

// General M x Ncols x K entry. The arm is selected by shape, never by the
// caller: rectangular (poz_run_rect) when padWaste >=
// POSEIDON_OZAKI_RECT_MIN_WASTE, else square: zero-pad to
// S = roundup(max(M,Ncols,K),16), repack A and B into square row-major
// scratch, run the square pipeline and crop back. POSEIDON_OZAKI_PATH forces
// either. {a,b,c}ColMajor select each operand's stride convention.
extern "C" void __poseidon_ozaki_dgemm_ex(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, cudaStream_t stream, int num_moduli) {
  if (M <= 0 || Ncols <= 0 || K <= 0)
    return;
  num_moduli = poz_checkNm(num_moduli);
  if (num_moduli == 0) {
    // Native path needs no padding: every {a,b,c}ColMajor combination maps
    // directly onto one cublasDgemm (see poz_native_dgemm).
    poz_native_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor,
                     cColMajor, alpha, beta, stream);
    return;
  }
  int mx = M > Ncols ? M : Ncols;
  if (K > mx)
    mx = K;
  int S = ((mx + 15) / 16) * 16;
  {
    const double padWaste = (double)S * (double)S * (double)S /
                            ((double)M * (double)Ncols * (double)K);
    const int mode = poz_pathMode();
    if (mode == 1 || (mode == 0 && padWaste >= poz_rectMinWaste())) {
      poz_withModuli(num_moduli, [&](auto nm) {
        poz_run_rect<decltype(nm)::value>(C, A, B, M, Ncols, K, lda, ldb, ldc,
                                          aColMajor, bColMajor, cColMajor,
                                          alpha, beta, stream);
      });
      return;
    }
  }
  poz_ensure(S);
  size_t plane = (size_t)S * S;
  POZ_CK(cudaMemsetAsync(g_scratch.As, 0, plane * sizeof(double), stream));
  POZ_CK(cudaMemsetAsync(g_scratch.Bs, 0, plane * sizeof(double), stream));
  dim3 b(16, 16);
  poz_pack<<<dim3((K + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      g_scratch.As, A, M, K, lda, aColMajor, S);
  poz_pack<<<dim3((Ncols + 15) / 16, (K + 15) / 16), b, 0, stream>>>(
      g_scratch.Bs, B, K, Ncols, ldb, bColMajor, S);
  poz_withModuli(num_moduli, [&](auto nm) {
    poz_run_square<decltype(nm)::value>(g_scratch.Cs, g_scratch.As,
                                        g_scratch.Bs, S, 1.0, 0.0, stream);
  });
  poz_crop<<<dim3((Ncols + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      C, g_scratch.Cs, M, Ncols, ldc, cColMajor, S, alpha, beta);
}
