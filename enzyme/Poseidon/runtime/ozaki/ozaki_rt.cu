//===- ozaki_rt.cu - Ozaki-II GEMM runtime --------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
// (OzakiII.h). The pipeline layout (one strided-batched INT8 GEMM,
// layout-aware rectangular arm) is original to this implementation.
//
// GEMMul8 (https://github.com/RIKEN-RCCS/GEMMul8): Copyright (c) 2025- RIKEN
// R-CCS, MIT License. Its numerical conventions are followed with
// acknowledgment.
//
// An FP64 GEMM as row/column scaling shifts -> num_moduli int8 residue planes
// -> one strided-batched INT8 tensor-core GEMM -> double-double CRT. The
// residue and CRT kernels are instantiated per modulus count
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
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "../GemmRuntime.h"
#include "OzakiII.h"

namespace {
constexpr int POZ_MAXNM = (int)kOzakiIIMaxModuli;
// Runtime-configured constants (uploaded per num_moduli).
__constant__ double poz_QHi[POZ_MAXNM];
__constant__ double poz_QLo[POZ_MAXNM];
__constant__ double poz_PHi;
__constant__ double poz_PLo;
__constant__ double poz_InvP;
__constant__ float poz_log2P;

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
// A 32-lane tree over threadIdx.x of a 32-wide block row. A 64-lane wavefront
// holds two such rows, and each half sums in this order (hip_as_cuda.h), so
// HIP gives the CUDA shifts.
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
  __shared__ double sh[kOzakiIINormThreads / 32];
  const double *v = src + (size_t)blockIdx.x * ld;
  double s = 0.0;
  for (int k = threadIdx.x; k < K; k += kOzakiIINormThreads)
    s = __fma_ru(v[k], v[k], s);
  s = poz_warpSumRU(s);
  if ((threadIdx.x & 31) == 0)
    sh[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = (threadIdx.x < kOzakiIINormThreads / 32) ? sh[threadIdx.x] : 0.0;
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

// K < kOzakiIINormSplitK: the whole vector in one tile.
__global__ void poz_sftStrided(const double *src, size_t ld, int R, int K,
                               short *sft) {
  const double s = poz_stridedTile(src, ld, R, 0, K);
  const int r = blockIdx.x * 32 + threadIdx.y;
  if (threadIdx.x == 0 && r < R)
    sft[r] = poz_shiftOf(s);
}

// K >= kOzakiIINormSplitK: one partial per (vector, 256-long chunk) ...
__global__ void poz_sumStridedPart(const double *src, size_t ld, int R, int K,
                                   double *part) {
  const int k0 = blockIdx.y * kOzakiIINormChunk;
  const double s =
      poz_stridedTile(src, ld, R, k0, min(K, k0 + kOzakiIINormChunk));
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
    poz_sftContig<<<R, kOzakiIINormThreads, 0, stream>>>(src, ld, K, sft);
    return;
  }
  const int gx = (R + 31) / 32;
  if (K < kOzakiIINormSplitK) {
    poz_sftStrided<<<gx, dim3(32, 32), 0, stream>>>(src, ld, R, K, sft);
    return;
  }
  const int nChunk = (K + kOzakiIINormChunk - 1) / kOzakiIINormChunk;
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

struct PozScratch {
  int Np = 0;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  // Strided-vector partial sums of squares: Np*ceil(Np/256) doubles.
  double *part = nullptr;
  // Padded square F64 operand/result buffers for the square-padded pipeline:
  // an M x Ncols x K source is repacked into As/Bs (S x S, S = pad of
  // max(M,Ncols,K)) and the result cropped back out of Cs.
  double *As = nullptr, *Bs = nullptr, *Cs = nullptr;
};
PozScratch g_scratch;
OzakiIIConstants g_consts;
int g_uploadedNm = 0;

static inline int pozPad(int N) {
  return ((N + kOzakiIIPadTile - 1) / kOzakiIIPadTile) * kOzakiIIPadTile;
}
static inline size_t pozChunks(int K) {
  return (size_t)((K + kOzakiIINormChunk - 1) / kOzakiIINormChunk);
}

static void poz_checkK(int K) {
  if (K < kOzakiIIMaxK)
    return;
  fflush(stdout);
  fprintf(stderr,
          "Poseidon GEMM runtime: __poseidon_ozaki_dgemm with K = %d: its "
          "INT32 residue products overflow from K = %d on\n",
          K, kOzakiIIMaxK);
  abort();
}

void poz_ensure(int N) {
  int Np = pozPad(N);
  if (g_scratch.Np == Np)
    return;
  if (g_scratch.Ar) {
    POSEIDON_GEMM_CK(cudaFree(g_scratch.Ar));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.Br));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.Cr));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.sA));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.sB));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.part));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.As));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.Bs));
    POSEIDON_GEMM_CK(cudaFree(g_scratch.Cs));
  }
  size_t plane = (size_t)Np * Np;
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.Ar, POZ_MAXNM * plane));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.Br, POZ_MAXNM * plane));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.Cr, POZ_MAXNM * plane * 4));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.sA, Np * 2));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.sB, Np * 2));
  POSEIDON_GEMM_CK(
      cudaMalloc(&g_scratch.part, (size_t)Np * pozChunks(Np) * sizeof(double)));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.As, plane * sizeof(double)));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.Bs, plane * sizeof(double)));
  POSEIDON_GEMM_CK(cudaMalloc(&g_scratch.Cs, plane * sizeof(double)));
  g_scratch.Np = Np;
}
void poz_uploadConsts(int nm) {
  if (g_uploadedNm == nm)
    return;
  g_consts = ozakiIIConstants(nm);
  POSEIDON_GEMM_CK(
      cudaMemcpyToSymbol(poz_QHi, g_consts.qhi, nm * sizeof(double)));
  POSEIDON_GEMM_CK(
      cudaMemcpyToSymbol(poz_QLo, g_consts.qlo, nm * sizeof(double)));
  POSEIDON_GEMM_CK(cudaMemcpyToSymbol(poz_PHi, &g_consts.pHi, sizeof(double)));
  POSEIDON_GEMM_CK(cudaMemcpyToSymbol(poz_PLo, &g_consts.pLo, sizeof(double)));
  POSEIDON_GEMM_CK(
      cudaMemcpyToSymbol(poz_InvP, &g_consts.invP, sizeof(double)));
  POSEIDON_GEMM_CK(
      cudaMemcpyToSymbol(poz_log2P, &g_consts.log2P, sizeof(float)));
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
  poz_checkK(N);
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
  cublasHandle_t h = poseidonCublas();
  POSEIDON_GEMM_CB(cublasSetStream(h, stream));
  const int32_t i_one = 1, i_zero = 0;
  POSEIDON_GEMM_CB(cublasGemmStridedBatchedEx(
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

// 0 by shape, 1 rectangular, 2 square-padded (__poseidon_ozaki_set_path).
int g_pathMode = 0;

// Column chunking bounds the residue scratch, POZ_MAXNM*(Kp + 4*Mp) bytes per
// column plus the shift and its partial sums, to this. It is numerically
// inert: A's shift is a per-row property of the whole operand and B's a
// per-column property of one column, so every chunking returns the same bits.
constexpr double kPozRectScratchBytes = 512.0 * 1024 * 1024;

static int poz_rectChunk(int Mp, int Kp, int Ncols) {
  double perCol = (double)POZ_MAXNM * ((double)Kp + 4.0 * (double)Mp) +
                  8.0 * (double)pozChunks(Kp) + 2.0;
  long c = (long)(kPozRectScratchBytes / perCol);
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
    POSEIDON_GEMM_CK(cudaFree(g_rect.Ar));
    POSEIDON_GEMM_CK(cudaFree(g_rect.Br));
    POSEIDON_GEMM_CK(cudaFree(g_rect.Cr));
    POSEIDON_GEMM_CK(cudaFree(g_rect.sA));
    POSEIDON_GEMM_CK(cudaFree(g_rect.sB));
    POSEIDON_GEMM_CK(cudaFree(g_rect.part));
    g_rect = PozRectScratch();
  }
  POSEIDON_GEMM_CK(cudaMalloc(&g_rect.Ar, (size_t)POZ_MAXNM * Mp * Kp));
  POSEIDON_GEMM_CK(cudaMalloc(&g_rect.Br, (size_t)POZ_MAXNM * Ncp * Kp));
  POSEIDON_GEMM_CK(
      cudaMalloc(&g_rect.Cr, (size_t)POZ_MAXNM * Ncp * Mp * sizeof(int32_t)));
  POSEIDON_GEMM_CK(cudaMalloc(&g_rect.sA, (size_t)Mp * sizeof(short)));
  POSEIDON_GEMM_CK(cudaMalloc(&g_rect.sB, (size_t)Ncp * sizeof(short)));
  POSEIDON_GEMM_CK(
      cudaMalloc(&g_rect.part,
                 (size_t)std::max(Mp, Ncp) * pozChunks(Kp) * sizeof(double)));
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
  poz_checkK(K);
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

  cublasHandle_t h = poseidonCublas();
  POSEIDON_GEMM_CB(cublasSetStream(h, stream));
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
    POSEIDON_GEMM_CB(cublasGemmStridedBatchedEx(
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
    fflush(stdout);
    fprintf(stderr,
            "Poseidon GEMM runtime: __poseidon_ozaki_dgemm with num_moduli = "
            "%d: it takes %u to %d moduli\n",
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

extern "C" void __poseidon_ozaki_set_path(int mode) {
  if (mode < 0 || mode > 2) {
    fflush(stdout);
    fprintf(stderr,
            "Poseidon GEMM runtime: __poseidon_ozaki_set_path(%d): expected 0 "
            "(by shape), 1 (rectangular) or 2 (square-padded)\n",
            mode);
    abort();
  }
  g_pathMode = mode;
}

// The pipeline is chosen by shape: an N x N x N row-major product with leading
// dimensions N runs the square pipeline in place; any other runs the
// rectangular one (poz_run_rect) when padWaste reaches kOzakiIIRectMinWaste,
// else it is zero-padded to S = roundup(max(M,Ncols,K),16), repacked into
// square row-major scratch, run square and cropped back.
extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int M, int Ncols, int K,
                                       int lda, int ldb, int ldc, int aColMajor,
                                       int bColMajor, int cColMajor,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli) {
  if (M <= 0 || Ncols <= 0 || K <= 0)
    return;
  int mx = M > Ncols ? M : Ncols;
  if (K > mx)
    mx = K;
  int S = pozPad(mx);
  const double padWaste = (double)S * (double)S * (double)S /
                          ((double)M * (double)Ncols * (double)K);
  const bool squareInPlace = M == Ncols && Ncols == K && lda == K &&
                             ldb == Ncols && ldc == Ncols && !aColMajor &&
                             !bColMajor && !cColMajor;
  if (g_pathMode != 1 && squareInPlace)
    poz_withModuli(num_moduli, [&](auto nm) {
      poz_run_square<decltype(nm)::value>(C, A, B, M, alpha, beta, stream);
    });
  else if (g_pathMode == 1 ||
           (g_pathMode == 0 && padWaste >= kOzakiIIRectMinWaste))
    poz_withModuli(num_moduli, [&](auto nm) {
      poz_run_rect<decltype(nm)::value>(C, A, B, M, Ncols, K, lda, ldb, ldc,
                                        aColMajor, bColMajor, cColMajor, alpha,
                                        beta, stream);
    });
  else {
    poz_ensure(S);
    size_t plane = (size_t)S * S;
    POSEIDON_GEMM_CK(
        cudaMemsetAsync(g_scratch.As, 0, plane * sizeof(double), stream));
    POSEIDON_GEMM_CK(
        cudaMemsetAsync(g_scratch.Bs, 0, plane * sizeof(double), stream));
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
  POSEIDON_GEMM_CK(cudaGetLastError());
}
