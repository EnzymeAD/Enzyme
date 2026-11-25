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
// (OzakiII.h). The pipeline layout (one strided-batched INT8 GEMM per chunk
// of columns, operands read in their own layouts) is original to this
// implementation.
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

// Each operand is quantized, per modulus, into a row-major (R x Kp) int8
// plane whose row is the operand's outer index (A: the output row, B: the
// output column) and whose column is the contraction index k. Read
// column-major with leading dimension Kp these are A^T and B, so the batched
// TN INT8 GEMM yields the row-major C. Per operand only the source layout
// differs: contiguous in k ("direct") or in the outer index ("transposed":
// aColMajor for A, !bColMajor for B).
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

// The device scratch of a dispatch, one allocation: the residue planes of A
// (Mp x Kp) and of a chunk of up to Ncp columns of B, their INT32 products,
// the shifts, and the partial sums of squares of strided vectors.
struct PozScratch {
  int Mp = 0, Kp = 0, Ncp = 0;
  // The device held no more than Ncp columns for this Mp and Kp.
  bool columnsCapped = false;
  void *block = nullptr;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  double *part = nullptr;
};
PozScratch g_scratch;

// Points the buffers of s into base (null: sizes them only) and returns the
// bytes they take, each 256-byte aligned.
static size_t pozLayout(PozScratch &s, char *base) {
  size_t at = 0;
  auto take = [&](auto *&p, size_t bytes) {
    p = base ? reinterpret_cast<std::remove_reference_t<decltype(p)>>(base + at)
             : nullptr;
    at += (bytes + 255) & ~(size_t)255;
  };
  take(s.Cr, (size_t)POZ_MAXNM * s.Ncp * s.Mp * sizeof(int32_t));
  take(s.Ar, (size_t)POZ_MAXNM * s.Mp * s.Kp);
  take(s.Br, (size_t)POZ_MAXNM * s.Ncp * s.Kp);
  take(s.part,
       (size_t)std::max(s.Mp, s.Ncp) * pozChunks(s.Kp) * sizeof(double));
  take(s.sA, (size_t)s.Mp * sizeof(short));
  take(s.sB, (size_t)s.Ncp * sizeof(short));
  return at;
}

// Makes g_scratch hold an Mp x Kp operand A and the columns of B, and returns
// how many columns a chunk takes. Scratch that holds them already is reused.
// It asks for all Ncols columns; when the device cannot hold them, it asks
// again for half as many, down to one tile of kOzakiIIPadTile columns. The
// chunking is numerically inert: A's shift is a property of a whole row and
// B's of one whole column, so every chunk count returns the same bits. Only an
// out-of-memory status is retried. CUDA documents which errors leave the
// context unusable (cudaErrorIllegalAddress and its kind), and
// cudaErrorMemoryAllocation is not one of them; HIP's documentation says
// nothing either way, so there the retry relies on each call's own status:
// any other failure, including a later call's on a damaged context, is fatal.
static int poz_ensure(int Mp, int Kp, int Ncols) {
  PozScratch &s = g_scratch;
  const int want = pozPad(Ncols);
  if (s.block && s.Mp >= Mp && s.Kp >= Kp && (s.Ncp >= want || s.columnsCapped))
    return std::min(Ncols, s.Ncp);
  PozScratch next;
  next.Mp = std::max(Mp, s.Mp);
  next.Kp = std::max(Kp, s.Kp);
  next.Ncp = s.columnsCapped ? want : std::max(want, s.Ncp);
  if (s.block)
    POSEIDON_GEMM_CK(cudaFree(s.block));
  s = PozScratch();
  for (;;) {
    const size_t bytes = pozLayout(next, nullptr);
    const cudaError_t e = cudaMalloc(&next.block, bytes);
    if (e == cudaSuccess)
      break;
    if (e != cudaErrorMemoryAllocation)
      poseidonGemmFatal("cudaMalloc", __FILE__, __LINE__,
                        cudaGetErrorString(e));
    // The failure is the last error of this thread; the entry checks that.
    (void)cudaGetLastError();
    if (next.Ncp == kOzakiIIPadTile) {
      size_t freeBytes = 0, totalBytes = 0;
      POSEIDON_GEMM_CK(cudaMemGetInfo(&freeBytes, &totalBytes));
      fflush(stdout);
      fprintf(stderr,
              "Poseidon GEMM runtime: __poseidon_ozaki_dgemm needs %zu bytes "
              "of device memory for a %d x %d operand and %d columns of the "
              "other, and the device has %zu bytes free\n",
              bytes, next.Mp, next.Kp, next.Ncp, freeBytes);
      abort();
    }
    next.Ncp = std::max(kOzakiIIPadTile,
                        next.Ncp / 2 / kOzakiIIPadTile * kOzakiIIPadTile);
    next.columnsCapped = true;
  }
  pozLayout(next, static_cast<char *>(next.block));
  s = next;
  return std::min(Ncols, s.Ncp);
}

// C(M x Ncols) = alpha*(A@B) + beta*C, A(M x K), B(K x Ncols), each operand
// row- or column-major per the {a,b,c}ColMajor flags and read in place: every
// buffer is sized to the real extents, rounded up to kOzakiIIPadTile for the
// INT8 leading dimension. A is scaled and quantized once for the whole
// dispatch because its residues are a property of A alone; B and C go by
// chunks of columns (poz_ensure).
template <int NM>
static void poz_run(double *C, const double *A, const double *B, int M,
                    int Ncols, int K, int lda, int ldb, int ldc, int aColMajor,
                    int bColMajor, int cColMajor, double alpha, double beta,
                    cudaStream_t stream) {
  poz_checkK(K);
  poz_uploadConsts(NM);
  const int Kp = pozPad(K), Mp = pozPad(M);
  const int chunk = poz_ensure(Mp, Kp, Ncols);
  const PozScratch &sc = g_scratch;
  const dim3 tile(32, 8);

  // A, once for the whole dispatch.
  const int tr = aColMajor ? 1 : 0; // col-major A is contiguous in m, not k
  poz_shifts(A, lda, M, K, /*contiguous=*/!tr, sc.sA, sc.part, stream);
  const dim3 g = tr ? dim3((Mp + 31) / 32, (Kp + 31) / 32)
                    : dim3((Kp + 31) / 32, (Mp + 31) / 32);
  poz_fill<NM><<<g, tile, 0, stream>>>(A, sc.sA, sc.Ar, M, K, lda, tr, Mp, Kp);

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

    poz_shifts(Bc, ldb, nc, K, /*contiguous=*/!trB, sc.sB, sc.part, stream);
    const dim3 gB = trB ? dim3((ncp + 31) / 32, (Kp + 31) / 32)
                        : dim3((Kp + 31) / 32, (ncp + 31) / 32);
    poz_fill<NM>
        <<<gB, tile, 0, stream>>>(Bc, sc.sB, sc.Br, nc, K, ldb, trB, ncp, Kp);
    // Column-major C^T(ncp x Mp) = op_T(Br) * op_N(Ar) = B^T*A^T, so the int32
    // result read row-major with row stride ncp is the row-major product.
    POSEIDON_GEMM_CB(cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, ncp, Mp, Kp, &i_one, sc.Br, CUDA_R_8I, Kp,
        (long long)ncp * Kp, sc.Ar, CUDA_R_8I, Kp, (long long)Mp * Kp, &i_zero,
        sc.Cr, CUDA_R_32I, ncp, (long long)ncp * Mp, NM, CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT));
    poz_crt<NM>
        <<<dim3((nc + 31) / 32, (M + 31) / 32), dim3(8, 32), 0, stream>>>(
            sc.Cr, sc.sA, sc.sB, Cc, M, nc, Mp, ncp, ldc, cColMajor, alpha,
            beta);
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

extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int M, int Ncols, int K,
                                       int lda, int ldb, int ldc, int aColMajor,
                                       int bColMajor, int cColMajor,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli) {
  if (M <= 0 || Ncols <= 0 || K <= 0)
    return;
  poz_withModuli(num_moduli, [&](auto nm) {
    poz_run<decltype(nm)::value>(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor,
                                 bColMajor, cColMajor, alpha, beta, stream);
  });
  POSEIDON_GEMM_CK(cudaGetLastError());
}
