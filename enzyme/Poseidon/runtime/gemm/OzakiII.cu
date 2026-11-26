//===- OzakiII.cu - Ozaki-II GEMM runtime ---------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file reimplements the algorithm of K. Ozaki, Y. Uchino and T. Imamura,
// Ozaki Scheme II (arXiv:2504.08009), in the fast mode of GEMMul8 v3.1.1,
// whose operand scaling GEMMul8's documentation attributes to S. Kawakami and
// D. Takahashi, Improved Scaling for Fast Mode of Ozaki Scheme II
// (arXiv:2606.29129); no code is taken from GEMMul8
// (https://github.com/RIKEN-RCCS/GEMMul8, RIKEN R-CCS, MIT License). Where
// the algorithm leaves a choice to the implementation, the choice is GEMMul8
// v3.1.1's, each value computed here from its defining formula, so that the
// same operands give the same result bit for bit:
// - the moduli, the scaling constants and the CRT constants
//   (src/oz2/common/table.hpp);
// - the shift formula, its rounding and the truncation (src/oz2/scaling/fast);
// - the residue representatives and the byte-weight residue method
//   (src/oz2/mod/mod_core.hpp; SchemeConstants.h).
//
// The pipeline layout (one strided-batched INT8 GEMM per chunk of columns,
// operands read in their own layouts) is this implementation's own.
//
// An FP64 GEMM as row/column scaling shifts -> num_moduli int8 residue planes
// -> one strided-batched INT8 tensor-core GEMM -> double-double CRT. The
// residue and CRT kernels are instantiated per modulus count
// (kOzakiIIMinModuli..14), with the moduli and their byte weights as
// immediates.
//
// Operand scaling (GEMMul8 v3.1.1's fast mode). For each row a of A and each
// column b of B, s = ||a||_2^2 is accumulated in FP64 with every operation
// rounded upward, and the shift is
//   sft = floor(log2P(nm) - log2(s)/2),  log2P(nm) = rd(log2(P-1)/2 - 1/2),
// evaluated in float with directed rounding plus a 2^-23 allowance for the
// hardware log2 (shiftOf). The scaled operand is trunc(2^sft * a), so
// ||a'||_2 <= 2^log2P <= sqrt((P-1)/2) and, by Cauchy-Schwarz, every entry of
// the integer product lies in the symmetric CRT window [-(P-1)/2, (P-1)/2].
// The sum-of-squares reduction order is fixed per operand layout (contiguous
// or strided vector, matching GEMMul8's), so the shifts are reproducible. A
// row whose sum of squares overflows (|a| beyond ~1e154) or underflows to zero
// is outside the fast mode's range, as in GEMMul8.
//
// The per-modulus products are one gpublasGemmStridedBatchedEx(GPUBLAS_OP_T,
// GPUBLAS_OP_N, GPU_R_8I -> GPU_R_32I, GPUBLAS_COMPUTE_32I) call. Residues are
// laid out for its TN form: A residues row-major (column-major view A^T) and
// B residues transposed (column-major view B), so the batched product is the
// column-major C^T = B^T*A^T, i.e. the row-major C the CRT pass reads. Because
// the accumulation is INT32 and |sum_k r_a*r_b| <= K*128*128 < 2^31 for
// K < 2^17, the result is bit-identical to any other correct INT8 GEMM engine.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "GemmRuntime.h"
#include "SchemeConstants.h"

using namespace poseidon::gemm;

namespace {
constexpr int kMaxModuli = (int)kOzakiIIMaxModuli;

// f(std::integral_constant<int, i>) for i = 0..NM-1, so that modulus i and its
// byte weights are compile-time constants in f.
template <class F, int... I>
__device__ __forceinline__ void
forEachModulus(F &&f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>()), ...);
}
template <int NM, class F> __device__ __forceinline__ void forModuli(F &&f) {
  forEachModulus(f, std::make_integer_sequence<int, NM>());
}

// ---------------------------------------------------------------------------
// Scaling shifts.
// ---------------------------------------------------------------------------
// A 32-lane tree over threadIdx.x of a 32-wide block row. A 64-lane wavefront
// holds two such rows, and each half sums in this order (DeviceApi.h), so
// HIP gives the CUDA shifts.
__device__ __forceinline__ double warpSumUp(double s) {
  s = __dadd_ru(s, gpuShflDownSync(0xffffffffu, s, 16));
  s = __dadd_ru(s, gpuShflDownSync(0xffffffffu, s, 8));
  s = __dadd_ru(s, gpuShflDownSync(0xffffffffu, s, 4));
  s = __dadd_ru(s, gpuShflDownSync(0xffffffffu, s, 2));
  s = __dadd_ru(s, gpuShflDownSync(0xffffffffu, s, 1));
  return s;
}

// floor(log2P - log2(s)/2) for s = ||a||_2^2 (an upper bound). Each rounding
// is directed so the shift can only come out smaller: log2P is rounded down,
// the mantissa of s up, and 2^-23 covers the error of the hardware log2.
__device__ __forceinline__ short shiftOf(double s, float log2P) {
  if (s == 0.0)
    return 0;
  const int e = ilogb(s);
  const float m = __double2float_ru(scalbn(s, -e));
  const float mu0 = __fsub_rd(log2P, (float)e * 0.5f);
  const float mu1 = __fadd_ru(__log2f(m) * 0.5f, 0x1p-23f);
  return (short)__float2int_rd(__fsub_rd(mu0, mu1));
}

// Contiguous vectors (vector r is src[r*ld + k], k < K): one block per vector,
// thread t folds k = t, t+256, ..., then a warp tree and a tree over the
// eight warp sums.
__global__ void shiftsContiguousKernel(const double *src, size_t ld, int K,
                                       short *sft, float log2P) {
  __shared__ double sh[kOzakiIINormThreads / 32];
  const double *v = src + (size_t)blockIdx.x * ld;
  double s = 0.0;
  for (int k = threadIdx.x; k < K; k += kOzakiIINormThreads)
    s = __fma_ru(v[k], v[k], s);
  s = warpSumUp(s);
  if ((threadIdx.x & 31) == 0)
    sh[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = (threadIdx.x < kOzakiIINormThreads / 32) ? sh[threadIdx.x] : 0.0;
    s = warpSumUp(s);
    if (threadIdx.x == 0)
      sft[blockIdx.x] = shiftOf(s, log2P);
  }
}

// Strided vectors (vector r is src[k*ld + r]), 32 vectors per 32x32 block:
// thread (x, y) folds k = k0+y, k0+y+32, ... < k1 of vector 32*blockIdx.x + x;
// a shared-memory transpose puts a vector's 32 phases in one warp.
__device__ __forceinline__ double stridedTileSum(const double *src, size_t ld,
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
  return warpSumUp(sh[threadIdx.x][threadIdx.y]);
}

// K < kOzakiIINormSplitK: the whole vector in one tile.
__global__ void shiftsStridedKernel(const double *src, size_t ld, int R, int K,
                                    short *sft, float log2P) {
  const double s = stridedTileSum(src, ld, R, 0, K);
  const int r = blockIdx.x * 32 + threadIdx.y;
  if (threadIdx.x == 0 && r < R)
    sft[r] = shiftOf(s, log2P);
}

// K >= kOzakiIINormSplitK: one partial per (vector, 256-long chunk) ...
__global__ void stridedPartialSumsKernel(const double *src, size_t ld, int R,
                                         int K, double *part) {
  const int k0 = blockIdx.y * kOzakiIINormChunk;
  const double s =
      stridedTileSum(src, ld, R, k0, min(K, k0 + kOzakiIINormChunk));
  const int r = blockIdx.x * 32 + threadIdx.y;
  if (threadIdx.x == 0 && r < R)
    part[(size_t)r * gridDim.y + blockIdx.y] = s;
}

// ... then lane x of a vector's warp folds chunks x, x+32, ... and a warp tree
// finishes it. Block (32, 8): one warp per vector.
__global__ void shiftsStridedReduceKernel(const double *part, int R, int nChunk,
                                          short *sft, float log2P) {
  const int r = blockIdx.x * 8 + threadIdx.y;
  if (r >= R)
    return; // warp-uniform
  double s = 0.0;
  for (int c = threadIdx.x; c < nChunk; c += 32)
    s = __dadd_ru(s, part[(size_t)r * nChunk + c]);
  s = warpSumUp(s);
  if (threadIdx.x == 0)
    sft[r] = shiftOf(s, log2P);
}

// Shifts of the R vectors (length K) of one operand. `part` holds
// R*ceil(K/256) doubles for the strided K >= 1024 case.
void computeShifts(const double *src, size_t ld, int R, int K, bool contiguous,
                   short *sft, double *part, float log2P, gpuStream_t stream) {
  if (contiguous) {
    shiftsContiguousKernel<<<R, kOzakiIINormThreads, 0, stream>>>(src, ld, K,
                                                                  sft, log2P);
    return;
  }
  const int gx = (R + 31) / 32;
  if (K < kOzakiIINormSplitK) {
    shiftsStridedKernel<<<gx, dim3(32, 32), 0, stream>>>(src, ld, R, K, sft,
                                                         log2P);
    return;
  }
  const int nChunk = (K + kOzakiIINormChunk - 1) / kOzakiIINormChunk;
  stridedPartialSumsKernel<<<dim3(gx, nChunk), dim3(32, 32), 0, stream>>>(
      src, ld, R, K, part);
  shiftsStridedReduceKernel<<<(R + 7) / 8, dim3(32, 8), 0, stream>>>(
      part, R, nChunk, sft, log2P);
}

// Scaled operand entry: trunc(2^sft * x), exact (scalbn only moves the
// exponent and |result| < 2^55). scalbn(x, sft), not x * scalbn(1.0, sft): sft
// can exceed 1023 for a row whose norm is tiny, where scalbn(1.0, sft) is inf.
__device__ __forceinline__ long quantize(double x, int sft) {
  return __double2ll_rz(scalbn(x, sft));
}

// The residues of v modulo the first NM moduli, one per residue plane. v is 0
// outside the operand, and the residues of 0 are 0.
template <int NM>
__device__ __forceinline__ void storeResidues(int8_t *dst, size_t plane,
                                              size_t off, long v) {
  forModuli<NM>([&](auto i) {
    constexpr int I = decltype(i)::value;
    constexpr OzakiIIModulus m = ozakiIIModulus(I);
    dst[(size_t)I * plane + off] = ozakiIIResidueI8(v, m);
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
// Blocks in y are numbered from y0 (forGridY).
template <int NM>
__global__ void
fillResiduesKernel(const double *src, const short *sft, int8_t *dst, int R,
                   int K, int ld, int transposed, int Rp, int Kp, unsigned y0) {
  __shared__ long sh[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int by = blockIdx.y + y0;
  const size_t plane = (size_t)Rp * Kp;
  if (!transposed) {
    const int kBase = blockIdx.x * 32, rBase = by * 32;
    const int k = kBase + tx;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = rBase + ty + 8 * j;
      if (r >= Rp || k >= Kp)
        continue;
      const bool inb = (r < R && k < K);
      const long v = inb ? quantize(src[(size_t)r * ld + k], sft[r]) : 0L;
      storeResidues<NM>(dst, plane, (size_t)r * Kp + k, v);
    }
  } else {
    const int rBase = blockIdx.x * 32, kBase = by * 32;
    {
      const int r = rBase + tx;
      const short s = (r < R) ? sft[r] : (short)0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int kl = ty + 8 * j, k = kBase + kl;
        sh[kl][tx] =
            (r < R && k < K) ? quantize(src[(size_t)k * ld + r], s) : 0L;
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int rl = ty + 8 * j, r = rBase + rl, k = kBase + tx;
      if (r >= Rp || k >= Kp)
        continue;
      storeResidues<NM>(dst, plane, (size_t)r * Kp + k, sh[tx][rl]);
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
__device__ __forceinline__ void crtQuad(const int32_t *Cr, size_t off,
                                        size_t plane, const OzakiIIConstants &q,
                                        double *out) {
  double x[4], y[4];
  forModuli<NM>([&](auto i) {
    constexpr int I = decltype(i)::value;
    constexpr OzakiIIModulus m = ozakiIIModulus(I);
    const int4 raw =
        *reinterpret_cast<const int4 *>(Cr + (size_t)I * plane + off);
    const int *rawv = reinterpret_cast<const int *>(&raw);
    const double qhi = q.qhi[I], qlo = q.qlo[I];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const double d = (double)ozakiIIResidueI32(rawv[j], m);
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
    const double quot = rint(q.invP * x[j]);
    if constexpr (NM <= (int)kOzakiIIPIsDouble)
      out[j] = fma(q.pHi, quot, x[j]);
    else
      out[j] = fma(q.pLo, quot, fma(q.pHi, quot, x[j]) + y[j]);
  }
}

// C = alpha*AB + beta*C; beta == 0 never reads C.
__device__ __forceinline__ double axpby(double alpha, double ab, double beta,
                                        double c) {
  return (beta == 0.0) ? alpha * ab : fma(beta, c, alpha * ab);
}

// The four products at (r, c0 .. c0 + 3) of the M x Ncols region, CRT
// recombined from one int4 per residue plane and scaled back by the shifts;
// those past Ncols are left at zero. ldCr is a multiple of 16 and c0 one of 4,
// so every load is 16-byte aligned and stays inside the padded plane.
template <int NM>
__device__ __forceinline__ void
productQuad(const int32_t *Cr, const short *sftA, const short *sftB, int r,
            int c0, int Ncols, int Mp, int ldCr, const OzakiIIConstants &q,
            double prod[4]) {
  double crtv[4];
  crtQuad<NM>(Cr, (size_t)r * ldCr + c0, (size_t)Mp * ldCr, q, crtv);
  const short sa = sftA[r];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int c = c0 + j;
    if (c >= Ncols)
      break;
    prod[j] = scalbn(crtv[j], -(sa + sftB[c]));
  }
}

// The CRT recombine of the M x Ncols region into C = alpha * AB + beta * C.
// Block (8, 32): each thread computes four columns of one row (four
// independent accumulator pairs keep loads in flight), so a block covers a
// 32 x 32 tile.
//   row-major C   : the (r, c) mapping is already coalesced; written out.
//   column-major C: consecutive c are ldc apart, so the tile is transposed
//                   through shared memory and written, and read for beta != 0,
//                   with consecutive r in a warp.
template <int NM>
__global__ void
crtKernel(const int32_t *Cr, const short *sftA, const short *sftB, double *C,
          int M, int Ncols, int Mp, int ldCr, int ldc, int cColMajor,
          double alpha, double beta, const OzakiIIConstants q, unsigned y0) {
  __shared__ double sh[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int cBase = blockIdx.x * 32, rBase = (blockIdx.y + y0) * 32;
  const int r = rBase + ty, c0 = cBase + 4 * tx;
  double prod[4] = {0, 0, 0, 0};
  const bool live = (r < M && c0 < Ncols);
  if (live)
    productQuad<NM>(Cr, sftA, sftB, r, c0, Ncols, Mp, ldCr, q, prod);
  if (!cColMajor) {
    if (!live)
      return;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int c = c0 + j;
      if (c >= Ncols)
        return;
      const size_t o = (size_t)r * ldc + c;
      C[o] = axpby(alpha, prod[j], beta, C[o]);
    }
    return;
  }
  // Thread tid = tx + 8 * ty takes row tid & 31 of the tile, so the 32 lanes
  // of a warp walk 32 consecutive rows of one column.
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
    C[o] = axpby(alpha, sh[rl][cl], beta, C[o]);
  }
}

static inline int padded(int N) {
  return ((N + kOzakiIIPadTile - 1) / kOzakiIIPadTile) * kOzakiIIPadTile;
}
static inline size_t normChunks(int K) {
  return (size_t)((K + kOzakiIINormChunk - 1) / kOzakiIINormChunk);
}

// The CRT constants of nm moduli, computed once per process. Kernels take them
// as arguments, so every device has them.
static const OzakiIIConstants &constantsOf(int nm) {
  static const std::array<OzakiIIConstants, kMaxModuli + 1> table = [] {
    std::array<OzakiIIConstants, kMaxModuli + 1> t{};
    for (int n = (int)kOzakiIIMinModuli; n <= kMaxModuli; ++n)
      t[n] = ozakiIIConstants(n);
    return t;
  }();
  return table[nm];
}

static void checkModuli(int nm) {
  if (nm < (int)kOzakiIIMinModuli || nm > kMaxModuli)
    fail("__poseidon_ozakiii_dgemm with num_moduli = %d: it takes %u to %d "
         "moduli",
         nm, kOzakiIIMinModuli, kMaxModuli);
}

// The device scratch of a dispatch, one allocation: the residue planes of A
// (Mp x Kp) and of a chunk of up to Ncp columns of B, their INT32 products,
// the shifts, and the partial sums of squares of strided vectors.
struct Scratch {
  int Mp = 0, Kp = 0, Ncp = 0;
  // The device held no more than Ncp columns for this Mp and Kp.
  bool columnsCapped = false;
  void *block = nullptr;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  double *part = nullptr;
};
struct OzakiDevice : Device {
  Scratch scratch;
};

// Points the buffers of s into base (null: sizes them only) and returns the
// bytes they take, each 256-byte aligned.
static size_t layoutScratch(Scratch &s, char *base) {
  size_t at = 0;
  auto take = [&](auto *&p, size_t bytes) {
    p = base ? reinterpret_cast<std::remove_reference_t<decltype(p)>>(base + at)
             : nullptr;
    at += (bytes + 255) & ~(size_t)255;
  };
  take(s.Cr, (size_t)kMaxModuli * s.Ncp * s.Mp * sizeof(int32_t));
  take(s.Ar, (size_t)kMaxModuli * s.Mp * s.Kp);
  take(s.Br, (size_t)kMaxModuli * s.Ncp * s.Kp);
  take(s.part,
       (size_t)std::max(s.Mp, s.Ncp) * normChunks(s.Kp) * sizeof(double));
  take(s.sA, (size_t)s.Mp * sizeof(short));
  take(s.sB, (size_t)s.Ncp * sizeof(short));
  return at;
}

// Makes the scratch of `dev` hold an Mp x Kp operand A and the columns of B,
// and returns how many columns a chunk takes. Scratch that holds them already
// is reused. It asks for all Ncols columns; when the device cannot hold them,
// it asks again for half as many, down to one tile of kOzakiIIPadTile columns.
// The chunking is numerically inert: A's shift is a property of a whole row and
// B's of one whole column, so every chunk count returns the same bits. Only an
// out-of-memory status is retried. CUDA documents which errors leave the
// context unusable (cudaErrorIllegalAddress and its kind), and
// gpuErrorMemoryAllocation is not one of them; HIP's documentation says
// nothing either way, so there the retry relies on each call's own status:
// any other failure, including a later call's on a damaged context, is fatal.
static int ensureScratch(OzakiDevice &dev, int Mp, int Kp, int Ncols) {
  Scratch &s = dev.scratch;
  const int want = padded(Ncols);
  if (s.block && s.Mp >= Mp && s.Kp >= Kp && (s.Ncp >= want || s.columnsCapped))
    return std::min(Ncols, s.Ncp);
  Scratch next;
  next.Mp = std::max(Mp, s.Mp);
  next.Kp = std::max(Kp, s.Kp);
  next.Ncp = s.columnsCapped ? want : std::max(want, s.Ncp);
  if (s.block) {
    drainScratch(dev);
    POSEIDON_GEMM_CK(gpuFree(s.block));
  }
  s = Scratch();
  for (;;) {
    const size_t bytes = layoutScratch(next, nullptr);
    const gpuError_t e = gpuMalloc(&next.block, bytes);
    if (e == gpuSuccess)
      break;
    if (e != gpuErrorMemoryAllocation)
      failCall("gpuMalloc", __FILE__, __LINE__, gpuGetErrorString(e));
    // The failure is the last error of this thread; the entry checks that.
    (void)gpuGetLastError();
    if (next.Ncp == kOzakiIIPadTile) {
      size_t freeBytes = 0, totalBytes = 0;
      POSEIDON_GEMM_CK(gpuMemGetInfo(&freeBytes, &totalBytes));
      fail("__poseidon_ozakiii_dgemm needs %zu bytes of device memory for a %d "
           "x %d operand and %d columns of the other, and the device has %zu "
           "bytes free",
           bytes, next.Mp, next.Kp, next.Ncp, freeBytes);
    }
    next.Ncp = std::max(kOzakiIIPadTile,
                        next.Ncp / 2 / kOzakiIIPadTile * kOzakiIIPadTile);
    next.columnsCapped = true;
  }
  layoutScratch(next, static_cast<char *>(next.block));
  s = next;
  return std::min(Ncols, s.Ncp);
}

// The residues of the R vectors of length K of one operand into NM planes of
// Rp x Kp: their shifts (computeShifts), then the scaled vectors' residues.
// `transposed` is the operand being contiguous along R rather than K.
template <int NM>
static void residues(const double *src, int ld, int R, int K, int transposed,
                     short *sft, int8_t *dst, int Rp, int Kp, double *part,
                     float log2P, gpuStream_t stream) {
  computeShifts(src, ld, R, K, /*contiguous=*/!transposed, sft, part, log2P,
                stream);
  const unsigned gr = (Rp + 31) / 32, gk = (Kp + 31) / 32;
  forGridY(transposed ? gk : gr, [&](unsigned y0, unsigned n) {
    fillResiduesKernel<NM>
        <<<dim3(transposed ? gr : gk, n), dim3(32, 8), 0, stream>>>(
            src, sft, dst, R, K, ld, transposed, Rp, Kp, y0);
  });
}

// The NM INT32 products of the residue planes in one strided-batched INT8
// GEMM: column-major C^T (ncp x Mp) = op_T(Br) op_N(Ar) = B^T A^T, so the
// result read row-major with row stride ncp is the row-major product.
static void int8Products(gpublasHandle_t h, const Scratch &sc, int NM, int ncp,
                         int Mp, int Kp) {
  const int32_t one = 1, zero = 0;
  POSEIDON_GEMM_CB(gpublasGemmStridedBatchedEx(
      h, GPUBLAS_OP_T, GPUBLAS_OP_N, ncp, Mp, Kp, &one, sc.Br, GPU_R_8I, Kp,
      (long long)ncp * Kp, sc.Ar, GPU_R_8I, Kp, (long long)Mp * Kp, &zero,
      sc.Cr, GPU_R_32I, ncp, (long long)ncp * Mp, NM, GPUBLAS_COMPUTE_32I,
      GPUBLAS_GEMM_DEFAULT));
}

// C (M x Ncols) = alpha * (A B) + beta * C for A (M x K) and B (K x Ncols),
// each operand row- or column-major per its flag and read in place. A is
// scaled and quantized once for the whole call, its residues being a property
// of A alone; B and C go by chunks of columns (ensureScratch), a pointer
// offset under either storage order.
template <int NM>
static void multiply(Call<OzakiDevice> &call, double *C, const double *A,
                     const double *B, int M, int Ncols, int K, int lda, int ldb,
                     int ldc, int aColMajor, int bColMajor, int cColMajor,
                     double alpha, double beta) {
  if (K >= kOzakiIIMaxK)
    fail("__poseidon_ozakiii_dgemm with K = %d: its INT32 residue products "
         "overflow from K = %d on",
         K, kOzakiIIMaxK);
  const gpuStream_t stream = call.stream;
  const gpublasHandle_t h = call.blas();
  const OzakiIIConstants &q = constantsOf(NM);
  const int Kp = padded(K), Mp = padded(M);
  const int chunk = ensureScratch(call.dev, Mp, Kp, Ncols);
  const Scratch &sc = call.dev.scratch;

  // A column-major A is contiguous along M, a row-major B along Ncols.
  residues<NM>(A, lda, M, K, /*transposed=*/aColMajor != 0, sc.sA, sc.Ar, Mp,
               Kp, sc.part, q.log2P, stream);
  for (int j0 = 0; j0 < Ncols; j0 += chunk) {
    const int nc = std::min(chunk, Ncols - j0), ncp = padded(nc);
    const double *Bc = bColMajor ? B + (size_t)j0 * ldb : B + j0;
    double *Cc = cColMajor ? C + (size_t)j0 * ldc : C + j0;
    residues<NM>(Bc, ldb, nc, K, /*transposed=*/!bColMajor, sc.sB, sc.Br, ncp,
                 Kp, sc.part, q.log2P, stream);
    int8Products(h, sc, NM, ncp, Mp, Kp);
    forGridY((M + 31) / 32, [&](unsigned y0, unsigned n) {
      crtKernel<NM><<<dim3((nc + 31) / 32, n), dim3(8, 32), 0, stream>>>(
          sc.Cr, sc.sA, sc.sB, Cc, M, nc, Mp, ncp, ldc, cColMajor, alpha, beta,
          q, y0);
    });
  }
}

// f(std::integral_constant<int, nm>) for a modulus count nm in
// kOzakiIIMinModuli..14 (checkModuli), so that f launches the kernels
// instantiated for nm.
template <class F, int... I>
void withModuliOf(int nm, F &&f, std::integer_sequence<int, I...>) {
  constexpr int lo = (int)kOzakiIIMinModuli;
  (void)((nm == lo + I && (f(std::integral_constant<int, lo + I>()), true)) ||
         ...);
}
template <class F> void withModuli(int nm, F &&f) {
  withModuliOf(
      nm, f,
      std::make_integer_sequence<int,
                                 kMaxModuli - (int)kOzakiIIMinModuli + 1>());
}
} // namespace

extern "C" void __poseidon_ozakiii_dgemm(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, gpuStream_t stream, int num_moduli) {
  checkModuli(num_moduli);
  runEntry<OzakiDevice>("__poseidon_ozakiii_dgemm", C, M, Ncols, K, ldc,
                        cColMajor, beta, stream,
                        /*scratch=*/true, [&](Call<OzakiDevice> &call) {
                          withModuli(num_moduli, [&](auto nm) {
                            multiply<decltype(nm)::value>(
                                call, C, A, B, M, Ncols, K, lda, ldb, ldc,
                                aColMajor, bColMajor, cColMajor, alpha, beta);
                          });
                        });
}
