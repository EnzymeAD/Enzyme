// ozaki_rt.cu: Poseidon host-dispatch runtime for Ozaki Scheme II.
//
// Provenance: this file is an independent implementation of the published
// Ozaki Scheme II algorithm (K. Ozaki, Y. Uchino, T. Imamura,
// arXiv:2504.08009). No code is derived from the reference GEMMul8 library
// (MIT licensed); GEMMul8 is used only as a benchmark reference and as the
// layout-compatibility target for the shared cuBLAS INT8 strided-batched
// entry point. The Barrett-multiply residue reduction and the coalesced
// bit-pattern magnitude scan are original to this implementation.
//
// Exposes the host entry points the Poseidon host-module pass substitutes for
// a raised FP64 GEMM kernel launch: row/column magnitude scan -> num_moduli
// int8 residue planes -> one strided-batched INT8 tensor-core GEMM ->
// double-double CRT, at a solver-chosen num_moduli. The CRT constants and the
// integer scaling budget (beta) are recomputed per nm at runtime; the modulus
// and moduli-count tables are shared with the compiler through OzakiII.h.
//
// The per-modulus products are one cublasGemmStridedBatchedEx(CUBLAS_OP_T,
// CUBLAS_OP_N, CUDA_R_8I -> CUDA_R_32I, CUBLAS_COMPUTE_32I) call. Residues are
// laid out for its TN form: A residues row-major (column-major view A^T) and
// B residues transposed (column-major view B), so the batched product is the
// column-major C^T = B^T*A^T, i.e. the row-major C the CRT pass reads. Because
// the accumulation is INT32 and the scaling budget keeps every partial sum
// below 2^31, the result is bit-identical to any other correct INT8 GEMM
// engine at the same nm.
//
// Semantics: C = alpha*(A@B) + beta*C. __poseidon_ozaki_dgemm handles the
// square, row-major, leading-dim-N, no-transpose case; __poseidon_ozaki_dgemm_ex
// handles general M x Ncols x K layout-aware shapes by padding to a square.
// Scratch buffers (sized for the max 14 moduli) and the per-nm CRT constants
// are cached across calls. Forward dispatch only.
//
// num_moduli == 0 selects the native cuBLAS DGEMM path (no emulation), which
// the solver routes to when the cost model prices it below every Ozaki rung.
//
// Operand scaling rule: two arms, selected by the POSEIDON_OZAKI_SCALE
// environment variable or __poseidon_ozaki_set_scale_rule().
//
//   "maxbeta" (default): one global bit budget
//   beta = clamp((bitlen(P)-3-ceil(log2 K))/2, 1, 50) applied to the per-row /
//   per-column max, so the scaled inner product is bounded by the worst case
//   K*max_a*max_b <= 2^(bitlen(P)-3).
//
//   "norm2": the Cauchy-Schwarz rule of the published scheme: the shift is
//   chosen so each row's / column's scaled 2-norm reaches sqrt((P-1)/2), hence
//   |sum a'*b'| <= ||a'||_2*||b'||_2 <= (P-1)/2, the symmetric-CRT window with
//   no slack given away.
//
// The arms differ only in how sftA/sftB are computed; the residue fill, the
// batched INT8 GEMM and the CRT recombine are shared.
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../../OzakiII.h"

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
// Runtime-configured CRT constants (uploaded per num_moduli).
__constant__ int poz_nm;
__constant__ int poz_P[POZ_MAXNM];
// Barrett magics for the moduli: poz_M[i] = 2^64/p + 1 (Lemire fastmod) and
// poz_T32[i] = 2^32 mod p, which together turn every residue into multiplies.
__constant__ unsigned long long poz_M[POZ_MAXNM];
__constant__ unsigned int poz_T32[POZ_MAXNM];
__constant__ double poz_QHi[POZ_MAXNM];
__constant__ double poz_QLo[POZ_MAXNM];
__constant__ double poz_PHi;
__constant__ double poz_PLo;
__constant__ double poz_InvP;

// Lazily-created, process-lifetime cuBLAS handle (same caching convention as
// the Ozaki scratch/constants); the stream is set on every dispatch.
cublasHandle_t poz_cublasHandle() {
  static cublasHandle_t h = nullptr;
  if (!h)
    POZ_CB(cublasCreate(&h));
  return h;
}

// n mod d for 32-bit n, using Lemire's fastmod magic M = 2^64/d + 1: two
// multiplies instead of the ~70-instruction integer-division sequence NVCC
// emits for a runtime divisor. (A double-reciprocal quotient would be shorter
// still, but FP64 is exactly the pipeline these GPUs deprioritize, and it
// measured slower than the division it replaced.)
__device__ __forceinline__ unsigned poz_fastmod(unsigned n,
                                                unsigned long long M,
                                                unsigned d) {
  return (unsigned)__umul64hi(M * (unsigned long long)n, (unsigned long long)d);
}

// |a| < 2^beta <= 2^50, so |a| splits into two 32-bit halves that are each
// reduced with fastmod and recombined through 2^32 mod p; the recombination
// stays below 2^16, so a third fastmod finishes it. The sign is reapplied
// afterwards to reproduce C's truncating % exactly.
__device__ __forceinline__ int8_t poz_modI8(long a, int p, unsigned long long M,
                                            unsigned t32) {
  unsigned long long ua = (unsigned long long)(a < 0 ? -a : a);
  unsigned rhi = poz_fastmod((unsigned)(ua >> 32), M, (unsigned)p);
  unsigned rlo = poz_fastmod((unsigned)ua, M, (unsigned)p);
  long r = (long)poz_fastmod(rhi * t32 + rlo, M, (unsigned)p);
  if (a < 0)
    r = -r;
  int h = p / 2;
  if (r > h)
    r -= p;
  else if (r < -h)
    r += p;
  if ((p & 1) == 0 && r == h)
    r -= p;
  return (int8_t)r;
}

// The residue-GEMM output is int32, so one fastmod on |raw| suffices.
__device__ __forceinline__ int poz_modI32(int raw, int p, unsigned long long M) {
  unsigned ur = (unsigned)(raw < 0 ? -raw : raw);
  int r = (int)poz_fastmod(ur, M, (unsigned)p);
  if (raw < 0)
    r = -r;
  int h = p / 2;
  if (r > h)
    r -= p;
  else if (r < -h)
    r += p;
  return r;
}

// Row maxima of |A| and column maxima of |B|, both read along the CONTIGUOUS
// axis: 32x8 threads walk a 32x32 tile, the A reduction runs across the warp
// (shuffle) and the B reduction down the tile (shared), so each matrix is
// streamed exactly once at full coalescing. Maxima are folded with atomicMax on
// the IEEE bit pattern of the absolute value, which is monotone over the
// non-negative doubles, so the result is order-independent and identical to a
// sequential scan for finite inputs.
__device__ __forceinline__ unsigned long long poz_absKey(double v) {
  return (unsigned long long)__double_as_longlong(fabs(v));
}

__global__ void poz_absMax(const double *A, const double *B,
                           unsigned long long *mxA, unsigned long long *mxB,
                           int N) {
  __shared__ unsigned long long shB[8][32];
  int c = blockIdx.x * 32 + threadIdx.x;
  int rBase = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  unsigned long long bAcc = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int r = rBase + ty + 8 * j;
    unsigned long long a = 0, b = 0;
    if (r < N && c < N) {
      a = poz_absKey(A[(size_t)r * N + c]);
      b = poz_absKey(B[(size_t)r * N + c]);
    }
    // Row max of A: reduce across the 32 lanes of this warp (blockDim.x == 32).
    for (int off = 16; off; off >>= 1) {
      unsigned long long o = __shfl_xor_sync(0xffffffffu, a, off);
      a = a > o ? a : o;
    }
    if (tx == 0 && r < N)
      atomicMax(&mxA[r], a);
    bAcc = bAcc > b ? bAcc : b;
  }
  // Column max of B: fold the 8 y-rows in shared, then one atomic per column.
  shB[ty][tx] = bAcc;
  __syncthreads();
  if (ty == 0 && c < N) {
    unsigned long long m = shB[0][tx];
#pragma unroll
    for (int j = 1; j < 8; j++)
      m = m > shB[j][tx] ? m : shB[j][tx];
    atomicMax(&mxB[c], m);
  }
}

__global__ void poz_shifts(const unsigned long long *mxA,
                           const unsigned long long *mxB, short *sftA,
                           short *sftB, int N, int beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  double ma = __longlong_as_double((long long)mxA[i]);
  double mb = __longlong_as_double((long long)mxB[i]);
  sftA[i] = (short)(ma > 0 ? beta - 1 - ilogb(ma) : 0);
  sftB[i] = (short)(mb > 0 ? beta - 1 - ilogb(mb) : 0);
}

// norm2 arm: same 32x8-over-32x32-tile traversal as poz_absMax, but each tile
// also folds the sum of squares of its 32 A-row / 32 B-column entries into a
// per-(row, blockIdx.x) / per-(column, blockIdx.y) partial. Partials, not
// atomicAdd: a float sum folded by atomics depends on block scheduling and the
// shift could flip by one between runs of the same binary. The two-stage form
// reduces in a fixed lane order.
// The sums of squares are accumulated in FP32, not FP64: they are consumed by
// a floor() to an integer power of two, so the norm only has to be trustworthy
// as a bound, and FP32 summation of K terms is relative-accurate to K*2^-24,
// which poz_run_square's capEff margin absorbs; in FP64 the scan would run on
// the deprioritized FP64 pipe. Extreme ranges (a square that overflows or
// flushes in FP32) are caught in poz_sftFromNorm and fall back to the
// conservative max*sqrt(K) bound.
__global__ void poz_scanNorm(const double *A, const double *B,
                             unsigned long long *mxA, unsigned long long *mxB,
                             float *psA, float *psB, int N, int gridX,
                             int gridY) {
  __shared__ unsigned long long shB[8][32];
  __shared__ float shBs[8][32];
  int c = blockIdx.x * 32 + threadIdx.x;
  int rBase = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  unsigned long long bAcc = 0;
  float bSq = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int r = rBase + ty + 8 * j;
    unsigned long long a = 0, b = 0;
    float avf = 0.0f, bvf = 0.0f;
    if (r < N && c < N) {
      double av = A[(size_t)r * N + c];
      double bv = B[(size_t)r * N + c];
      a = poz_absKey(av);
      b = poz_absKey(bv);
      avf = (float)av;
      bvf = (float)bv;
    }
    float aSq = avf * avf;
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      unsigned long long o = __shfl_xor_sync(0xffffffffu, a, off);
      a = a > o ? a : o;
      aSq += __shfl_xor_sync(0xffffffffu, aSq, off); // butterfly: fixed order
    }
    if (tx == 0 && r < N) {
      atomicMax(&mxA[r], a);
      psA[(size_t)r * gridX + blockIdx.x] = aSq;
    }
    bAcc = bAcc > b ? bAcc : b;
    bSq += bvf * bvf;
  }
  shB[ty][tx] = bAcc;
  shBs[ty][tx] = bSq;
  __syncthreads();
  if (ty == 0 && c < N) {
    unsigned long long m = shB[0][tx];
    float s = shBs[0][tx];
#pragma unroll
    for (int j = 1; j < 8; j++) {
      m = m > shB[j][tx] ? m : shB[j][tx];
      s += shBs[j][tx];
    }
    atomicMax(&mxB[c], m);
    psB[(size_t)c * gridY + blockIdx.y] = s;
  }
}

// Reduce the per-tile partials in a fixed order and turn each 2-norm into the
// integer shift. `capEff` already carries the round-to-nearest and summation
// safety margins (see poz_run_square).
__device__ __forceinline__ short poz_sftFromNorm(double s2, double mx,
                                                 double capEff, int K) {
  if (!(mx > 0.0))
    return 0; // all-zero row/column: residues are zero, sft irrelevant
  // s2 overflows only for ||a||2 > 1.3e154 and flushes to zero only for
  // |a| < 1.5e-162 throughout; max*sqrt(K) is a valid (conservative) upper
  // bound on ||a||2 in both cases, so the arm degrades to the worst-case bound
  // rather than producing a wrong shift.
  double nrm = (s2 > 0.0 && isfinite(s2)) ? sqrt(s2) : mx * sqrt((double)K);
  int sft = (int)floor(log2(capEff) - log2(nrm));
  // log2 is faithful to <1 ulp, so the floor can be off by one either way.
  // Repair it exactly; both loops terminate in at most one step in practice.
#pragma unroll
  for (int t = 0; t < 3; t++)
    if (scalbn(nrm, sft) > capEff)
      --sft;
#pragma unroll
  for (int t = 0; t < 3; t++)
    if (scalbn(nrm, sft + 1) <= capEff)
      ++sft;
  return (short)sft;
}

__global__ void poz_shiftsNorm(const float *psA, const float *psB,
                               const unsigned long long *mxA,
                               const unsigned long long *mxB, short *sftA,
                               short *sftB, int N, int gridX, int gridY,
                               double capEff, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  // One thread per row/column over ceil(N/32) partials: N·ceil(N/32) FP64 adds
  // for the whole matrix, so this reduction can stay in FP64 for free.
  double sa = 0.0, sb = 0.0;
  for (int j = 0; j < gridX; j++)
    sa += (double)psA[(size_t)i * gridX + j];
  for (int j = 0; j < gridY; j++)
    sb += (double)psB[(size_t)i * gridY + j];
  sftA[i] = poz_sftFromNorm(sa, __longlong_as_double((long long)mxA[i]), capEff,
                            K);
  sftB[i] = poz_sftFromNorm(sb, __longlong_as_double((long long)mxB[i]), capEff,
                            K);
}

// Residue fill for the batched TN INT8 GEMM. A residues keep the row-major
// (row, k) layout, so their column-major view with leading dimension Np is A^T
// and CUBLAS_OP_N reproduces it. B residues are written TRANSPOSED, at
// (k_out * Np + k_contract), so their column-major view is B itself and
// CUBLAS_OP_T reproduces B^T. Ozaki's operands are square after padding, so the
// pad region {r >= N or k >= N} is closed under transposition: zeroing offA and
// offB for exactly those threads clears every padded entry of both planes once.
__global__ void poz_fillRes(const double *A, const double *B, const short *sftA,
                            const short *sftB, int8_t *Ar, int8_t *Br, int N,
                            int Np) {
  // 32x8 threads over a 32x32 tile. A's residues are written straight out
  // (the source's contiguous axis is also the residue plane's). B's scaled
  // integers are staged in shared and written back with the tile transposed,
  // so consecutive lanes still store consecutive bytes.
  __shared__ long shB[32][33];
  int kBase = blockIdx.x * 32, rBase = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  size_t plane = (size_t)Np * Np;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int rl = ty + 8 * j, r = rBase + rl, k = kBase + tx;
    bool inb = (r < N && k < N);
    long ap = 0, bp = 0;
    if (inb) {
      // scalbn(x, sft), NOT x * scalbn(1.0, sft): under the norm2 arm sft can
      // legally exceed 1023 for a row whose 2-norm is subnormal-small, and
      // scalbn(1.0, 1077) is inf while scalbn(x, 1077) is exact. Identical for
      // every shift the maxbeta arm produces on normal data.
      ap = llround(scalbn(A[(size_t)r * N + k], sftA[r]));
      bp = llround(scalbn(B[(size_t)r * N + k], sftB[k]));
    }
    shB[rl][tx] = bp;
    if (r < Np && k < Np) {
      size_t offA = (size_t)r * Np + k;
      for (int i = 0; i < poz_nm; i++)
        Ar[i * plane + offA] =
            inb ? poz_modI8(ap, poz_P[i], poz_M[i], poz_T32[i]) : (int8_t)0;
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int kl = ty + 8 * j, k = kBase + kl, r = rBase + tx;
    if (k >= Np || r >= Np)
      continue;
    bool inb = (r < N && k < N);
    long bp = shB[tx][kl];
    size_t offB = (size_t)k * Np + r;
    for (int i = 0; i < poz_nm; i++)
      Br[i * plane + offB] =
          inb ? poz_modI8(bp, poz_P[i], poz_M[i], poz_T32[i]) : (int8_t)0;
  }
}

// Four columns per thread, loaded as one int4 per residue plane: the recombine
// is a pure memory stream over nm int32 planes, so the four independent
// accumulator pairs are what keeps enough loads in flight to saturate it. Np is
// a multiple of 16, so every int4 load is 16-byte aligned and any read past
// column N still lands inside the padded plane.
__global__ void poz_crt(const int32_t *Cr, const short *sftA, const short *sftB,
                        double *C, int N, int Np, double alpha, double beta) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (r >= N || c0 >= N)
    return;
  size_t off = (size_t)r * Np + c0, plane = (size_t)Np * Np;
  double x[4] = {0, 0, 0, 0}, y[4] = {0, 0, 0, 0};
  for (int i = 0; i < poz_nm; i++) {
    int4 raw = *reinterpret_cast<const int4 *>(Cr + i * plane + off);
    const int *rawv = reinterpret_cast<const int *>(&raw);
    int p = poz_P[i];
    unsigned long long M = poz_M[i];
    double qhi = poz_QHi[i], qlo = poz_QLo[i];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      double d = (double)poz_modI32(rawv[j], p, M);
      x[j] = fma(qhi, d, x[j]);
      y[j] = fma(qlo, d, y[j]);
    }
  }
  short sa = sftA[r];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int c = c0 + j;
    if (c >= N)
      return;
    // The quotient must come from x+y, not from the error-free part x alone.
    // y is the accumulated qPiLo residual, up to ~2^-29 of P, so a quotient
    // taken from x is wrong whenever the exact product lands within 2^-29·P of
    // ±P/2 and the answer is then off by exactly P. The maxbeta arm never gets
    // closer than P/8 and cannot hit it; the norm2 arm deliberately runs the
    // Cauchy-Schwarz bound up to (P-1)/2, where a rank-1 or near-parallel
    // operand pair saturates it exactly. One FP64 add narrows the band from
    // 2^-28.4 to 2^-52. The reconstruction itself is unchanged.
    double quot = rint(poz_InvP * (x[j] + y[j]));
    double crtv = fma(poz_PLo, quot, fma(poz_PHi, quot, x[j]) + y[j]);
    double prod = scalbn(crtv, -(sa + sftB[c]));
    size_t o = (size_t)r * N + c;
    C[o] = (beta == 0.0) ? alpha * prod : alpha * prod + beta * C[o];
  }
}

// Host-side per-nm CRT constant computation: qPi_i = (M_i·y_i) mod P, qPiHi
// truncated to 41 bits for the FMA recombine, budget = ⌊log2 P⌋−3 for the
// scaling beta.
static long pozModInv(long a, long m) {
  long o_r = ((a % m) + m) % m, r = m, o_s = 1, s = 0;
  while (r) {
    long q = o_r / r, t;
    t = o_r - q * r; o_r = r; r = t;
    t = o_s - q * s; o_s = s; s = t;
  }
  return ((o_s % m) + m) % m;
}
static int pozBitLen(__int128 v) {
  int n = 0;
  while (v) { ++n; v >>= 1; }
  return n;
}
struct PozConsts {
  int nm = 0, budgetBits = 0;
  int mod[POZ_MAXNM];
  unsigned long long magic[POZ_MAXNM];
  unsigned t32[POZ_MAXNM];
  double qhi[POZ_MAXNM], qlo[POZ_MAXNM], pHi, pLo, invP;
  // norm2 arm: the largest scaled 2-norm a row/column may reach,
  // sqrt((P-1)/2). Then |Σ a'·b'| <= ||a'||₂·||b'||₂ <= (P-1)/2, the exact
  // symmetric-CRT window. long double (64-bit significand) carries P <= 2^111
  // to a relative 2^-64, far finer than the integer floor that consumes it.
  double capNrm = 0.0;
};
static void pozComputeConsts(int nm, PozConsts &c) {
  c.nm = nm;
  __int128 P = 1;
  for (int i = 0; i < nm; i++) {
    unsigned p = (unsigned)kOzakiIIModuli[i];
    c.mod[i] = kOzakiIIModuli[i];
    c.magic[i] = ~0ull / p + 1;
    c.t32[i] = (unsigned)((1ull << 32) % p);
    P *= kOzakiIIModuli[i];
  }
  double pHiPos = (double)P;
  c.pHi = -pHiPos;
  c.pLo = -(double)(P - (__int128)pHiPos);
  c.invP = 1.0 / (double)P;
  c.budgetBits = pozBitLen(P) - 3;
  c.capNrm = (double)sqrtl(((long double)P - 1.0L) / 2.0L);
  // Truncate every qPi at ONE global granularity (2^(bitlen(P)-41)): per-index
  // granularities make the recombine x = sum qhi_i*d_i round at ulp(x) when the
  // e_i bitlens spread (nm=12/13), deploying LESS accurately than nm=10.
  int pbl = pozBitLen(P);
  for (int i = 0; i < nm; i++) {
    long p = kOzakiIIModuli[i];
    __int128 Mi = P / p;
    long yi = pozModInv((long)(Mi % p), p);
    __int128 e = (Mi * (__int128)yi) % P;
    __int128 hiI = e;
    if (pbl > 41) { int sh = pbl - 41; hiI = (e >> sh) << sh; }
    c.qhi[i] = (double)hiI;
    c.qlo[i] = (double)(e - hiI);
  }
}

struct PozScratch {
  int Np = 0;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  unsigned long long *mA = nullptr, *mB = nullptr; // |.| maxima, atomicMax keys
  // norm2 arm: per-(row, tile-column) and per-(column, tile-row) partial sums
  // of squares, FP32 (see poz_scanNorm). Np*ceil(Np/32) floats each
  // (8.4 MB at N=8192).
  float *pA = nullptr, *pB = nullptr;
  // Padded square F64 operand/result buffers for the non-square (_ex) path:
  // a non-square M×Ncols×K source is repacked into As/Bs (S×S, S=pad of
  // max(M,Ncols,K)) so the square pipeline can run, then Cs is cropped back.
  double *As = nullptr, *Bs = nullptr, *Cs = nullptr;
};
PozScratch g_scratch;
PozConsts g_consts;
int g_uploadedNm = 0;

// Scaling arm: 0 = maxbeta (default), 1 = norm2. Read once from
// POSEIDON_OZAKI_SCALE; __poseidon_ozaki_set_scale_rule overrides in-process so
// a harness can A/B both arms in one binary.
int g_scaleRule = -1;
int poz_scaleRule() {
  if (g_scaleRule < 0) {
    const char *e = getenv("POSEIDON_OZAKI_SCALE");
    if (!e || !*e)
      g_scaleRule = 0;
    else if (!strcmp(e, "norm2"))
      g_scaleRule = 1;
    else if (!strcmp(e, "maxbeta"))
      g_scaleRule = 0;
    else {
      // Never guess: an unrecognised arm name is a configuration error, and
      // silently running the other rule would misattribute every number.
      printf("poseidon_ozaki_rt: FATAL unknown POSEIDON_OZAKI_SCALE='%s' "
             "(expected 'maxbeta' or 'norm2')\n",
             e);
      abort();
    }
  }
  return g_scaleRule;
}

// INT8 tensor-core GEMM leading dimensions must be multiples of 4; the residue
// planes are therefore held at Np = roundup(N, 16), which also keeps every
// per-modulus plane base 16-byte aligned. For the sizes the solver dispatches
// (multiples of 16 in ozp, and the _ex path's own S = roundup(.,16)) Np == N,
// so the padding costs nothing.
static inline int pozPad(int N) { return ((N + 15) / 16) * 16; }

void poz_ensure(int N) {
  int Np = pozPad(N);
  if (g_scratch.Np == Np)
    return;
  if (g_scratch.Ar) {
    cudaFree(g_scratch.Ar); cudaFree(g_scratch.Br); cudaFree(g_scratch.Cr);
    cudaFree(g_scratch.sA); cudaFree(g_scratch.sB);
    cudaFree(g_scratch.mA); cudaFree(g_scratch.mB);
    cudaFree(g_scratch.As); cudaFree(g_scratch.Bs); cudaFree(g_scratch.Cs);
    if (g_scratch.pA) { cudaFree(g_scratch.pA); cudaFree(g_scratch.pB); }
    g_scratch.pA = g_scratch.pB = nullptr;
  }
  size_t plane = (size_t)Np * Np;
  POZ_CK(cudaMalloc(&g_scratch.Ar, POZ_MAXNM * plane));
  POZ_CK(cudaMalloc(&g_scratch.Br, POZ_MAXNM * plane));
  POZ_CK(cudaMalloc(&g_scratch.Cr, POZ_MAXNM * plane * 4));
  POZ_CK(cudaMalloc(&g_scratch.sA, Np * 2));
  POZ_CK(cudaMalloc(&g_scratch.sB, Np * 2));
  POZ_CK(cudaMalloc(&g_scratch.mA, Np * sizeof(unsigned long long)));
  POZ_CK(cudaMalloc(&g_scratch.mB, Np * sizeof(unsigned long long)));
  POZ_CK(cudaMalloc(&g_scratch.As, plane * sizeof(double)));
  POZ_CK(cudaMalloc(&g_scratch.Bs, plane * sizeof(double)));
  POZ_CK(cudaMalloc(&g_scratch.Cs, plane * sizeof(double)));
  if (poz_scaleRule() == 1) {
    size_t tiles = (size_t)((Np + 31) / 32);
    POZ_CK(cudaMalloc(&g_scratch.pA, (size_t)Np * tiles * sizeof(float)));
    POZ_CK(cudaMalloc(&g_scratch.pB, (size_t)Np * tiles * sizeof(float)));
  }
  g_scratch.Np = Np;
}
void poz_uploadConsts(int nm) {
  if (g_uploadedNm == nm)
    return;
  pozComputeConsts(nm, g_consts);
  POZ_CK(cudaMemcpyToSymbol(poz_nm, &nm, sizeof(int)));
  POZ_CK(cudaMemcpyToSymbol(poz_P, g_consts.mod, nm * sizeof(int)));
  POZ_CK(cudaMemcpyToSymbol(poz_M, g_consts.magic,
                            nm * sizeof(unsigned long long)));
  POZ_CK(cudaMemcpyToSymbol(poz_T32, g_consts.t32, nm * sizeof(unsigned)));
  POZ_CK(cudaMemcpyToSymbol(poz_QHi, g_consts.qhi, nm * sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_QLo, g_consts.qlo, nm * sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_PHi, &g_consts.pHi, sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_PLo, &g_consts.pLo, sizeof(double)));
  POZ_CK(cudaMemcpyToSymbol(poz_InvP, &g_consts.invP, sizeof(double)));
  g_uploadedNm = nm;
}

// Repack a rows×cols (layout-aware) source into the top-left of an S×S
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

// Crop the M×Ncols top-left of an S×S row-major result back into a layout-aware
// destination, applying C = alpha*Cs + beta*C only over the useful region.
__global__ void poz_crop(double *C, const double *Cs, int M, int Ncols, int ldc,
                         int cColMajor, int S, double alpha, double beta) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= Ncols)
    return;
  size_t dst = cColMajor ? (size_t)j * ldc + i : (size_t)i * ldc + j;
  C[dst] = alpha * Cs[(size_t)i * S + j] + beta * C[dst];
}

// num_moduli selects the Ozaki-II precision (1..14); the per-nm CRT constants
// and scaling budget are computed and uploaded here (cached). Square N x N x N,
// NN, row-major pipeline: C = alpha*(A@B)+beta*C.
static void poz_run_square(double *C, const double *A, const double *B, int N,
                           double alpha, double beta, cudaStream_t stream,
                           int num_moduli) {
  poz_ensure(N);
  poz_uploadConsts(num_moduli);
  int Np = g_scratch.Np;
  int log2K = (int)ceil(log2((double)N));
  int beta_exp = (g_consts.budgetBits - log2K) / 2;
  if (beta_exp > 50) beta_exp = 50;
  if (beta_exp < 1) beta_exp = 1;
  size_t plane = (size_t)Np * Np;
  dim3 tile(32, 8), gTile((Np + 31) / 32, (Np + 31) / 32);
  POZ_CK(cudaMemsetAsync(g_scratch.mA, 0, Np * sizeof(unsigned long long),
                         stream));
  POZ_CK(cudaMemsetAsync(g_scratch.mB, 0, Np * sizeof(unsigned long long),
                         stream));
  // s32 residue-GEMM accumulator: |sum_k r_a*r_b| <= K*128*128 = K*2^14 must
  // stay under 2^31, i.e. K < 131072.
  if (N >= 131072) { // 2^31 / 2^14, written out: 1<<31 overflows a signed int
    printf("poseidon_ozaki_rt: FATAL K=%d >= 131072 — the s32 residue-GEMM "
           "accumulator would overflow; needs an intermediate reduction.\n",
           N);
    abort();
  }
  int tilesN = (N + 31) / 32;
  if (poz_scaleRule() == 1 && !g_scratch.pA) {
    // Reachable when the arm is switched in-process after the scratch set was
    // sized for maxbeta (an A/B harness); poz_ensure covers the normal case.
    size_t tiles = (size_t)((Np + 31) / 32);
    POZ_CK(cudaMalloc(&g_scratch.pA, (size_t)Np * tiles * sizeof(float)));
    POZ_CK(cudaMalloc(&g_scratch.pB, (size_t)Np * tiles * sizeof(float)));
  }
  if (poz_scaleRule() == 1) {
    // norm2: the shift comes from each row's / column's 2-norm, so the padded
    // extent never enters and the real reduction length K = N is what the
    // round-to-nearest margin is sized against.
    // Margin: the FP32 sums of squares are relative-accurate to K·2^-24, so a
    // (1 - K·2^-24) haircut on the cap covers a norm under-estimate (K·2^-25
    // on the norm itself, doubled); the 2^-20 floor covers the FP64 reduction
    // and the sqrtl of the cap. The -0.5·sqrt(K) term is what makes ROUND-TO-
    // NEAREST quantization legal: each rounded element can exceed its scaled
    // value by 0.5, so ||a'||₂ can exceed 2^sft·||a||₂ by up to 0.5·sqrt(K).
    // Together they cost ~1e-3 bits.
    double relMargin = std::max(1.0 / 1048576.0, (double)N / 16777216.0);
    double capEff =
        g_consts.capNrm * (1.0 - relMargin) - 0.5 * std::sqrt((double)N);
    poz_scanNorm<<<dim3(tilesN, tilesN), tile, 0, stream>>>(
        A, B, g_scratch.mA, g_scratch.mB, g_scratch.pA, g_scratch.pB, N, tilesN,
        tilesN);
    poz_shiftsNorm<<<(N + 127) / 128, 128, 0, stream>>>(
        g_scratch.pA, g_scratch.pB, g_scratch.mA, g_scratch.mB, g_scratch.sA,
        g_scratch.sB, N, tilesN, tilesN, capEff, N);
  } else {
    poz_absMax<<<dim3(tilesN, tilesN), tile, 0, stream>>>(
        A, B, g_scratch.mA, g_scratch.mB, N);
    poz_shifts<<<(N + 127) / 128, 128, 0, stream>>>(
        g_scratch.mA, g_scratch.mB, g_scratch.sA, g_scratch.sB, N, beta_exp);
  }
  poz_fillRes<<<gTile, tile, 0, stream>>>(A, B, g_scratch.sA, g_scratch.sB,
                                          g_scratch.Ar, g_scratch.Br, N, Np);
  // One strided-batched INT8 tensor-core GEMM over all num_moduli residue
  // planes. Column-major C^T(Np×Np) = op(Br)^T · op(Ar) = B^T·A^T, so the int32
  // result read row-major is the row-major product the CRT pass expects.
  cublasHandle_t h = poz_cublasHandle();
  POZ_CB(cublasSetStream(h, stream));
  const int32_t i_one = 1, i_zero = 0;
  POZ_CB(cublasGemmStridedBatchedEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, Np, Np, Np, &i_one, g_scratch.Br, CUDA_R_8I,
      Np, (long long)plane, g_scratch.Ar, CUDA_R_8I, Np, (long long)plane,
      &i_zero, g_scratch.Cr, CUDA_R_32I, Np, (long long)plane, num_moduli,
      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
  poz_crt<<<dim3((N + 127) / 128, (N + 7) / 8), dim3(32, 8), 0, stream>>>(
      g_scratch.Cr, g_scratch.sA, g_scratch.sB, C, N, Np, alpha, beta);
}

// Rectangular arm. The square pipeline pads every operand to
// S = roundup(max(M,Ncols,K),16), so a skinny product pays
// padWaste = S^3/(M*Ncols*K) times the useful work and the elementwise passes
// over the padded plane dominate. This arm sizes every buffer to the real
// M x K / K x Ncols / M x Ncols extents (rounded up by at most 15 columns for
// the INT8 leading-dimension requirement) and repacks no operand.
//
// Residue layout: both operands are quantized into the same form, per modulus
// a row-major (R x Kp) int8 plane whose row index is the operand's outer index
// and whose column index is the contraction index k,
//     A: R = M,     outer = the output row m
//     B: R = Ncols, outer = the output column c
// Read column-major with leading dimension Kp, A's plane is A^T (K x M) and
// B's plane is B (K x Ncols), the TN form the batched INT8 entry needs, so the
// same cublasGemmStridedBatchedEx(OP_T, OP_N) the square path issues produces
// the column-major C^T(Ncols x M), i.e. row-major C with row stride Ncp. The
// scaling shift is indexed by the destination row for both operands, so one
// scan / shift / fill kernel serves both; the only per-operand difference is
// whether the source's contiguous axis is k ("direct") or the destination row
// ("transposed"), which is aColMajor for A and !bColMajor for B.
//
// Numerics are the square arm's with the padding removed from the bound: the
// maxbeta budget uses ceil(log2 K) on the real reduction length and the norm2
// cap uses the real K in its round-to-nearest margin, so a rung is at least as
// accurate here as there.

// One thread per output row of a residue plane; `transposed` selects the
// source addressing. Rows of the destination beyond R are not written: cuBLAS
// never reads them (they only exist so the plane stride stays 16-byte aligned)
// and the fill zeroes their residues anyway.
//   direct     (transposed==0): src(r,k) = src[r*ld + k]   (contiguous in k)
//   transposed (transposed==1): src(r,k) = src[k*ld + r]   (contiguous in r)
// Both traversals fold the FP32 sums of squares in a fixed lane order, for the
// reproducibility reason poz_scanNorm gives.
__global__ void poz_scanRect(const double *src, unsigned long long *mx,
                             float *ps, int R, int K, int ld, int transposed,
                             int nPart, int wantNorm) {
  __shared__ unsigned long long shM[8][32];
  __shared__ float shS[8][32];
  const int tx = threadIdx.x, ty = threadIdx.y;
  if (!transposed) {
    const int kBase = blockIdx.x * 32, rBase = blockIdx.y * 32;
    const int k = kBase + tx;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = rBase + ty + 8 * j;
      unsigned long long a = 0;
      float sq = 0.0f;
      if (r < R && k < K) {
        double v = src[(size_t)r * ld + k];
        a = poz_absKey(v);
        float vf = (float)v;
        sq = vf * vf;
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        unsigned long long o = __shfl_xor_sync(0xffffffffu, a, off);
        a = a > o ? a : o;
        sq += __shfl_xor_sync(0xffffffffu, sq, off); // butterfly: fixed order
      }
      if (tx == 0 && r < R) {
        atomicMax(&mx[r], a);
        if (wantNorm)
          ps[(size_t)r * nPart + blockIdx.x] = sq;
      }
    }
  } else {
    const int rBase = blockIdx.x * 32, kBase = blockIdx.y * 32;
    const int r = rBase + tx;
    unsigned long long acc = 0;
    float accS = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int k = kBase + ty + 8 * j;
      if (r < R && k < K) {
        double v = src[(size_t)k * ld + r];
        unsigned long long a = poz_absKey(v);
        acc = acc > a ? acc : a;
        float vf = (float)v;
        accS += vf * vf;
      }
    }
    shM[ty][tx] = acc;
    shS[ty][tx] = accS;
    __syncthreads();
    if (ty == 0 && r < R) {
      unsigned long long m = shM[0][tx];
      float s = shS[0][tx];
#pragma unroll
      for (int j = 1; j < 8; j++) {
        m = m > shM[j][tx] ? m : shM[j][tx];
        s += shS[j][tx];
      }
      atomicMax(&mx[r], m);
      if (wantNorm)
        ps[(size_t)r * nPart + blockIdx.y] = s;
    }
  }
}

// Both scaling arms in one launch (the branch is block-uniform): rule 0 =
// maxbeta on the row/column max, rule 1 = norm2 on the reduced 2-norm.
__global__ void poz_shiftsRect(const unsigned long long *mx, const float *ps,
                               short *sft, int R, int nPart, int rule,
                               int beta_exp, double capEff, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R)
    return;
  const double m = __longlong_as_double((long long)mx[i]);
  if (rule == 0) {
    sft[i] = (short)(m > 0 ? beta_exp - 1 - ilogb(m) : 0);
    return;
  }
  double s = 0.0;
  for (int j = 0; j < nPart; j++)
    s += (double)ps[(size_t)i * nPart + j];
  sft[i] = poz_sftFromNorm(s, m, capEff, K);
}

// Residue fill into the (Rp x Kp) planes. `transposed` has the same meaning as
// in poz_scanRect; in the transposed case the scaled integers are staged in
// shared and written back with the tile transposed, so BOTH the source read
// and the int8 store stay coalesced. Every destination element with r >= R or
// k >= K is written as an explicit zero: the k pad is inside the contraction
// range the batched GEMM is issued over, so it must not carry garbage, and the
// r pad keeps the residues of the (never-read) alignment rows well defined.
__global__ void poz_fillRect(const double *src, const short *sft, int8_t *dst,
                             int R, int K, int ld, int transposed, int Rp,
                             int Kp) {
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
      // scalbn(x, sft), NOT x * scalbn(1.0, sft): see poz_fillRes.
      const long v = inb ? llround(scalbn(src[(size_t)r * ld + k], sft[r])) : 0L;
      const size_t off = (size_t)r * Kp + k;
      for (int i = 0; i < poz_nm; i++)
        dst[i * plane + off] =
            inb ? poz_modI8(v, poz_P[i], poz_M[i], poz_T32[i]) : (int8_t)0;
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
            (r < R && k < K) ? llround(scalbn(src[(size_t)k * ld + r], s)) : 0L;
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int rl = ty + 8 * j, r = rBase + rl, k = kBase + tx;
      if (r >= Rp || k >= Kp)
        continue;
      const bool inb = (r < R && k < K);
      const long v = sh[tx][rl];
      const size_t off = (size_t)r * Kp + k;
      for (int i = 0; i < poz_nm; i++)
        dst[i * plane + off] =
            inb ? poz_modI8(v, poz_P[i], poz_M[i], poz_T32[i]) : (int8_t)0;
    }
  }
}

// CRT recombine over the M x Ncols useful region only. Four columns per thread
// as one int4 per residue plane, exactly as poz_crt: ldCr is a multiple of 16
// and each thread's c0 is a multiple of 4, so every load is 16-byte aligned and
// stays inside the padded plane. Block is (8, 32): 8 threads x 4 columns cover
// 32 columns, 32 threads cover 32 rows.
//   row-major C   : the natural (r, c) mapping is already coalesced, write out.
//   column-major C: consecutive c are ldc apart, so the tile is transposed
//                   through shared and written with consecutive r in a warp.
//                   The beta!=0 read of C happens in that same transposed
//                   phase, so it is coalesced too.
__global__ void poz_crtRect(const int32_t *Cr, const short *sftA,
                            const short *sftB, double *C, int M, int Ncols,
                            int Mp, int ldCr, int ldc, int cColMajor,
                            double alpha, double beta) {
  __shared__ double sh[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y; // blockDim = (8, 32)
  const int cBase = blockIdx.x * 32, rBase = blockIdx.y * 32;
  const int r = rBase + ty, c0 = cBase + 4 * tx;
  const size_t plane = (size_t)Mp * ldCr;
  double prod[4] = {0, 0, 0, 0};
  const bool live = (r < M && c0 < Ncols);
  if (live) {
    const size_t off = (size_t)r * ldCr + c0;
    double x[4] = {0, 0, 0, 0}, y[4] = {0, 0, 0, 0};
    for (int i = 0; i < poz_nm; i++) {
      int4 raw = *reinterpret_cast<const int4 *>(Cr + i * plane + off);
      const int *rawv = reinterpret_cast<const int *>(&raw);
      const int p = poz_P[i];
      const unsigned long long Mg = poz_M[i];
      const double qhi = poz_QHi[i], qlo = poz_QLo[i];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        double d = (double)poz_modI32(rawv[j], p, Mg);
        x[j] = fma(qhi, d, x[j]);
        y[j] = fma(qlo, d, y[j]);
      }
    }
    const short sa = sftA[r];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int c = c0 + j;
      if (c >= Ncols)
        break;
      // Quotient from x+y, not x alone: see poz_crt.
      const double quot = rint(poz_InvP * (x[j] + y[j]));
      const double crtv = fma(poz_PLo, quot, fma(poz_PHi, quot, x[j]) + y[j]);
      prod[j] = scalbn(crtv, -(sa + sftB[c]));
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
      C[o] = (beta == 0.0) ? alpha * prod[j] : alpha * prod[j] + beta * C[o];
    }
    return;
  }
  // Transposed write-out: stage (row, col) in shared, then re-linearize the
  // block so a WARP walks 32 consecutive rows of one column. tid&31 is the row,
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
    const double v = alpha * sh[rl][cl];
    C[o] = (beta == 0.0) ? v : v + beta * C[o];
  }
}

// Runtime knobs for the rectangular arm. Parsed once; an unparseable value
// aborts rather than silently selecting the other behaviour, the same
// convention POSEIDON_OZAKI_SCALE follows.
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
// Threshold on padWaste = S^3/(M*Ncols*K). 1.05 is "square up to the 16-column
// alignment rounding": an exactly square dispatch has padWaste 1.0 and the
// worst a square shape rounds up to is (S/N)^3 <= 1.05 for N >= 320, so every
// genuinely square _ex call stays on the square arm bit for bit. The
// rectangular arm is faster even at padWaste 1.0 (it skips the poz_pack /
// poz_crop FP64 repacks), so the threshold buys conservatism, not speed;
// POSEIDON_OZAKI_RECT_MIN_WASTE=1 recovers that.
static double poz_rectMinWaste() {
  static double v = poz_envDouble("POSEIDON_OZAKI_RECT_MIN_WASTE", 1.05);
  return v;
}
static size_t poz_rectCapBytes() {
  static size_t v = (size_t)poz_envLong("POSEIDON_OZAKI_RECT_CAP_MB", 512) << 20;
  return v;
}

// Internal column chunking: the rectangular arm's cost is linear in the column
// count, so the only reason to split is the residue scratch, POZ_MAXNM*(Kp +
// 4*Mp) bytes per column plus the norm2 partials. Chunking is numerically
// inert (A's shift is a per-row property of the whole operand and B's a
// per-column property of one column, so every chunking returns the same bits),
// which makes the split point a pure memory/launch-overhead trade.
long g_rectChunk = -1; // <0 = not yet read from the environment
static int poz_rectChunk(int Mp, int Kp, int Ncols) {
  if (g_rectChunk < 0)
    g_rectChunk = poz_envLong("POSEIDON_OZAKI_RECT_CHUNK", 0);
  const long forced = g_rectChunk;
  if (forced > 0)
    return (int)std::min<long>(forced, Ncols);
  double perCol = (double)POZ_MAXNM * ((double)Kp + 4.0 * (double)Mp) +
                  4.0 * (double)((Kp + 31) / 32) + 10.0;
  long c = (long)((double)poz_rectCapBytes() / perCol);
  c &= ~127L; // keep every chunk but the last a multiple of 128 columns
  if (c < 128)
    c = 128;
  if (c > Ncols)
    c = Ncols;
  return (int)c;
}

// Cross-dispatch reuse of the quantized A planes, for a caller that knows its
// A operand is loop invariant. Opt-in and explicit: the runtime cannot verify
// the promise, and inferring it from the pointer alone would be wrong for B
// (same buffer, new contents, every call). The reuse is dropped whenever
// anything the residues depend on changes: the operand descriptor, the modulus
// count, the scaling arm, or a scratch reallocation.
struct PozPinA {
  const double *ptr = nullptr;
  int M = 0, K = 0, lda = 0, colMajor = 0;
  bool pinned = false;
  // state of the planes currently sitting in g_rect.Ar
  bool filled = false;
  int nm = 0, rule = -1;
  unsigned long long gen = 0;
};
PozPinA g_pinA;
unsigned long long g_rectGen = 0; // bumped on every scratch (re)allocation

struct PozRectScratch {
  int Mp = 0, Kp = 0, Ncp = 0, tiles = 0;
  int8_t *Ar = nullptr, *Br = nullptr;
  int32_t *Cr = nullptr;
  short *sA = nullptr, *sB = nullptr;
  unsigned long long *mA = nullptr, *mB = nullptr;
  float *pA = nullptr, *pB = nullptr;
};
PozRectScratch g_rect;

// Grow-only: an alternating shape sequence (the last chunk of a dispatch is
// short) must not free and re-allocate on every call.
static void poz_ensureRect(int Mp, int Kp, int Ncp) {
  const bool norm = (poz_scaleRule() == 1);
  if (g_rect.Mp >= Mp && g_rect.Kp >= Kp && g_rect.Ncp >= Ncp &&
      (!norm || g_rect.pA))
    return;
  Mp = std::max(Mp, g_rect.Mp);
  Kp = std::max(Kp, g_rect.Kp);
  Ncp = std::max(Ncp, g_rect.Ncp);
  const int tiles = (Kp + 31) / 32;
  if (g_rect.Ar) {
    cudaFree(g_rect.Ar); cudaFree(g_rect.Br); cudaFree(g_rect.Cr);
    cudaFree(g_rect.sA); cudaFree(g_rect.sB);
    cudaFree(g_rect.mA); cudaFree(g_rect.mB);
    if (g_rect.pA) { cudaFree(g_rect.pA); cudaFree(g_rect.pB); }
    g_rect = PozRectScratch();
  }
  POZ_CK(cudaMalloc(&g_rect.Ar, (size_t)POZ_MAXNM * Mp * Kp));
  POZ_CK(cudaMalloc(&g_rect.Br, (size_t)POZ_MAXNM * Ncp * Kp));
  POZ_CK(cudaMalloc(&g_rect.Cr, (size_t)POZ_MAXNM * Ncp * Mp * sizeof(int32_t)));
  POZ_CK(cudaMalloc(&g_rect.sA, (size_t)Mp * sizeof(short)));
  POZ_CK(cudaMalloc(&g_rect.sB, (size_t)Ncp * sizeof(short)));
  POZ_CK(cudaMalloc(&g_rect.mA, (size_t)Mp * sizeof(unsigned long long)));
  POZ_CK(cudaMalloc(&g_rect.mB, (size_t)Ncp * sizeof(unsigned long long)));
  if (norm) {
    POZ_CK(cudaMalloc(&g_rect.pA, (size_t)Mp * tiles * sizeof(float)));
    POZ_CK(cudaMalloc(&g_rect.pB, (size_t)Ncp * tiles * sizeof(float)));
  }
  g_rect.Mp = Mp; g_rect.Kp = Kp; g_rect.Ncp = Ncp; g_rect.tiles = tiles;
  ++g_rectGen; // any cached A planes lived in the buffer that was just freed
}

// Rectangular Ozaki-II: C(M x Ncols) = alpha*(A@B) + beta*C, A(M x K),
// B(K x Ncols), each operand row- or column-major per the {a,b,c}ColMajor
// flags. Columns are chunked internally; A is scanned and quantized once for
// the whole dispatch because its residues are a property of A alone.
static void poz_run_rect(double *C, const double *A, const double *B, int M,
                         int Ncols, int K, int lda, int ldb, int ldc,
                         int aColMajor, int bColMajor, int cColMajor,
                         double alpha, double beta, cudaStream_t stream,
                         int num_moduli) {
  // s32 residue-GEMM accumulator: |Σ_k r_a·r_b| <= K·128·128 must stay under
  // 2^31 (same bound the square path enforces, on the REAL K here).
  if (K >= 131072) {
    printf("poseidon_ozaki_rt: FATAL K=%d >= 131072 — the s32 residue-GEMM "
           "accumulator would overflow; needs an intermediate reduction.\n",
           K);
    abort();
  }
  poz_uploadConsts(num_moduli);
  const int Kp = pozPad(K), Mp = pozPad(M);
  const int rule = poz_scaleRule();
  const int chunk = poz_rectChunk(Mp, Kp, Ncols);
  poz_ensureRect(Mp, Kp, pozPad(chunk));

  const int log2K = (int)ceil(log2((double)K));
  int beta_exp = (g_consts.budgetBits - log2K) / 2;
  if (beta_exp > 50) beta_exp = 50;
  if (beta_exp < 1) beta_exp = 1;
  // Same margin recipe as poz_run_square, on the real reduction length.
  const double relMargin = std::max(1.0 / 1048576.0, (double)K / 16777216.0);
  const double capEff =
      g_consts.capNrm * (1.0 - relMargin) - 0.5 * std::sqrt((double)K);
  const int nPart = (Kp + 31) / 32;
  const dim3 tile(32, 8);

  // A, once for the whole dispatch (and, if the caller pinned it, once for the
  // whole run of dispatches).
  const bool reuseA = g_pinA.pinned && g_pinA.filled && g_pinA.ptr == A &&
                      g_pinA.M == M && g_pinA.K == K && g_pinA.lda == lda &&
                      g_pinA.colMajor == aColMajor && g_pinA.nm == num_moduli &&
                      g_pinA.rule == rule && g_pinA.gen == g_rectGen;
  if (!reuseA) {
    const int tr = aColMajor ? 1 : 0; // col-major A is contiguous in m, not k
    POZ_CK(cudaMemsetAsync(g_rect.mA, 0, (size_t)Mp * sizeof(unsigned long long),
                           stream));
    const dim3 g = tr ? dim3((Mp + 31) / 32, (Kp + 31) / 32)
                      : dim3((Kp + 31) / 32, (Mp + 31) / 32);
    poz_scanRect<<<g, tile, 0, stream>>>(A, g_rect.mA, g_rect.pA, M, K, lda, tr,
                                         nPart, rule == 1);
    poz_shiftsRect<<<(M + 127) / 128, 128, 0, stream>>>(
        g_rect.mA, g_rect.pA, g_rect.sA, M, nPart, rule, beta_exp, capEff, K);
    poz_fillRect<<<g, tile, 0, stream>>>(A, g_rect.sA, g_rect.Ar, M, K, lda, tr,
                                         Mp, Kp);
    if (g_pinA.pinned && g_pinA.ptr == A && g_pinA.M == M && g_pinA.K == K &&
        g_pinA.lda == lda && g_pinA.colMajor == aColMajor) {
      g_pinA.filled = true;
      g_pinA.nm = num_moduli;
      g_pinA.rule = rule;
      g_pinA.gen = g_rectGen;
    }
  }

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

    POZ_CK(cudaMemsetAsync(g_rect.mB, 0,
                           (size_t)ncp * sizeof(unsigned long long), stream));
    const dim3 gB = trB ? dim3((ncp + 31) / 32, (Kp + 31) / 32)
                        : dim3((Kp + 31) / 32, (ncp + 31) / 32);
    poz_scanRect<<<gB, tile, 0, stream>>>(Bc, g_rect.mB, g_rect.pB, nc, K, ldb,
                                          trB, nPart, rule == 1);
    poz_shiftsRect<<<(nc + 127) / 128, 128, 0, stream>>>(
        g_rect.mB, g_rect.pB, g_rect.sB, nc, nPart, rule, beta_exp, capEff, K);
    poz_fillRect<<<gB, tile, 0, stream>>>(Bc, g_rect.sB, g_rect.Br, nc, K, ldb,
                                          trB, ncp, Kp);
    // Column-major C^T(ncp x Mp) = op_T(Br) · op_N(Ar) = B^T·A^T, so the int32
    // result read row-major with row stride ncp is the row-major product.
    POZ_CB(cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N, ncp, Mp, Kp, &i_one, g_rect.Br, CUDA_R_8I,
        Kp, (long long)ncp * Kp, g_rect.Ar, CUDA_R_8I, Kp, (long long)Mp * Kp,
        &i_zero, g_rect.Cr, CUDA_R_32I, ncp, (long long)ncp * Mp, num_moduli,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    poz_crtRect<<<dim3((nc + 31) / 32, (M + 31) / 32), dim3(8, 32), 0, stream>>>(
        g_rect.Cr, g_rect.sA, g_rect.sB, Cc, M, nc, Mp, ncp, ldc, cColMajor,
        alpha, beta);
  }
}

// General layout-aware native DGEMM: logical C(M×Ncols) = alpha*(A@B) + beta*C
// with A(M×K), B(K×Ncols). Each operand is either column-major (X[c*ld + r])
// or row-major (X[r*ld + c]), the convention poz_pack/poz_crop use.
// cuBLAS is column-major, and a row-major rows×cols matrix with row stride ld
// is the column-major cols×rows TRANSPOSE with the same ld, so every layout
// combination maps to one cublasDgemm:
//   - column-major C: C = op(A)·op(B), op = N for col-major storage, T for
//     row-major storage;
//   - row-major C: compute the column-major C^T (Ncols×M) = op(B)·op(A) with
//     the ops flipped (T for col-major storage, N for row-major storage).
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
} // namespace

// Select the operand scaling arm in-process: 0 = maxbeta (default), 1 = norm2.
// Overrides POSEIDON_OZAKI_SCALE. Must be called before the first dispatch,
// because the scratch set is sized for the selected arm.
extern "C" void __poseidon_ozaki_set_scale_rule(int rule) {
  if (rule != 0 && rule != 1) {
    printf("poseidon_ozaki_rt: FATAL __poseidon_ozaki_set_scale_rule(%d): "
           "expected 0 (maxbeta) or 1 (norm2)\n",
           rule);
    abort();
  }
  g_scaleRule = rule;
}

// Select the _ex shape arm in-process: 0 = auto (shape-driven, default),
// 1 = force rectangular, 2 = force square-padded. Overrides
// POSEIDON_OZAKI_PATH. Harness/diagnostic entry: the emission side never calls
// it, and "auto" is what any real dispatch runs.
extern "C" void __poseidon_ozaki_set_path(int mode) {
  if (mode < 0 || mode > 2) {
    printf("poseidon_ozaki_rt: FATAL __poseidon_ozaki_set_path(%d): expected 0 "
           "(auto), 1 (rect) or 2 (square)\n",
           mode);
    abort();
  }
  g_pathMode = mode;
}

// Promise that the A operand described here does not change until unpinned,
// letting the rectangular arm reuse its quantized residue planes across
// dispatches. The runtime cannot verify the promise, so it is opt-in and never
// inferred; pinning a different descriptor or changing num_moduli / the
// scaling arm invalidates the cached planes. Rectangular arm only.
extern "C" void __poseidon_ozaki_pin_a(const double *A, int M, int K, int lda,
                                       int aColMajor) {
  if (!A || M <= 0 || K <= 0) {
    printf("poseidon_ozaki_rt: FATAL __poseidon_ozaki_pin_a(%p, M=%d, K=%d): "
           "expected a non-null operand with positive extents\n",
           (const void *)A, M, K);
    abort();
  }
  g_pinA = PozPinA();
  g_pinA.ptr = A;
  g_pinA.M = M;
  g_pinA.K = K;
  g_pinA.lda = lda;
  g_pinA.colMajor = aColMajor;
  g_pinA.pinned = true;
}
extern "C" void __poseidon_ozaki_unpin_a(void) { g_pinA = PozPinA(); }

// Override the rectangular arm's internal column chunk (0 = size it from the
// scratch budget, which is what a real dispatch does). Harness entry.
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
  // nm==0 => native cuBLAS DGEMM; out-of-range => full-precision Ozaki (14).
  if (num_moduli < 0 || num_moduli > 14)
    num_moduli = 14;
  if (transA || transB || (lda && lda != N) || (ldb && ldb != N) ||
      (ldc && ldc != N)) {
    printf("poseidon_ozaki_rt: unsupported GEMM shape (transA=%d transB=%d "
           "lda=%d ldb=%d ldc=%d N=%d)\n",
           transA, transB, lda, ldb, ldc, N);
    return;
  }
  if (num_moduli == 0) {
    // Row-major square NN GEMM == column-major C^T = alpha*B^T·A^T + beta*C^T.
    poz_native_dgemm(C, A, B, N, N, N, N, N, N, /*aColMajor=*/0,
                     /*bColMajor=*/0, /*cColMajor=*/0, alpha, beta, stream);
    return;
  }
  poz_run_square(C, A, B, N, alpha, beta, stream, num_moduli);
}

// General M×Ncols×K entry. Two arms, selected internally by shape and never by
// the caller (the emission side's argument list is fixed):
//   RECTANGULAR (padWaste >= POSEIDON_OZAKI_RECT_MIN_WASTE): buffers sized
//     M×K / K×Ncols / M×Ncols, columns chunked inside the runtime, A quantized
//     once per dispatch. See poz_run_rect.
//   SQUARE (square-ish shapes, and forced by POSEIDON_OZAKI_PATH=square):
//     zero-pad to S=roundup(max(M,Ncols,K),16), repack A/B (layout-aware) into
//     square row-major scratch, run the square pipeline, crop back. The padding
//     waste (S^3 vs M·Ncols·K) is what the cost model charges for this arm.
// {a,b,c}ColMajor select each operand's stride convention in both arms.
extern "C" void __poseidon_ozaki_dgemm_ex(double *C, const double *A,
                                          const double *B, int M, int Ncols,
                                          int K, int lda, int ldb, int ldc,
                                          int aColMajor, int bColMajor,
                                          int cColMajor, double alpha,
                                          double beta, cudaStream_t stream,
                                          int num_moduli) {
  if (M <= 0 || Ncols <= 0 || K <= 0)
    return;
  // nm==0 => native cuBLAS DGEMM; out-of-range => full-precision Ozaki (14).
  if (num_moduli < 0 || num_moduli > 14)
    num_moduli = 14;
  if (num_moduli == 0) {
    // Native path needs no padding: every {a,b,c}ColMajor combination maps
    // directly onto one cublasDgemm (see poz_native_dgemm).
    poz_native_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor,
                     cColMajor, alpha, beta, stream);
    return;
  }
  int mx = M > Ncols ? M : Ncols;
  if (K > mx) mx = K;
  int S = ((mx + 15) / 16) * 16;
  {
    const double padWaste = (double)S * (double)S * (double)S /
                            ((double)M * (double)Ncols * (double)K);
    const int mode = poz_pathMode();
    if (mode == 1 || (mode == 0 && padWaste >= poz_rectMinWaste())) {
      poz_run_rect(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor,
                   cColMajor, alpha, beta, stream, num_moduli);
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
  poz_run_square(g_scratch.Cs, g_scratch.As, g_scratch.Bs, S, 1.0, 0.0, stream,
                 num_moduli);
  poz_crop<<<dim3((Ncols + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      C, g_scratch.Cs, M, Ncols, ldc, cColMajor, S, alpha, beta);
}
