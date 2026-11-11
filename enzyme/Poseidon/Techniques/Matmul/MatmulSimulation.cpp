//=- MatmulSimulation.cpp - What the GEMM runtimes return, on the host ----=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiled without floating-point contraction and with the rounding mode
// honored (CMakeLists.txt): every product and sum is rounded where and how the
// runtime rounds it.
//
//===----------------------------------------------------------------------===//

#include "MatmulSimulation.h"

#include "runtime/ozaki/OzakiII.h"
#include "runtime/tcec/tcec_scale.h"

#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// The Ozaki-II shifts are rounded upward and downward, as the runtime's are.
#pragma STDC FENV_ACCESS ON

namespace poseidon {

const char *tcecBackendName(TcecBackend backend) {
  switch (backend) {
  case TcecBackend::BuiltIn:
    return "built-in";
  case TcecBackend::CuMpSGEMM:
    return "cuMpSGEMM";
  }
  return "";
}

// An FP32 value rounded to the nearest FP16 value, ties to even, with FP16's
// subnormals and overflow, as __float2half.
static float toHalf(float v) {
  if (v == 0.f || !std::isfinite(v))
    return v;
  int q = std::max(std::ilogb(v), -14) - 10;
  double r = std::ldexp(std::nearbyint(std::ldexp((double)v, -q)), q);
  return std::fabs(r) > 65504.0 ? std::copysign(INFINITY, v) : (float)r;
}

static float toFloatTowardZero(double x) {
  float f = (float)x;
  if (std::fabs((double)f) > std::fabs(x))
    f = std::nextafter(f, 0.f);
  return f;
}

constexpr unsigned kMmaK = 16;

// One m16n16k16 MMA of an output element: `acc` plus the products of the n <=
// 16 FP16 pairs of a[] and b[]. The products are exact in FP64, and so is
// their sum while they lie within 2^27 of each other.
static float mma(float acc, const float *a, const float *b, unsigned n) {
  double s = 0.0;
  for (unsigned k = 0; k < n; ++k)
    s += (double)a[k] * (double)b[k];
  return toFloatTowardZero((double)acc + s);
}

namespace {
// An operand as the runtime hands it to the backend: scaled by 2^scale
// (ptc_absmax, ptc_scales), narrowed to FP32 (ptc_narrow), and split into an
// FP16 high slice and the FP16 residual times 2^11 (ptc_split; cuMpSGEMM's
// load_matrix_sync splits the same way). Rows of A, columns of B, each K long.
struct SplitOperand {
  int scale = 0;
  std::vector<float> hi, lo;

  SplitOperand(const double *X, unsigned vectors, unsigned K, bool columns,
               unsigned ld) {
    double maxAbs = 0.0;
    for (unsigned i = 0; i < vectors * K; ++i)
      maxAbs = std::max(maxAbs, std::fabs(X[i]));
    scale = ptc_scaleExponent(maxAbs);
    hi.resize((size_t)vectors * K);
    lo.resize((size_t)vectors * K);
    for (unsigned r = 0; r < vectors; ++r)
      for (unsigned k = 0; k < K; ++k) {
        double x = columns ? X[(size_t)k * ld + r] : X[(size_t)r * ld + k];
        float v = (float)ptc_ldexp(x, scale);
        float h = toHalf(v);
        hi[(size_t)r * K + k] = h;
        lo[(size_t)r * K + k] = toHalf((v - h) * 2048.f);
      }
  }
};
} // namespace

void simulateTcecDgemm(TcecBackend backend, const double *A, const double *B,
                       unsigned M, unsigned N, unsigned K, double *C) {
  const SplitOperand a(A, M, K, /*columns=*/false, K);
  const SplitOperand b(B, N, K, /*columns=*/true, N);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j) {
      const float *ah = &a.hi[(size_t)i * K], *al = &a.lo[(size_t)i * K];
      const float *bh = &b.hi[(size_t)j * K], *bl = &b.lo[(size_t)j * K];
      double v = 0.0;
      switch (backend) {
      case TcecBackend::BuiltIn: {
        // Three cublasGemmEx calls, each accumulating over all of K in the
        // tensor core; the third adds into the second with beta = 1. The
        // widen combines the two in FP64 (ptc_widen).
        float main = 0.f, corr = 0.f, corr2 = 0.f;
        for (unsigned k = 0; k < K; k += kMmaK) {
          unsigned n = std::min(kMmaK, K - k);
          main = mma(main, ah + k, bh + k, n);
          corr = mma(corr, ah + k, bl + k, n);
          corr2 = mma(corr2, al + k, bh + k, n);
        }
        corr = corr2 + corr;
        v = (double)main + (double)corr / 2048.0;
        break;
      }
      case TcecBackend::CuMpSGEMM: {
        // mma_rn_sync: the main term of each block from a zero accumulator,
        // added in FP32; the two corrections, low A slice first, chained in
        // the tensor core; store_matrix_sync combines them in FP32.
        float main = 0.f, corr = 0.f;
        for (unsigned k = 0; k < K; k += kMmaK) {
          unsigned n = std::min(kMmaK, K - k);
          main = main + mma(0.f, ah + k, bh + k, n);
          corr = mma(corr, al + k, bh + k, n);
          corr = mma(corr, ah + k, bl + k, n);
        }
        v = (double)(main + corr / 2048.f);
        break;
      }
      }
      C[(size_t)i * N + j] = ptc_ldexp(v, -(a.scale + b.scale));
    }
}

namespace {
// What runtime/ozaki/ozaki_rt.cu computes once per modulus count
// (pozComputeConsts): -P as a double-double, 1/P, and the CRT weights
// q_i = (P/p_i)((P/p_i)^-1 mod p_i) split at one granularity.
struct OzakiIIConstants {
  unsigned nm = 0;
  float log2P = 0.f;
  double qhi[kOzakiIIMaxModuli], qlo[kOzakiIIMaxModuli];
  double pHi = 0.0, pLo = 0.0, invP = 0.0;
};

// Restores the rounding mode on scope exit.
class RoundingScope {
  int saved;

public:
  explicit RoundingScope(int mode) : saved(std::fegetround()) {
    std::fesetround(mode);
  }
  ~RoundingScope() { std::fesetround(saved); }
};
} // namespace

static long modInverse(long a, long m) {
  long r0 = ((a % m) + m) % m, r = m, s0 = 1, s = 0;
  while (r) {
    long q = r0 / r, t = r0 - q * r;
    r0 = r;
    r = t;
    t = s0 - q * s;
    s0 = s;
    s = t;
  }
  return ((s0 % m) + m) % m;
}

static int bitLength(__int128 v) {
  int n = 0;
  for (; v; v >>= 1)
    ++n;
  return n;
}

// 1/P rounded to nearest even, by long division.
static double reciprocalNearest(__int128 P) {
  __int128 r = 1;
  int e = 0;
  for (; r < P; r <<= 1)
    ++e;
  uint64_t q = 0; // floor(2^(e+53) / P), 54 bits
  for (int i = 0; i < 54; ++i) {
    q <<= 1;
    if (r >= P) {
      r -= P;
      q |= 1;
    }
    r <<= 1;
  }
  bool round = q & 1, sticky = r != 0;
  q >>= 1;
  if (round && (sticky || (q & 1)))
    ++q;
  return std::ldexp((double)q, -(e + 52));
}

static OzakiIIConstants ozakiIIConstants(unsigned nm) {
  OzakiIIConstants c;
  c.nm = nm;
  __int128 P = 1;
  int rho = 0; // bound on the sum of the residue magnitudes
  for (unsigned i = 0; i < nm; ++i) {
    P *= kOzakiIIModuli[i];
    rho += kOzakiIIModuli[i] / 2;
  }
  double pHi = (double)P;
  c.pHi = -pHi;
  c.pLo = -(double)(P - (__int128)pHi);
  c.invP = reciprocalNearest(P);
  c.log2P = kOzakiIILog2P[nm];
  // The high parts keep 53 - ceil(log2 rho) bits below bitlen(P), so their
  // sum weighted by the residues is exact.
  int G = bitLength(P) - (53 - bitLength(rho - 1));
  for (unsigned i = 0; i < nm; ++i) {
    long p = kOzakiIIModuli[i];
    __int128 Mi = P / p;
    __int128 e = (Mi * modInverse((long)(Mi % p), p)) % P;
    if (nm <= kOzakiIIPIsDouble) {
      c.qhi[i] = (double)e;
      c.qlo[i] = 0.0;
    } else {
      __int128 hi = (e >> G) << G;
      c.qhi[i] = (double)hi;
      c.qlo[i] = (double)(e - hi);
    }
  }
  return c;
}

// The sums-of-squares reduction of ozaki_rt.cu (poz_shifts): its block shapes
// decide which partial sums are rounded.
constexpr unsigned kNormThreads = 256; // POZ_NRM_THREADS
constexpr unsigned kNormSplitK = 1024; // POZ_NRM_SPLIT_K
constexpr unsigned kNormChunk = 256;   // POZ_NRM_KCHUNK

// poz_warpSumRU: lane 0's sum of 32 lanes, each step rounded upward.
static double warpSumUp(double *lane) {
  for (unsigned d = 16; d; d >>= 1)
    for (unsigned i = 0; i < d; ++i)
      lane[i] = lane[i] + lane[i + d];
  return lane[0];
}

// ||vector r||_2^2 rounded upward in the runtime's order: poz_sftContig for a
// contiguous vector, poz_sftStrided or poz_sumStridedPart and
// poz_sftStridedReduce for a strided one.
static double sumOfSquaresUp(const OzakiIIOperand &X, unsigned r, unsigned K) {
  RoundingScope up(FE_UPWARD);
  auto at = [&](unsigned k) {
    return X.contiguous ? X.src[(size_t)r * X.ld + k]
                        : X.src[(size_t)k * X.ld + r];
  };
  double lane[32];
  if (X.contiguous) {
    double warps[32] = {};
    for (unsigned w = 0; w < kNormThreads / 32; ++w) {
      for (unsigned l = 0; l < 32; ++l) {
        double s = 0.0;
        for (unsigned k = 32 * w + l; k < K; k += kNormThreads)
          s = std::fma(at(k), at(k), s);
        lane[l] = s;
      }
      warps[w] = warpSumUp(lane);
    }
    return warpSumUp(warps);
  }
  // poz_stridedTile: phase y of the 32 sums k = k0 + y, k0 + y + 32, ...
  auto tile = [&](unsigned k0, unsigned k1) {
    for (unsigned y = 0; y < 32; ++y) {
      double s = 0.0;
      for (unsigned k = k0 + y; k < k1; k += 32)
        s = std::fma(at(k), at(k), s);
      lane[y] = s;
    }
    return warpSumUp(lane);
  };
  if (K < kNormSplitK)
    return tile(0, K);
  std::vector<double> part((K + kNormChunk - 1) / kNormChunk);
  for (unsigned c = 0; c < part.size(); ++c)
    part[c] = tile(c * kNormChunk, std::min(K, (c + 1) * kNormChunk));
  for (unsigned x = 0; x < 32; ++x) {
    double s = 0.0;
    for (size_t c = x; c < part.size(); c += 32)
      s = s + part[c];
    lane[x] = s;
  }
  return warpSumUp(lane);
}

// poz_shiftOf: floor(log2P - log2(s)/2) with each rounding directed so that
// the shift can only come out smaller. The device's __log2f is an
// approximation the 2^-23 allowance covers; the host's log2 stands in for it.
static int shiftOf(double s, float log2P) {
  if (s == 0.0)
    return 0;
  int e = std::ilogb(s);
  float m, mu0, mu1, d;
  {
    RoundingScope up(FE_UPWARD);
    m = (float)std::scalbn(s, -e);
  }
  {
    RoundingScope down(FE_DOWNWARD);
    mu0 = log2P - (float)e * 0.5f;
  }
  float half = std::log2(m) * 0.5f;
  {
    RoundingScope up(FE_UPWARD);
    mu1 = half + 0x1p-23f;
  }
  {
    RoundingScope down(FE_DOWNWARD);
    d = mu0 - mu1;
  }
  return (short)std::floor(d);
}

// poz_crtQuad on the residues of v: x accumulates the high parts of the CRT
// weights, y the low ones, and v = x + y - round(x/P) P is formed with P as a
// double-double. The residues are the symmetric representatives poz_modI32
// returns ([-128, 127] for 256), which depend on v alone, not on the INT8
// operand residues whose products the runtime sums.
static double crtRecombine(const OzakiIIConstants &c, __int128 v) {
  double x = 0.0, y = 0.0;
  for (unsigned i = 0; i < c.nm; ++i) {
    int p = kOzakiIIModuli[i], h = p / 2;
    int r = (int)(v % p);
    if (r > h)
      r -= p;
    else if (r < -h)
      r += p;
    if ((p & 1) == 0 && r == h)
      r -= p;
    double d = (double)r;
    if (i == 0) {
      x = c.qhi[0] * d;
      y = c.qlo[0] * d;
    } else {
      x = std::fma(c.qhi[i], d, x);
      y = std::fma(c.qlo[i], d, y);
    }
  }
  double quot = std::rint(c.invP * x);
  return c.nm <= kOzakiIIPIsDouble
             ? std::fma(c.pHi, quot, x)
             : std::fma(c.pLo, quot, std::fma(c.pHi, quot, x) + y);
}

namespace {
// An operand's vectors as the runtime quantizes them (poz_quant): vector r is
// trunc(2^shift[r] x), exact, of magnitude at most 2^log2P < 2^55.
struct QuantizedOperand {
  std::vector<int> shift;
  std::vector<int64_t> q;
  std::vector<bool> finite;

  QuantizedOperand(const OzakiIIConstants &c, const OzakiIIOperand &X,
                   unsigned K)
      : shift(X.R), q((size_t)X.R * K), finite(X.R) {
    for (unsigned r = 0; r < X.R; ++r) {
      double s = sumOfSquaresUp(X, r, K);
      finite[r] = std::isfinite(s);
      if (!finite[r])
        continue;
      shift[r] = shiftOf(s, c.log2P);
      for (unsigned k = 0; k < K; ++k) {
        double x = X.contiguous ? X.src[(size_t)r * X.ld + k]
                                : X.src[(size_t)k * X.ld + r];
        q[(size_t)r * K + k] = (int64_t)std::scalbn(x, shift[r]);
      }
    }
  }
};
} // namespace

void simulateOzakiIIDgemm(unsigned nm, const OzakiIIOperand &A,
                          const OzakiIIOperand &B, unsigned K, double *P) {
  // The runtime's INT32 residue products hold K * 2^14; the exact sum below
  // holds K * 2^110.
  assert(nm >= kOzakiIIMinModuli && nm <= kOzakiIIMaxModuli && K < 131072 &&
         "outside the runtime's domain");
  const OzakiIIConstants c = ozakiIIConstants(nm);
  const QuantizedOperand a(c, A, K), b(c, B, K);
  for (unsigned i = 0; i < A.R; ++i)
    for (unsigned j = 0; j < B.R; ++j) {
      double &out = P[(size_t)i * B.R + j];
      if (!a.finite[i] || !b.finite[j]) {
        out = std::numeric_limits<double>::quiet_NaN();
        continue;
      }
      const int64_t *qa = &a.q[(size_t)i * K], *qb = &b.q[(size_t)j * K];
      __int128 v = 0;
      for (unsigned k = 0; k < K; ++k)
        v += (__int128)qa[k] * qb[k];
      out = std::scalbn(crtRecombine(c, v), -(a.shift[i] + b.shift[j]));
    }
}

} // namespace poseidon
