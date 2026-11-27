//===- Simulation.cpp - What the GEMM runtimes return, on the host --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiled without floating-point contraction and with the rounding mode
// honored (CMakeLists.txt): every product and sum is rounded where and how the
// runtime rounds it.
//
//===----------------------------------------------------------------------===//

#include "Simulation.h"

#include "runtime/gemm/SchemeConstants.h"

#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// The Ozaki-II shifts are rounded upward and downward, as the runtime's are.
// GCC has no such pragma and takes -frounding-math (CMakeLists.txt) instead.
#ifdef __clang__
#pragma STDC FENV_ACCESS ON
#endif

namespace poseidon {

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
// An operand as the runtime hands it to cuMpSGEMM, scaled by 2^scale
// (maxMagnitudeKernel, scaleExponentsKernel) and narrowed to FP32 (pack), and
// as cuMpSGEMM's load_matrix_sync splits it: an FP16 high slice and the FP16
// residual times 2^11, each zero below `operandMin`. Rows of A, columns of B,
// each K long.
struct SplitOperand {
  int scale = 0;
  std::vector<float> hi, lo;

  SplitOperand(const double *X, unsigned vectors, unsigned K, bool columns,
               unsigned ld, double operandMin) {
    auto kept = [&](float h) {
      return std::fabs(h) < operandMin ? std::copysign(0.f, h) : h;
    };
    double maxAbs = 0.0;
    for (unsigned i = 0; i < vectors * K; ++i)
      maxAbs = std::max(maxAbs, std::fabs(X[i]));
    scale = tcecScaleExponent(maxAbs);
    hi.resize((size_t)vectors * K);
    lo.resize((size_t)vectors * K);
    for (unsigned r = 0; r < vectors; ++r)
      for (unsigned k = 0; k < K; ++k) {
        double x = columns ? X[(size_t)k * ld + r] : X[(size_t)r * ld + k];
        float v = (float)tcecScale(x, scale);
        float h = toHalf(v);
        hi[(size_t)r * K + k] = kept(h);
        lo[(size_t)r * K + k] = kept(toHalf((v - h) * 2048.f));
      }
  }
};
} // namespace

void simulateTcecDgemm(const double *A, const double *B, unsigned M, unsigned N,
                       unsigned K, double operandMin, double *C) {
  const SplitOperand a(A, M, K, /*columns=*/false, K, operandMin);
  const SplitOperand b(B, N, K, /*columns=*/true, N, operandMin);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j) {
      const float *ah = &a.hi[(size_t)i * K], *al = &a.lo[(size_t)i * K];
      const float *bh = &b.hi[(size_t)j * K], *bl = &b.lo[(size_t)j * K];
      // mma_rn_sync: the main term of each block from a zero accumulator,
      // added in FP32; the two corrections, low A slice first, chained in the
      // tensor core; store_matrix_sync combines them in FP32.
      float main = 0.f, corr = 0.f;
      for (unsigned k = 0; k < K; k += kMmaK) {
        unsigned n = std::min(kMmaK, K - k);
        main = main + mma(0.f, ah + k, bh + k, n);
        corr = mma(corr, al + k, bh + k, n);
        corr = mma(corr, ah + k, bl + k, n);
      }
      double v = (double)(main + corr / 2048.f);
      C[(size_t)i * N + j] = tcecScale(v, -(a.scale + b.scale));
    }
}

namespace {
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

// warpSumUp: lane 0's sum of 32 lanes, each step rounded upward.
static double warpSumUp(double *lane) {
  for (unsigned d = 16; d; d >>= 1)
    for (unsigned i = 0; i < d; ++i)
      lane[i] = lane[i] + lane[i + d];
  return lane[0];
}

// ||vector r||_2^2 rounded upward in the runtime's order:
// shiftsContiguousKernel for a contiguous vector, shiftsStridedKernel or
// stridedPartialSumsKernel and shiftsStridedReduceKernel for a strided one.
static double sumOfSquaresUp(const OzakiIIOperand &X, unsigned r, unsigned K) {
  RoundingScope up(FE_UPWARD);
  auto at = [&](unsigned k) {
    return X.contiguous ? X.src[(size_t)r * X.ld + k]
                        : X.src[(size_t)k * X.ld + r];
  };
  double lane[32];
  if (X.contiguous) {
    double warps[32] = {};
    for (unsigned w = 0; w < kOzakiIINormThreads / 32; ++w) {
      for (unsigned l = 0; l < 32; ++l) {
        double s = 0.0;
        for (unsigned k = 32 * w + l; k < K; k += kOzakiIINormThreads)
          s = std::fma(at(k), at(k), s);
        lane[l] = s;
      }
      warps[w] = warpSumUp(lane);
    }
    return warpSumUp(warps);
  }
  // stridedTileSum: phase y of the 32 sums k = k0 + y, k0 + y + 32, ...
  auto tile = [&](unsigned k0, unsigned k1) {
    for (unsigned y = 0; y < 32; ++y) {
      double s = 0.0;
      for (unsigned k = k0 + y; k < k1; k += 32)
        s = std::fma(at(k), at(k), s);
      lane[y] = s;
    }
    return warpSumUp(lane);
  };
  constexpr unsigned chunk = kOzakiIINormChunk;
  if (K < (unsigned)kOzakiIINormSplitK)
    return tile(0, K);
  std::vector<double> part((K + chunk - 1) / chunk);
  for (unsigned c = 0; c < part.size(); ++c)
    part[c] = tile(c * chunk, std::min(K, (c + 1) * chunk));
  for (unsigned x = 0; x < 32; ++x) {
    double s = 0.0;
    for (size_t c = x; c < part.size(); c += 32)
      s = s + part[c];
    lane[x] = s;
  }
  return warpSumUp(lane);
}

// shiftOf: floor(log2P - log2(s)/2) with each rounding directed so that
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

// crtQuad on the residues of v: x accumulates the high parts of the CRT
// weights, y the low ones, and v = x + y - round(x/P) P is formed with P as a
// double-double. The residues are the symmetric representatives
// ozakiIIResidueI32 returns ([-128, 127] for 256), which depend on v alone, not
// on the INT8 operand residues whose products the runtime sums.
static double crtRecombine(const OzakiIIConstants &c, __int128 v) {
  double x = 0.0, y = 0.0;
  const unsigned __int128 mag = v < 0 ? -(unsigned __int128)v : v;
  for (unsigned i = 0; i < c.nm; ++i) {
    const uint32_t p = (uint32_t)kOzakiIIModuli[i];
    double d = (double)ozakiIISymmetric((uint32_t)(mag % p), v < 0, p);
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
// An operand's vectors as the runtime quantizes them (quantize): vector r is
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
  assert(nm >= kOzakiIIMinModuli && nm <= kOzakiIIMaxModuli &&
         K < (unsigned)kOzakiIIMaxK && "outside the runtime's domain");
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
