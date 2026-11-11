//=- MatmulSimulation.cpp - What the GEMM runtimes return, on the host ----=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compiled without floating-point contraction (CMakeLists.txt): every product
// and sum is rounded where the runtime rounds it.
//
//===----------------------------------------------------------------------===//

#include "MatmulSimulation.h"

#include "runtime/tcec/tcec_scale.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

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

} // namespace poseidon
