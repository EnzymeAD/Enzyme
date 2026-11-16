// Reference routines for expansion_reference_identity.cpp and the
// expansion_fast_flags harness: MultiFloats.jl's two-limb FP32 routines (at
// Val{2}), which the materializer's two-component addition, subtraction,
// multiplication and square root reproduce.
//
// Provenance: MultiFloats.jl, https://github.com/dzhang314/MultiFloats.jl,
// commit 28041a3e17e8ba0d6d3e75c49cc83f737c5e8701 (version 3.2.6+8),
// transcribed from Julia into C++ statement for statement:
//   two_sum         src/MultiFloats.jl 313-321
//   fast_two_sum    src/MultiFloats.jl 1047-1052
//   one_prod        src/MultiFloats.jl 1064
//   two_prod        src/MultiFloats.jl 1070-1074
//   mfadd, Val{2}   src/mfadd.jl 10-24
//   mfmul, Val{2}   src/mfmul.jl 10-23
//   x - y           src/MultiFloats.jl 1102 (x + (-y)), 1020 (limbwise -)
//   sqrt            src/MultiFloats.jl 1454, 1448-1449: sqrt_r returns a
//                   MultiFloat that iszero (653-654, 647) as it is, else
//                   mfsqrt, Val{2} (1416-1421), with div_r (1188) and sqrt_r
//                   on Float32 (1196, the correctly rounded square root).
//
// MultiFloats.jl is distributed under the following license.
//
// MIT License
//
// Copyright (c) 2019-2026 David K. Zhang
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Compile without floating-point contraction.

// clang-format off
#ifndef EXPANSION_MATCH_MULTIFLOATS_SUBSET_H
#define EXPANSION_MATCH_MULTIFLOATS_SUBSET_H

#include <cmath>

namespace mfref {

inline void two_sum(float a, float b, float &sum, float &err) {
  sum = a + b;
  float a_prime = sum - b;
  float b_prime = sum - a_prime;
  float a_err = a - a_prime;
  float b_err = b - b_prime;
  err = a_err + b_err;
}

inline void fast_two_sum(float a, float b, float &sum, float &b_err) {
  sum = a + b;
  float b_prime = sum - a;
  b_err = b - b_prime;
}

inline float one_prod(float a, float b) { return a * b; }

inline void two_prod(float a, float b, float &prod, float &err) {
  prod = one_prod(a, b);
  err = std::fma(a, b, -prod);
}

// mfadd(x, y, Val{2})
inline void mfadd(const float *x, const float *y, float *out) {
  float a, b, c, d;
  two_sum(x[0], y[0], a, b);
  two_sum(x[1], y[1], c, d);
  fast_two_sum(a, c, a, c);
  b += d;
  b += c;
  fast_two_sum(a, b, a, b);
  out[0] = a;
  out[1] = b;
}

// x - y = x + (-y)
inline void mfsub(const float *x, const float *y, float *out) {
  const float ny[2] = {-y[0], -y[1]};
  mfadd(x, ny, out);
}

// mfmul(x, y, Val{2})
inline void mfmul(const float *x, const float *y, float *out) {
  float p00, e00;
  two_prod(x[0], y[0], p00, e00);
  float p01 = one_prod(x[0], y[1]);
  float p10 = one_prod(x[1], y[0]);
  p01 += p10;
  e00 += p01;
  fast_two_sum(p00, e00, p00, e00);
  out[0] = p00;
  out[1] = e00;
}

// mfsqrt(x, Val{2})
inline void mfsqrt(const float *x, float *out) {
  float x_hi = x[0], x_lo = x[1];
  float y = std::sqrt(x_hi);
  float r = std::fma(-y, y, x_hi) + x_lo;
  fast_two_sum(y, r / (y + y), out[0], out[1]);
}

// sqrt_r on a two-limb MultiFloat: ifelse(iszero(x), x, mfsqrt(x)).
inline void sqrt_r(const float *x, float *out) {
  if (x[0] == 0.0f && x[1] == 0.0f) {
    out[0] = x[0];
    out[1] = x[1];
    return;
  }
  mfsqrt(x, out);
}

} // namespace mfref

#endif // EXPANSION_MATCH_MULTIFLOATS_SUBSET_H
// clang-format on
