//=- UniformReal.cpp - Uniform reals that do not depend on the library ----=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// std::uniform_real_distribution<double> as libstdc++ 14 computes it, the
// mapping the build's standard library used, written out so that results do
// not depend on the library: std::generate_canonical<double, 53> takes as many
// draws as cover 53 bits, sums them weighted by powers of the number R of
// values a draw spans and divides by R to that power, rounding each operation
// to double; a result that rounds to 1 becomes the largest double below 1; and
// U in [0, 1) becomes U * (B - A) + A.
//
//===----------------------------------------------------------------------===//

#include "UniformReal.h"

using namespace poseidon;

// minstd_rand0 draws from [1, 2147483646]: R = 2147483646, two draws, R * R
// rounded to double.
static constexpr double kMinstdRange = 2147483646.0;
static constexpr double kMinstdRangeSquared = 0x1.fffffffp+61;
// mt19937_64 draws from [0, 2^64 - 1]: R = 2^64, one draw.
static constexpr double kMT64Range = 0x1p+64;
static constexpr double kBelowOne = 0x1.fffffffffffffp-1;

static double scale(double U, double A, double B) {
  if (U >= 1.0)
    U = kBelowOne;
  double Span = B - A;
  double Scaled = U * Span;
  return Scaled + A;
}

double poseidon::uniformReal(std::minstd_rand0 &G, double A, double B) {
  std::minstd_rand0::result_type First = G();
  std::minstd_rand0::result_type Second = G();
  double Sum = double(First - std::minstd_rand0::min());
  double Next = double(Second - std::minstd_rand0::min()) * kMinstdRange;
  Sum += Next;
  return scale(Sum / kMinstdRangeSquared, A, B);
}

double poseidon::uniformReal(std::mt19937_64 &G, double A, double B) {
  double Draw = double(G() - std::mt19937_64::min());
  return scale(Draw / kMT64Range, A, B);
}
