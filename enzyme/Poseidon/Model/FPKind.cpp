//===- FPKind.cpp - Number formats and rounding to them -------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPKind.h"

#include "llvm/ADT/APFloat.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace llvm;

namespace poseidon {

const char *fpKindName(FPKind k) {
  static constexpr const char *kNames[] = {
      "invalid",
#define FP_FORMAT(Kind, Name, Dtype, ...) Dtype,
#include "Model/FPFormat.def"
  };
  return kNames[unsigned(k)];
}

static const fltSemantics *semanticsForKind(FPKind k) {
  static constexpr const fltSemantics &(*kSemantics[])() = {
      nullptr,
#define FP_FORMAT(Kind, Name, Dtype, CostType, CType, Words, Storage,          \
                  Semantics, ...)                                              \
  Semantics,
#include "Model/FPFormat.def"
  };
  return kSemantics[unsigned(k)] ? &kSemantics[unsigned(k)]() : nullptr;
}

namespace {
struct Format {
  int precision = 0, minExponent = 0;
  double subnormalUlp = 0.0;
  uint64_t maxFinite = 0;
};
} // namespace

static Format formatOf(FPKind k) {
  Format f;
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return f;
  assert(APFloat::semanticsHasInf(*sem) && "overflow does not round to inf");
  f.precision = APFloat::semanticsPrecision(*sem);
  f.minExponent = APFloat::semanticsMinExponent(*sem);
  f.subnormalUlp = std::ldexp(1.0, f.minExponent - f.precision + 1);
  APFloat largest = APFloat::getLargest(*sem);
  bool losesInfo;
  largest.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
  f.maxFinite = largest.bitcastToAPInt().getZExtValue();
  return f;
}

Rounding::Rounding(FPKind k) : kind(k) {
  assert(k != FPKind::INT8 && k != FPKind::INT32 &&
         "an integer format has no floating-point rounding");
  static const Format formats[] = {
      formatOf(FPKind::Invalid),
#define FP_FORMAT(Kind, ...) formatOf(FPKind::Kind),
#include "Model/FPFormat.def"
  };
  const Format &f = formats[unsigned(k)];
  precision = f.precision;
  minExponent = f.minExponent;
  subnormalUlp = f.subnormalUlp;
  maxFinite = f.maxFinite;
}

// Round to nearest, ties to even, as APFloat::convert does, computed with
// integer arithmetic on the binary64 encoding: exact, independent of the
// floating-point environment, and cheap enough for every multiply-accumulate
// the matrix accuracy model simulates. memcpy rather than llvm::bit_cast keeps
// this one call per value in a build without optimization.
// The fields of an IEEE binary64: the stored fraction bits below the implicit
// leading bit, the exponent bias, and the biased exponent of an infinity or a
// NaN once unbiased.
static constexpr int kF64Digits = std::numeric_limits<double>::digits;
static constexpr int kF64FractionBits = kF64Digits - 1;
static constexpr int kF64Bias = std::numeric_limits<double>::max_exponent - 1;
static constexpr int kF64SpecialExponent = kF64Bias + 1;
static constexpr unsigned kF64SignBit = 63;

double Rounding::operator()(double x) const {
  if (precision == 0)
    return x;
  uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  const uint64_t sign = bits & (uint64_t(1) << kF64SignBit);
  const uint64_t mag = bits ^ sign;
  const int exponent = int(mag >> kF64FractionBits) - kF64Bias;
  uint64_t r;
  if (exponent == kF64SpecialExponent) {
    // Infinity and NaN keep APFloat's encoding.
    bool losesInfo;
    APFloat ap(x);
    ap.convert(*semanticsForKind(kind), APFloat::rmNearestTiesToEven,
               &losesInfo);
    ap.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
    return ap.convertToDouble();
  }
  if (exponent >= minExponent) {
    // A carry out of the kept significand bits moves the exponent up.
    const unsigned drop = kF64Digits - precision;
    const uint64_t half = uint64_t(1) << (drop - 1);
    r = (mag + half - 1 + ((mag >> drop) & 1)) >> drop << drop;
    if (r > maxFinite)
      r = uint64_t(0x7ff) << kF64FractionBits; // infinity
  } else {
    // A subnormal result is a multiple of subnormalUlp. A binary64 subnormal
    // lies far below any of these formats' and rounds to zero here.
    const int drop = kF64Digits - precision + (minExponent - exponent);
    r = 0;
    if (drop <= int(kF64SignBit)) {
      const uint64_t significand =
          (mag & ((uint64_t(1) << kF64FractionBits) - 1)) |
          (uint64_t(1) << kF64FractionBits);
      const uint64_t half = uint64_t(1) << (drop - 1);
      const double ulps =
          (double)((significand + half - 1 + ((significand >> drop) & 1)) >>
                   drop);
      const double rounded = ulps * subnormalUlp;
      std::memcpy(&r, &rounded, sizeof(r));
    }
  }
  r |= sign;
  double result;
  std::memcpy(&result, &r, sizeof(result));
  return result;
}

double roundToPrec(double x, FPKind k) { return Rounding(k)(x); }

double minSubnormalForKind(FPKind k) {
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return 0.0;
  APFloat sm = APFloat::getSmallest(*sem);
  bool li;
  sm.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &li);
  return sm.convertToDouble();
}

} // namespace poseidon
