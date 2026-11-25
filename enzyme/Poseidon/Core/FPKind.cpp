//===- FPKind.cpp - Number formats and rounding to them -------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPKind.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace llvm;

namespace poseidon {

const char *fpKindName(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return "f16";
  case FPKind::BF16:
    return "bf16";
  case FPKind::TF32:
    return "tf32";
  case FPKind::F32:
    return "f32";
  case FPKind::F64:
    return "f64";
  case FPKind::S8:
    return "s8";
  case FPKind::S32:
    return "s32";
  case FPKind::Invalid:
    return "invalid";
  }
  llvm_unreachable("unknown FPKind value");
}

// nullptr for kinds with no FP rounding semantics (F64, Invalid, S8/S32).
static const fltSemantics *semanticsForKind(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return &APFloat::IEEEhalf();
  case FPKind::BF16:
    return &APFloat::BFloat();
  case FPKind::TF32:
    return &APFloat::FloatTF32();
  case FPKind::F32:
    return &APFloat::IEEEsingle();
  case FPKind::F64:
  case FPKind::Invalid:
  case FPKind::S8:
  case FPKind::S32:
    return nullptr;
  }
  return nullptr;
}

Rounding::Rounding(FPKind k) : kind(k) {
  assert(k != FPKind::S8 && k != FPKind::S32 &&
         "an integer format has no floating-point rounding");
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return;
  struct Format {
    int precision, minExponent;
    double subnormalUlp;
    uint64_t maxFinite;
  };
  auto formatOf = [](FPKind narrow) {
    const fltSemantics &sem = *semanticsForKind(narrow);
    assert(APFloat::semanticsHasInf(sem) && "overflow does not round to inf");
    Format f;
    f.precision = APFloat::semanticsPrecision(sem);
    f.minExponent = APFloat::semanticsMinExponent(sem);
    f.subnormalUlp = std::ldexp(1.0, f.minExponent - f.precision + 1);
    APFloat largest = APFloat::getLargest(sem);
    bool losesInfo;
    largest.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
    f.maxFinite = largest.bitcastToAPInt().getZExtValue();
    return f;
  };
  static const Format formats[] = {
      formatOf(FPKind::F16), formatOf(FPKind::BF16), formatOf(FPKind::TF32),
      formatOf(FPKind::F32)};
  const Format &f = formats[(int)k - (int)FPKind::F16];
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
