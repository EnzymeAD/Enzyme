//===- NumberFormat.cpp - Number formats and rounding to them -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NumberFormat.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace llvm;

namespace poseidon {

// The LLVMType column as one function type (getInt8Ty returns an IntegerType);
// null for an expansion.
template <auto Get> static Type *storage(LLVMContext &C) {
  if constexpr (Get == nullptr)
    return nullptr;
  else
    return Get(C);
}

static constexpr NumberFormatInfo kFormats[] = {
#define NUMBER_FORMAT(Kind, Name, Dtype, CostType, CType, Words, LLVMType,     \
                      Semantics, Components, Matrix)                           \
  {NumberFormat::Kind, Name,      Dtype,      CostType, CType, Words,          \
   storage<LLVMType>,  Semantics, Components, Matrix},
#include "Core/NumberFormat.def"
};

const NumberFormatInfo &formatOf(NumberFormat K) {
  assert(K != NumberFormat::Invalid && "Invalid has no row");
  return kFormats[unsigned(K) - 1];
}

// A format no graph value has (TF32, the integer formats, an expansion) has no
// C type, so a float is FP32 and an i8 is no format.
static bool isGraphFormat(const NumberFormatInfo &F) {
  return !F.CType.empty();
}

const NumberFormatInfo *formatOf(Type *T) {
  const NumberFormatInfo *F = find_if(kFormats, [&](const NumberFormatInfo &F) {
    return isGraphFormat(F) && F.LLVMType(T->getContext()) == T;
  });
  return F == std::end(kFormats) ? nullptr : F;
}

const NumberFormatInfo *formatOfDtype(StringRef Dtype) {
  const NumberFormatInfo *F = find_if(kFormats, [&](const NumberFormatInfo &F) {
    return isGraphFormat(F) && F.Dtype == Dtype;
  });
  return F == std::end(kFormats) ? nullptr : F;
}

NumberFormat matrixFormatOf(Type *T) {
  const NumberFormatInfo *F = formatOf(T);
  return F && F->Matrix ? F->Kind : NumberFormat::Invalid;
}

Type *storageTypeOf(LLVMContext &ctx, NumberFormat k) {
  const NumberFormatInfo &F = formatOf(k);
  assert(!F.Components && "an expansion has no LLVM type");
  return F.LLVMType(ctx);
}

const char *dtypeName(NumberFormat k) {
  return k == NumberFormat::Invalid ? "invalid" : formatOf(k).Dtype.data();
}

static const fltSemantics *semanticsForKind(NumberFormat k) {
  if (k == NumberFormat::Invalid)
    return nullptr;
  const NumberFormatInfo &F = formatOf(k);
  return F.Semantics ? &F.Semantics() : nullptr;
}

namespace {
struct Format {
  int precision = 0, minExponent = 0;
  double subnormalUlp = 0.0;
  uint64_t maxFinite = 0;
};
} // namespace

static Format roundingFormatOf(NumberFormat k) {
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

Rounding::Rounding(NumberFormat k) : kind(k) {
  assert(k != NumberFormat::INT8 && k != NumberFormat::INT32 &&
         "an integer format has no floating-point rounding");
  static const Format formats[] = {
      roundingFormatOf(NumberFormat::Invalid),
#define NUMBER_FORMAT(Kind, ...) roundingFormatOf(NumberFormat::Kind),
#include "Core/NumberFormat.def"
  };
  const Format &f = formats[unsigned(k)];
  precision = f.precision;
  minExponent = f.minExponent;
  subnormalUlp = f.subnormalUlp;
  maxFinite = f.maxFinite;
}

// The fields of an IEEE binary64: the stored fraction bits below the implicit
// leading bit, the exponent bias, and the biased exponent of an infinity or a
// NaN once unbiased.
static constexpr int kF64Digits = std::numeric_limits<double>::digits;
static constexpr int kF64FractionBits = kF64Digits - 1;
static constexpr int kF64Bias = std::numeric_limits<double>::max_exponent - 1;
static constexpr int kF64SpecialExponent = kF64Bias + 1;
static constexpr unsigned kF64SignBit = 63;

// Round to nearest, ties to even, as APFloat::convert does, computed with
// integer arithmetic on the binary64 encoding: exact, independent of the
// floating-point environment, and cheap enough for every multiply-accumulate
// the matrix accuracy model simulates. memcpy rather than llvm::bit_cast keeps
// this one call per value in a build without optimization.
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

double roundToPrec(double x, NumberFormat k) { return Rounding(k)(x); }

double minSubnormalOf(NumberFormat k) {
  const fltSemantics *sem = semanticsForKind(k);
  if (!sem)
    return 0.0;
  APFloat sm = APFloat::getSmallest(*sem);
  bool li;
  sm.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &li);
  return sm.convertToDouble();
}

} // namespace poseidon
