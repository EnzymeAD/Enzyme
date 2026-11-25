//===- FPKind.h - Number formats and rounding to them ---------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The operand and accumulator formats of a matrix product, and rounding a
// double to them.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FPKIND_H
#define POSEIDON_FPKIND_H

#include <cstdint>

namespace poseidon {

/// The scalar formats the pass computes in, and the integer formats of the INT8
/// tensor-core path.
enum class FPKind {
  Invalid = 0,
  F16 = 1,
  BF16 = 2,
  TF32 = 3,
  F32 = 4,
  F64 = 5,
  /// 8-bit signed integer, the INT8 tensor-core operand.
  S8 = 6,
  /// 32-bit signed integer, the INT8 tensor-core accumulator.
  S32 = 7,
};
/// The format's name as diagnostics and cost-model rows spell it.
const char *fpKindName(FPKind k);

/// `x` rounded to nearest, ties to even, to the format `k` and returned as a
/// double; overflow gives an infinity and a value below the format's range a
/// subnormal or zero. `x` itself for F64 and Invalid; `k` must not be an
/// integer format.
double roundToPrec(double x, FPKind k);
/// roundToPrec(x, k) with the format of `k` looked up once, for a loop that
/// rounds many values to it.
class Rounding {
public:
  /// The rounding to `k`, which must not be an integer format.
  explicit Rounding(FPKind k);
  /// roundToPrec(x, kind).
  double operator()(double x) const;

private:
  /// The format rounded to.
  FPKind kind;
  /// 0 for the kinds every double is already in (F64, Invalid).
  int precision = 0;
  /// The format's smallest normal exponent.
  int minExponent = 0;
  /// The spacing of the format's subnormals, as a double.
  double subnormalUlp = 0.0;
  /// The binary64 encoding of the largest finite value of the format.
  uint64_t maxFinite = 0;
};
/// Smallest positive (subnormal) magnitude representable in `k`, as a double.
/// Returns 0 for F64/Invalid/integer kinds (no underflow constraint).
double minSubnormalForKind(FPKind k);

} // namespace poseidon
#endif // POSEIDON_FPKIND_H
