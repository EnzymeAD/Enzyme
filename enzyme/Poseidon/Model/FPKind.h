//===- FPKind.h - Number formats and rounding to them ---------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The number formats of the pass, and rounding a double to them.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_FPKIND_H
#define POSEIDON_MODEL_FPKIND_H

#include <cstdint>

namespace poseidon {

/// A row of FPFormat.def; Invalid for none.
enum class FPKind {
  Invalid,
#define FP_FORMAT(Kind, ...) Kind,
#include "Model/FPFormat.def"
};
/// The format's name as diagnostics and cost-model rows spell it.
const char *fpKindName(FPKind k);

/// `x` rounded to nearest, ties to even, to the format `k` and returned as a
/// double; overflow gives an infinity and a value below the format's range a
/// subnormal or zero. `x` itself for a format without Semantics and for
/// Invalid; `k` must not be an integer format.
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
  /// 0 for a format without Semantics and for Invalid.
  int precision = 0;
  /// The format's smallest normal exponent.
  int minExponent = 0;
  /// The spacing of the format's subnormals, as a double.
  double subnormalUlp = 0.0;
  /// The binary64 encoding of the largest finite value of the format.
  uint64_t maxFinite = 0;
};
/// Smallest positive (subnormal) magnitude representable in `k`, as a double;
/// 0 for a format without Semantics and for Invalid.
double minSubnormalForKind(FPKind k);

} // namespace poseidon
#endif // POSEIDON_MODEL_FPKIND_H
