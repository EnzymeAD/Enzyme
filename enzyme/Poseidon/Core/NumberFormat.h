//===- NumberFormat.h - Number formats and rounding to them ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The number formats of the pass: their names, the LLVM types their values
// are held in, and rounding a double to them.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_NUMBERFORMAT_H
#define POSEIDON_CORE_NUMBERFORMAT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Type.h"

#include <cstdint>

namespace llvm {
class LLVMContext;
struct fltSemantics;
} // namespace llvm

namespace poseidon {

/// A row of NumberFormat.def; Invalid for none.
enum class NumberFormat {
  Invalid,
#define NUMBER_FORMAT(Kind, ...) Kind,
#include "Core/NumberFormat.def"
};
/// The format's name as diagnostics and cost-model rows spell it.
const char *dtypeName(NumberFormat k);

/// `x` rounded to nearest, ties to even, to the format `k` and returned as a
/// double; overflow gives an infinity and a value below the format's range a
/// subnormal or zero. `x` itself for a format without Semantics and for
/// Invalid; `k` must not be an integer format.
double roundToPrec(double x, NumberFormat k);
/// roundToPrec(x, k) with the format of `k` looked up once, for a loop that
/// rounds many values to it.
class Rounding {
public:
  /// The rounding to `k`, which must not be an integer format.
  explicit Rounding(NumberFormat k);
  /// roundToPrec(x, kind).
  double operator()(double x) const;

private:
  /// The format rounded to.
  NumberFormat kind;
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
double minSubnormalOf(NumberFormat k);

/// A row of NumberFormat.def.
struct NumberFormatInfo {
  NumberFormat Kind;
  llvm::StringLiteral Name, Dtype, CostType, CType, Words;
  llvm::Type *(*LLVMType)(llvm::LLVMContext &);
  /// What roundToPrec rounds to; null for none.
  const llvm::fltSemantics &(*Semantics)();
  unsigned Components;
  bool Matrix;
};

/// The row of `K`, which must not be Invalid.
const NumberFormatInfo &formatOf(NumberFormat K);
/// The format a value of type `T` of the graph has, or null when `T` is not a
/// floating-point type of one.
const NumberFormatInfo *formatOf(llvm::Type *T);
/// The format a graph dtype names, or null.
const NumberFormatInfo *formatOfDtype(llvm::StringRef Dtype);
/// Component count of an FP32 expansion format, or 0 if `K` is not one.
inline unsigned expansionComponents(NumberFormat K) {
  return formatOf(K).Components;
}

/// The matrix-model kind of `T`; Invalid for a type of no such kind.
NumberFormat matrixFormatOf(llvm::Type *T);
/// The LLVM type a value of kind `k` is stored in: TF32 in float, S8 and S32 in
/// i8 and i32. `k` must be neither Invalid nor an expansion.
llvm::Type *storageTypeOf(llvm::LLVMContext &ctx, NumberFormat k);

} // namespace poseidon
#endif // POSEIDON_CORE_NUMBERFORMAT_H
