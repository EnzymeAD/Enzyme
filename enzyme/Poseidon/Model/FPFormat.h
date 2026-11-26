//===- FPFormat.h - the formats a precision change uses -------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The names and LLVM types of the formats of FPFormat.def, and a precision
// change: the nodes it moves from one format to another.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_FPFORMAT_H
#define POSEIDON_MODEL_FPFORMAT_H

#include "Model/FPKind.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Type.h"

namespace llvm {
class LLVMContext;
} // namespace llvm

namespace poseidon {

class FPLLValue;

/// A row of FPFormat.def.
struct FPFormatInfo {
  FPKind Kind;
  llvm::StringLiteral Name, Dtype, CostType, CType, Words;
  llvm::Type *(*LLVMType)(llvm::LLVMContext &);
  unsigned Components;
  bool Matrix;
};

/// The row of `K`, which must not be Invalid.
const FPFormatInfo &formatOf(FPKind K);
/// The format a value of type `T` of the graph has, or null when `T` is not a
/// floating-point type of one.
const FPFormatInfo *formatOf(llvm::Type *T);
/// The format a graph dtype names, or null.
const FPFormatInfo *formatOfDtype(llvm::StringRef Dtype);
/// Component count of an FP32 expansion format, or 0 if `K` is not one.
inline unsigned expansionComponents(FPKind K) { return formatOf(K).Components; }

/// The matrix-model kind of `T`; Invalid for a type of no such kind.
FPKind fpKindFromType(llvm::Type *T);
/// The LLVM type a value of kind `k` is stored in: TF32 in float, S8 and S32 in
/// i8 and i32. `k` must be neither Invalid nor an expansion.
llvm::Type *llvmTypeForFPKind(llvm::LLVMContext &ctx, FPKind k);

/// The nodes of one tier of a precision candidate and the format they move
/// from and to. The nodes belong to the site's graph.
struct PrecisionChange {
  /// The operations of the tier.
  llvm::SetVector<FPLLValue *> nodes;
  /// The format they are computed in as written.
  FPKind oldType;
  /// The format the change computes them in.
  FPKind newType;

  explicit PrecisionChange(llvm::SetVector<FPLLValue *> &nodes, FPKind oldType,
                           FPKind newType)
      : nodes(nodes), oldType(oldType), newType(newType) {}
};

} // namespace poseidon
#endif // POSEIDON_MODEL_FPFORMAT_H
