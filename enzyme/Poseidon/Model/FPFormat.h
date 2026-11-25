//===- FPFormat.h - the formats a precision change uses -------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The formats an operation of the expression graph can be computed in, plain
// IEEE types and FP32 expansions, their precisions and LLVM types, and a
// precision change: the nodes it moves from one format to another.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_FP_FORMAT_H
#define POSEIDON_MODEL_FP_FORMAT_H

#include "Core/FPKind.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Type.h"

namespace llvm {
class LLVMContext;
} // namespace llvm

namespace poseidon {

class FPLLValue;

enum class PrecisionChangeType {
  BF16,
  FP16,
  FP32,
  FP64,
  FP80,
  FP128,
  Expansion2,
  Expansion3,
  Expansion4
};
// Component count of an FP32 expansion type, or 0 if `t` is not one.
inline unsigned expansionComponents(PrecisionChangeType t) {
  switch (t) {
  case PrecisionChangeType::Expansion2:
    return 2;
  case PrecisionChangeType::Expansion3:
    return 3;
  case PrecisionChangeType::Expansion4:
    return 4;
  default:
    return 0;
  }
}

// A format of the pass: its precision change type, its kind in the matrix
// model (Invalid for none), and its names as a graph dtype, as a cost-model
// type column, in C and in reports. An FP32 expansion has a precision change
// type and a report name only.
struct FPFormatInfo {
  PrecisionChangeType Precision;
  FPKind Kind;
  llvm::StringLiteral Dtype;
  llvm::StringLiteral CostType;
  llvm::StringLiteral CType;
  llvm::StringLiteral Name;
  llvm::Type *(*LLVMType)(llvm::LLVMContext &);
};

const FPFormatInfo &formatOf(PrecisionChangeType P);
// The format of `T`, or null when `T` is not a floating-point type of one.
const FPFormatInfo *formatOf(llvm::Type *T);
// The format a graph dtype names, or null.
const FPFormatInfo *formatOfDtype(llvm::StringRef Dtype);

llvm::Type *getLLVMFPType(PrecisionChangeType type, llvm::LLVMContext &context);
PrecisionChangeType getPrecisionChangeType(llvm::Type *type);
llvm::StringRef getPrecisionChangeTypeString(PrecisionChangeType type);

FPKind fpKindFromType(llvm::Type *T);
// The LLVM type a value of kind `k` is stored in: TF32 in float, S8 and S32 in
// i8 and i32.
llvm::Type *llvmTypeForFPKind(llvm::LLVMContext &ctx, FPKind k);

struct PrecisionChange {
  llvm::SetVector<FPLLValue *> nodes;
  PrecisionChangeType oldType;
  PrecisionChangeType newType;

  explicit PrecisionChange(llvm::SetVector<FPLLValue *> &nodes,
                           PrecisionChangeType oldType,
                           PrecisionChangeType newType)
      : nodes(nodes), oldType(oldType), newType(newType) {}
};

} // namespace poseidon
#endif // POSEIDON_MODEL_FP_FORMAT_H
