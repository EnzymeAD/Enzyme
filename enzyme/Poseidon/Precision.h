//=- Precision.h - Precision change utilities for Poseidon ----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for handling precision changes in the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PRECISION_H
#define POSEIDON_PRECISION_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace poseidon {

class FPNode;
class FPLLValue;
struct Subgraph;
class CandidateSubgraph;

enum class PrecisionChangeType {
  BF16,
  FP16,
  FP32,
  FP64,
  FP80,
  FP128,
  Expansion2,
  // Wider FP32 expansions; appended so no existing enumerator value shifts.
  Expansion3,
  Expansion4
};
// Component count of an FP32 expansion type, or 0 if `t` is not one; the one
// place the family's parameterization is written down.
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
unsigned getMPFRPrec(PrecisionChangeType type);
llvm::Type *getLLVMFPType(PrecisionChangeType type, llvm::LLVMContext &context);
PrecisionChangeType getPrecisionChangeType(llvm::Type *type);
llvm::StringRef getPrecisionChangeTypeString(PrecisionChangeType type);

enum class FPKind {
  Invalid = 0,
  F16 = 1,
  BF16 = 2,
  TF32 = 3,
  F32 = 4,
  F64 = 5,
  S8 = 6,  // 8-bit signed integer (INT8 tensor-core operand)
  S32 = 7, // 32-bit signed integer (INT8 tensor-core accumulator)
};
const char *fpKindName(FPKind k);
FPKind fpKindFromStr(llvm::StringRef tok);
FPKind fpKindFromType(llvm::Type *T);

double roundToPrec(double x, FPKind k);
// Smallest positive (subnormal) magnitude representable in `k`, as a double.
// Returns 0 for F64/Invalid/integer kinds (no underflow constraint).
double minSubnormalForKind(FPKind k);

struct PrecisionChange {
  llvm::SetVector<FPLLValue *> nodes;
  PrecisionChangeType oldType;
  PrecisionChangeType newType;

  explicit PrecisionChange(llvm::SetVector<FPLLValue *> &nodes,
                           PrecisionChangeType oldType,
                           PrecisionChangeType newType)
      : nodes(nodes), oldType(oldType), newType(newType) {}
};

struct PTCandidate {
  llvm::SmallVector<PrecisionChange, 1> changes;
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  // Double, not InstructionCost: see RewriteCandidate::CompCost. Native
  // cost-model rows are below 1.0 and would round to zero here.
  double CompCost = std::numeric_limits<double>::max();
  std::string desc;
  llvm::MapVector<FPNode *, double> perOutputAccCost;
  std::unordered_map<FPNode *, llvm::SmallVector<double, 4>> errors;

  explicit PTCandidate(llvm::SmallVector<PrecisionChange> changes,
                       const std::string &desc)
      : changes(std::move(changes)), desc(desc) {}

  void apply(Subgraph &subgraph, llvm::ValueToValueMapTy *VMap = nullptr);
};

void changePrecision(llvm::Instruction *I, PrecisionChange &change,
                     llvm::MapVector<llvm::Value *, llvm::Value *> &oldToNew);

double getCompCost(Subgraph &subgraph, PTCandidate &pt);

void setUnifiedAccuracyCost(
    CandidateSubgraph &CS,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

} // namespace poseidon
#endif // POSEIDON_PRECISION_H