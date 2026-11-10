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
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "Core/FPKind.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace poseidon {

class FPNode;
class FPLLValue;
struct Subgraph;
class CandidateSubgraph;
struct Options;
class Session;
struct FunctionProfileHeader;

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

FPKind fpKindFromType(llvm::Type *T);

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
  // Its position when enumerated, the j of its PT<i>_<j> id; see
  // RewriteCandidate::ordinal.
  size_t ordinal = 0;
  llvm::MapVector<FPNode *, double> perOutputAccCost;
  std::unordered_map<FPNode *, llvm::SmallVector<double, 4>> errors;

  explicit PTCandidate(llvm::SmallVector<PrecisionChange> changes,
                       const std::string &desc)
      : changes(std::move(changes)), desc(desc) {}

  void apply(Session &S, Subgraph &subgraph,
             llvm::ValueToValueMapTy *VMap = nullptr);
};

void changePrecision(llvm::Instruction *I, PrecisionChange &change,
                     llvm::MapVector<llvm::Value *, llvm::Value *> &oldToNew);

double getCompCost(Session &S, Subgraph &subgraph, PTCandidate &pt);

void setUnifiedAccuracyCost(
    const Options &O, CandidateSubgraph &CS,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

// The precision candidates of `subgraph`, appended to `CSs` unpriced: two- and
// three-tier assignments of its operations by profiled sensitivity, over the
// formats and expansions the target, the cost model and the options allow.
void proposePrecisionCandidates(
    Session &S, llvm::Function &F, Subgraph &subgraph,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const FunctionProfileHeader &profileHeader, bool expansionSafe,
    llvm::SmallVectorImpl<CandidateSubgraph> &CSs);

} // namespace poseidon
#endif // POSEIDON_PRECISION_H