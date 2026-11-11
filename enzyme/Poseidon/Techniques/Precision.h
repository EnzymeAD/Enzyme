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
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "Model/FPFormat.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace poseidon {

class FPNode;
class FPLLValue;
struct Subgraph;
class PrecisionUnit;
struct Options;
class Session;
struct FunctionProfileHeader;

struct PTCandidate {
  llvm::SmallVector<PrecisionChange, 1> changes;
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  // Per execution, in cost-model units: see RewriteCandidate::CompCost.
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

bool demoteFPCastPHIs(llvm::Function &F);

double getCompCost(Session &S, Subgraph &subgraph, PTCandidate &pt);

void setUnifiedAccuracyCost(
    const Options &O, PrecisionUnit &CS,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

// The precision candidates of `subgraph`, appended to `CSs` unpriced: two- and
// three-tier assignments of its operations by profiled sensitivity, over the
// formats and expansions the target, the cost model and the options allow.
void proposePrecisionCandidates(
    Session &S, llvm::Function &F, Subgraph &subgraph,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const FunctionProfileHeader &profileHeader, bool expansionSafe,
    llvm::SmallVectorImpl<PrecisionUnit> &CSs);

} // namespace poseidon
#endif // POSEIDON_PRECISION_H