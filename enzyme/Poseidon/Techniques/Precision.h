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
#include "llvm/Support/JSON.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "Model/CostModel.h"
#include "Model/FPFormat.h"
#include "Model/FPNode.h"
#include "Unit.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace poseidon {

class FPNode;
class FPLLValue;
struct Subgraph;
class PrecisionUnit;
class RewriteUnit;
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
             llvm::ValueToValueMapTy *VMap = nullptr, FPGraph *graph = nullptr);
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

class PrecisionUnit : public SiteUnit {
public:
  Subgraph *subgraph;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  uint64_t executions = 0;
  llvm::MapVector<FPNode *, double> perOutputInitialAccCost;
  llvm::SmallVector<PTCandidate, 8> candidates;
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;

  using RewriteUnitSet = std::set<RewriteUnit *>;
  struct RepriceKey {
    size_t candidateIndex;
    RewriteUnitSet RewriteUnits;
    bool operator==(const RepriceKey &other) const;
  };

  struct RepriceKeyHash {
    std::size_t operator()(const RepriceKey &key) const;
  };

  std::unordered_map<RepriceKey, double, RepriceKeyHash> compCostDeltaCache;
  std::unordered_map<RepriceKey, double, RepriceKeyHash> accCostDeltaCache;

  explicit PrecisionUnit(CostModelState &CM, Subgraph &subgraph)
      : SiteUnit(UnitKind::PT), subgraph(&subgraph) {
    initialCompCost =
        getCompCost(CM, {subgraph.outputs.begin(), subgraph.outputs.end()},
                    subgraph.inputs, &subgraph.opExec, subgraph.execNormalizer);
  }

  void apply(Session &S, size_t candidateIndex);
  double getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);
  double getAdjustedCompCostDelta(Session &S, size_t candidateIndex,
                                  llvm::ArrayRef<SolutionStep> steps);
  double getAdjustedAccCostDelta(
      size_t candidateIndex, llvm::ArrayRef<SolutionStep> steps,
      std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
          &valueToNodeMap,
      std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::PT);
  }
  size_t numCandidates() const override { return candidates.size(); }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  // Priced over the outputs the algebraic units of its subgraph leave.
  bool pricedAlone() const override { return false; }
  size_t memoizedPricings() const override { return compCostDeltaCache.size(); }
  bool consumesFootprint() const override { return false; }
  bool pricedAgainst(const OptimizationUnit &earlier) const override;
  const llvm::SetVector<llvm::Instruction *> *footprint() const override;
  std::string candidateName(size_t i) const override {
    return candidates[i].desc;
  }
  size_t ordinalOf(size_t i) const override { return candidates[i].ordinal; }
  std::string choiceText(size_t i) const override;
  llvm::json::Object stepJSON(size_t i) override;
  void writeReportStep(llvm::raw_ostream &OS, size_t i,
                       llvm::StringRef pad) override;
  void materialize(size_t i) override;
  void price() override;
  void scaleErrors(double factor) override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override { return "CS"; }
  llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const override {
    return dropped;
  }
  std::string selectionText(size_t i) const override {
    return candidates[i].desc;
  }
  void addReportSummary(llvm::json::Array &summary) override;
  void writeReportSummary(llvm::raw_ostream &OS) override;
  void addCuratedRewrites(llvm::json::Array &rewrites) override;
};

} // namespace poseidon
#endif // POSEIDON_PRECISION_H