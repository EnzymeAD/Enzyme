//===- Precision.h - precision candidates of an FP subgraph ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The precision unit of an FP subgraph: assignments of its operations to
// narrower or wider formats and to floating-point expansions.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PRECISION_H
#define POSEIDON_PRECISION_H

#include "Model/CostModel.h"
#include "Model/FPFormat.h"
#include "Model/FPNode.h"
#include "Unit.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

namespace poseidon {

class PrecisionUnit;
class RewriteUnit;
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

  explicit PTCandidate(llvm::SmallVector<PrecisionChange> changes,
                       const std::string &desc)
      : changes(std::move(changes)), desc(desc) {}

  // Materializes the changes on `subgraph`, or, given `VMap`, on its clone.
  // `limbs`, if given, receives the limbs of the two-component change and then
  // those of the wider ones.
  void apply(Session &S, Subgraph &subgraph,
             llvm::ValueToValueMapTy *VMap = nullptr, FPGraph *graph = nullptr,
             llvm::SmallVectorImpl<llvm::Value *> *limbs = nullptr);
};

bool demoteFPCastPHIs(llvm::Function &F);

// The precision candidates of `subgraph`, appended to `CSs` unpriced: two- and
// three-tier assignments of its operations by profiled sensitivity, over the
// formats and expansions the target, the cost model and the options allow.
llvm::Error proposePrecisionCandidates(
    Session &S, llvm::Function &F, Subgraph &subgraph, FPGraph &G,
    const FunctionProfileHeader &profileHeader, bool expansionSafe,
    llvm::SmallVectorImpl<PrecisionUnit> &CSs);

class PrecisionUnit : public SubgraphUnit<PTCandidate> {
public:
  llvm::MapVector<FPNode *, double> perOutputInitialAccCost;

  using RewriteUnitSet = std::set<RewriteUnit *>;
  struct RepriceKey {
    size_t candidateIndex;
    RewriteUnitSet RewriteUnits;
    bool operator<(const RepriceKey &other) const {
      return std::tie(candidateIndex, RewriteUnits) <
             std::tie(other.candidateIndex, other.RewriteUnits);
    }
  };

  std::map<RepriceKey, double> compCostDeltaCache;
  std::map<RepriceKey, double> accCostDeltaCache;

  explicit PrecisionUnit(Subgraph &subgraph)
      : SubgraphUnit(UnitKind::PT, subgraph) {}

  // Prices the subgraph as written.
  llvm::Error priceAsWritten(CostModelState &CM);

  double getCompCostDelta(size_t candidateIndex);
  // Memoizes the cost delta of each (candidate, steps) request.
  void priceRequests(
      Session &S,
      llvm::ArrayRef<std::pair<size_t, llvm::ArrayRef<SolutionStep>>> requests);
  RepriceKey repriceKey(size_t candidateIndex,
                        llvm::ArrayRef<SolutionStep> steps) const;

  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::PT);
  }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  // Priced over the outputs the algebraic units of its subgraph leave.
  bool pricedAlone() const override { return false; }
  size_t memoizedPricings() const override { return compCostDeltaCache.size(); }
  void priceAhead(llvm::ArrayRef<llvm::ArrayRef<SolutionStep>> priors) override;
  bool consumesFootprint() const override { return false; }
  bool pricedAgainst(const OptimizationUnit &earlier) const override;
  std::string candidateName(size_t i) const override {
    return candidates[i].desc;
  }
  std::string choiceText(size_t i) const override;
  llvm::Error materialize(size_t i) override;
  llvm::Error price() override;
  llvm::Error takePricingError() override { return pricingError.take(); }
  void scaleErrors(double factor) override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override { return "CS"; }
  std::string selectionText(size_t i) const override {
    return candidates[i].desc;
  }

private:
  // What a repricing by the solve could not price, whose cost it took as 0.
  DeferredError pricingError;
};

} // namespace poseidon
#endif // POSEIDON_PRECISION_H