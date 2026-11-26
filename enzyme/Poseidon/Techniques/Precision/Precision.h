//===- Precision.h - precision candidates of an FP subgraph ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The precision unit of an FP subgraph: assignments of its operations to
// narrower or wider formats and to floating-point expansions.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_PRECISION_PRECISION_H
#define POSEIDON_TECHNIQUES_PRECISION_PRECISION_H

#include "Core/FPNode.h"
#include "Core/NumberFormat.h"
#include "Core/SubgraphUnit.h"
#include "Core/Technique.h"
#include "Model/CostModel.h"
#include "Model/Evaluators.h"
#include "Support/Diagnostics.h"

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
class AlgebraicUnit;
class Session;
struct SiteProfileHeader;

/// A precision candidate: an assignment of the subgraph's operations to
/// formats, one change per tier.
struct PrecisionCandidate {
  /// The tiers, each a set of nodes and the format they move to.
  llvm::SmallVector<PrecisionChange, 1> changes;
  /// The accuracy cost of the outputs with the changes applied, on the scale
  /// of the unit's initialAccCost; NaN until priced.
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  /// The cost of the outputs with the changes applied per execution, in
  /// cost-model units.
  double CompCost = std::numeric_limits<double>::max();
  /// The name reports and pins use: the operations each tier holds as shares
  /// ("All FP64(0%) + FP32(100%)").
  std::string desc;
  /// Its position when enumerated, the j of its P<i>_<j> id. Pricing drops
  /// candidates and a table hit prices none, so the position in `candidates`
  /// differs between the two.
  size_t ordinal = 0;
  /// Each output's part of accuracyCost, by the output's node.
  llvm::MapVector<FPNode *, double> perOutputAccCost;

  /// A candidate of the tiers `changes`, named `desc`.
  explicit PrecisionCandidate(llvm::SmallVector<PrecisionChange> changes,
                              const std::string &desc)
      : changes(std::move(changes)), desc(desc) {}

  /// Materializes the changes on `subgraph`, or, given `VMap`, on its clone.
  /// `components`, if given, receives the components of the two-component
  /// change and then those of the wider ones.
  void apply(Session &S, Subgraph &subgraph,
             llvm::ValueToValueMapTy *VMap = nullptr, FPGraph *graph = nullptr,
             llvm::SmallVectorImpl<llvm::Value *> *components = nullptr);
};

/// Demotes each FP64 PHI of `F` whose every incoming value is an extension of
/// an FP32 value or an FP32-representable constant to an FP32 PHI; whether one
/// was demoted.
bool demoteFPCastPHIs(llvm::Function &F);

/// The precision candidates of `subgraph`, appended to `precisionUnits`
/// unpriced: two- and three-tier assignments of its operations by profiled
/// sensitivity, over the formats and expansions the target, the cost model and
/// the options allow.
llvm::Error proposePrecisionCandidates(
    Session &S, llvm::Function &F, Subgraph &subgraph, FPGraph &G,
    const SiteProfileHeader &profileHeader, bool expansionSafe,
    llvm::SmallVectorImpl<PrecisionUnit> &precisionUnits);

/// The precision candidates of one subgraph. A candidate is repriced against
/// the outputs the selected rewrites of the same subgraph leave.
class PrecisionUnit : public SubgraphUnit<PrecisionCandidate> {
public:
  /// The kind of the precision units.
  static constexpr UnitKindInfo Info = {"precision change", "P",
                                        "a precision change", true};
  /// Each output's part of initialAccCost, by the output's node.
  llvm::MapVector<FPNode *, double> perOutputInitialAccCost;

  using AlgebraicUnitSet = std::set<AlgebraicUnit *>;
  /// A repricing: a candidate, by position, and the algebraic units of the same
  /// subgraph among the steps it is priced after.
  struct RepriceKey {
    /// The candidate's position in `candidates`.
    size_t candidateIndex;
    /// The algebraic units of the subgraph the candidate is priced after.
    AlgebraicUnitSet AlgebraicUnits;
    bool operator<(const RepriceKey &other) const {
      return std::tie(candidateIndex, AlgebraicUnits) <
             std::tie(other.candidateIndex, other.AlgebraicUnits);
    }
  };

  /// Each repricing's change to the profiled run's cost, in cost-model units.
  std::map<RepriceKey, double> compCostDeltaCache;
  /// Each repricing's accuracy cost less the outputs' as written, on the scale
  /// of initialAccCost.
  std::map<RepriceKey, double> accCostDeltaCache;

  /// An unbound unit over `subgraph`, which must outlive it.
  explicit PrecisionUnit(Subgraph &subgraph) : SubgraphUnit(Info, subgraph) {}

  /// Prices the subgraph as written.
  llvm::Error priceAsWritten(CostModelState &CM);

  /// The change candidate `candidateIndex` makes to the profiled run's cost
  /// over the whole subgraph, in cost-model units. The unit must be priced.
  double getCompCostDelta(size_t candidateIndex);
  std::optional<double> pricedCostDelta(size_t i) override {
    if (candidates[i].CompCost == std::numeric_limits<double>::max())
      return std::nullopt;
    return getCompCostDelta(i);
  }
  /// Memoizes the cost delta of each (candidate, steps) request.
  void priceRequests(
      Session &S,
      llvm::ArrayRef<std::pair<size_t, llvm::ArrayRef<SolutionStep>>> requests);
  /// The key of candidate `candidateIndex` priced after `steps`.
  RepriceKey repriceKey(size_t candidateIndex,
                        llvm::ArrayRef<SolutionStep> steps) const;

  static bool classof(const OptimizationUnit *U) {
    return &llvm::cast<SiteUnit>(U)->info() == &Info;
  }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  /// Priced over the outputs the algebraic units of its subgraph leave.
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
  std::string keyHeader() const override { return "P"; }
  std::string selectionText(size_t i) const override {
    return candidates[i].desc;
  }

private:
  /// What a repricing by the solve could not price, whose cost it took as 0.
  DeferredError pricingError;
};

/// What the precision technique keeps for a site.
struct PrecisionSiteState : TechniqueState {
  static inline char ID;
  /// The precision units proposed for the site.
  llvm::SmallVector<PrecisionUnit, 4> units;
  /// Whether the site's floating-point operations are rounded as written, so
  /// that floating-point expansions can be proposed for them.
  bool expansionSafe = true;
};

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_PRECISION_PRECISION_H
