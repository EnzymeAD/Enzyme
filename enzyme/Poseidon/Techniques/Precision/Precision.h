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
#include "Core/Technique.h"
#include "Model/Evaluators.h"
#include "Model/SubgraphUnit.h"
#include "Support/Diagnostics.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace poseidon {

struct CostModelState;
class Session;

/// A precision candidate: an assignment of the subgraph's operations to
/// formats, one change per tier.
struct PrecisionCandidate {
  /// The tiers, each a set of nodes and the format they move to.
  llvm::SmallVector<FormatChange, 1> changes;
  /// The accuracy cost of the outputs with the changes applied, on the scale
  /// of the unit's initialAccCost; NaN until priced.
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  /// The cost of the outputs with the changes applied per execution, in
  /// cost-model units, once priced.
  std::optional<double> compCost;
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
  explicit PrecisionCandidate(llvm::SmallVector<FormatChange> changes,
                              const std::string &desc)
      : changes(std::move(changes)), desc(desc) {}

  /// Materializes the changes on `subgraph`, or, given `VMap`, on its clone.
  /// `components`, if given, receives the components of the two-component
  /// change and then those of the wider ones.
  void apply(Session &S, Subgraph &subgraph,
             llvm::ValueToValueMapTy *VMap = nullptr, FPGraph *graph = nullptr,
             llvm::SmallVectorImpl<llvm::Value *> *components = nullptr);
};

/// The profiled execution counts a precision unit weighs the instructions of
/// its subgraph by.
struct MeasuredFrequencies {
  /// Measured execution counts of the subgraph's instructions over the
  /// profiling run, which survive materialization. Empty prices exactly as an
  /// unweighted walk.
  llvm::DenseMap<const llvm::Instruction *, uint64_t> opExec;
  /// The same counts by loop depth.
  std::map<unsigned, uint64_t> depthExec;
  /// The execution count the counts are divided by: the first output's.
  uint64_t execNormalizer = 0;
  /// Frequency scale for synthetic output-boundary casts: they execute at the
  /// insertion point's frequency (the output's consumer), not the body's.
  /// min(1, threadsPerLaunch * launchCount / exec(output)); 1 for elementwise.
  double outBoundaryFreqScale = 1.0;

  /// The measured count of the deepest class at or below `d` (the shallowest
  /// class outside every recorded loop) over the normalizer; 1.0 without one.
  double freqWeightAtDepth(unsigned d) const {
    if (depthExec.empty() || execNormalizer == 0)
      return 1.0;
    auto it = depthExec.upper_bound(d);
    uint64_t e = (it == depthExec.begin()) ? depthExec.begin()->second
                                           : std::prev(it)->second;
    return (double)e / (double)execNormalizer;
  }
};

/// The precision candidates of one subgraph. A candidate is repriced against
/// the outputs the selected rewrites of the same subgraph leave.
class PrecisionUnit : public SubgraphUnit<PrecisionUnit, PrecisionCandidate> {
public:
  /// The kind of the precision units.
  static constexpr UnitKindInfo Info = {"precision change", "P",
                                        "a precision change", true};
  /// Each output's part of initialAccCost, by the output's node.
  llvm::MapVector<FPNode *, double> perOutputInitialAccCost;
  /// The counts the subgraph's instructions are priced by.
  MeasuredFrequencies frequencies;

  /// A repricing: a candidate, by position, and the units rewriting outputs
  /// of the same subgraph among the steps it is priced after.
  struct RepriceKey {
    /// The candidate's position in `candidates`.
    size_t candidateIndex;
    /// The units rewriting outputs of the subgraph that the candidate is
    /// priced after.
    std::set<const SiteUnit *> rewrites;
    bool operator<(const RepriceKey &other) const {
      return std::tie(candidateIndex, rewrites) <
             std::tie(other.candidateIndex, other.rewrites);
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
  /// over the whole subgraph, in cost-model units. The candidate must be
  /// priced.
  double getCompCostDelta(size_t candidateIndex) const;
  /// Memoizes the cost delta of each (candidate, steps) request.
  void priceRequests(
      Session &S,
      llvm::ArrayRef<std::pair<size_t, llvm::ArrayRef<SolutionStep>>> requests);
  /// The key of candidate `candidateIndex` priced after `steps`.
  RepriceKey repriceKey(size_t candidateIndex,
                        llvm::ArrayRef<SolutionStep> steps) const;

  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  /// Priced over what the rewrites of its subgraph's outputs leave of it.
  bool pricedAlone() const override { return false; }
  size_t memoizedPricings() const override { return compCostDeltaCache.size(); }
  void priceAhead(llvm::ArrayRef<llvm::ArrayRef<SolutionStep>> priors) override;
  bool consumesFootprint() const override { return false; }
  bool pricedAgainst(const OptimizationUnit &earlier) const override;
  std::string candidateName(size_t i) const override {
    return candidates[i].desc;
  }
  std::string selectionText(size_t i) const override {
    return candidates[i].desc;
  }
  std::string choiceText(size_t i) const override;
  llvm::Error materialize(size_t i) override;
  llvm::Error price() override;
  llvm::Error takePricingError() override { return pricingError.take(); }
  void scaleErrors(double factor) override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override { return "P"; }
  llvm::Error explain(Explanation &E, UnitExplanation &U) override;
  void remarkApplied(llvm::OptimizationRemarkEmitter &ORE, Explanation &E,
                     const UnitExplanation &U, const Alternative &A,
                     llvm::StringRef ErrorChange,
                     llvm::StringRef CostChange) const override;
  std::string rivalReason(const Alternative &A,
                          const Alternative &Rival) const override;

private:
  /// What a repricing by the solve could not price, whose cost it took as 0.
  DeferredError pricingError;
};

/// The technique of precision changes and floating-point expansions.
const Technique &precisionTechnique();

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_PRECISION_PRECISION_H
