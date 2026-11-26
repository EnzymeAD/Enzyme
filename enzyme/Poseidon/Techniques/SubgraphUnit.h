//===- SubgraphUnit.h - a unit over one FP subgraph -----------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the units of the elementwise techniques share: a unit whose candidates
// rewrite the operations of one FP subgraph, each priced on the subgraph's
// samples against the subgraph as written.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_SUBGRAPH_UNIT_H
#define POSEIDON_TECHNIQUES_SUBGRAPH_UNIT_H

#include "Core/FPNode.h"
#include "Core/Unit.h"
#include "Model/Sampling.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace poseidon {

/// A subgraph's outputs as written, priced on its samples.
struct BaselineAccuracy {
  /// One output as written; the pointers are into the samples, which must
  /// outlive it.
  struct Output {
    /// The output's node, of the site's graph.
    const FPNode *node;
    /// The output's values at the points.
    const SubgraphSamples::Reference *ref;
    /// Its mean absolute error there, in the units of the value.
    double error;
    /// The output's gradient; its error is weighted by |grad|.
    double grad;
  };
  /// The samples' points, owned by the samples.
  llvm::ArrayRef<SubgraphSamples::Point> points;
  /// One entry per output, in the order they were given.
  llvm::SmallVector<Output, 4> outputs;
  /// The outputs' mean absolute errors weighted by |gradient|, summed.
  double cost = 0.0;
};

/// The baseline of the outputs `nodes`, of gradients `grads`, on `samples`; an
/// error saying why the subgraph cannot be priced when there are no points,
/// every sample of an output scores NaN, or the weighted sum is not finite.
llvm::Expected<BaselineAccuracy>
baselineAccuracy(SubgraphSamples &samples, llvm::ArrayRef<const FPNode *> nodes,
                 llvm::ArrayRef<double> grads, bool print);

/// A unit whose candidates rewrite the operations of one FP subgraph, each
/// priced against the subgraph as written. `Candidate` has an accuracyCost and
/// an ordinal.
template <typename Candidate> class SubgraphUnit : public SiteUnit {
public:
  /// The subgraph, of the site the unit is bound to.
  Subgraph *subgraph;
  /// The accuracy cost of the outputs as written: each output's mean absolute
  /// error on the samples times |its gradient|, summed, and multiplied by the
  /// site's error scale once scaleErrors has run. NaN until priced; 0 when the
  /// subgraph cannot be priced.
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  /// The cost of the outputs as written per execution, in cost-model units.
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  /// How often the profiling run executed the unit's outputs: a cost per
  /// execution times this is the profiled run's.
  uint64_t executions = 0;
  /// The candidates kept so far; pricing drops those it cannot price.
  llvm::SmallVector<Candidate> candidates;
  /// Why the subgraph could not be priced; empty when it was.
  std::string unpricedReason;
  /// The candidates pricing dropped: each one's name and why.
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;

  /// Candidate i's accuracy cost less the subgraph's as written, on the scale
  /// of initialAccCost.
  double getAccCostDelta(size_t i) const {
    return candidates[i].accuracyCost - initialAccCost;
  }

  size_t numCandidates() const override { return candidates.size(); }
  size_t ordinalOf(size_t i) const override { return candidates[i].ordinal; }
  llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const override {
    return dropped;
  }
  const llvm::SetVector<llvm::Instruction *> *footprint() const override {
    return subgraph ? &subgraph->operations : nullptr;
  }
  void scaleErrors(double factor) override {
    initialAccCost *= factor;
    for (Candidate &c : candidates)
      c.accuracyCost *= factor;
  }

  /// The baseline of the outputs `nodes`, of gradients `grads`, with
  /// initialAccCost set to its cost; none when the subgraph cannot be priced,
  /// which drops every candidate and keeps the subgraph as written.
  std::optional<BaselineAccuracy>
  priceBaseline(SubgraphSamples &samples, llvm::ArrayRef<const FPNode *> nodes,
                llvm::ArrayRef<double> grads, bool print) {
    llvm::Expected<BaselineAccuracy> B =
        baselineAccuracy(samples, nodes, grads, print);
    if (B) {
      initialAccCost = B->cost;
      return std::move(*B);
    }
    unpricedReason = llvm::toString(B.takeError());
    if (print)
      llvm::errs() << "[poseidon] subgraph cannot be priced (" << unpricedReason
                   << "); discarding its " << candidates.size() << ' '
                   << info().noun
                   << " candidate(s) and keeping the original body\n";
    candidates.clear();
    initialAccCost = 0.0;
    return std::nullopt;
  }

protected:
  /// An unbound unit of kind `kind` over `subgraph`, which must outlive it.
  SubgraphUnit(UnitKind kind, Subgraph &subgraph)
      : SiteUnit(kind), subgraph(&subgraph) {}
};

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_SUBGRAPH_UNIT_H
