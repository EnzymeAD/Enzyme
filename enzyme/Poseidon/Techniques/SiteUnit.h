//===- SiteUnit.h - an optimization unit of a site ------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the pass, beyond the solver, needs of an optimization unit: the site it
// belongs to, its footprint as instructions, its candidates by name, the fixed
// formats it appears in, and its materialization. Every kind of unit derives
// from SiteUnit; the solver sees the OptimizationUnit base.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_SITE_UNIT_H
#define POSEIDON_TECHNIQUES_SITE_UNIT_H

#include "Model/FPNode.h"
#include "Model/Sampling.h"
#include "Solve/DP.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class DebugLoc;
class Instruction;
class Value;
class raw_ostream;
} // namespace llvm

namespace poseidon {

class Session;
class FPGraph;
struct SiteState;

/// The kinds of unit, by the letter -poseidon-pin names them with.
enum class UnitKind { Rewrite, Precision, Matmul };

/// What the pass's fixed formats call one kind of unit.
struct UnitKindInfo {
  /// The kind the row describes; rows are listed in UnitKind order.
  UnitKind kind;
  /// What the solver's progress lines call the kind.
  llvm::StringLiteral noun;
  /// The letter of the ids <letter><i>_<j>, of the pins <letter><i>=<name>
  /// and of the DP table's steps.
  llvm::StringLiteral letter;
  /// What the duplicate-pin error says the unit already has.
  llvm::StringLiteral appliedNoun;
  /// Pricing may drop candidates: a candidate's ordinal is not its position,
  /// and a DP table step names its candidate by key as well as by position.
  bool pricingDrops;
};

/// The kinds, in the order the fixed formats list them.
llvm::ArrayRef<UnitKindInfo> unitKinds();
/// The row of `kind`.
const UnitKindInfo &infoOf(UnitKind kind);

class SiteUnit;
/// The unit a step takes a candidate of.
SiteUnit &unitOf(const SolutionStep &step);

/// An optimization unit of one site: the candidates one technique proposes
/// for a subgraph or a matrix product of it, how each is named, priced and
/// applied. A unit is used only after bind.
class SiteUnit : public OptimizationUnit {
public:
  /// Every unit of the pass is a SiteUnit.
  static bool classof(const OptimizationUnit *) { return true; }

  /// Binds the unit to the session and the site it was proposed for, as the
  /// site's `indexInKind`-th unit of its kind. Once. `graph` is the site as its
  /// graph: the caller converts, since the site's type is defined above this
  /// layer.
  void bind(Session &S, SiteState &site, FPGraph &graph, size_t indexInKind);
  /// The session, site and graph bind gave; the unit must be bound.
  Session &session() const {
    assert(boundSession && "unit read before its site was sealed");
    return *boundSession;
  }
  /// The site the unit was proposed for.
  SiteState &site() const {
    assert(boundSite && "unit read before its site was sealed");
    return *boundSite;
  }
  /// The site's expression graph, which owns the unit's nodes.
  FPGraph &graph() const {
    assert(boundGraph && "unit read before its site was sealed");
    return *boundGraph;
  }
  /// The i of R<i>, P<i>, M<i>, and the DP table's itemIndex.
  size_t indexInKind() const {
    assert(boundSite && "unit read before its site was sealed");
    return boundIndex;
  }

  UnitKind unitKind() const { return UnitKind(kind()); }
  /// The row of the unit's kind.
  const UnitKindInfo &info() const { return infoOf(unitKind()); }
  /// The noun of the unit's kind ("rewrite", "precision change").
  llvm::StringRef kindName() const final { return info().noun; }

  /// The instructions the unit's candidates edit or consume; null for none.
  virtual const llvm::SetVector<llvm::Instruction *> *footprint() const = 0;
  /// The outputs each candidate replaces.
  virtual llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const { return {}; }

  /// Whether the footprints share an instruction; a footprint does not
  /// overlap itself, and a unit without one overlaps none.
  bool footprintOverlaps(const OptimizationUnit &other) const final;
  /// Whether the two units' rewritten outputs share a value.
  bool rewritesSameOutputAs(const OptimizationUnit &other) const final;

  /// Candidate i's name: what a pin names it by and the DP table keys it by.
  virtual std::string candidateName(size_t i) const = 0;
  /// The j of candidate i's id <unit><i>_<j>.
  virtual size_t ordinalOf(size_t i) const { return i; }
  /// Candidate i's id, <unit><i>_<j>.
  std::string candidateId(size_t i) const;
  /// The position of the candidate whose ordinal is `ordinal`, unless pricing
  /// dropped it.
  std::optional<size_t> positionOfOrdinal(size_t ordinal) const;
  /// The candidates pricing dropped, by name, each with why.
  virtual llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const {
    return {};
  }
  /// Candidate i as the line selecting it by -poseidon-pin names it.
  virtual std::string selectionText(size_t i) const = 0;
  /// Candidate i as the DP table print and the materialization log name it.
  virtual std::string choiceText(size_t i) const = 0;
  /// Rewrites the site's IR with candidate i. A candidate that cannot be built
  /// as selected is an error, and the compile stops.
  virtual llvm::Error materialize(size_t i) = 0;

  /// Prices the candidates and drops those the accuracy model refuses; a unit
  /// priced when proposed does nothing.
  virtual llvm::Error price() { return llvm::Error::success(); }
  /// The first error the solve met pricing the unit's candidates, which ends
  /// the compile once the solve returns.
  virtual llvm::Error takePricingError() { return llvm::Error::success(); }
  /// Multiplies every error the unit carries by `factor`.
  virtual void scaleErrors(double factor) = 0;
  /// The -poseidon-print block of the priced candidates.
  virtual void printPriced(llvm::raw_ostream &OS) = 0;
  /// The unit's line of the DP table key's candidate set; each candidate
  /// follows on its own line as two spaces and its name.
  virtual std::string keyHeader() const = 0;

protected:
  /// A unit of kind `kind`, unbound.
  explicit SiteUnit(UnitKind kind) : OptimizationUnit(unsigned(kind)) {}

private:
  Session *boundSession = nullptr;
  SiteState *boundSite = nullptr;
  FPGraph *boundGraph = nullptr;
  size_t boundIndex = 0;
};

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

/// A source position; line 0 when there is none.
struct SourceLoc {
  /// The file as the debug location names it.
  std::string file;
  /// 1-based; line 0 is no position, column 0 an unknown column.
  unsigned line = 0, col = 0;
  /// Whether the location names a position.
  bool valid() const { return line != 0; }
};
/// The position of a debug location; none for an empty one.
SourceLoc locOf(const llvm::DebugLoc &DL);
/// The position of `V`'s debug location; none for a value that is no
/// instruction or has none.
SourceLoc locOf(const llvm::Value *V);
/// A position as the reports write it: "file", "line" and "col".
llvm::json::Object locJSON(const SourceLoc &L);
/// Sets o[key] to the position L, unless L is no position.
void putLoc(llvm::json::Object &o, llvm::StringRef key, const SourceLoc &L);
/// Sets o[key] to the source location of the instruction V, unless it has none.
void putLoc(llvm::json::Object &o, llvm::StringRef key, llvm::Value *V);
/// An instruction's source location as file:line:col; empty without one.
std::string getSourceLocationStr(llvm::Value *V);
/// A value as the IR prints it.
std::string getValueStr(llvm::Value *V);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_SITE_UNIT_H
