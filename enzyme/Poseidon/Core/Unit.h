//===- Unit.h - the interface every optimization unit implements ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An optimization unit offers candidates, each with a cost and an error change
// against the code as written, and has a footprint, the instructions its
// candidates edit or consume. The solver sees the OptimizationUnit base; the
// rest of the pass sees SiteUnit, which every technique's unit derives from:
// the site it belongs to, its footprint as instructions, its candidates by
// name, the fixed formats it appears in, and its materialization. The cost and
// error changes are the whole contract between a technique's models and the
// solver.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_UNIT_H
#define POSEIDON_CORE_UNIT_H

#include "Core/FPNode.h"
#include "Core/Session.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class Instruction;
class Value;
class raw_ostream;
} // namespace llvm

namespace poseidon {

class Explanation;
struct SiteState;
struct UnitExplanation;
class OptimizationUnit;

/// A candidate taken: candidate `candidateIndex` of `unit`.
struct SolutionStep {
  /// The unit; owned by the site it was proposed for.
  OptimizationUnit *unit;
  /// The candidate's index among the unit's candidates.
  size_t candidateIndex;

  /// Candidate `candidateIndex` of `unit`.
  SolutionStep(OptimizationUnit *unit, size_t candidateIndex)
      : unit(unit), candidateIndex(candidateIndex) {}
};

/// A place in the code the solver chooses for: the candidates it offers, their
/// prices, and how its steps combine with other units'.
class OptimizationUnit {
public:
  virtual ~OptimizationUnit();

  /// What the solver's progress lines call the unit's kind.
  virtual llvm::StringRef kindName() const = 0;

  /// How many candidates the unit offers.
  virtual size_t numCandidates() const = 0;

  /// Candidate i's cost change against the code as written, in cost-model units
  /// over the profiled run, given the steps already in the strategy.
  virtual double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) = 0;
  /// Candidate i's error change, on the application's scale, given the steps
  /// already in the strategy.
  virtual double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) = 0;
  /// The cost change, in cost-model units over the profiled run, by which
  /// candidate i's price can be off: a price measured as a wall-clock ratio
  /// repeats only to within it. 0 for a price exact on the model's terms.
  virtual double costTieBand(size_t i) const { return 0.0; }
  /// True when costDelta and errorDelta do not read `prior`. Only such a unit's
  /// step can be displaced, its deltas charged back exactly.
  virtual bool pricedAlone() const { return true; }
  /// The pricings the unit has memoized so far.
  virtual size_t memoizedPricings() const { return 0; }
  /// Called before the solver asks for the deltas of every candidate against
  /// each of `priors`, for a unit that prices several of them at once.
  virtual void priceAhead(llvm::ArrayRef<llvm::ArrayRef<SolutionStep>> priors) {
  }

  /// True when taking a candidate consumes the footprint (a raise replaces the
  /// loop that computes a product); false when it edits the footprint in place
  /// and composes with the other such units on it.
  virtual bool consumesFootprint() const = 0;
  /// Whether the two footprints share an instruction. A footprint does not
  /// overlap itself: a product is not in conflict with itself.
  virtual bool footprintOverlaps(const OptimizationUnit &other) const = 0;
  /// Whether the two units rewrite one output.
  virtual bool rewritesSameOutputAs(const OptimizationUnit &other) const = 0;
  /// Whether this unit's deltas depend on whether `earlier` was taken.
  virtual bool pricedAgainst(const OptimizationUnit &earlier) const {
    return false;
  }
  /// Whether every later unit admits and prices a step of this unit exactly
  /// as a step of `earlier`.
  virtual bool sameRewriteTargetAs(const OptimizationUnit &earlier) const {
    return false;
  }

protected:
  OptimizationUnit() = default;
  OptimizationUnit(const OptimizationUnit &) = default;
  OptimizationUnit(OptimizationUnit &&) = default;
  OptimizationUnit &operator=(const OptimizationUnit &) = default;
  OptimizationUnit &operator=(OptimizationUnit &&) = default;
};

/// What the pass's fixed formats call one kind of unit. Each kind is one
/// object, whose address identifies the kind: a unit class's classof compares
/// a unit's info() with its own.
struct UnitKindInfo {
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
  /// site's `indexInKind`-th unit of its kind. Once.
  void bind(Session &S, SiteState &site, size_t indexInKind);
  /// The session bind gave; the unit must be bound.
  Session &session() const {
    assert(boundSession && "unit read before its site was sealed");
    return *boundSession;
  }
  /// The site the unit was proposed for.
  SiteState &site() const {
    assert(boundSite && "unit read before its site was sealed");
    return *boundSite;
  }
  /// The site's expression graph, which owns the unit's nodes: the site.
  FPGraph &graph() const;
  /// The i of A<i>, P<i>, M<i>, and the DP table's itemIndex.
  size_t indexInKind() const {
    assert(boundSite && "unit read before its site was sealed");
    return boundIndex;
  }

  /// The unit's kind, which its technique registers (Core/Technique.h).
  const UnitKindInfo &info() const { return *kindInfo; }
  /// The noun of the unit's kind ("algebraic", "precision change").
  llvm::StringRef kindName() const final { return info().noun; }

  /// The FP subgraph the unit's candidates rewrite; null for a unit over
  /// something else.
  virtual const Subgraph *subgraphOf() const { return nullptr; }
  /// The instructions the unit's candidates edit or consume; null for none.
  virtual const llvm::SetVector<llvm::Instruction *> *footprint() const = 0;
  /// The outputs each candidate replaces.
  virtual llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const { return {}; }
  /// Removes from `left`, a copy of the unit's subgraph, the outputs a step of
  /// the unit replaces and the operations it erases with them.
  virtual void removeRewritten(Subgraph &left) const {}

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
  droppedCandidates() const = 0;
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
  /// Candidate i's cost change against the code as written, in cost-model
  /// units over the profiled run, as priced on its own; none for a
  /// candidate pricing left without a cost.
  virtual std::optional<double> pricedCostDelta(size_t i) = 0;
  /// Multiplies every error the unit carries by `factor`.
  virtual void scaleErrors(double factor) = 0;
  /// The -poseidon-print block of the priced candidates.
  virtual void printPriced(llvm::raw_ostream &OS) = 0;
  /// The unit's line of the DP table key's candidate set; each candidate
  /// follows on its own line as two spaces and its name.
  virtual std::string keyHeader() const = 0;
  /// Adds the unit's candidates to `U`, the entry of its subgraph or of the
  /// unit itself in the explanation `E` of the site's solve; an error is one
  /// `E` reports.
  virtual llvm::Error explain(Explanation &E, UnitExplanation &U) = 0;

protected:
  /// A unit of kind `kind`, unbound.
  explicit SiteUnit(const UnitKindInfo &kind) : kindInfo(&kind) {}

private:
  const UnitKindInfo *kindInfo;
  Session *boundSession = nullptr;
  SiteState *boundSite = nullptr;
  size_t boundIndex = 0;
};

} // namespace poseidon
#endif // POSEIDON_CORE_UNIT_H
