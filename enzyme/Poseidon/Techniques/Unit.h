//===- Unit.h - an optimization unit of a site ----------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_TECHNIQUES_UNIT_H
#define POSEIDON_TECHNIQUES_UNIT_H

#include "Core/Solver.h"
#include "Model/FPNode.h"
#include "Model/Sampling.h"

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
struct FunctionFPState;

// The kinds of unit, by the letter -poseidon-apply-rewrites names them with.
enum class UnitKind { R, PT, M };

// What the pass's fixed formats call one kind of unit.
struct UnitKindInfo {
  UnitKind kind;
  // The DP table's itemType.
  llvm::StringLiteral tag;
  // What the solver's progress lines call the kind.
  llvm::StringLiteral noun;
  // The <unit> of the ids <unit><i>_<j> and of the pins <unit><i>=<name>.
  llvm::StringLiteral idPrefix;
  // What the duplicate-pin error says the unit already has.
  llvm::StringLiteral appliedNoun;
  // Pricing may drop candidates: a candidate's ordinal is not its position,
  // and a DP table step names its candidate by key as well as by position.
  bool pricingDrops;
};

// The kinds, in the order the fixed formats list them.
llvm::ArrayRef<UnitKindInfo> unitKinds();
const UnitKindInfo &infoOf(UnitKind kind);

class SiteUnit;
// The unit a step takes a candidate of.
SiteUnit &unitOf(const SolutionStep &step);

class SiteUnit : public OptimizationUnit {
public:
  // Every unit of the pass is a SiteUnit.
  static bool classof(const OptimizationUnit *) { return true; }

  // Binds the unit to the session and the site it was proposed for, as the
  // site's `indexInKind`-th unit of its kind. Once.
  void bind(Session &S, FunctionFPState &site, FPGraph &graph,
            size_t indexInKind);
  Session &session() const {
    assert(boundSession && "unit read before its site was sealed");
    return *boundSession;
  }
  FunctionFPState &site() const {
    assert(boundSite && "unit read before its site was sealed");
    return *boundSite;
  }
  FPGraph &graph() const {
    assert(boundGraph && "unit read before its site was sealed");
    return *boundGraph;
  }
  // The i of R<i>, PT<i>, M<i>, and the DP table's itemIndex.
  size_t indexInKind() const {
    assert(boundSite && "unit read before its site was sealed");
    return boundIndex;
  }

  UnitKind unitKind() const { return UnitKind(kind()); }
  const UnitKindInfo &info() const { return infoOf(unitKind()); }
  llvm::StringRef kindName() const final { return info().noun; }

  // The instructions the unit's candidates edit or consume; null for none.
  virtual const llvm::SetVector<llvm::Instruction *> *footprint() const = 0;
  // The outputs each candidate replaces.
  virtual llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const { return {}; }

  bool footprintOverlaps(const OptimizationUnit &other) const final;
  bool rewritesSameOutputAs(const OptimizationUnit &other) const final;

  // Candidate i's name: what a pin names it by and the DP table keys it by.
  virtual std::string candidateName(size_t i) const = 0;
  // The j of candidate i's id <unit><i>_<j>.
  virtual size_t ordinalOf(size_t i) const { return i; }
  // Candidate i's id, <unit><i>_<j>.
  std::string candidateId(size_t i) const;
  // The position of the candidate whose ordinal is `ordinal`, unless pricing
  // dropped it.
  std::optional<size_t> positionOfOrdinal(size_t ordinal) const;
  // The candidates pricing dropped, by name, each with why.
  virtual llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const {
    return {};
  }
  // Candidate i as the line selecting it by -poseidon-apply-rewrites names it.
  virtual std::string selectionText(size_t i) const = 0;
  // Candidate i as the DP table print and the materialization log name it.
  virtual std::string choiceText(size_t i) const = 0;
  // Rewrites the site's IR with candidate i.
  virtual void materialize(size_t i) = 0;

  // Prices the candidates and drops those the accuracy model refuses; a unit
  // priced when proposed does nothing.
  virtual void price() {}
  // Multiplies every error the unit carries by `factor`.
  virtual void scaleErrors(double factor) = 0;
  // The -poseidon-print block of the priced candidates.
  virtual void printPriced(llvm::raw_ostream &OS) = 0;
  // The unit's line of the DP table key's candidate set; each candidate
  // follows on its own line as two spaces and its name.
  virtual std::string keyHeader() const = 0;

protected:
  explicit SiteUnit(UnitKind kind) : OptimizationUnit(unsigned(kind)) {}

private:
  Session *boundSession = nullptr;
  FunctionFPState *boundSite = nullptr;
  FPGraph *boundGraph = nullptr;
  size_t boundIndex = 0;
};

// A subgraph's outputs as written, priced on its samples.
struct BaselineAccuracy {
  struct Output {
    const FPNode *node;
    // The output's values at the points.
    const SubgraphSamples::Reference *ref;
    // Its mean error there.
    double error;
    double grad;
  };
  llvm::ArrayRef<SubgraphSamples::Point> points;
  llvm::SmallVector<Output, 4> outputs;
  // The outputs' errors weighted by |gradient|, summed.
  double cost = 0.0;
};

// The baseline of the outputs `nodes`, of gradients `grads`, on `samples`; an
// error saying why the subgraph cannot be priced when there are no points,
// every sample of an output scores NaN, or the weighted sum is not finite.
llvm::Expected<BaselineAccuracy>
baselineAccuracy(SubgraphSamples &samples, llvm::ArrayRef<const FPNode *> nodes,
                 llvm::ArrayRef<double> grads, bool print);

// A unit whose candidates rewrite the operations of one FP subgraph, each
// priced against the subgraph as written. `Candidate` has an accuracyCost and
// an ordinal.
template <typename Candidate> class SubgraphUnit : public SiteUnit {
public:
  Subgraph *subgraph;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  // Per execution, in cost-model units.
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  uint64_t executions = 0;
  llvm::SmallVector<Candidate> candidates;
  // Why the subgraph could not be priced; empty when it was.
  std::string unpricedReason;
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;

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

  // The baseline of the outputs `nodes`, of gradients `grads`, with
  // initialAccCost set to its cost; none when the subgraph cannot be priced,
  // which drops every candidate and keeps the subgraph as written.
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
  SubgraphUnit(UnitKind kind, Subgraph &subgraph)
      : SiteUnit(kind), subgraph(&subgraph) {}
};

// A source position; line 0 when there is none.
struct SourceLoc {
  std::string file;
  unsigned line = 0, col = 0;
  bool valid() const { return line != 0; }
};
// The position of a debug location, or of an instruction's.
SourceLoc locOf(const llvm::DebugLoc &DL);
SourceLoc locOf(const llvm::Value *V);
// A position as the reports write it: "file", "line" and "col".
llvm::json::Object locJSON(const SourceLoc &L);
// Sets o[key] to the position L, unless L is no position.
void putLoc(llvm::json::Object &o, llvm::StringRef key, const SourceLoc &L);
// Sets o[key] to the source location of the instruction V, unless it has none.
void putLoc(llvm::json::Object &o, llvm::StringRef key, llvm::Value *V);
// An instruction's source location as file:line:col; empty without one.
std::string getSourceLocationStr(llvm::Value *V);
// A value as the IR prints it.
std::string getValueStr(llvm::Value *V);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_UNIT_H
