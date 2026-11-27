//===- Algebraic.h - algebraic rewrites from Herbie -----------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The algebraic units of an FP subgraph: Herbie's rewrites of its outputs,
// one output or an array of them per unit.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_ALGEBRAIC_ALGEBRAIC_H
#define POSEIDON_TECHNIQUES_ALGEBRAIC_ALGEBRAIC_H

#include "Core/FPNode.h"
#include "Core/Technique.h"
#include "Model/CostModel.h"
#include "Model/SubgraphUnit.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace llvm {
class LoopInfo;
class PassBuilder;
} // namespace llvm

namespace poseidon {

struct Options;

/// An expression Herbie returned for a core.
struct AlgebraicCandidate {
  /// Per execution, in cost-model units, once priced; the solver puts the
  /// change over the profiled run on its axis.
  std::optional<double> compCost;
  /// Herbie's cost of the expression over its cost of the core as written.
  double herbieCost = std::numeric_limits<double>::quiet_NaN();
  /// Herbie's accuracy estimate: 1 less its average error bits over the
  /// format's bits.
  double herbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  /// The accuracy cost of the outputs with the expression in place, on the
  /// scale of the unit's initialAccCost; NaN until priced.
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  /// The FPCore expression, over the core's symbols.
  std::string expr;
  /// Its position among the alternatives Herbie returned for its core, the j of
  /// its A<i>_<j> id. Pricing drops candidates and a table hit prices none, so
  /// the position in `candidates` differs between the two.
  size_t ordinal = 0;

  /// A candidate of Herbie's relative cost `cost` and accuracy `accuracy`.
  AlgebraicCandidate(double cost, double accuracy, std::string expression)
      : herbieCost(cost), herbieAccuracy(accuracy),
        expr(std::move(expression)) {}
};

/// How often the conversions on the boundary of a rewrite in another precision
/// execute in the emitted code, per execution of the rewritten outputs; a
/// conversion not listed runs with the outputs. A conversion of a value
/// computed outside the outputs' loop is hoisted to that value (`hoisted`).
/// A value the rewrite itself carries across the back edge stays in the
/// rewrite's precision inside the loop when the rewrite reads it only
/// converted and returns converted results: its conversion (`carried`) and the
/// results' (`output`) then run once per entry of the loop.
struct BoundaryCastFreq {
  /// Per input value computed outside the loop, how often its conversion runs
  /// per execution of the outputs.
  llvm::SmallDenseMap<const llvm::Value *, double, 8> hoisted;
  /// Per loop-carried input, how often its conversion runs per execution of
  /// the outputs.
  llvm::SmallDenseMap<const llvm::Value *, double, 2> carried;
  /// How often the results' conversions run per execution of the outputs.
  double output = 1.0;
};

/// The rewrites Herbie returned for one core: one output of a subgraph, or a
/// group of its outputs as an array, at one FPCore precision.
class AlgebraicUnit : public SubgraphUnit<AlgebraicUnit, AlgebraicCandidate> {
public:
  /// The kind of the algebraic units.
  static constexpr UnitKindInfo Info = {"algebraic", "A",
                                        "an algebraic rewrite", true};
  /// The core's FPCore body, over the subgraph's input symbols.
  std::string expr;
  /// Herbie's accuracy estimate of the core as written, as for a candidate.
  double initialHerbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  /// The outputs and the operations only they use, in an order that erases
  /// each user before what it uses: what a candidate replaces.
  llvm::SetVector<llvm::Instruction *> erasableInsts;
  /// The outputs one candidate replaces, in array element order: the one
  /// output of a per-output core, all of them for an array core.
  llvm::SmallVector<llvm::Value *, 1> roots;
  /// The gradient of each of `roots`, in the same order.
  llvm::SmallVector<double, 1> rootGrads;
  /// Whether Herbie returned an expression for the core.
  bool herbieAnswered = false;
  /// The FPCore precision the core is searched at: "binary64" or "binary32".
  std::string herbiePrecision;
  /// Herbie's status for the core, or "timeout" or "crashed" for a run that
  /// gave no answer.
  std::string herbieStatus;
  /// The bits Herbie measures the core's error in.
  double herbieBits = std::numeric_limits<double>::quiet_NaN();

  /// The unit of `outputs` with their gradients `grads`.
  explicit AlgebraicUnit(Subgraph &subgraph,
                         llvm::ArrayRef<llvm::Value *> outputs,
                         llvm::ArrayRef<double> grads, std::string expr,
                         uint64_t executions)
      : SubgraphUnit(Info, subgraph), expr(expr),
        roots(outputs.begin(), outputs.end()),
        rootGrads(grads.begin(), grads.end()) {
    this->executions = executions;
    findErasableInstructions();
  }

  /// Prices the outputs as written and the operations a candidate erases.
  llvm::Error priceAsWritten(CostModelState &CM);

  /// Whether the core is an array of several outputs.
  bool isArray() const { return roots.size() > 1; }

  /// The change candidate `candidateIndex` makes to the profiled run's cost,
  /// in cost-model units: its cost less the erased operations', times the
  /// executions. The candidate must be priced.
  double getCompCostDelta(size_t candidateIndex) const;
  /// How often a candidate's boundary conversions run, given the loops `LI`
  /// of the site's function.
  BoundaryCastFreq boundaryCastFreq(const llvm::LoopInfo &LI) const;

  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  bool consumesFootprint() const override { return false; }
  /// The searches of one output, and any two cores with the same outputs, are
  /// admitted and priced alike by every later unit.
  bool sameRewriteTargetAs(const OptimizationUnit &earlier) const override;
  llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const override {
    return roots;
  }
  void removeRewritten(Subgraph &left) const override;
  std::string candidateName(size_t i) const override {
    return candidates[i].expr;
  }
  std::string selectionText(size_t i) const override {
    return expr + " -> " + candidates[i].expr;
  }
  std::string choiceText(size_t i) const override;
  llvm::Error materialize(size_t i) override;
  llvm::Error price() override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override {
    return std::string(Info.letter) + " " + herbiePrecision + " " + expr;
  }
  llvm::Error explain(Explanation &E, UnitExplanation &U) override;
  void remarkApplied(llvm::OptimizationRemarkEmitter &ORE, Explanation &E,
                     const UnitExplanation &U, const Alternative &A,
                     llvm::StringRef ErrorChange,
                     llvm::StringRef CostChange) const override;
  std::string rivalReason(const Alternative &A,
                          const Alternative &Rival) const override;

private:
  void findErasableInstructions();
  /// What the operations a candidate erases cost per execution.
  double erasableCost = std::numeric_limits<double>::quiet_NaN();
};

/// The technique of algebraic rewrites from Herbie.
const Technique &algebraicTechnique();

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_ALGEBRAIC_ALGEBRAIC_H
