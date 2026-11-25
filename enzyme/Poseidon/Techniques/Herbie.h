//===- Herbie.h - algebraic rewrites from Herbie --------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The algebraic units of an FP subgraph: Herbie's rewrites of its outputs,
// one output or an array of them per unit.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_HERBIE_H
#define POSEIDON_HERBIE_H

#include "Model/CostModel.h"
#include "Model/FPNode.h"
#include "Unit.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace llvm {
class LoopInfo;
class PassBuilder;
} // namespace llvm

namespace poseidon {

struct Options;
class Session;

struct RewriteCandidate {
  // Per execution, in cost-model units; the solver puts the change over the
  // profiled run on its axis.
  double CompCost = std::numeric_limits<double>::max();
  double herbieCost = std::numeric_limits<double>::quiet_NaN();
  double herbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  std::string expr;
  // Its position among the alternatives Herbie returned for its core, the j of
  // its R<i>_<j> id. Pricing drops candidates and a table hit prices none, so
  // the position in `candidates` differs between the two.
  size_t ordinal = 0;

  RewriteCandidate(double cost, double accuracy, std::string expression)
      : herbieCost(cost), herbieAccuracy(accuracy),
        expr(std::move(expression)) {}
};

// How often the conversions on the boundary of a rewrite in another precision
// execute in the emitted code, per execution of the rewritten outputs; a
// conversion not listed runs with the outputs. A conversion of a value
// computed outside the outputs' loop is hoisted to that value (`hoisted`).
// A value the rewrite itself carries across the back edge stays in the
// rewrite's precision inside the loop when the rewrite reads it only
// converted and returns converted results: its conversion (`carried`) and the
// results' (`output`) then run once per entry of the loop.
struct BoundaryCastFreq {
  llvm::SmallDenseMap<const llvm::Value *, double, 8> hoisted;
  llvm::SmallDenseMap<const llvm::Value *, double, 2> carried;
  double output = 1.0;
};

class RewriteUnit : public SubgraphUnit<RewriteCandidate> {
public:
  std::string expr;
  double grad = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  llvm::SetVector<llvm::Instruction *> erasableInsts;
  // The outputs one candidate replaces, in array element order: the one
  // output of a per-output core, all of them for an array core.
  llvm::SmallVector<llvm::Value *, 1> roots;
  llvm::SmallVector<double, 1> rootGrads;
  bool herbieAnswered = false;
  std::string herbiePrecision;
  std::string herbieStatus;
  double herbieBits = std::numeric_limits<double>::quiet_NaN();

  // The unit of `outputs` with their gradients `grads`; its gradient is that
  // of a single output, and the sum of magnitudes of an array's.
  explicit RewriteUnit(CostModelState &CM, Subgraph &subgraph,
                       llvm::ArrayRef<llvm::Value *> outputs,
                       llvm::ArrayRef<double> grads, std::string expr,
                       uint64_t executions)
      : SubgraphUnit(UnitKind::R, subgraph), expr(expr),
        roots(outputs.begin(), outputs.end()),
        rootGrads(grads.begin(), grads.end()) {
    this->executions = executions;
    grad = grads.front();
    if (grads.size() > 1) {
      grad = 0.0;
      for (double g : grads)
        grad += std::fabs(g);
    }
    initialCompCost = getCompCost(
        CM, llvm::SmallVector<llvm::Value *>(outputs.begin(), outputs.end()),
        subgraph.inputs);
    findErasableInstructions();
  }

  bool isArray() const { return roots.size() > 1; }

  double getCompCostDelta(size_t candidateIndex);
  BoundaryCastFreq boundaryCastFreq(
      const llvm::LoopInfo &LI,
      const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
          &valueToNodeMap) const;

  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::R);
  }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  bool consumesFootprint() const override { return false; }
  // The searches of one output, and any two cores with the same outputs, are
  // admitted and priced alike by every later unit.
  bool sameRewriteTargetAs(const OptimizationUnit &earlier) const override;
  llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const override {
    return roots;
  }
  std::string candidateName(size_t i) const override {
    return candidates[i].expr;
  }
  std::string choiceText(size_t i) const override;
  void materialize(size_t i) override;
  void price() override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override {
    return "CO " + herbiePrecision + " " + expr;
  }
  std::string selectionText(size_t i) const override {
    return expr + " -> " + candidates[i].expr;
  }

private:
  void findErasableInstructions();
};

// Prints how many Herbie answers the compile ran fresh and read from the
// cache.
void reportHerbieAnswers(Session &S);

// Herbie's rewrites of the outputs of `subgraph`, the site's subgraph number
// `subgraphCounter`, appended to `COs` unpriced. Its inputs are named from
// `symbolCounter`, which the site's subgraphs share.
void proposeHerbieCandidates(
    Session &S, llvm::Function &F, Subgraph &subgraph, int subgraphCounter,
    int &symbolCounter,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVectorImpl<RewriteUnit> &COs);

} // namespace poseidon
#endif // POSEIDON_HERBIE_H
