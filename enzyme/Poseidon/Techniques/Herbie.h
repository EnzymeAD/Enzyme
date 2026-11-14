//=- Herbie.h - Herbie integration utilities ------------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for integrating with the Herbie tool for
// floating-point expression optimization.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_HERBIE_H
#define POSEIDON_HERBIE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Core/CacheKey.h"
#include "Model/CostModel.h"
#include "Model/FPCore.h"
#include "Model/FPNode.h"
#include "Unit.h"

namespace llvm {
class LoopInfo;
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
      : herbieCost(cost), herbieAccuracy(accuracy), expr(expression) {}
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

class RewriteUnit : public SiteUnit {
public:
  Subgraph *subgraph;
  llvm::Value *oldOutput;
  std::string expr;
  double grad = std::numeric_limits<double>::quiet_NaN();
  uint64_t executions = 0;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieCost = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  llvm::SmallVector<RewriteCandidate> candidates;
  llvm::SetVector<llvm::Instruction *> erasableInsts;
  // The outputs one candidate replaces, in array element order: the single
  // oldOutput for a per-output core, all of them for an array core.
  llvm::SmallVector<llvm::Value *, 1> roots;
  llvm::SmallVector<double, 1> rootGrads;
  bool herbieAnswered = false;
  std::string herbiePrecision;
  std::string herbieStatus;
  double herbieBits = std::numeric_limits<double>::quiet_NaN();
  std::string unpricedReason;
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;

  explicit RewriteUnit(CostModelState &CM, Subgraph &subgraph,
                       llvm::Value *oldOutput, std::string expr, double grad,
                       uint64_t executions)
      : SiteUnit(UnitKind::R), subgraph(&subgraph), oldOutput(oldOutput),
        expr(expr), grad(grad), executions(executions), roots({oldOutput}),
        rootGrads({grad}) {
    initialCompCost = getCompCost(CM, {oldOutput}, subgraph.inputs);
    findErasableInstructions();
  }

  explicit RewriteUnit(CostModelState &CM, Subgraph &subgraph,
                       llvm::ArrayRef<llvm::Value *> outputs,
                       llvm::ArrayRef<double> grads, std::string expr,
                       uint64_t executions)
      : SiteUnit(UnitKind::R), subgraph(&subgraph), oldOutput(outputs.front()),
        expr(expr), executions(executions),
        roots(outputs.begin(), outputs.end()),
        rootGrads(grads.begin(), grads.end()) {
    grad = 0.0;
    for (double g : grads)
      grad += std::fabs(g);
    initialCompCost = getCompCost(
        CM, llvm::SmallVector<llvm::Value *>(outputs.begin(), outputs.end()),
        subgraph.inputs);
    findErasableInstructions();
  }

  bool isArray() const { return roots.size() > 1; }

  void apply(size_t candidateIndex,
             std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                 &valueToNodeMap,
             std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
             bool print);
  double getCompCostDelta(CostModelState &CM, size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);
  BoundaryCastFreq boundaryCastFreq(
      const llvm::LoopInfo &LI,
      const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
          &valueToNodeMap) const;

  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::R);
  }
  size_t numCandidates() const override { return candidates.size(); }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  bool consumesFootprint() const override { return false; }
  // The searches of one output, and any two cores with the same outputs, are
  // admitted and priced alike by every later unit.
  bool sameRewriteTargetAs(const OptimizationUnit &earlier) const override;
  const llvm::SetVector<llvm::Instruction *> *footprint() const override;
  llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const override {
    return roots;
  }
  std::string candidateName(size_t i) const override {
    return candidates[i].expr;
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
  std::string keyHeader() const override {
    return "CO " + herbiePrecision + " " + expr;
  }
  llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const override {
    return dropped;
  }
  std::string selectionText(size_t i) const override {
    return expr + " -> " + candidates[i].expr;
  }
  void addReportSummary(llvm::json::Array &summary) override;
  void writeReportSummary(llvm::raw_ostream &OS) override;
  void addCuratedRewrites(llvm::json::Array &rewrites) override;

private:
  void findErasableInstructions();
};

// The parts of a cache key that say which Herbie searched: its binary and the
// platform it ranks candidates by.
// `F` is the site a mistake in the platform is reported at.
void addHerbieKeyParts(Session &S, const llvm::Function &F, CacheKey &key);

// Prints how many Herbie answers the compile ran fresh, read from the cache
// and replayed from -poseidon-import-herbie-answers.
void reportHerbieAnswers(Session &S);

// Collects Herbie's rewrites of `inputExprs`, cores of the site `F`, into
// `COs`, unpriced.
bool improveViaHerbie(Session &S, const std::vector<std::string> &inputExprs,
                      std::vector<RewriteUnit> &COs, int subgraphIdx,
                      const llvm::Function &F);

std::string getPrecondition(
    const Options &O, const llvm::SmallSet<std::string, 8> &args,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap,
    const std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

void setUnifiedAccuracyCost(
    const Options &O, RewriteUnit &CO,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

double getCompCost(
    CostModelState &CM, const std::string &expr, llvm::Module *M,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    const llvm::FastMathFlags &FMF,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr,
    const BoundaryCastFreq *Boundary = nullptr);

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
