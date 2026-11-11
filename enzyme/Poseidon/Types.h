//=- Types.h - AST node declarations for Poseidon -------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AST node classes for representing floating-point
// expressions in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TYPES_H
#define POSEIDON_TYPES_H

#include "CostModel.h"
#include "Model/FPNode.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "Utils.h"

#include "Techniques/Precision.h"
#include "Techniques/Unit.h"

namespace poseidon {

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
  bool sharesRoot(const RewriteUnit &other) const;

  void apply(size_t candidateIndex,
             std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                 &valueToNodeMap,
             std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
             bool print);
  double getCompCostDelta(CostModelState &CM, size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);

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

class PrecisionUnit : public SiteUnit {
public:
  Subgraph *subgraph;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  uint64_t executions = 0;
  llvm::MapVector<FPNode *, double> perOutputInitialAccCost;
  llvm::SmallVector<PTCandidate, 8> candidates;
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;

  using RewriteUnitSet = std::set<RewriteUnit *>;
  struct CacheKey {
    size_t candidateIndex;
    RewriteUnitSet RewriteUnits;
    bool operator==(const CacheKey &other) const;
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey &key) const;
  };

  std::unordered_map<CacheKey, double, CacheKeyHash> compCostDeltaCache;
  std::unordered_map<CacheKey, double, CacheKeyHash> accCostDeltaCache;

  explicit PrecisionUnit(CostModelState &CM, Subgraph &subgraph)
      : SiteUnit(UnitKind::PT), subgraph(&subgraph) {
    initialCompCost =
        getCompCost(CM, {subgraph.outputs.begin(), subgraph.outputs.end()},
                    subgraph.inputs, &subgraph.opExec, subgraph.execNormalizer);
  }

  void apply(Session &S, size_t candidateIndex);
  double getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);
  double getAdjustedCompCostDelta(Session &S, size_t candidateIndex,
                                  llvm::ArrayRef<SolutionStep> steps);
  double getAdjustedAccCostDelta(
      size_t candidateIndex, llvm::ArrayRef<SolutionStep> steps,
      std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
          &valueToNodeMap,
      std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);

  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::PT);
  }
  size_t numCandidates() const override { return candidates.size(); }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep> prior) override;
  // Priced over the outputs the algebraic units of its subgraph leave.
  bool pricedAlone() const override { return false; }
  size_t memoizedPricings() const override { return compCostDeltaCache.size(); }
  bool consumesFootprint() const override { return false; }
  bool pricedAgainst(const OptimizationUnit &earlier) const override;
  const llvm::SetVector<llvm::Instruction *> *footprint() const override;
  std::string candidateName(size_t i) const override {
    return candidates[i].desc;
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
  std::string keyHeader() const override { return "CS"; }
  llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const override {
    return dropped;
  }
  std::string selectionText(size_t i) const override {
    return candidates[i].desc;
  }
  void addReportSummary(llvm::json::Array &summary) override;
  void writeReportSummary(llvm::raw_ostream &OS) override;
  void addCuratedRewrites(llvm::json::Array &rewrites) override;
};

} // namespace poseidon
#endif // POSEIDON_TYPES_H