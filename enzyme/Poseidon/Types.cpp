//=- Types.cpp - AST node implementations for Poseidon --------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST node classes for representing floating-point
// expressions in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <optional>

#include "Herbie.h"
#include "Model/CostModel.h"
#include "Pass/Session.h"
#include "Solvers.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"
#include "Techniques/Precision.h"
#include "Types.h"
#include "Utils.h"

using namespace llvm;

namespace poseidon {

void RewriteUnit::apply(
    size_t candidateIndex,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap, bool print) {
  if (candidateIndex >= candidates.size()) {
    if (print)
      llvm::errs() << "RewriteUnit::apply: candidateIndex " << candidateIndex
                   << " out of range (size " << candidates.size()
                   << "), skipping\n";
    return;
  }
  for (Value *root : roots) {
    auto *oldInst = dyn_cast<Instruction>(root);
    if (!oldInst || !oldInst->getParent()) {
      if (print)
        llvm::errs() << "RewriteUnit::apply: oldOutput "
                     << "is not in a basic block (already erased?), skipping\n";
      return;
    }
  }

  auto parsedNode =
      parseHerbieExpr(candidates[candidateIndex].expr, valueToNodeMap,
                      symbolToValueMap, nullptr, &subgraph->inputs);
  SmallVector<std::shared_ptr<FPNode>, 1> elements;
  if (isArray()) {
    if (parsedNode->op != "array" ||
        parsedNode->operands.size() != roots.size())
      report_fatal_error("Poseidon: an array rewrite does not have one element "
                         "per output of its subgraph");
    elements.append(parsedNode->operands.begin(), parsedNode->operands.end());
  } else {
    elements.push_back(parsedNode);
  }

  SmallVector<Value *, 4> newOutputs;
  if (isArray()) {
    DominatorTree DT(*cast<Instruction>(oldOutput)->getFunction());
    std::optional<IRBuilder<>> builder;
    newOutputs = emitSharedElements(
        elements,
        [&](size_t k) -> IRBuilder<> & {
          auto *oldInst = cast<Instruction>(roots[k]);
          builder.emplace(oldInst->getParent(),
                          ++BasicBlock::iterator(oldInst));
          builder->setFastMathFlags(oldInst->getFastMathFlags());
          return *builder;
        },
        nullptr, &DT);
  }

  for (auto [k, root, element] : llvm::enumerate(roots, elements)) {
    auto *oldInst = cast<Instruction>(root);
    IRBuilder<> builder(oldInst->getParent(), ++BasicBlock::iterator(oldInst));
    builder.setFastMathFlags(oldInst->getFastMathFlags());

    Value *newOutput = isArray() ? newOutputs[k] : element->getLLValue(builder);
    assert(newOutput && "Failed to get value from parsed node");
    if (isArray() && isa<Instruction>(newOutput))
      builder.SetInsertPoint(cast<Instruction>(newOutput)->getNextNode());

    if (newOutput->getType() != root->getType())
      newOutput =
          builder.CreateFPCast(newOutput, root->getType(), "herbie.fpcast");

    root->replaceAllUsesWith(newOutput);
    symbolToValueMap[valueToNodeMap[root]->symbol] = newOutput;
    valueToNodeMap[newOutput] = std::make_shared<FPLLValue>(
        newOutput, "__no", valueToNodeMap[root]->dtype);
  }

  for (auto *I : erasableInsts) {
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
    subgraph->operations.remove(I); // Avoid a second removal
    cast<FPLLValue>(valueToNodeMap[I].get())->value = nullptr;
  }

  subgraph->outputs_rewritten += roots.size();
}

bool RewriteUnit::sharesRoot(const RewriteUnit &other) const {
  for (Value *root : roots)
    if (llvm::is_contained(other.roots, root))
      return true;
  return false;
}

// Lower is better
double RewriteUnit::getCompCostDelta(CostModelState &CM,
                                     size_t candidateIndex) {
  double erasableCost = 0.0;

  for (auto *I : erasableInsts) {
    erasableCost += getInstructionCompCost(CM, I);
  }

  return (candidates[candidateIndex].CompCost - erasableCost) *
         (double)executions;
}

void RewriteUnit::findErasableInstructions() {
  SmallPtrSet<Value *, 8> visited;
  SmallPtrSet<Instruction *, 8> exprInsts;
  for (Value *root : roots)
    collectExprInsts(root, subgraph->inputs, exprInsts, visited);
  visited.clear();

  // Seeded in IR order, not in the order the pointers happened to hash: the
  // topological sort below is stable, so its input order decides the order the
  // report lists these instructions in and the order their costs are summed.
  SetVector<Instruction *> instsToProcess;
  for (Instruction &I :
       instructions(*cast<Instruction>(oldOutput)->getFunction()))
    if (exprInsts.contains(&I))
      instsToProcess.insert(&I);

  SmallVector<Instruction *, 8> instsToProcessSorted;
  reverseTopoSort(instsToProcess, instsToProcessSorted);

  erasableInsts.clear();
  for (Value *root : roots)
    erasableInsts.insert(cast<Instruction>(root));

  for (auto *I : instsToProcessSorted) {
    if (erasableInsts.contains(I))
      continue;

    bool usedOutside = false;
    for (auto user : I->users()) {
      if (auto *userI = dyn_cast<Instruction>(user)) {
        if (erasableInsts.contains(userI)) {
          continue;
        }
      }
      usedOutside = true;
      break;
    }

    if (!usedOutside) {
      erasableInsts.insert(I);
    }
  }
}

bool PrecisionUnit::CacheKey::operator==(const CacheKey &other) const {
  return candidateIndex == other.candidateIndex &&
         RewriteUnits == other.RewriteUnits;
}

std::size_t PrecisionUnit::CacheKeyHash::operator()(const CacheKey &key) const {
  std::size_t seed = std::hash<size_t>{}(key.candidateIndex);
  for (const auto *ao : key.RewriteUnits) {
    seed ^= std::hash<const RewriteUnit *>{}(ao) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

void PrecisionUnit::apply(Session &S, size_t candidateIndex) {
  if (candidateIndex >= candidates.size()) {
    llvm_unreachable("Invalid candidate index");
  }

  candidates[candidateIndex].apply(S, *subgraph);
}

// Lower is better
double PrecisionUnit::getCompCostDelta(size_t candidateIndex) {
  // TODO: adjust this based on erasured instructions
  return (candidates[candidateIndex].CompCost - initialCompCost) *
         (double)executions;
}

// Lower is better
double PrecisionUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

// Lower is better
double RewriteUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

double PrecisionUnit::getAdjustedCompCostDelta(Session &S,
                                               size_t candidateIndex,
                                               ArrayRef<SolutionStep> steps) {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }

  CacheKey key{candidateIndex, RewriteUnits};

  auto cacheIt = compCostDeltaCache.find(key);
  if (cacheIt != compCostDeltaCache.end()) {
    return cacheIt->second;
  }

  Subgraph newSubgraph = *this->subgraph;

  for (auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      const auto &CO = *co;
      if (CO.subgraph == subgraph) {
        newSubgraph.operations.remove_if(
            [&CO](Instruction *I) { return CO.erasableInsts.contains(I); });
        for (Value *root : CO.roots)
          newSubgraph.outputs.remove(cast<Instruction>(root));
      }
    }
  }

  if (newSubgraph.outputs.empty()) {
    compCostDeltaCache[key] = 0;
    return 0;
  }

  double initialCompCost = getCompCost(
      S.CostModel, {newSubgraph.outputs.begin(), newSubgraph.outputs.end()},
      newSubgraph.inputs, &newSubgraph.opExec, newSubgraph.execNormalizer);

  double candidateCompCost =
      getCompCost(S, newSubgraph, candidates[candidateIndex]);

  double adjustedCostDelta =
      (candidateCompCost - initialCompCost) * (double)executions;
  // Per-candidate pricing audit: with a one-point Pareto table these are the
  // ONLY record of why each candidate lost (all-dominated frontiers must be
  // auditable, not silent).
  if (S.Opts.Diagnostics.Print)
    llvm::errs() << "CS candidate '" << candidates[candidateIndex].desc
                 << "': initial=" << initialCompCost
                 << " cand=" << candidateCompCost << " exec=" << executions
                 << " Δcost=" << adjustedCostDelta
                 << " Δacc=" << getAccCostDelta(candidateIndex) << "\n";

  compCostDeltaCache[key] = adjustedCostDelta;
  return adjustedCostDelta;
}

double PrecisionUnit::getAdjustedAccCostDelta(
    size_t candidateIndex, ArrayRef<SolutionStep> steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }

  CacheKey key{candidateIndex, RewriteUnits};

  auto cacheIt = accCostDeltaCache.find(key);
  if (cacheIt != accCostDeltaCache.end()) {
    return cacheIt->second;
  }

  double totalCandidateAccCost = 0.0;
  double totalInitialAccCost = 0.0;

  SmallPtrSet<FPNode *, 8> stepNodes;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      const auto &CO = *co;
      if (CO.subgraph == subgraph) {
        for (Value *root : CO.roots) {
          auto it = valueToNodeMap.find(root);
          assert(it != valueToNodeMap.end() && it->second);
          stepNodes.insert(it->second.get());
        }
      }
    }
  }

  for (auto &[node, cost] : perOutputInitialAccCost) {
    if (!stepNodes.count(node)) {
      totalInitialAccCost += cost;
    }
  }

  for (auto &[node, cost] : candidates[candidateIndex].perOutputAccCost) {
    if (!stepNodes.count(node)) {
      totalCandidateAccCost += cost;
    }
  }

  double adjustedAccCostDelta = totalCandidateAccCost - totalInitialAccCost;

  accCostDeltaCache[key] = adjustedAccCostDelta;
  return adjustedAccCostDelta;
}

double RewriteUnit::costDelta(size_t i, ArrayRef<SolutionStep>) {
  return getCompCostDelta(session().CostModel, i);
}

double RewriteUnit::errorDelta(size_t i, ArrayRef<SolutionStep>) {
  return getAccCostDelta(i);
}

bool RewriteUnit::sameRewriteTargetAs(const OptimizationUnit &earlier) const {
  const auto *other = dyn_cast<RewriteUnit>(&earlier);
  return other && other->subgraph == subgraph &&
         other->roots.size() == roots.size() &&
         llvm::all_of(other->roots,
                      [&](Value *r) { return llvm::is_contained(roots, r); });
}

const SetVector<Instruction *> *RewriteUnit::footprint() const {
  return subgraph ? &subgraph->operations : nullptr;
}

double PrecisionUnit::costDelta(size_t i, ArrayRef<SolutionStep> prior) {
  return getAdjustedCompCostDelta(session(), i, prior);
}

double PrecisionUnit::errorDelta(size_t i, ArrayRef<SolutionStep> prior) {
  return getAdjustedAccCostDelta(i, prior, site().valueToNodeMap,
                                 site().symbolToValueMap);
}

bool PrecisionUnit::pricedAgainst(const OptimizationUnit &earlier) const {
  const auto *co = dyn_cast<RewriteUnit>(&earlier);
  return co && co->subgraph == subgraph;
}

const SetVector<Instruction *> *PrecisionUnit::footprint() const {
  return subgraph ? &subgraph->operations : nullptr;
}

std::string RewriteUnit::choiceText(size_t i) const {
  const RewriteCandidate &cand = candidates[i];
  return expr + " --(" + std::to_string(cand.ordinal) + ")-> " + cand.expr;
}

json::Object RewriteUnit::stepJSON(size_t i) {
  json::Object stepObj;
  stepObj["type"] = "rewrite";
  stepObj["original_expr"] = expr;
  auto &cand = candidates[i];
  stepObj["rewritten_expr"] = cand.expr;
  stepObj["herbie_cost"] = jsonFloat(cand.herbieCost);
  stepObj["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
  stepObj["initial_herbie_cost"] = jsonFloat(initialHerbieCost);
  stepObj["initial_herbie_accuracy"] = jsonFloat(initialHerbieAccuracy);
  stepObj["computation_cost_delta"] = getCompCostDelta(session().CostModel, i);
  stepObj["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(i));
  stepObj["gradient"] = jsonFloat(grad);
  stepObj["executions"] = static_cast<int64_t>(executions);

  SmallVector<Instruction *, 8> insts;
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    insts.push_back(I);
  for (auto *I : erasableInsts)
    insts.push_back(I);
  stepObj["source_locations"] = getSourceLocationsJSON(insts);

  json::Array affectedIR;
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    affectedIR.push_back(getValueStr(I));
  for (auto *I : erasableInsts) {
    if (I != oldOutput)
      affectedIR.push_back(getValueStr(I));
  }
  stepObj["affected_instructions"] = std::move(affectedIR);
  return stepObj;
}

void RewriteUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  auto &cand = candidates[i];
  OS << pad << "[Rewrite] " << expr << "  -->  " << cand.expr << "\n";
  if (!std::isnan(initialHerbieAccuracy) && !std::isnan(cand.herbieAccuracy))
    OS << pad << "  Herbie accuracy: " << initialHerbieAccuracy << " -> "
       << cand.herbieAccuracy << " bits\n";
  OS << pad << "  Gradient: " << grad << ", Executions: " << executions << "\n";

  SmallSet<std::string, 4> seen;
  auto printLoc = [&](Instruction *I) {
    auto loc = getSourceLocationStr(I);
    if (!loc.empty() && seen.insert(loc).second)
      OS << pad << "  Source: " << loc << "\n";
  };
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    printLoc(I);
  for (auto *I : erasableInsts)
    printLoc(I);

  OS << pad << "  Affected IR:\n";
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    OS << pad << "    " << *I << "\n";
  for (auto *I : erasableInsts) {
    if (I != oldOutput)
      OS << pad << "    " << *I << "\n";
  }
}

void RewriteUnit::materialize(size_t i) {
  apply(i, site().valueToNodeMap, site().symbolToValueMap,
        session().Opts.Diagnostics.Print);
}

std::string PrecisionUnit::choiceText(size_t i) const {
  const PTCandidate &cand = candidates[i];
  return "CS: " + cand.desc + " (#" + std::to_string(cand.ordinal) + ")";
}

json::Object PrecisionUnit::stepJSON(size_t i) {
  auto &cand = candidates[i];
  json::Object stepObj;
  stepObj["type"] = "precision_change";
  stepObj["description"] = cand.desc;
  stepObj["candidate_index"] = static_cast<int64_t>(cand.ordinal);

  json::Array changes;
  for (const auto &change : cand.changes) {
    json::Object changeObj;
    changeObj["from"] = getPrecisionChangeTypeString(change.oldType);
    changeObj["to"] = getPrecisionChangeTypeString(change.newType);
    changeObj["num_operations"] = static_cast<int64_t>(change.nodes.size());

    SmallVector<Instruction *, 8> insts;
    json::Array nodeIR;
    for (auto *node : change.nodes) {
      if (auto *I = dyn_cast<Instruction>(node->value)) {
        insts.push_back(I);
        nodeIR.push_back(getValueStr(I));
      }
    }
    changeObj["source_locations"] = getSourceLocationsJSON(insts);
    changeObj["affected_instructions"] = std::move(nodeIR);
    changes.push_back(std::move(changeObj));
  }
  stepObj["changes"] = std::move(changes);
  return stepObj;
}

void PrecisionUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  auto &cand = candidates[i];
  OS << pad << "[Precision] " << cand.desc << " (#" << cand.ordinal << ")\n";
  for (const auto &change : cand.changes) {
    OS << pad << "  " << getPrecisionChangeTypeString(change.oldType) << " -> "
       << getPrecisionChangeTypeString(change.newType) << " for "
       << change.nodes.size() << " operations\n";
    SmallSet<std::string, 4> seen;
    for (auto *node : change.nodes) {
      if (auto *I = dyn_cast<Instruction>(node->value)) {
        auto loc = getSourceLocationStr(I);
        if (!loc.empty() && seen.insert(loc).second)
          OS << pad << "    Source: " << loc << "\n";
      }
    }
  }
}

void PrecisionUnit::materialize(size_t i) { apply(session(), i); }

void RewriteUnit::price() {
  if (!herbieAnswered)
    return;
  FunctionFPState &st = site();
  Module *M = st.F->getParent();
  const FastMathFlags FMF = cast<Instruction>(oldOutput)->getFastMathFlags();
  for (RewriteCandidate &c : candidates)
    c.CompCost = getCompCost(session().CostModel, c.expr, M, st.valueToNodeMap,
                             st.symbolToValueMap, FMF, &subgraph->inputs);
  setUnifiedAccuracyCost(session().Opts, *this, st.valueToNodeMap,
                         st.symbolToValueMap);
}

void PrecisionUnit::price() {
  for (PTCandidate &c : candidates)
    c.CompCost = getCompCost(session(), *subgraph, c);
  setUnifiedAccuracyCost(session().Opts, *this, site().valueToNodeMap,
                         site().symbolToValueMap);
}

void RewriteUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &cand : candidates)
    cand.accuracyCost *= factor;
}

void PrecisionUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &[node, cost] : perOutputInitialAccCost)
    cost *= factor;
  for (auto &cand : candidates) {
    cand.accuracyCost *= factor;
    for (auto &[node, cost] : cand.perOutputAccCost)
      cost *= factor;
  }
}

void RewriteUnit::printPriced(raw_ostream &OS) {
  OS << "\n################################\n";
  OS << "Initial AccuracyCost: " << initialAccCost << "\n";
  OS << "Initial ComputationCost: " << initialCompCost << "\n";
  OS << "Initial HerbieCost: " << initialHerbieCost << "\n";
  OS << "Initial HerbieAccuracy: " << initialHerbieAccuracy << "\n";
  OS << "Initial Expression: " << expr << "\n";
  OS << "Grad: " << grad << "\n\n";
  OS << "Candidates:\n";
  OS << "Δ AccCost\t\tΔ "
        "CompCost\t\tHerbieCost\t\tAccuracy\t\tExpression\n";
  OS << "--------------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t";
    // A compile that reads its DP table from the cache prices no candidate.
    if (candidate.CompCost == std::numeric_limits<double>::max())
      OS << "unpriced";
    else
      OS << getCompCostDelta(session().CostModel, i);
    OS << "\t\t" << candidate.herbieCost << "\t\t" << candidate.herbieAccuracy
       << "\t\t" << candidate.expr << "\n";
  }
  OS << "################################\n\n";
}

void RewriteUnit::addReportSummary(json::Array &summary) {
  json::Object co;
  co["original_expr"] = expr;
  co["num_candidates"] = static_cast<int64_t>(candidates.size());
  co["gradient"] = jsonFloat(grad);
  co["executions"] = static_cast<int64_t>(executions);
  co["initial_accuracy_cost"] = jsonFloat(initialAccCost);
  co["initial_computation_cost"] = initialCompCost;
  if (auto *I = dyn_cast<Instruction>(oldOutput)) {
    auto loc = getSourceLocationJSON(I);
    if (!loc.empty())
      co["source_location"] = std::move(loc);
  }
  summary.push_back(std::move(co));
}

void RewriteUnit::writeReportSummary(raw_ostream &textOut) {
  textOut << "Expression #" << indexInKind() << ": " << expr << "\n";
  textOut << "  Gradient: " << grad << ", Executions: " << executions << "\n";
  if (auto *I = dyn_cast<Instruction>(oldOutput)) {
    auto loc = getSourceLocationStr(I);
    if (!loc.empty())
      textOut << "  Source: " << loc << "\n";
  }
  textOut << "  Candidates: " << candidates.size() << "\n";
  for (size_t j = 0; j < candidates.size(); ++j) {
    auto &cand = candidates[j];
    textOut << "    [" << cand.ordinal << "] " << cand.expr;
    if (!std::isnan(cand.herbieAccuracy))
      textOut << "  (accuracy: " << cand.herbieAccuracy << " bits)";
    textOut << "\n";
  }
}

void RewriteUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &cand = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    auto compDelta = getCompCostDelta(session().CostModel, ci);
    double compDeltaVal = compDelta;

    if (std::isnan(accDelta))
      continue;

    // comp < 0 means faster, acc < 0 means more accurate
    std::string category;
    double efficiency = 0.0;
    if (compDeltaVal <= 0 && accDelta <= 0) {
      category = "free_win";
      efficiency = std::abs(accDelta) + std::abs(compDeltaVal);
    } else if (compDeltaVal <= 0 && accDelta > 0) {
      category = "speed_for_accuracy";
      efficiency = (accDelta > 1e-30) ? std::abs(compDeltaVal) / accDelta
                                      : std::abs(compDeltaVal);
    } else if (compDeltaVal > 0 && accDelta <= 0) {
      category = "accuracy_for_speed";
      efficiency = (compDeltaVal > 0) ? std::abs(accDelta) / compDeltaVal
                                      : std::abs(accDelta);
    } else {
      continue;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["original_expr"] = expr;
    rw["rewritten_expr"] = cand.expr;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDeltaVal;
    rw["accuracy_cost_delta"] = jsonFloat(accDelta);
    rw["gradient"] = jsonFloat(grad);
    rw["executions"] = static_cast<int64_t>(executions);
    rw["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
    rw["initial_herbie_accuracy"] = jsonFloat(initialHerbieAccuracy);

    if (auto *I = dyn_cast<Instruction>(oldOutput)) {
      auto loc = getSourceLocationJSON(I);
      if (!loc.empty())
        rw["source_location"] = std::move(loc);
    }

    rewrites.push_back(std::move(rw));
  }
}

// Top-N per-candidate deltas keep an all-dominated frontier auditable (the
// table then collapses to the no-op point).
static constexpr size_t kMaxCandRows = 40;

// The candidates' positions by cost change, cheapest first; equal changes keep
// their positions' order.
static SmallVector<size_t, 40> byCostDelta(PrecisionUnit &CS) {
  SmallVector<size_t, 40> order(CS.candidates.size());
  for (size_t ci = 0; ci < order.size(); ++ci)
    order[ci] = ci;
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    return CS.getCompCostDelta(a) < CS.getCompCostDelta(b);
  });
  return order;
}

void PrecisionUnit::addReportSummary(json::Array &summary) {
  json::Object cs;
  cs["num_candidates"] = static_cast<int64_t>(candidates.size());
  cs["initial_accuracy_cost"] = jsonFloat(initialAccCost);
  cs["initial_computation_cost"] = initialCompCost;
  SmallVector<size_t, 40> order = byCostDelta(*this);
  json::Array cands;
  for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
    size_t ci = order[k];
    json::Object c;
    c["id"] = candidateId(ci);
    c["description"] = candidates[ci].desc;
    c["computation_cost_delta"] = getCompCostDelta(ci);
    c["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(ci));
    cands.push_back(std::move(c));
  }
  cs["candidates"] = std::move(cands);
  summary.push_back(std::move(cs));
}

void PrecisionUnit::writeReportSummary(raw_ostream &textOut) {
  textOut << "Subgraph #" << indexInKind() << ": " << candidates.size()
          << " candidates (Δcost, Δacc; sorted by Δcost)\n";
  SmallVector<size_t, 40> order = byCostDelta(*this);
  for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
    size_t ci = order[k];
    textOut << "  [" << candidateId(ci) << "] " << candidates[ci].desc
            << ": Δcost=" << getCompCostDelta(ci)
            << " Δacc=" << format("%.6e", getAccCostDelta(ci)) << "\n";
  }
}

void PrecisionUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &pt = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    auto compDelta = getCompCostDelta(ci);
    double compDeltaVal = compDelta;

    if (std::isnan(accDelta))
      continue;

    std::string category;
    double efficiency = 0.0;
    if (compDeltaVal <= 0 && accDelta <= 0) {
      category = "free_win";
      efficiency = std::abs(accDelta) + std::abs(compDeltaVal);
    } else if (compDeltaVal <= 0 && accDelta > 0) {
      category = "speed_for_accuracy";
      efficiency = (accDelta > 1e-30) ? std::abs(compDeltaVal) / accDelta
                                      : std::abs(compDeltaVal);
    } else if (compDeltaVal > 0 && accDelta <= 0) {
      category = "accuracy_for_speed";
      efficiency = (compDeltaVal > 0) ? std::abs(accDelta) / compDeltaVal
                                      : std::abs(accDelta);
    } else {
      // Emitted even though dominated: an all-dominated frontier must
      // leave a record of why each candidate lost.
      category = "dominated";
      efficiency = 0.0;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["type"] = "precision_change";
    rw["description"] = pt.desc;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDeltaVal;
    rw["accuracy_cost_delta"] = jsonFloat(accDelta);

    SmallVector<Instruction *, 16> ptInsts;
    for (const auto &change : pt.changes) {
      for (auto *node : change.nodes) {
        if (auto *I = dyn_cast<Instruction>(node->value))
          ptInsts.push_back(I);
      }
    }
    rw["source_locations"] = getSourceLocationsJSON(ptInsts);
    json::Array affectedIR;
    for (auto *I : ptInsts)
      affectedIR.push_back(getValueStr(I));
    rw["affected_instructions"] = std::move(affectedIR);

    rewrites.push_back(std::move(rw));
  }
}

void PrecisionUnit::printPriced(raw_ostream &OS) {
  OS << "\n################################\n";
  OS << "Initial AccuracyCost: " << initialAccCost << "\n";
  OS << "Initial ComputationCost: " << initialCompCost << "\n";
  OS << "Candidates:\n";
  OS << "Δ AccCost\t\tΔ CompCost\t\tDescription\n"
     << "---------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t";
    if (candidate.CompCost == std::numeric_limits<double>::max())
      OS << "unpriced";
    else
      OS << getCompCostDelta(i);
    OS << "\t\t" << candidate.desc << "\n";
  }
  OS << "################################\n\n";
}

} // namespace poseidon
