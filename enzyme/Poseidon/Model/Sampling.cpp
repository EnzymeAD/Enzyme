//=- Sampling.cpp - input sampling for the accuracy model -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sampling.h"
#include "Core/Options.h"
#include "Core/Parallel.h"
#include "Core/UniformReal.h"
#include "Evaluators.h"
#include "FPCore.h"
#include "FPNode.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using namespace llvm;

namespace poseidon {

static constexpr double kCancellationThreshold = 1e-12;
static constexpr double kCancellationFraction = 0.5;
static constexpr size_t kJointDrawFactor = 64;
static constexpr size_t kMinJointSamples = 32;

double sampleError(double goldVal, double result) {
  return std::fabs(goldVal - result);
}

CancellationPlan
buildCancellationPlan(const Options &O, const Subgraph &subgraph,
                      const std::unordered_map<Value *, std::shared_ptr<FPNode>>
                          &valueToNodeMap) {
  CancellationPlan plan;
  const auto &inputs = subgraph.inputs;

  // Smallest magnitude the profiler observed for a node. A range straddling
  // zero can be arbitrarily small (treated as 0); otherwise the closer
  // endpoint.
  auto minMagnitude = [](FPNode *n) -> double {
    double lb = n->getLowerBound(), ub = n->getUpperBound();
    if (!std::isfinite(lb) || !std::isfinite(ub) || lb > ub)
      return std::numeric_limits<double>::infinity();
    if (lb <= 0.0 && ub >= 0.0)
      return 0.0;
    return std::min(std::fabs(lb), std::fabs(ub));
  };

  bool nearSingular = false;
  for (Instruction *I : subgraph.operations) {
    auto it = valueToNodeMap.find(I);
    if (it == valueToNodeMap.end())
      continue;
    FPNode *node = it->second.get();

    if (node->op == "-" && node->operands.size() == 2) {
      // One entry per unordered pair.
      auto *a = dyn_cast<FPLLValue>(node->operands[0].get());
      auto *b = dyn_cast<FPLLValue>(node->operands[1].get());
      if (a && b && a->value != b->value && inputs.count(a->value) &&
          inputs.count(b->value)) {
        bool seen = false;
        for (const auto &pr : plan.pairs)
          if ((pr.first == a->value && pr.second == b->value) ||
              (pr.first == b->value && pr.second == a->value)) {
            seen = true;
            break;
          }
        if (!seen)
          plan.pairs.push_back({a->value, b->value});
      }
    } else if (node->op == "sqrt" && node->operands.size() >= 1) {
      if (minMagnitude(node->operands[0].get()) < kCancellationThreshold)
        nearSingular = true;
    } else if (node->op == "/" && node->operands.size() == 2) {
      if (minMagnitude(node->operands[1].get()) < kCancellationThreshold)
        nearSingular = true;
    }
  }

  plan.active = nearSingular && !plan.pairs.empty();
  if (O.Diagnostics.Print && plan.active)
    llvm::errs() << "[poseidon] cancellation-aware sampling active: "
                 << plan.pairs.size() << " leaf-subtraction pair(s), "
                 << "near-singular denominator/radicand profiled\n";
  return plan;
}

static void collectLeaves(const FPNode *node,
                          SmallPtrSetImpl<const FPNode *> &seen,
                          SetVector<const FPLLValue *> &leaves) {
  if (isa<FPConst>(node) || !seen.insert(node).second)
    return;
  if (auto *LL = dyn_cast<FPLLValue>(node); LL && node->operands.empty()) {
    leaves.insert(LL);
    return;
  }
  for (const auto &operand : node->operands)
    collectLeaves(operand.get(), seen, leaves);
}

// A subgraph's inputs are often functions of the same few values (every
// intermediate a function of a temperature T, say), so drawing each in its
// own profiled range pairs values that never occur together: terms
// that cancel in the data overflow an exp, and an input and its log disagree.
// Each sample draws the leaves the inputs are computed from (loads, arguments,
// globals) in their profiled ranges, evaluates each input from them as the
// program does, and is kept if every input lies in its own profiled range, the
// box Herbie searched.
void getSampledPoints(
    const Options &O, ArrayRef<Value *> inputs,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints,
    const CancellationPlan *plan, size_t count, unsigned seedOffset) {
  const size_t numSamples = count ? count : O.Model.NumSamples;
  std::minstd_rand0 gen(O.Model.RandomSeed + seedOffset);
  sampledPoints.clear();

  MapVector<Value *, SmallVector<double, 2>> box;
  SmallPtrSet<const FPNode *, 32> seen;
  SetVector<const FPLLValue *> leaves;
  SmallVector<FPNode *, 8> computed;
  for (const auto input : inputs) {
    FPNode *node = valueToNodeMap.at(input).get();

    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
      // [INF, -INF] is the initial range: the profiler never observed this
      // input. Refuse to sample rather than invent a bound.
      llvm::errs() << "Poseidon: input to sampled subgraph has unprofiled "
                      "bounds ["
                   << lower << ", " << upper
                   << "]; refusing to sample it. This subgraph cannot be "
                      "priced and will keep its original body. Input value: "
                   << *input << "\n";
      return;
    }

    box.insert({input, {lower, upper}});
    collectLeaves(node, seen, leaves);
    if (!node->operands.empty())
      computed.push_back(node);
  }

  MapVector<Value *, SmallVector<double, 2>> leafBox;
  for (const FPLLValue *leaf : leaves) {
    double lower = leaf->getLowerBound(), upper = leaf->getUpperBound();
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
      llvm::errs() << "Poseidon: value a subgraph input is computed from has "
                      "unprofiled bounds ["
                   << lower << ", " << upper
                   << "]; refusing to sample the subgraph. It cannot be priced "
                      "and will keep its original body. Value: "
                   << *leaf->value << "\n";
      return;
    }
    leafBox.insert({leaf->value, {lower, upper}});
  }

  // Two leaves the subgraph subtracts, drawn independently, never come close,
  // so in a share of the samples each such pair is moved to a common
  // near-coincident value, with one log-uniform separation scale shared across
  // all pairs so every subtracted difference is small together.
  SmallVector<std::pair<Value *, Value *>, 4> pairs;
  if (plan && plan->active)
    for (const auto &pr : plan->pairs)
      if (leafBox.count(pr.first) && leafBox.count(pr.second))
        pairs.push_back(pr);
  const size_t numCoincidence =
      pairs.empty() ? 0
                    : static_cast<size_t>(kCancellationFraction *
                                          static_cast<double>(numSamples));
  double maxRange = 0.0;
  for (const auto &pr : pairs) {
    const auto &range = leafBox.find(pr.first)->second;
    double r = range[1] - range[0];
    if (std::isfinite(r) && r > maxRange)
      maxRange = r;
  }
  if (!(maxRange > 0.0))
    maxRange = 1.0;

  const size_t budget = kJointDrawFactor * numSamples;
  size_t drawn = 0;
  auto drawLeaves = [&](MapVector<Value *, double> &leafPoint) {
    for (const auto &entry : leafBox)
      leafPoint.insert(
          {entry.first, uniformReal(gen, entry.second[0], entry.second[1])});
  };
  // Whether the inputs computed from `leafPoint` lie in the box, then
  // `point`.
  auto evaluate = [&](const MapVector<Value *, double> &leafPoint,
                      MapVector<Value *, double> &point,
                      SmallVectorImpl<double> &results) {
    if (!computed.empty())
      getFPValues(computed, leafPoint, results);
    size_t next = 0;
    for (const auto &entry : box) {
      FPNode *node = valueToNodeMap.at(entry.first).get();
      double v;
      if (!node->operands.empty())
        v = results[next++];
      else if (isa<FPConst>(node))
        v = node->getLowerBound();
      else
        v = leafPoint.lookup(entry.first);
      if (!(v >= entry.second[0] && v <= entry.second[1]))
        return false;
      point.insert({entry.first, v});
    }
    return true;
  };

  // The coincidence stratum: a draw takes more numbers from `gen` while fewer
  // than numCoincidence points are kept.
  SmallVector<double, 8> results;
  while (sampledPoints.size() < numCoincidence &&
         sampledPoints.size() < numSamples && drawn < budget) {
    ++drawn;
    MapVector<Value *, double> leafPoint;
    drawLeaves(leafPoint);

    const double scaleLo = 1e-18;
    double logScale = uniformReal(gen, std::log(scaleLo), std::log(maxRange));
    double scale = std::exp(logScale);
    for (const auto &pr : pairs) {
      double anchor = leafPoint[pr.first];
      double jitter = uniformReal(gen, 0.5, 1.5);
      double sign = uniformReal(gen, 0.0, 1.0) < 0.5 ? -1.0 : 1.0;
      // Clamped into the leaf's own profiled range so no out-of-distribution
      // value is fabricated.
      const auto &range = leafBox.find(pr.second)->second;
      leafPoint[pr.second] =
          std::clamp(anchor + sign * scale * jitter, range[0], range[1]);
    }

    MapVector<Value *, double> point;
    if (evaluate(leafPoint, point, results))
      sampledPoints.push_back(std::move(point));
  }

  // Past it every draw takes the same numbers from `gen`, so a block of draws
  // is drawn in order ahead of its evaluations, which run concurrently, and
  // kept in order up to the first numSamples in the box.
  const size_t numLeaves = leafBox.size();
  SmallVector<double, 0> leafValues;
  SmallVector<MapVector<Value *, double>, 0> points;
  SmallVector<char, 0> inBox;
  while (sampledPoints.size() < numSamples && drawn < budget) {
    // As many draws as the points still missing take at the share kept so
    // far, and a margin, so that few are evaluated past the last one kept.
    const double missing = numSamples - sampledPoints.size();
    const double kept = (sampledPoints.size() + 1.0) / (drawn + 1.0);
    const size_t n = std::min<size_t>(
        budget - drawn, std::max(64.0, std::ceil(1.25 * missing / kept)));
    leafValues.resize(n * numLeaves);
    for (size_t i = 0; i < n; ++i)
      for (auto [l, entry] : enumerate(leafBox))
        leafValues[i * numLeaves + l] =
            uniformReal(gen, entry.second[0], entry.second[1]);
    points.assign(n, {});
    inBox.assign(n, 0);
    forEachBlock(O.Model.Threads, n, 64, [&](size_t begin, size_t end) {
      SmallVector<double, 8> blockResults;
      for (size_t i = begin; i < end; ++i) {
        MapVector<Value *, double> leafPoint;
        for (auto [l, entry] : enumerate(leafBox))
          leafPoint.insert({entry.first, leafValues[i * numLeaves + l]});
        inBox[i] = evaluate(leafPoint, points[i], blockResults);
      }
    });
    for (size_t i = 0; i < n && sampledPoints.size() < numSamples; ++i) {
      ++drawn;
      if (inBox[i])
        sampledPoints.push_back(std::move(points[i]));
    }
  }

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] joint sampling: " << leaves.size()
                 << " leaves, " << sampledPoints.size() << " of " << drawn
                 << " draws inside the profiled input box\n";
  if (sampledPoints.size() < kMinJointSamples) {
    llvm::errs() << "Poseidon: joint sampling kept " << sampledPoints.size()
                 << " of " << drawn
                 << " draws inside the subgraph's profiled input box; it "
                    "cannot be priced and will keep its original body.\n";
    sampledPoints.clear();
  }
}

struct SubgraphSamples::RoundReference {
  Reference values;
  SmallVector<char, 0> computed;
};

SubgraphSamples::SubgraphSamples(
    const Options &O, const Subgraph &subgraph,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap)
    : O(O), subgraph(subgraph), valueToNodeMap(valueToNodeMap) {}

SubgraphSamples::~SubgraphSamples() = default;

ArrayRef<SubgraphSamples::Point> SubgraphSamples::points() {
  if (!sampled) {
    sampled = true;
    CancellationPlan plan = buildCancellationPlan(O, subgraph, valueToNodeMap);
    getSampledPoints(O, subgraph.inputs.getArrayRef(), valueToNodeMap, base,
                     &plan);
  }
  return base;
}

// The exact and the as-written value of `node` at `point`, and the error of
// the latter, into entry `i` of `R`.
static void evaluateReference(FPNode *node, const SubgraphSamples::Point &point,
                              SubgraphSamples::Reference &R, size_t i) {
  SmallVector<double, 1> results;
  getMPFRValues({node}, point, results, 53);
  R.gold[i] = results[0];
  getFPValues({node}, point, results);
  R.value[i] = results[0];
  R.error[i] = sampleError(R.gold[i], R.value[i]);
}

const SubgraphSamples::Reference &
SubgraphSamples::reference(const FPNode *node) {
  auto [it, fresh] = references.try_emplace(node);
  Reference &R = it->second;
  if (!fresh)
    return R;
  ArrayRef<Point> at = points();
  R.gold.resize(at.size());
  R.value.resize(at.size());
  R.error.resize(at.size());
  forEachBlock(O.Model.Threads, at.size(), 8, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i)
      evaluateReference(const_cast<FPNode *>(node), at[i], R, i);
  });
  return R;
}

ArrayRef<SubgraphSamples::Point> SubgraphSamples::round(unsigned round) {
  assert(round > 0 && "round 0 is points()");
  while (rounds.size() < round) {
    rounds.emplace_back();
    getSampledPoints(O, subgraph.inputs.getArrayRef(), valueToNodeMap,
                     rounds.back(), nullptr, O.Model.NumSamples, rounds.size());
  }
  return rounds[round - 1];
}

const SubgraphSamples::Reference &
SubgraphSamples::roundReference(const FPNode *node, unsigned round,
                                ArrayRef<size_t> which) {
  ArrayRef<Point> at = this->round(round);
  auto &perRound = roundReferences[node];
  while (perRound.size() < round)
    perRound.push_back(std::make_unique<RoundReference>());
  RoundReference &RR = *perRound[round - 1];
  if (RR.computed.empty()) {
    RR.values.gold.resize(at.size());
    RR.values.value.resize(at.size());
    RR.values.error.resize(at.size());
    RR.computed.resize(at.size(), 0);
  }
  SmallVector<size_t, 0> todo;
  for (size_t p : which)
    if (!RR.computed[p]) {
      RR.computed[p] = 1;
      todo.push_back(p);
    }
  forEachBlock(O.Model.Threads, todo.size(), 8, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i)
      evaluateReference(const_cast<FPNode *>(node), at[todo[i]], RR.values,
                        todo[i]);
  });
  return RR.values;
}

SubgraphSamples &
samplesOf(const Options &O, Subgraph &subgraph,
          const std::unordered_map<Value *, std::shared_ptr<FPNode>>
              &valueToNodeMap) {
  if (!subgraph.samples)
    subgraph.samples =
        std::make_shared<SubgraphSamples>(O, subgraph, valueToNodeMap);
  return *subgraph.samples;
}

} // namespace poseidon
