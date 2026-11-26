//===- Sampling.cpp - input sampling for the accuracy model ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sampling.h"
#include "Evaluators.h"
#include "FPCore.h"
#include "FPNode.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Parallel.h"
#include "Support/UniformReal.h"
#include "Support/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <random>

using namespace llvm;

namespace poseidon {

static constexpr double kCancellationThreshold = 1e-12;
static constexpr double kCancellationFraction = 0.5;
static constexpr size_t kJointDrawFactor = 64;
static constexpr size_t kMinJointSamples = 32;
// Draws and exact references are evaluated concurrently in tasks of this many.
static constexpr size_t kDrawsPerTask = 64;
static constexpr size_t kReferencesPerTask = 8;
// The independent draws of one block exceed the points still missing at the
// share kept so far by this factor.
static constexpr double kDrawMargin = 1.25;

double sampleError(double goldVal, double result) {
  return std::fabs(goldVal - result);
}

namespace {
// Cancellation-aware sampling plan: `active` iff the subgraph divides by or
// takes the sqrt of a quantity profiled near zero; `pairs` are the leaf
// subtraction operands whose near-coincidence produces it (independent
// marginal sampling never reproduces that configuration).
struct CancellationPlan {
  bool active = false;
  SmallVector<std::pair<Value *, Value *>, 4> pairs;
};
} // namespace

static CancellationPlan
buildCancellationPlan(const Options &O, const Subgraph &subgraph,
                      const DenseMap<Value *, FPNode *> &valueToNodeMap) {
  CancellationPlan plan;
  const SetVector<Value *> &inputs = subgraph.inputs;

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
    FPNode *node = it->second;

    if (node->is(fpcore::Sub) && node->operands.size() == 2) {
      // One entry per unordered pair.
      auto *a = dyn_cast<FPLLValue>(node->operands[0]);
      auto *b = dyn_cast<FPLLValue>(node->operands[1]);
      if (a && b && a->value != b->value && inputs.count(a->value) &&
          inputs.count(b->value)) {
        bool seen = any_of(plan.pairs, [&](const auto &pr) {
          return (pr.first == a->value && pr.second == b->value) ||
                 (pr.first == b->value && pr.second == a->value);
        });
        if (!seen)
          plan.pairs.push_back({a->value, b->value});
      }
    } else if (node->is(fpcore::Sqrt) && node->operands.size() >= 1) {
      if (minMagnitude(node->operands[0]) < kCancellationThreshold)
        nearSingular = true;
    } else if (node->is(fpcore::Div) && node->operands.size() == 2) {
      if (minMagnitude(node->operands[1]) < kCancellationThreshold)
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
    collectLeaves(operand, seen, leaves);
}

namespace {
using Box = MapVector<Value *, SmallVector<double, 2>>;

// Draws points for the inputs of a subgraph. A subgraph's inputs are often
// functions of the same few values (every intermediate a function of a
// temperature T, say), so drawing each in its own profiled range pairs values
// that never occur together: terms that cancel in the data overflow an exp,
// and an input and its log disagree. Each sample draws the leaves the inputs
// are computed from (loads, arguments, globals) in their profiled ranges,
// evaluates each input from them as the program does, and is kept if every
// input lies in its own profiled range, the box Herbie searched.
class JointSampler {
public:
  JointSampler(const Options &O,
               const DenseMap<Value *, FPNode *> &valueToNodeMap,
               unsigned seedOffset)
      : O(O), valueToNodeMap(valueToNodeMap),
        gen(O.Model.RandomSeed + seedOffset) {}

  // The boxes of `inputs` and of the leaves they are computed from; an error
  // naming a value whose profiled range holds no interval of finite values.
  Error collectBoxes(ArrayRef<Value *> inputs);
  // Up to `target` points with each pair of `pairs` near-coincident.
  void drawCoincident(ArrayRef<std::pair<Value *, Value *>> pairs,
                      size_t target,
                      SmallVectorImpl<SubgraphSamples::Point> &out);
  // Points with independent leaves until `target` are kept.
  void drawIndependent(size_t target,
                       SmallVectorImpl<SubgraphSamples::Point> &out);

  const Box &leafRanges() const { return leafBox; }
  size_t numLeaves() const { return leafBox.size(); }
  size_t draws() const { return drawn; }

private:
  // Whether the inputs computed from `leafPoint` lie in the box, then
  // `point`.
  bool evaluate(const SubgraphSamples::Point &leafPoint,
                SubgraphSamples::Point &point,
                SmallVectorImpl<double> &results) const;

  const Options &O;
  const DenseMap<Value *, FPNode *> &valueToNodeMap;
  std::minstd_rand0 gen;
  Box box, leafBox;
  SmallVector<FPNode *, 8> computed;
  size_t drawn = 0;
};
} // namespace

// An error unless [lower, upper] is an interval of finite values; the profile
// gives an infinite bound when the run saw an infinity, and an empty range
// when it saw no value but NaN.
static Error checkFinite(double lower, double upper, const Value &V,
                         StringRef what) {
  if (std::isfinite(lower) && std::isfinite(upper) && lower <= upper)
    return Error::success();
  std::string text;
  raw_string_ostream os(text);
  os << "the profiled range [" << lower << ", " << upper << "] of " << what << V
     << " is not an interval of finite values";
  return createStringError(inconvertibleErrorCode(), text);
}

Error JointSampler::collectBoxes(ArrayRef<Value *> inputs) {
  SmallPtrSet<const FPNode *, 32> seen;
  SetVector<const FPLLValue *> leaves;
  for (Value *input : inputs) {
    FPNode *node = valueToNodeMap.at(input);
    double lower = node->getLowerBound(), upper = node->getUpperBound();
    if (Error E = checkFinite(lower, upper, *input, "the subgraph input "))
      return E;
    box.insert({input, {lower, upper}});
    collectLeaves(node, seen, leaves);
    if (!node->operands.empty())
      computed.push_back(node);
  }
  for (const FPLLValue *leaf : leaves) {
    double lower = leaf->getLowerBound(), upper = leaf->getUpperBound();
    if (Error E = checkFinite(lower, upper, *leaf->value,
                              "the value a subgraph input is computed from, "))
      return E;
    leafBox.insert({leaf->value, {lower, upper}});
  }
  return Error::success();
}

bool JointSampler::evaluate(const SubgraphSamples::Point &leafPoint,
                            SubgraphSamples::Point &point,
                            SmallVectorImpl<double> &results) const {
  if (!computed.empty())
    getFPValues(computed, leafPoint, results);
  size_t next = 0;
  for (const auto &[input, range] : box) {
    FPNode *node = valueToNodeMap.at(input);
    double v;
    if (!node->operands.empty())
      v = results[next++];
    else if (isa<FPConst>(node))
      v = node->getLowerBound();
    else
      v = leafPoint.lookup(input);
    if (!(v >= range[0] && v <= range[1]))
      return false;
    point.insert({input, v});
  }
  return true;
}

// Two leaves the subgraph subtracts, drawn independently, never come close, so
// each such pair is moved to a common near-coincident value, with one
// log-uniform separation scale shared across all pairs so every subtracted
// difference is small together. A draw takes more numbers from `gen` than an
// independent one, so these draws run one at a time.
void JointSampler::drawCoincident(
    ArrayRef<std::pair<Value *, Value *>> pairs, size_t target,
    SmallVectorImpl<SubgraphSamples::Point> &out) {
  const size_t budget = kJointDrawFactor * O.Model.NumSamples;
  double maxRange = 0.0;
  for (const auto &pr : pairs) {
    const SmallVector<double, 2> &range = leafBox.find(pr.first)->second;
    double r = range[1] - range[0];
    if (std::isfinite(r) && r > maxRange)
      maxRange = r;
  }
  if (!(maxRange > 0.0))
    maxRange = 1.0;

  constexpr double kSmallestSeparation = 1e-18;
  SmallVector<double, 8> results;
  while (out.size() < target && drawn < budget) {
    ++drawn;
    SubgraphSamples::Point leafPoint;
    for (const auto &[leaf, range] : leafBox)
      leafPoint.insert({leaf, uniformReal(gen, range[0], range[1])});
    double scale = std::exp(
        uniformReal(gen, std::log(kSmallestSeparation), std::log(maxRange)));
    for (const auto &pr : pairs) {
      double anchor = leafPoint[pr.first];
      double jitter = uniformReal(gen, 0.5, 1.5);
      double sign = uniformReal(gen, 0.0, 1.0) < 0.5 ? -1.0 : 1.0;
      // Clamped into the leaf's own profiled range so no out-of-distribution
      // value is fabricated.
      const SmallVector<double, 2> &range = leafBox.find(pr.second)->second;
      leafPoint[pr.second] =
          std::clamp(anchor + sign * scale * jitter, range[0], range[1]);
    }
    SubgraphSamples::Point point;
    if (evaluate(leafPoint, point, results))
      out.push_back(std::move(point));
  }
}

// Every draw takes the same numbers from `gen`, so a block of draws is drawn in
// order ahead of its evaluations, which run concurrently, and kept in order up
// to the first `target` in the box.
void JointSampler::drawIndependent(
    size_t target, SmallVectorImpl<SubgraphSamples::Point> &out) {
  const size_t budget = kJointDrawFactor * O.Model.NumSamples;
  const size_t numLeaves = leafBox.size();
  SmallVector<double, 0> leafValues;
  SmallVector<SubgraphSamples::Point, 0> points;
  SmallVector<char, 0> inBox;
  while (out.size() < target && drawn < budget) {
    // As many draws as the points still missing take at the share kept so
    // far, and a margin, so that few are evaluated past the last one kept.
    const double missing = target - out.size();
    const double kept = (out.size() + 1.0) / (drawn + 1.0);
    const size_t n = std::min<size_t>(
        budget - drawn,
        std::max<double>(kDrawsPerTask,
                         std::ceil(kDrawMargin * missing / kept)));
    leafValues.resize(n * numLeaves);
    for (size_t i = 0; i < n; ++i)
      for (auto [l, entry] : enumerate(leafBox))
        leafValues[i * numLeaves + l] =
            uniformReal(gen, entry.second[0], entry.second[1]);
    points.assign(n, {});
    inBox.assign(n, 0);
    forEachBlock(
        O.Model.Threads, n, kDrawsPerTask, [&](size_t begin, size_t end) {
          SmallVector<double, 8> blockResults;
          for (size_t i = begin; i < end; ++i) {
            SubgraphSamples::Point leafPoint;
            for (auto [l, entry] : enumerate(leafBox))
              leafPoint.insert({entry.first, leafValues[i * numLeaves + l]});
            inBox[i] = evaluate(leafPoint, points[i], blockResults);
          }
        });
    for (size_t i = 0; i < n && out.size() < target; ++i) {
      ++drawn;
      if (inBox[i])
        out.push_back(std::move(points[i]));
    }
  }
}

// -poseidon-num-samples points in the profiled box of `inputs`, the share
// kCancellationFraction of them near the coincidences `plan` names when it is
// active; an error saying why when the box holds too few.
static Expected<SmallVector<SubgraphSamples::Point, 4>>
drawPoints(const Options &O, ArrayRef<Value *> inputs,
           const DenseMap<Value *, FPNode *> &valueToNodeMap,
           const CancellationPlan *plan, unsigned seedOffset = 0) {
  const size_t numSamples = O.Model.NumSamples;
  JointSampler sampler(O, valueToNodeMap, seedOffset);
  if (Error E = sampler.collectBoxes(inputs))
    return std::move(E);

  SmallVector<std::pair<Value *, Value *>, 4> pairs;
  if (plan && plan->active)
    for (const auto &pr : plan->pairs)
      if (sampler.leafRanges().count(pr.first) &&
          sampler.leafRanges().count(pr.second))
        pairs.push_back(pr);
  SmallVector<SubgraphSamples::Point, 4> points;
  if (!pairs.empty())
    sampler.drawCoincident(
        pairs,
        std::min(numSamples,
                 static_cast<size_t>(kCancellationFraction *
                                     static_cast<double>(numSamples))),
        points);
  sampler.drawIndependent(numSamples, points);

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] joint sampling: " << sampler.numLeaves()
                 << " leaves, " << points.size() << " of " << sampler.draws()
                 << " draws inside the profiled input box\n";
  if (points.size() < kMinJointSamples)
    return createStringError(inconvertibleErrorCode(),
                             "joint sampling kept " + Twine(points.size()) +
                                 " of " + Twine(sampler.draws()) +
                                 " draws inside the subgraph's profiled "
                                 "input box");
  return points;
}

struct SubgraphSamples::RoundReference {
  Reference values;
  SmallVector<char, 0> computed;
};

SubgraphSamples::SubgraphSamples(const Options &O, const Subgraph &subgraph,
                                 const FPGraph &G)
    : O(O), subgraph(subgraph), valueToNodeMap(G.valueToNodeMap) {}

SubgraphSamples::~SubgraphSamples() = default;

Expected<ArrayRef<SubgraphSamples::Point>> SubgraphSamples::points() {
  if (!sampled) {
    sampled = true;
    CancellationPlan plan = buildCancellationPlan(O, subgraph, valueToNodeMap);
    Expected<SmallVector<Point, 4>> drawn =
        drawPoints(O, subgraph.inputs.getArrayRef(), valueToNodeMap, &plan);
    if (drawn)
      base = std::move(*drawn);
    else
      unsampled = toString(drawn.takeError());
  }
  if (unsampled)
    return createStringError(inconvertibleErrorCode(), *unsampled);
  return ArrayRef<Point>(base);
}

// The exact and the as-written value of `node` at `point`, and the error of
// the latter, into entry `i` of `R`.
std::optional<double> SubgraphSamples::Reference::meanError() const {
  double sum = 0.0;
  unsigned count = 0;
  for (double e : error)
    if (!std::isnan(e)) {
      sum += e;
      ++count;
    }
  if (count == 0)
    return std::nullopt;
  return sum / count;
}

// Returns whether the exact value did not settle.
static bool evaluateReference(FPNode *node, const SubgraphSamples::Point &point,
                              SubgraphSamples::Reference &R, size_t i) {
  SmallVector<double, 1> results;
  bool unsettled = getMPFRValues({node}, point, results);
  R.gold[i] = results[0];
  getFPValues({node}, point, results);
  R.value[i] = results[0];
  R.error[i] = sampleError(R.gold[i], R.value[i]);
  return unsettled;
}

void SubgraphSamples::fillReferences(const FPNode *node, ArrayRef<Point> at,
                                     ArrayRef<size_t> which,
                                     Reference &R) const {
  if (R.gold.empty()) {
    R.gold.resize(at.size());
    R.value.resize(at.size());
    R.error.resize(at.size());
  }
  std::atomic<unsigned> unsettled{0};
  forEachBlock(O.Model.Threads, which.size(), kReferencesPerTask,
               [&](size_t begin, size_t end) {
                 for (size_t i = begin; i < end; ++i)
                   unsettled += evaluateReference(const_cast<FPNode *>(node),
                                                  at[which[i]], R, which[i]);
               });
  if (!unsettled)
    return;
  const Instruction &At = *subgraph.outputs.front();
  diagnose(At,
           "the exact value of an expression of this subgraph did not settle "
           "within " +
               Twine(kMaxMPFRPrecision) + " bits at " + Twine(unsettled) +
               " of " + Twine(which.size()) +
               " sample points, which the accuracy model counts as NaN",
           DS_Warning);
}

const SubgraphSamples::Reference &
SubgraphSamples::reference(const FPNode *node) {
  auto [it, fresh] = references.try_emplace(node);
  Reference &R = it->second;
  if (!fresh)
    return R;
  ArrayRef<Point> at = cantFail(points());
  fillReferences(node, at, to_vector<0>(seq<size_t>(0, at.size())), R);
  return R;
}

ArrayRef<SubgraphSamples::Point> SubgraphSamples::round(unsigned round) {
  assert(round > 0 && "round 0 is points()");
  while (rounds.size() < round) {
    Expected<SmallVector<Point, 4>> drawn =
        drawPoints(O, subgraph.inputs.getArrayRef(), valueToNodeMap,
                   /*plan=*/nullptr, rounds.size() + 1);
    rounds.emplace_back();
    if (drawn)
      rounds.back() = std::move(*drawn);
    else
      consumeError(drawn.takeError());
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
  if (RR.computed.empty())
    RR.computed.resize(at.size(), 0);
  SmallVector<size_t, 0> todo;
  for (size_t p : which)
    if (!RR.computed[p]) {
      RR.computed[p] = 1;
      todo.push_back(p);
    }
  fillReferences(node, at, todo, RR.values);
  return RR.values;
}

SubgraphSamples &samplesOf(const Options &O, Subgraph &subgraph,
                           const FPGraph &G) {
  if (!subgraph.samples)
    subgraph.samples = std::make_shared<SubgraphSamples>(O, subgraph, G);
  return *subgraph.samples;
}

} // namespace poseidon
