//=- Sampling.cpp - input sampling for the accuracy model -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sampling.h"
#include "Evaluators.h"
#include "Flags.h"
#include "Types.h"
#include "Utils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <limits>
#include <random>

using namespace llvm;

namespace poseidon {

static constexpr double kCancellationThreshold = 1e-12;
static constexpr double kCancellationFraction = 0.5;

double sampleError(double goldVal, double result) {
  return std::fabs(goldVal - result);
}

// The gate is workload-driven: only a profiled sqrt argument or denominator
// within kCancellationThreshold of zero activates the plan.
CancellationPlan
buildCancellationPlan(const Subgraph &subgraph,
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

    // FPNode ops use Herbie symbols ("-", "/", "sqrt"), not LLVM opcode names.
    if (node->op == "-" && node->operands.size() == 2) {
      // A subtraction of two subgraph inputs is a cancellation site; one entry
      // per unordered pair.
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
  if (flags::Print && plan.active)
    llvm::errs() << "[poseidon] cancellation-aware sampling active: "
                 << plan.pairs.size() << " leaf-subtraction pair(s), "
                 << "near-singular denominator/radicand profiled\n";
  return plan;
}

void getSampledPoints(
    ArrayRef<Value *> inputs,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints,
    const CancellationPlan *plan) {
  std::default_random_engine gen;
  gen.seed(flags::RandomSeed);
  std::uniform_real_distribution<> dis;

  MapVector<Value *, SmallVector<double, 2>> hypercube;
  for (const auto input : inputs) {
    const auto node = valueToNodeMap.at(input);

    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper) {
      // [INF, -INF] are the FPLLValue initialisers: the profiler never observed
      // this input (possible even with a non-zero gradient when the output
      // shadow is degenerate). Refuse to sample rather than invent a bound; the
      // caller leaves an unpriced subgraph alone.
      llvm::errs() << "Poseidon: input to sampled subgraph has unprofiled "
                      "bounds ["
                   << lower << ", " << upper
                   << "]; refusing to sample it. This subgraph cannot be "
                      "priced and will keep its original body. Input value: "
                   << *input << "\n";
      sampledPoints.clear();
      return;
    }

    hypercube.insert({input, {lower, upper}});
  }

  sampledPoints.clear();
  sampledPoints.resize(flags::NumSamples);

  // Close-encounter stratum: each sample first draws every input independently,
  // then overrides each cancelling pair to a common near-coincident value with
  // one log-uniform separation scale shared across all pairs, so every
  // subtracted difference is small together.
  size_t numCoincidence = 0;
  if (plan && plan->active && !plan->pairs.empty()) {
    numCoincidence = static_cast<size_t>(
        kCancellationFraction * static_cast<double>(flags::NumSamples));
  }

  double maxRange = 1.0;
  if (numCoincidence > 0) {
    maxRange = 0.0;
    for (const auto &pr : plan->pairs) {
      auto ai = hypercube.find(pr.first);
      if (ai == hypercube.end())
        continue;
      double r = ai->second[1] - ai->second[0];
      if (std::isfinite(r) && r > maxRange)
        maxRange = r;
    }
    if (!(maxRange > 0.0) || !std::isfinite(maxRange))
      maxRange = 1.0;
  }

  for (size_t i = 0; i < flags::NumSamples; ++i) {
    MapVector<Value *, double> point;
    for (const auto &entry : hypercube) {
      Value *val = entry.first;
      double lower = entry.second[0];
      double upper = entry.second[1];
      double sample = dis(gen, decltype(dis)::param_type{lower, upper});
      point.insert({val, sample});
    }

    if (i < numCoincidence) {
      const double scaleLo = 1e-18;
      double logScale = dis(gen, decltype(dis)::param_type{std::log(scaleLo),
                                                           std::log(maxRange)});
      double scale = std::exp(logScale);
      for (const auto &pr : plan->pairs) {
        auto bi = hypercube.find(pr.second);
        if (bi == hypercube.end())
          continue;
        double anchor = point[pr.first];
        double jitter = dis(gen, decltype(dis)::param_type{0.5, 1.5});
        double sign =
            (dis(gen, decltype(dis)::param_type{0.0, 1.0}) < 0.5) ? -1.0 : 1.0;
        double bVal = anchor + sign * scale * jitter;
        // Clamp the coincident operand into its own profiled box so no
        // out-of-distribution input is fabricated.
        double bLo = bi->second[0], bHi = bi->second[1];
        if (bVal < bLo)
          bVal = bLo;
        if (bVal > bHi)
          bVal = bHi;
        point[pr.second] = bVal;
      }
    }

    sampledPoints[i] = point;
  }
}

void getSampledPoints(
    const std::string &expr,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints) {
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SmallVector<Value *, 4> inputs;
  for (const auto &argStr : argStrSet) {
    inputs.push_back(symbolToValueMap.at(argStr));
  }

  getSampledPoints(inputs, valueToNodeMap, symbolToValueMap, sampledPoints);
}

} // namespace poseidon
