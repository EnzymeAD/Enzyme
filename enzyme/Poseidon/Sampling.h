//=- Sampling.h - input sampling for the accuracy model -------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The accuracy model scores a candidate on sampled inputs, so what is sampled
// decides what the score means: the profiled per-input box, plus a coincidence
// stratum for the cancellation regimes independent marginal draws never
// reproduce.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SAMPLING_H
#define POSEIDON_SAMPLING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace poseidon {

struct Subgraph;
class FPNode;

// Per-sample absolute accuracy error.
double sampleError(double goldVal, double result);

// Cancellation-aware sampling plan: `active` iff the subgraph divides by or
// takes the sqrt of a quantity profiled near zero; `pairs` are the leaf
// subtraction operands whose near-coincidence produces it (independent
// marginal sampling never reproduces that configuration).
struct CancellationPlan {
  bool active = false;
  llvm::SmallVector<std::pair<llvm::Value *, llvm::Value *>, 4> pairs;
};

CancellationPlan buildCancellationPlan(
    const Subgraph &subgraph,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap);

void getSampledPoints(
    llvm::ArrayRef<llvm::Value *> inputs,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap,
    const std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVector<llvm::MapVector<llvm::Value *, double>, 4> &sampledPoints,
    const CancellationPlan *plan = nullptr);

void getSampledPoints(
    const std::string &expr,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap,
    const std::unordered_map<std::string, llvm::Value *> &symbolToValueMap,
    llvm::SmallVector<llvm::MapVector<llvm::Value *, double>, 4>
        &sampledPoints);

} // namespace poseidon
#endif // POSEIDON_SAMPLING_H
