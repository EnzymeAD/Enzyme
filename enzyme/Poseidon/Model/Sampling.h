//=- Sampling.h - input sampling for the accuracy model -------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The inputs the accuracy model scores a candidate on: inputs evaluated from
// leaves drawn in their profiled ranges, as the program relates them, inside
// the profiled per-input box, plus a coincidence stratum for the cancellation
// regimes independent leaf draws never reproduce.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SAMPLING_H
#define POSEIDON_SAMPLING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

#include <deque>
#include <memory>
#include <unordered_map>

namespace poseidon {

struct Options;
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
    const Options &O, const Subgraph &subgraph,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap);

void getSampledPoints(
    const Options &O, llvm::ArrayRef<llvm::Value *> inputs,
    const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
        &valueToNodeMap,
    llvm::SmallVector<llvm::MapVector<llvm::Value *, double>, 4> &sampledPoints,
    const CancellationPlan *plan = nullptr, size_t count = 0,
    unsigned seedOffset = 0);

// The points one subgraph is priced at and what its original expressions
// evaluate to there. They depend on the subgraph's inputs and the seed alone,
// so every unit that prices a rewrite of the subgraph shares one copy. Not for
// concurrent use: each method spreads its own work over -poseidon-threads.
class SubgraphSamples {
public:
  using Point = llvm::MapVector<llvm::Value *, double>;

  // At each point, the expression's exact value (MPFR, converged to 53 bits),
  // its value as written, and the error of the latter.
  struct Reference {
    llvm::SmallVector<double, 4> gold, value, error;
  };

  SubgraphSamples(
      const Options &O, const Subgraph &subgraph,
      const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
          &valueToNodeMap);
  ~SubgraphSamples();

  // getSampledPoints over the subgraph's inputs with its cancellation plan;
  // empty when the profiled box cannot be sampled.
  llvm::ArrayRef<Point> points();
  // `node` at every one of points().
  const Reference &reference(const FPNode *node);

  // getSampledPoints without a cancellation plan and with seed offset
  // `round`, for `round` >= 1.
  llvm::ArrayRef<Point> round(unsigned round);
  // `node` at the points of `round` listed in `which`; an entry at any other
  // point of the round is not computed.
  const Reference &roundReference(const FPNode *node, unsigned round,
                                  llvm::ArrayRef<size_t> which);

private:
  struct RoundReference;

  const Options &O;
  const Subgraph &subgraph;
  const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
      &valueToNodeMap;
  bool sampled = false;
  llvm::SmallVector<Point, 4> base;
  std::unordered_map<const FPNode *, Reference> references;
  // rounds[r - 1] holds round r; a deque, as a round's points are read while
  // later rounds are drawn.
  std::deque<llvm::SmallVector<Point, 4>> rounds;
  std::unordered_map<const FPNode *,
                     llvm::SmallVector<std::unique_ptr<RoundReference>, 0>>
      roundReferences;
};

// The samples of `subgraph`, made on first use and kept in it until its
// `samples` is reset.
SubgraphSamples &
samplesOf(const Options &O, Subgraph &subgraph,
          const std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
              &valueToNodeMap);

} // namespace poseidon
#endif // POSEIDON_SAMPLING_H
