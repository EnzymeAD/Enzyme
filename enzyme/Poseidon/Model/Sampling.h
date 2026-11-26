//===- Sampling.h - input sampling for the accuracy model -----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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

#ifndef POSEIDON_MODEL_SAMPLING_H
#define POSEIDON_MODEL_SAMPLING_H

#include "Core/ErrorMoment.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>

namespace poseidon {

struct Options;
struct Subgraph;
class FPGraph;
class FPNode;

/// What the error of `result` against `goldVal` contributes to an accuracy
/// cost in moment `m`: |goldVal - result|, in the units of the value, or its
/// square.
double sampleError(double goldVal, double result, ErrorMoment m);

/// The points one subgraph is priced at and what its original expressions
/// evaluate to there. They depend on the subgraph's inputs and the seed alone,
/// so every unit that prices a rewrite of the subgraph shares one copy. Not for
/// concurrent use: each method spreads its own work over -poseidon-threads.
class SubgraphSamples {
public:
  /// A sample: a value for each input of the subgraph.
  using Point = llvm::MapVector<llvm::Value *, double>;

  /// At each point, the expression's exact value (MPFR, converged to 53 bits),
  /// its value as written, and the error of the latter in the subgraph's
  /// moment (sampleError); NaN where the exact value did not settle.
  struct Reference {
    /// One entry per point, in the order of the points.
    llvm::SmallVector<double, 4> gold, value, error;
    /// The mean of the errors that are not NaN; none when every one is.
    std::optional<double> meanError() const;
  };

  /// The samples of `subgraph` over the profiled ranges `G` records. Both
  /// must outlive the samples.
  SubgraphSamples(const Options &O, const Subgraph &subgraph, const FPGraph &G);
  ~SubgraphSamples();

  /// -poseidon-num-samples points in the profiled box of the subgraph's inputs,
  /// with a share near the cancellations it divides by or takes the root of;
  /// an error saying why when the box cannot be sampled. Drawn on the first
  /// call; the points live as long as the samples.
  llvm::Expected<llvm::ArrayRef<Point>> points();
  /// `node` at every one of points(), which has succeeded. A reference that
  /// does not settle within 1024 bits counts as NaN, and is reported once, with
  /// a count, as a warning at the subgraph's first output. Computed on the
  /// first call for `node`; it lives as long as the samples.
  const Reference &reference(const FPNode *node);

  /// As many points drawn without the cancellation share, from the seed offset
  /// by `round`, for `round` >= 1; empty when that draw keeps too few, which
  /// ends the search for more points.
  llvm::ArrayRef<Point> round(unsigned round);
  /// `node` at the points of `round` listed in `which`; an entry at any other
  /// point of the round is not computed.
  const Reference &roundReference(const FPNode *node, unsigned round,
                                  llvm::ArrayRef<size_t> which);

private:
  struct RoundReference;

  /// Fills entry `which[i]` of `R`, sized for `at` when empty, with the
  /// reference of `node` at `at[which[i]]`; warns of those that did not settle.
  void fillReferences(const FPNode *node, llvm::ArrayRef<Point> at,
                      llvm::ArrayRef<size_t> which, Reference &R) const;

  const Options &O;
  const Subgraph &subgraph;
  const llvm::DenseMap<llvm::Value *, FPNode *> &valueToNodeMap;
  bool sampled = false;
  llvm::SmallVector<Point, 4> base;
  /// Why points() has none.
  std::optional<std::string> unsampled;
  std::unordered_map<const FPNode *, Reference> references;
  /// rounds[r - 1] holds round r; a deque, as a round's points are read while
  /// later rounds are drawn.
  std::deque<llvm::SmallVector<Point, 4>> rounds;
  std::unordered_map<const FPNode *,
                     llvm::SmallVector<std::unique_ptr<RoundReference>, 0>>
      roundReferences;
};

/// The samples of `subgraph`, made on first use and kept in it until its
/// `samples` is reset.
SubgraphSamples &samplesOf(const Options &O, Subgraph &subgraph,
                           const FPGraph &G);

} // namespace poseidon
#endif // POSEIDON_MODEL_SAMPLING_H
