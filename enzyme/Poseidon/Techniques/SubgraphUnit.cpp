//===- SubgraphUnit.cpp - a unit over one FP subgraph ---------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubgraphUnit.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>

using namespace llvm;

namespace poseidon {

Expected<BaselineAccuracy> baselineAccuracy(SubgraphSamples &samples,
                                            ArrayRef<const FPNode *> nodes,
                                            ArrayRef<double> grads,
                                            bool print) {
  BaselineAccuracy B;
  Expected<ArrayRef<SubgraphSamples::Point>> drawn = samples.points();
  if (!drawn)
    return drawn.takeError();
  B.points = *drawn;
  for (auto [node, grad] : zip(nodes, grads)) {
    const SubgraphSamples::Reference &ref = samples.reference(node);
    std::optional<double> error = ref.meanError();
    if (!error)
      return createStringError(
          inconvertibleErrorCode(),
          "every sample of the original expression scored NaN");
    B.outputs.push_back({node, &ref, *error, grad});
    B.cost += *error * std::fabs(grad);
  }
  if (!std::isfinite(B.cost)) {
    // A non-finite baseline makes every candidate's delta NaN, and the DP
    // would quietly keep the subgraph as written.
    if (print)
      for (const BaselineAccuracy::Output &O : B.outputs)
        errs() << "[poseidon] output baseline accuracy cost " << O.error
               << ", gradient " << O.grad << '\n';
    return createStringError(
        inconvertibleErrorCode(),
        "the baseline accuracy cost is not finite; the site's profile is "
        "degenerate (a zero or null-space output shadow gives a gradient "
        "that cannot weight anything)");
  }
  return B;
}

} // namespace poseidon
