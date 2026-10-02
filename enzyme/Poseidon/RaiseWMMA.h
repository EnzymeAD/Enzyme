//=- RaiseWMMA.h - Scalar-loop to WMMA raising for Poseidon ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RAISE_WMMA_H
#define POSEIDON_RAISE_WMMA_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

#include "ProfileRead.h"
#include "Matmul.h"

namespace llvm {
class Function;
class LoopInfo;
class Module;
class ScalarEvolution;
} // namespace llvm

namespace poseidon {

void findScalarLoopMatmuls(llvm::Function &F, llvm::ScalarEvolution &SE,
                           llvm::LoopInfo &LI,
                           const FunctionProfileHeader &profileHeader,
                           llvm::SmallVectorImpl<AbstractMatmul> &out);

// Profile-gen only: collect one profiled clone's scalar-loop matmul reduction
// trip counts (profile-scale) as (slot, trip) pairs, so the profuse solve can
// price the Ozaki-II padding waste at one scale even under surrogate profiling.
// Must run on the clean clone: the profiling probes break the reduction shape.
void collectScalarLoopReductionTrips(
    llvm::Function &F, llvm::SmallVectorImpl<std::pair<size_t, unsigned>> &out);

void materializeScalarLoopRaise(const AbstractMatmul &m,
                                const CandidateMatmul::Option &opt);

} // namespace poseidon
#endif // POSEIDON_RAISE_WMMA_H
