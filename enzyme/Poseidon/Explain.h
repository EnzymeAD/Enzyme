//=- Explain.h - what a solve changed, considered and rejected -------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimization remarks (pass name "poseidon") and <func>.explain.json, built
// from the solver's own data after it has chosen and before anything is
// materialized. Output only: nothing here feeds back into a decision.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_EXPLAIN_H
#define POSEIDON_EXPLAIN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"

namespace poseidon {

struct FunctionFPState;
struct SolutionStep;
struct SolveRecord;

bool explainWanted(const llvm::Function &F);

void noteSiteCall(const llvm::Function *body, const llvm::DebugLoc &loc);

void explainNothingToOptimize(llvm::Function &F);

void explainSolve(llvm::Function &F, FunctionFPState &st,
                  llvm::ArrayRef<SolutionStep> steps, const SolveRecord &rec);

} // namespace poseidon
#endif // POSEIDON_EXPLAIN_H
