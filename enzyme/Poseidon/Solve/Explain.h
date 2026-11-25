//===- Explain.h - what a solve changed, considered and rejected ----------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

namespace poseidon {

struct FunctionFPState;
struct Options;
class Session;
struct SolutionStep;
struct SolveRecord;

bool explainWanted(const Options &O, const llvm::Function &F);

// The explanation of a site the pass leaves as written without solving it:
// `status` is what kept it from the solve and `reason` says why. F is the
// site's clone, or with `body` the site's body when no clone was made. `tau`
// > 0 is the site's accuracy target, from the site when `siteTarget`.
void explainLeftAlone(Session &S, llvm::Function &F, double tau,
                      bool siteTarget, llvm::StringRef status,
                      llvm::StringRef reason, llvm::StringRef profile = "",
                      bool body = false);

// A remark and the explanation of a site with nothing to optimize.
void explainNothingToOptimize(Session &S, llvm::Function &F, double tau,
                              bool siteTarget);

// A remark and the explanation of a site no profile has records of: `status`
// is "unprofiled" or "never_executed", `profile` the file looked at.
void explainUnprofiled(Session &S, llvm::Function &F, double tau,
                       bool siteTarget, llvm::StringRef status,
                       llvm::StringRef cause, llvm::StringRef profile);

void explainSolve(Session &S, llvm::Function &F, FunctionFPState &st,
                  llvm::ArrayRef<SolutionStep> steps, const SolveRecord &rec);

} // namespace poseidon
#endif // POSEIDON_EXPLAIN_H
