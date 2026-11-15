//=- Driver.h - the site actions of the Poseidon passes ---------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_DRIVER_H
#define POSEIDON_DRIVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class CallBase;
class CallInst;
class Module;
} // namespace llvm

namespace poseidon {

class Session;

// Profile-use: lower every marker collected from one function into a call of
// the optimized body.
bool lowerMarkers(Session &S, llvm::ArrayRef<llvm::CallInst *> markers,
                  llvm::SmallVectorImpl<llvm::CallBase *> &calls);

// Profile generation: instruments every site of M that is not deferred with
// the profiling probes and rewrites its marker into an __enzyme_autodiff
// request. Returns true if it instrumented a site.
bool prepareModule(Session &S, llvm::Module &M);

// Profile use for the sites this module declares with POSEIDON_OPTIMIZE (or
// names with -poseidon-kernels): each such kernel becomes a wrapper around its
// own body, and that body is solved and rewritten like a marked one.
bool optimizeAnnotatedSites(Session &S, llvm::Module &M);

// -poseidon-joint-dp: solve every deferred site under one shared budget.
bool solveDeferredSites(Session &S, llvm::Module &M,
                        llvm::FunctionAnalysisManager &FAM);

// Drop the canonicalized clones nothing calls any more. Runs once per module,
// after every site is lowered and inlined.
void finalizeModule(Session &S, llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_DRIVER_H
