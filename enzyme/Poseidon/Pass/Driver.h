//===- Driver.h - the site actions of the Poseidon passes -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_DRIVER_H
#define POSEIDON_DRIVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class CallBase;
class CallInst;
class Module;
} // namespace llvm

namespace poseidon {

class Session;

// Profile use: lower each of `markers` into a call of the optimized body,
// appending the calls made to `calls`.
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

// Drop the canonicalized clones nothing calls any more. Runs once per module,
// after every site is lowered and inlined.
void finalizeModule(Session &S, llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_DRIVER_H
