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
#include "llvm/Support/Error.h"

namespace llvm {
class CallBase;
class CallInst;
class Module;
} // namespace llvm

namespace poseidon {

class Session;
struct Site;

/// Profile use: each of `sites` solved and its call made a call of what the
/// solve chose, appending the calls made to `calls`; in a compile without a
/// profile, the site as written. A marker becomes a call of its body, a kernel
/// a wrapper around its own outlined body.
llvm::Expected<bool>
optimizeSites(Session &S, llvm::ArrayRef<Site> sites,
              llvm::SmallVectorImpl<llvm::CallBase *> &calls);

/// Profile generation: instruments every site of M that is not deferred with
/// the profiling probes and rewrites its marker into an __enzyme_autodiff
/// request. Returns true if it instrumented a site.
llvm::Expected<bool> prepareModule(Session &S, llvm::Module &M);

/// Drop the canonicalized clones nothing calls any more. Runs once per module,
/// after every site is lowered and inlined. An -poseidon-apply-rewrites item
/// that names no site of the module is an error.
llvm::Error finalizeModule(Session &S, llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_DRIVER_H
