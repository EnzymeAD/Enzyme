//===- PassInternal.h - declarations shared by the pass sources -----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The site actions of the Poseidon passes (Actions.cpp), and the module as
// written beside the rewritten one (AsWritten.cpp): the copy is compiled as a
// build without a profile would compile it, and takes from the rewritten
// module only the functions a rewrite reached.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PASS_PASS_INTERNAL_H
#define POSEIDON_PASS_PASS_INTERNAL_H

#include "Solve/Session.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {
class CallBase;
class CallInst;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

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
/// after every site is lowered and inlined. An -poseidon-pin item
/// that names no site of the module is an error.
llvm::Error finalizeModule(Session &S, llvm::Module &M);

/// A copy of `M`, in its context, whose values keep the use-list order of the
/// values they copy, which passes that iterate a value's users read: the
/// module as written, to compile as a build without the plugin would.
std::unique_ptr<llvm::Module> snapshotAsWritten(const llvm::Module &M);

/// Compiles `asWrittenModule`, the module as written, with the default pipeline
/// of `Phase` as a build without a profile, takes into it the functions in
/// `rewritten`, those a rewrite reached, and gives M its contents. M is left as
/// it is when the compile of `asWrittenModule` reports an error or a rewritten
/// function refers to what `asWrittenModule` does not hold alike.
void replaceWithAsWritten(llvm::Module &M,
                          std::unique_ptr<llvm::Module> asWrittenModule,
                          const Session &parent, llvm::PassBuilder &PB,
                          llvm::OptimizationLevel Level,
                          llvm::ThinOrFullLTOPhase Phase,
                          llvm::ArrayRef<llvm::Function *> rewritten);

} // namespace poseidon

#endif // POSEIDON_PASS_PASS_INTERNAL_H
