//===- Twin.h - the module as written, compiled beside the rewritten one --===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The twin is compiled as a build without a profile would compile it, and
// takes from the rewritten module only the functions a rewrite reached.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TWIN_H
#define POSEIDON_TWIN_H

#include "Session.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

#include <memory>

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace poseidon {

/// A copy of `M`, in its context, whose values keep the use-list order of the
/// values they copy, which passes that iterate a value's users read: the
/// module as written, to compile as a build without the plugin would.
std::unique_ptr<llvm::Module> snapshotAsWritten(const llvm::Module &M);

/// Compiles `twin`, the module as written, with the default pipeline of
/// `Phase` as a build without a profile, takes into it the functions in
/// `rewritten`, those a rewrite reached, and gives M its contents. M is left as
/// it is when the compile of `twin` reports an error or a rewritten function
/// refers to what `twin` does not hold alike.
void replaceWithAsWritten(llvm::Module &M, std::unique_ptr<llvm::Module> twin,
                          const Session &parent, llvm::PassBuilder &PB,
                          llvm::OptimizationLevel Level,
                          llvm::ThinOrFullLTOPhase Phase,
                          llvm::ArrayRef<llvm::Function *> rewritten);

} // namespace poseidon

#endif // POSEIDON_TWIN_H
