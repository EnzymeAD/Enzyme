//===- RealizeLibraryCall.h - host-side GEMM library dispatch -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A raised GEMM materialized as a host library call: the device sub-compilation
// records a descriptor in the manifest of its translation unit in the Poseidon
// cache dir, and the host sub-compilation (a separate cc1 process) reads it and
// rewrites the kernel launch into a call to the runtime helper.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_MATMUL_REALIZE_LIBRARY_CALL_H
#define POSEIDON_TECHNIQUES_MATMUL_REALIZE_LIBRARY_CALL_H

#include "Matmul.h"

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct Options;
class Session;

/// Map body params to wrapper launch-arg indices via primalArgs and record the
/// wrapper's host dispatch descriptor.
void writeGemmDescriptor(Session &S, llvm::Function &wrapper,
                         llvm::ArrayRef<llvm::Value *> primalArgs,
                         const ChosenLibraryCall &call);

/// For a joint solve, whose notes exist only after materialization:
/// addPendingGemmDispatch records the body-parameter to wrapper-argument map
/// when the site is split, and flushPendingGemmDispatches writes the
/// descriptors afterwards.
void addPendingGemmDispatch(Session &S, const llvm::Function *body,
                            llvm::Function &wrapper,
                            llvm::ArrayRef<llvm::Value *> primalArgs);
/// Writes the descriptor of each pending site whose body was given a library
/// call, and forgets every pending site.
void flushPendingGemmDispatches(Session &S);

/// Rewrite the launch in each launch stub that a host dispatch descriptor in
/// M's manifest matches into a call of the GEMM runtime. Host module only. An
/// error for a descriptor that does not parse.
llvm::Expected<bool> rewriteGemmStubBodies(const Options &O, llvm::Module &M);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_MATMUL_REALIZE_LIBRARY_CALL_H
