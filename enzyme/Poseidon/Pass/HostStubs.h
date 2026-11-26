//===- HostStubs.h - the host launch stubs of annotated kernels -----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host module's launch stubs of the kernels a descriptor in its manifest
// names, rewritten before inlining: for profile generation into a launch that
// supplies the profiling kernel's shadows, and for host library call.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PASS_HOST_STUBS_H
#define POSEIDON_PASS_HOST_STUBS_H

#include "Support/Utils.h"

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
} // namespace llvm

namespace poseidon {

/// Rewrites, on the host module and before inlining, each __device_stub__
/// launch stub that a descriptor in the module's manifest matches.
class HostStubPass : public PassParent<HostStubPass> {
public:
  /// Rewrites the stubs of `M`; nothing in a device module or a compile that
  /// hands no launch over, and nothing once the session has stopped.
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace poseidon
#endif // POSEIDON_PASS_HOST_STUBS_H
