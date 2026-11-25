//===- LaunchStubs.h - the host launch stubs of annotated kernels ---------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host module's launch stubs of the kernels a descriptor in its manifest
// names, rewritten before inlining: for profile generation into a launch that
// supplies the profiling kernel's shadows, and for host GEMM dispatch.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITES_LAUNCH_STUBS_H
#define POSEIDON_SITES_LAUNCH_STUBS_H

#include "IR/Utils.h"

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
} // namespace llvm

namespace poseidon {

struct Options;

/// Rewrite the launch in each launch stub that a profile generation descriptor
/// in M's manifest matches into a call of __poseidon_launch_profiled, which
/// allocates, zeroes and seeds the shadow buffers the profiling kernel takes
/// and launches it. Host module only.
bool rewriteProfileStubBodies(const Options &O, llvm::Module &M);

/// Rewrites, on the host module and before inlining, each __device_stub__
/// launch stub that a descriptor in the module's manifest matches.
class HostStubPass : public PassParent<HostStubPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace poseidon
#endif // POSEIDON_SITES_LAUNCH_STUBS_H
