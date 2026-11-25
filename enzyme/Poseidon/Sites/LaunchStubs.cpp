//===- LaunchStubs.cpp - the pass that rewrites host launch stubs ---------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaunchStubs.h"
#include "LaunchDescriptors.h"
#include "Pass/Session.h"
#include "Profile/ProfiledLaunch.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/HostDispatch.h"

using namespace llvm;

namespace poseidon {

llvm::PreservedAnalyses HostStubPass::run(llvm::Module &M,
                                          llvm::ModuleAnalysisManager &MAM) {
  Session &S = sessionOf(M, MAM);
  const Options &O = S.Opts;
  if (S.Stopped || Target::of(M).isDevice() || !handsOverLaunches(O))
    return llvm::PreservedAnalyses::all();
  // Profile generation: every launch of an annotated kernel goes through the
  // runtime helper that supplies its shadow buffers. Host dispatch
  // materializes only in the optimize phase: profile-gen builds instrument the
  // original kernel and do not link the runtime.
  bool changed = false;
  if (O.Profiling.generatesProfile()) {
    Expected<bool> rewritten = rewriteProfileStubBodies(O, M);
    if (!rewritten)
      return stopOnError(S, M, rewritten.takeError());
    changed = *rewritten;
    changed |= registerProfiledKernels(O, M);
  } else {
    Expected<bool> rewritten = rewriteGemmStubBodies(O, M);
    if (!rewritten)
      return stopOnError(S, M, rewritten.takeError());
    changed = *rewritten;
  }
  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace poseidon
