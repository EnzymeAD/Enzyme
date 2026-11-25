//===- SiteProfile.h - profile generation for a site ----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each site's body instrumented, and the call that reaches it made a
// reverse-mode request over the instrumented clone with the shadows it needs.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITE_PROFILE_H
#define POSEIDON_SITE_PROFILE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

namespace llvm {
class CallInst;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

class Session;

struct Site;

/// Instruments the site and makes the call that reaches it a derivative
/// request; whether it did, or the mistake that keeps the site from being
/// profiled. Markers of one body share `clones`' instrumented clone, and the
/// clones made take their ids from `nextSiteId`.
llvm::Expected<bool>
profileSite(Session &S, const Site &site,
            llvm::DenseMap<llvm::Function *, llvm::Function *> &clones,
            unsigned &nextSiteId);
/// Registers with the host profiler, around each call that passes a stack
/// object, that object's extent, so that a site reached through it can size
/// its shadow.
void registerStackArrays(Session &S, llvm::Module &M);
/// Registers before main the extent of every global the module defines.
void registerGlobals(llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_SITE_PROFILE_H
