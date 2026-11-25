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

namespace llvm {
class CallInst;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

class Session;

struct Site;

/// Instruments the site and makes the call that reaches it a derivative
/// request; whether it did. Markers of one body share `clones`' instrumented
/// clone, and the clones made take their ids from `nextSiteId`.
bool profileSite(Session &S, const Site &site,
                 llvm::DenseMap<llvm::Function *, llvm::Function *> &clones,
                 unsigned &nextSiteId);
void registerStackArrays(Session &S, llvm::Module &M);
void registerGlobals(llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_SITE_PROFILE_H
