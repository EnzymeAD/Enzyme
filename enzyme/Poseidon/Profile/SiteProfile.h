//===- SiteProfile.h - profile generation for a site ----------------------===//
// Each site's body instrumented, and the call that reaches it made a
// reverse-mode request over the instrumented clone with the shadows it needs.
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

bool profileSite(Session &S, llvm::CallInst *CI,
                 llvm::DenseMap<llvm::Function *, llvm::Function *> &clones,
                 unsigned &nextSiteId);
bool profileAnnotatedKernel(Session &S, llvm::Function &K,
                            unsigned &nextSiteId);
void registerStackArrays(Session &S, llvm::Module &M);
void registerGlobals(llvm::Module &M);
bool profileHostSite(Session &S, llvm::Function &B, unsigned &nextSiteId);

} // namespace poseidon

#endif // POSEIDON_SITE_PROFILE_H
