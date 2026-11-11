//===- KernelSites.h - site kernels and device call sites -----------------===//
// A site kernel made a wrapper around its own body, and a __poseidon_optimize
// call in device code made a marker over the copy of the function it calls.
#ifndef POSEIDON_KERNEL_SITES_H
#define POSEIDON_KERNEL_SITES_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {
class CallInst;
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

void markOutlineTransparent(llvm::Function *body);
llvm::CallInst *outlineKernelBody(llvm::Function &K, llvm::Function *&bodyOut);
llvm::Value *localObject(llvm::Value *p, const llvm::Function *caller);
bool pointsNowhere(llvm::Value *p);
std::string perCallSiteName(llvm::StringRef symbol);
bool lowerDeviceCallSites(llvm::Module &M);
void inlineKernelBody(llvm::CallInst *call, llvm::Function *body);

} // namespace poseidon

#endif // POSEIDON_KERNEL_SITES_H
