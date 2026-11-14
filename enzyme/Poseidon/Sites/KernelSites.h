//===- KernelSites.h - site kernels and device call sites -----------------===//
// A site kernel made a wrapper around its own body, and a __poseidon_optimize
// call in device code made a marker over the copy of the function it calls.
#ifndef POSEIDON_KERNEL_SITES_H
#define POSEIDON_KERNEL_SITES_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class Argument;
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

// What a function reaches through one of its pointer parameters, in bytes from
// the pointer it is passed: [lo, hi) read or written, [writeLo, writeHi)
// written, and the strictest alignment an access claims, `align` at offset
// `alignedAt`.
struct PointerExtent {
  int64_t lo = 0, hi = 0, writeLo = 0, writeHi = 0;
  uint64_t align = 1;
  int64_t alignedAt = 0;
  bool accessed() const { return lo < hi; }
  bool written() const { return writeLo < writeHi; }
};
// The extent of `A`, or none, with in `why` what the function does with the
// pointer that defeats it, phrased to follow "a pointer the call".
std::optional<PointerExtent> accessedExtent(llvm::Argument &A,
                                            std::string &why);

std::string perCallSiteName(llvm::StringRef symbol);
bool lowerDeviceCallSites(llvm::Module &M);
void inlineKernelBody(llvm::CallInst *call, llvm::Function *body);

} // namespace poseidon

#endif // POSEIDON_KERNEL_SITES_H
