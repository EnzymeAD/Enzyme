//===- HostSites.h - host functions and calls that are sites --------------===//
// POSEIDON_OPTIMIZE on a host function and __poseidon_optimize on a call: the
// body each site gets, which the later stages find by its kHostBody mark.
#ifndef POSEIDON_HOST_SITES_H
#define POSEIDON_HOST_SITES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm {
class CallBase;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct Options;

// The outlined body of a host site. Its value is the site's accuracy target,
// empty when the annotation carried none.
constexpr llvm::StringLiteral kHostBody = "poseidon-site-body";

std::string exactDouble(double v);
bool lowerPerCallSites(const Options &O, llvm::Module &M);
bool outlineHostSites(const Options &O, llvm::Module &M);
llvm::SmallVector<llvm::Function *, 4> hostSiteBodies(llvm::Module &M);
llvm::SmallVector<llvm::CallBase *, 4> callsOf(llvm::Function &F);
double takeHostBody(llvm::Function &B);

} // namespace poseidon

#endif // POSEIDON_HOST_SITES_H
