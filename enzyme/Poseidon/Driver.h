#ifndef POSEIDON_DRIVER_H
#define POSEIDON_DRIVER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class CallInst;
class Module;
class PassBuilder;
} // namespace llvm

namespace poseidon {

// Whether a called function is the Poseidon annotation.
bool isMarkerCall(llvm::StringRef calleeName);

// Profile-use: lower every marker collected from one function into a call of
// the optimized body.
bool lowerMarkers(llvm::ArrayRef<llvm::CallInst *> markers,
                  llvm::SmallVectorImpl<llvm::CallInst *> &calls);

// Runs before the host's own marker scan. Profile generation happens entirely
// here: each site is canonicalized, instrumented with the profiling probes and
// its marker rewritten into an ordinary __enzyme_autodiff request, which the
// host then lowers like any other.
void prepareModule(llvm::Module &M);

// Profile use for the sites this module declares with POSEIDON_OPTIMIZE (or
// names with -poseidon-kernels): each such kernel becomes a wrapper around its
// own body, and that body is solved and rewritten like a marked one.
bool optimizeAnnotatedSites(llvm::Module &M);

// -poseidon-joint-dp: solve every deferred site under one shared budget.
bool solveDeferredSites(llvm::Module &M, llvm::FunctionAnalysisManager &FAM);

// Drop the canonicalized clones nothing calls any more. Runs once per module,
// after the host has finished lowering and inlining every site.
void finalizeModule(llvm::Module &M);

void registerPasses(llvm::PassBuilder &PB);

} // namespace poseidon

#endif // POSEIDON_DRIVER_H
