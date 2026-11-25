//===- Utils.h - IR helpers of the Poseidon pass --------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shims over the LLVM versions the pass builds against, the simplification
// pipeline it runs, and small IR helpers.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SUPPORT_UTILS_H
#define POSEIDON_SUPPORT_UTILS_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/Error.h"

#include "Enzyme/PassUtils.h"

#include <string>

namespace llvm {
class PassBuilder;
class SCEVExpander;
class ScalarEvolution;
} // namespace llvm

namespace poseidon {

/// A PassBuilder without a target machine; LLVM 24 gives it the context.
llvm::PassBuilder makePassBuilder(llvm::LLVMContext &C);

/// A SCEVExpander that names the values it emits `name`; LLVM 22 dropped the
/// DataLayout argument of its constructor.
llvm::SCEVExpander makeSCEVExpander(llvm::ScalarEvolution &SE,
                                    const char *name);

/// A builder inserting at `IP` in `BB`; `IP` may be `BB->end()`. LLVM 24 takes
/// the position alone and deprecates the form that also names the block.
inline llvm::IRBuilder<> builderAt(llvm::BasicBlock *BB,
                                   llvm::BasicBlock::iterator IP) {
#if LLVM_VERSION_MAJOR >= 24
  return llvm::IRBuilder<>(IP);
#else
  return llvm::IRBuilder<>(BB, IP);
#endif
}

inline void setInsertPoint(llvm::IRBuilderBase &B, llvm::BasicBlock *BB,
                           llvm::BasicBlock::iterator IP) {
#if LLVM_VERSION_MAJOR >= 24
  B.SetInsertPoint(IP);
#else
  B.SetInsertPoint(BB, IP);
#endif
}

/// The operands an FP instruction computes from: a call's arguments, without
/// its callee.
inline auto fpOperands(llvm::Instruction &I) {
  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
    return CI->args();
  return I.operands();
}

/// A double as %g prints it.
std::string shortDouble(double v);
/// A double as text that reads back to the same bits, infinity included.
std::string exactDouble(double v);
/// The demangled name of `F`.
std::string displayName(const llvm::Function &F);
/// Whether `F` calls the function named `name` directly.
bool callsFunctionNamed(const llvm::Function &F, llvm::StringRef name);
/// Appends `insts` to `instsSorted` with each after those of `insts` it uses,
/// in `insts`' order where that leaves a choice.
void topoSort(const llvm::SetVector<llvm::Instruction *> &insts,
              llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);

/// Inlines the calls of `F` that `pick` selects, in instruction order, and
/// again in what that inlined, for a bounded number of rounds so that a
/// recursive callee cannot keep it going; returns how many calls it inlined.
unsigned inlineCallsWhile(llvm::Function &F,
                          llvm::function_ref<bool(llvm::CallBase &)> pick);

/// Runs the function simplification pipeline of `Level` that `PB` builds on F,
/// calling `afterPass`, when given, after each pass: the compile's own
/// pipeline, whose passes ask the compile's target, for instance whether a
/// branch on a loop-invariant condition may diverge.
void simplifyFunction(llvm::Function &F, llvm::PassBuilder &PB,
                      llvm::OptimizationLevel Level,
                      llvm::function_ref<void()> afterPass = nullptr);

/// Runs the target-independent function simplification pipeline of `Level` on
/// F. Without a target no branch diverges, so a loop may be versioned on a
/// loop-invariant condition.
void simplifyFunction(llvm::Function &F, llvm::OptimizationLevel Level);

} // namespace poseidon
#endif // POSEIDON_SUPPORT_UTILS_H
