//===- Utils.h - Utility functions for Poseidon optimization pass ---------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions for the Poseidon floating-point
// optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_UTILS_H
#define POSEIDON_UTILS_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <string>

namespace llvm {
class PassBuilder;
} // namespace llvm

namespace poseidon {

#if LLVM_VERSION_MAJOR >= 23
template <typename DerivedT>
using PassParent = llvm::RequiredPassInfoMixin<DerivedT>;
#else
template <typename DerivedT> using PassParent = llvm::PassInfoMixin<DerivedT>;
#endif

// LLVM 24 split BranchInst into CondBrInst and UncondBrInst.
inline bool isConditionalBranch(const llvm::Value *V) {
#if LLVM_VERSION_MAJOR >= 24
  return llvm::isa<llvm::CondBrInst>(V);
#else
  auto *BI = llvm::dyn_cast<llvm::BranchInst>(V);
  return BI && BI->isConditional();
#endif
}

inline llvm::Value *branchCondition(llvm::Value *V) {
#if LLVM_VERSION_MAJOR >= 24
  return llvm::cast<llvm::CondBrInst>(V)->getCondition();
#else
  return llvm::cast<llvm::BranchInst>(V)->getCondition();
#endif
}

// A builder inserting at `IP` in `BB`; `IP` may be `BB->end()`. LLVM 24 takes
// the position alone and deprecates the form that also names the block.
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

double stringToDouble(const std::string &str);
double literalToDouble(const std::string &str);
void topoSort(const llvm::SetVector<llvm::Instruction *> &insts,
              llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);
void reverseTopoSort(const llvm::SetVector<llvm::Instruction *> &insts,
                     llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);

void collectExprInsts(llvm::Value *V,
                      const llvm::SetVector<llvm::Value *> &inputs,
                      llvm::SmallPtrSetImpl<llvm::Instruction *> &exprInsts,
                      llvm::SmallPtrSetImpl<llvm::Value *> &visited);

// Runs the function simplification pipeline of `Level` that `PB` builds on F,
// calling `afterPass`, when given, after each pass: the compile's own
// pipeline, whose passes ask the compile's target, for instance whether a
// branch on a loop-invariant condition may diverge.
void simplifyFunction(llvm::Function &F, llvm::PassBuilder &PB,
                      llvm::OptimizationLevel Level,
                      llvm::function_ref<void()> afterPass = nullptr);

// Runs the target-independent function simplification pipeline of `Level` on
// F. Without a target no branch diverges, so a loop may be versioned on a
// loop-invariant condition.
void simplifyFunction(llvm::Function &F, llvm::OptimizationLevel Level);

// A mistake in what the compile was given that leaves nothing to compile with:
// an error diagnostic at `Loc` in `F`, then the compile ends with exit status 1
// and without crash diagnostics.
[[noreturn]] void reportUserError(const llvm::Function &F,
                                  const llvm::DiagnosticLocation &Loc,
                                  const llvm::Twine &Message);
[[noreturn]] void reportUserError(const llvm::Instruction &I,
                                  const llvm::Twine &Message);
// The same for a mistake no site is concerned in: a flag value, a plugin or a
// file of the whole compile.
[[noreturn]] void reportUserError(llvm::LLVMContext &Ctx,
                                  const llvm::Twine &Message);

} // namespace poseidon
#endif // POSEIDON_UTILS_H
