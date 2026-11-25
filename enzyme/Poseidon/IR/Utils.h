//===- Utils.h - IR helpers of the Poseidon pass --------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shims over the LLVM versions the pass builds against, the simplification
// pipeline it runs, how it reports a user error, and small IR helpers.
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
class SCEVExpander;
class ScalarEvolution;
} // namespace llvm

namespace poseidon {

#if LLVM_VERSION_MAJOR >= 23
template <typename DerivedT>
using PassParent = llvm::RequiredPassInfoMixin<DerivedT>;
#else
template <typename DerivedT> using PassParent = llvm::PassInfoMixin<DerivedT>;
#endif

// A PassBuilder without a target machine; LLVM 24 gives it the context.
llvm::PassBuilder makePassBuilder(llvm::LLVMContext &C);

// A SCEVExpander that names the values it emits `name`; LLVM 22 dropped the
// DataLayout argument of its constructor.
llvm::SCEVExpander makeSCEVExpander(llvm::ScalarEvolution &SE,
                                    const char *name);

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
// A double as %g prints it.
std::string shortDouble(double v);
// A double as text that reads back to the same bits, infinity included.
std::string exactDouble(double v);
// The demangled name of `F`.
std::string displayName(const llvm::Function &F);
double literalToDouble(const std::string &str);
void topoSort(const llvm::SetVector<llvm::Instruction *> &insts,
              llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);
void reverseTopoSort(const llvm::SetVector<llvm::Instruction *> &insts,
                     llvm::SmallVectorImpl<llvm::Instruction *> &instsSorted);

void collectExprInsts(llvm::Value *V,
                      const llvm::SetVector<llvm::Value *> &inputs,
                      llvm::SmallPtrSetImpl<llvm::Instruction *> &exprInsts,
                      llvm::SmallPtrSetImpl<llvm::Value *> &visited);

// Inlines the calls of `F` that `pick` selects, in instruction order, and again
// in what that inlined, for a bounded number of rounds so that a recursive
// callee cannot keep it going; returns how many calls it inlined.
unsigned inlineCallsWhile(llvm::Function &F,
                          llvm::function_ref<bool(llvm::CallBase &)> pick);

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

// A diagnostic of `Severity` at `Loc` in `F`, its message prefixed with
// "Poseidon: ".
void diagnose(const llvm::Function &F, const llvm::DiagnosticLocation &Loc,
              const llvm::Twine &Message, llvm::DiagnosticSeverity Severity);

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
