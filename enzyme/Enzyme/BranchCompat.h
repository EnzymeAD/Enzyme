//===- BranchCompat.h - Compatibility shims for branch instructions -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// LLVM replaced the single BranchInst class with two separate classes,
// UncondBrInst and CondBrInst, and removed BranchInst entirely (llvm-project
// 64fc793dd100, 5b4015e55961, 464639bdcc8c). That landed during the LLVM 24
// development cycle, so it cannot be detected from LLVM_VERSION_MAJOR; CMake
// probes for llvm::CondBrInst instead and defines ENZYME_SPLIT_BRANCH_INST.
//
// The two branch classes share no base other than Instruction, so the helpers
// below take and return Instruction*. Instruction already provides
// getNumSuccessors/getSuccessor/setSuccessor for terminators on every
// supported LLVM, which covers most of what BranchInst was used for; the rest
// (the condition, and creating branches) is spelled out here.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_BRANCH_COMPAT_H
#define ENZYME_BRANCH_COMPAT_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

/// The class(es) an isa<>/dyn_cast<> for "some kind of branch" has to name.
#ifdef ENZYME_SPLIT_BRANCH_INST
#define ENZYME_BRANCH_INSTS llvm::UncondBrInst, llvm::CondBrInst
#else
#define ENZYME_BRANCH_INSTS llvm::BranchInst
#endif

/// Whether V is a branch terminator, conditional or not.
inline bool isBranchInst(const llvm::Value *V) {
  return llvm::isa<ENZYME_BRANCH_INSTS>(V);
}

/// Whether V is a conditional branch.
inline bool isCondBranchInst(const llvm::Value *V) {
#ifdef ENZYME_SPLIT_BRANCH_INST
  return llvm::isa<llvm::CondBrInst>(V);
#else
  auto BI = llvm::dyn_cast<llvm::BranchInst>(V);
  return BI && BI->isConditional();
#endif
}

/// Whether V is an unconditional branch.
inline bool isUncondBranchInst(const llvm::Value *V) {
#ifdef ENZYME_SPLIT_BRANCH_INST
  return llvm::isa<llvm::UncondBrInst>(V);
#else
  auto BI = llvm::dyn_cast<llvm::BranchInst>(V);
  return BI && BI->isUnconditional();
#endif
}

/// dyn_cast<BranchInst>(V), as an Instruction* that is null unless V is a
/// branch.
inline llvm::Instruction *asBranchInst(llvm::Value *V) {
  return isBranchInst(V) ? llvm::cast<llvm::Instruction>(V) : nullptr;
}
inline const llvm::Instruction *asBranchInst(const llvm::Value *V) {
  return isBranchInst(V) ? llvm::cast<llvm::Instruction>(V) : nullptr;
}

/// As asBranchInst, but null unless V is a *conditional* branch.
inline llvm::Instruction *asCondBranchInst(llvm::Value *V) {
  return isCondBranchInst(V) ? llvm::cast<llvm::Instruction>(V) : nullptr;
}

/// The condition of a conditional branch. I must be one.
inline llvm::Value *getBranchCondition(const llvm::Instruction *I) {
#ifdef ENZYME_SPLIT_BRANCH_INST
  return llvm::cast<llvm::CondBrInst>(I)->getCondition();
#else
  return llvm::cast<llvm::BranchInst>(I)->getCondition();
#endif
}

/// Replace the condition of a conditional branch. I must be one.
inline void setBranchCondition(llvm::Instruction *I, llvm::Value *V) {
#ifdef ENZYME_SPLIT_BRANCH_INST
  llvm::cast<llvm::CondBrInst>(I)->setCondition(V);
#else
  llvm::cast<llvm::BranchInst>(I)->setCondition(V);
#endif
}

/// An unconditional branch to Dest, inserted before InsertBefore.
inline llvm::Instruction *createUncondBranch(llvm::BasicBlock *Dest,
                                             llvm::Instruction *InsertBefore) {
#if LLVM_VERSION_MAJOR >= 19
  // Instruction* insert positions were dropped along with BranchInst; the
  // iterator overload exists as far back as LLVM 19.
  auto IP = InsertBefore->getIterator();
#else
  auto IP = InsertBefore;
#endif
#ifdef ENZYME_SPLIT_BRANCH_INST
  return llvm::UncondBrInst::Create(Dest, IP);
#else
  return llvm::BranchInst::Create(Dest, IP);
#endif
}

#endif // ENZYME_BRANCH_COMPAT_H
