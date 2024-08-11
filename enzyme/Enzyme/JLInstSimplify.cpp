//=- JLInstSimplify.h - Additional instsimplifyrules for julia programs =//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains a utility LLVM pass for printing derived Activity Analysis
// results of a given function.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Support/Debug.h"

#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm-c/Core.h"
#include "llvm-c/DataTypes.h"

#include "llvm-c/ExternC.h"
#include "llvm-c/Types.h"

#include "JLInstSimplify.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "jl-inst-simplify"
namespace {

bool notCapturedBefore(llvm::Value *V, Instruction *inst) {
  Instruction *VI = dyn_cast<Instruction>(V);
  if (!VI)
    VI = &*inst->getParent()->getParent()->getEntryBlock().begin();
  else
    VI = VI->getNextNode();
  SmallPtrSet<BasicBlock *, 1> regionBetween;
  {
    SmallVector<BasicBlock *, 1> todo;
    todo.push_back(VI->getParent());
    while (todo.size()) {
      auto cur = todo.pop_back_val();
      if (regionBetween.count(cur))
        continue;
      regionBetween.insert(cur);
      if (cur == inst->getParent())
        continue;
      for (auto BB : successors(cur))
        todo.push_back(BB);
    }
  }
  SmallVector<Value *, 1> todo = {V};
  SmallPtrSet<Value *, 1> seen;
  while (todo.size()) {
    auto cur = todo.pop_back_val();
    if (seen.count(cur))
      continue;
    for (auto U : cur->users()) {
      auto UI = dyn_cast<Instruction>(U);
      if (!regionBetween.count(UI->getParent()))
        continue;
      if (UI->getParent() == VI->getParent()) {
        if (UI->comesBefore(VI))
          continue;
      }
      if (UI->getParent() == inst->getParent())
        if (inst->comesBefore(UI))
          continue;

      if (auto CI = dyn_cast<CallBase>(UI)) {
        auto fname = getFuncNameFromCall(CI);
        if (fname == "julia.pointer_from_objref")
          continue;
#if LLVM_VERSION_MAJOR >= 14
        for (size_t i = 0, size = CI->arg_size(); i < size; i++)
#else
        for (size_t i = 0, size = CI->getNumArgOperands(); i < size; i++)
#endif
        {
          if (cur == CI->getArgOperand(i)) {
            if (isNoCapture(CI, i))
              continue;
            return false;
          }
        }
        return true;
      }

      if (isa<CmpInst>(UI)) {
        continue;
      }
      if (isa<CastInst, GetElementPtrInst, LoadInst, PHINode>(UI)) {
        todo.push_back(UI);
        continue;
      }
      return false;
    }
  }
  return true;
}

static inline SetVector<llvm::Value *> getBaseObjects(llvm::Value *V,
                                                      bool offsetAllowed) {
  SetVector<llvm::Value *> results;

  SmallPtrSet<llvm::Value *, 2> seen;
  SmallVector<llvm::Value *, 1> todo = {V};

  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (seen.count(cur))
      continue;
    seen.insert(cur);
    auto obj = getBaseObject(cur, offsetAllowed);
    if (auto PN = dyn_cast<PHINode>(obj)) {
      for (auto &val : PN->incoming_values()) {
        todo.push_back(val);
      }
      continue;
    }
    if (auto SI = dyn_cast<SelectInst>(obj)) {
      todo.push_back(SI->getTrueValue());
      todo.push_back(SI->getFalseValue());
      continue;
    }
    results.insert(obj);
  }
  return results;
}

bool noaliased_or_arg(SetVector<llvm::Value *> &lhs_v,
                      SetVector<llvm::Value *> &rhs_v) {
  for (auto lhs : lhs_v) {
    auto lhs_na = isNoAlias(lhs);
    auto lhs_arg = isa<Argument>(lhs);

    // This LHS value is neither noalias or an argument
    if (!lhs_na && !lhs_arg)
      return false;

    for (auto rhs : rhs_v) {
      if (lhs == rhs)
        return false;
      if (isNoAlias(lhs))
        continue;
      if (!lhs_na && !isa<Argument>(rhs))
        return false;
    }
  }
  return true;
}

bool jlInstSimplify(llvm::Function &F, TargetLibraryInfo &TLI,
                    llvm::AAResults &AA, llvm::LoopInfo &LI) {
  bool changed = false;

  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto FI = dyn_cast<FreezeInst>(&I)) {
        if (FI->hasOneUse()) {
          bool allBranch = true;
          for (auto user : FI->users()) {
            if (!isa<BranchInst>(user)) {
              allBranch = false;
              break;
            }
          }
          if (allBranch) {
            FI->replaceAllUsesWith(FI->getOperand(0));
            changed = true;
            continue;
          }
        }
      }
      if (I.use_empty())
        continue;

      bool legal = false;
      ICmpInst::Predicate pred;
      if (auto cmp = dyn_cast<ICmpInst>(&I)) {
        pred = cmp->getPredicate();
        legal = true;
      } else if (auto CI = dyn_cast<CallBase>(&I)) {
        if (getFuncNameFromCall(CI) == "jl_mightalias") {
#if LLVM_VERSION_MAJOR >= 14
          size_t numargs = CI->arg_size();
#else
          size_t numargs = CI->getNumArgOperands();
#endif
          if (numargs == 2 && isa<PointerType>(I.getOperand(0)->getType()) &&
              isa<PointerType>(I.getOperand(0)->getType())) {
            legal = true;
            pred = ICmpInst::Predicate::ICMP_EQ;
          }
        }
      }

      if (legal) {
        auto lhs_v = getBaseObjects(I.getOperand(0), /*offsetAllowed*/ false);
        auto rhs_v = getBaseObjects(I.getOperand(1), /*offsetAllowed*/ false);
        if (lhs_v.size() == 1 && rhs_v.size() == 1 && lhs_v[0] == rhs_v[0]) {
          auto repval = ICmpInst::isTrueWhenEqual(pred)
                            ? ConstantInt::get(I.getType(), 1)
                            : ConstantInt::get(I.getType(), 0);
          I.replaceAllUsesWith(repval);
          changed = true;
          continue;
        }
        if (noaliased_or_arg(lhs_v, rhs_v)) {
          auto repval = ICmpInst::isTrueWhenEqual(pred)
                            ? ConstantInt::get(I.getType(), 0)
                            : ConstantInt::get(I.getType(), 1);
          I.replaceAllUsesWith(repval);
          changed = true;
          continue;
        }
        bool loadlegal = true;
        SmallVector<LoadInst *, 1> llhs, lrhs;
        for (auto lhs : lhs_v) {
          auto ld = dyn_cast<LoadInst>(lhs);
          if (!ld || !isa<PointerType>(ld->getType())) {
            loadlegal = false;
            break;
          }
          llhs.push_back(ld);
        }
        for (auto rhs : rhs_v) {
          auto ld = dyn_cast<LoadInst>(rhs);
          if (!ld || !isa<PointerType>(ld->getType())) {
            loadlegal = false;
            break;
          }
          lrhs.push_back(ld);
        }
        SetVector<Value *> llhs_s, lrhs_s;
        for (auto v : llhs) {
          for (auto obj :
               getBaseObjects(v->getOperand(0), /*offsetAllowed*/ false)) {
            llhs_s.insert(obj);
          }
        }
        for (auto v : lrhs) {
          for (auto obj :
               getBaseObjects(v->getOperand(0), /*offsetAllowed*/ false)) {
            lrhs_s.insert(obj);
          }
        }
        // TODO handle multi size
        if (llhs_s.size() == 1 && lrhs_s.size() == 1 && loadlegal) {
          auto lhsv = llhs_s[0];
          auto rhsv = lrhs_s[0];
          if ((isNoAlias(lhsv) && (isNoAlias(rhsv) || isa<Argument>(rhsv) ||
                                   notCapturedBefore(lhsv, &I))) ||
              (isNoAlias(rhsv) &&
               (isa<Argument>(lhsv) || notCapturedBefore(rhsv, &I)))) {
            bool legal = false;
            for (int i = 0; i < 2; i++) {
              Value *start = (i == 0) ? lhsv : rhsv;
              Instruction *starti = dyn_cast<Instruction>(start);
              if (!starti) {
                if (!isa<Argument>(start))
                  continue;
                starti = &cast<Argument>(start)
                              ->getParent()
                              ->getEntryBlock()
                              .front();
              }

              bool overwritten = false;
              allInstructionsBetween(
                  LI, starti, &I, [&](Instruction *I) -> bool {
                    if (!I->mayWriteToMemory())
                      return /*earlyBreak*/ false;

                    for (auto LI : llhs)
                      if (writesToMemoryReadBy(AA, TLI,
                                               /*maybeReader*/ LI,
                                               /*maybeWriter*/ I)) {
                        overwritten = true;
                        return /*earlyBreak*/ true;
                      }
                    for (auto LI : lrhs)
                      if (writesToMemoryReadBy(AA, TLI,
                                               /*maybeReader*/ LI,
                                               /*maybeWriter*/ I)) {
                        overwritten = true;
                        return /*earlyBreak*/ true;
                      }
                    return /*earlyBreak*/ false;
                  });
              if (!overwritten) {
                legal = true;
                break;
              }
            }

            if (legal && lhsv != rhsv) {
              auto repval = ICmpInst::isTrueWhenEqual(pred)
                                ? ConstantInt::get(I.getType(), 0)
                                : ConstantInt::get(I.getType(), 1);
              I.replaceAllUsesWith(repval);
              changed = true;
              continue;
            }
          }
        }
      }
    }
  return changed;
}

class JLInstSimplify final : public FunctionPass {
public:
  static char ID;
  JLInstSimplify() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    return jlInstSimplify(F, TLI, AA, LI);
  }
};

} // namespace

FunctionPass *createJLInstSimplifyPass() { return new JLInstSimplify(); }

extern "C" void LLVMAddJLInstSimplifyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createJLInstSimplifyPass());
}

char JLInstSimplify::ID = 0;

static RegisterPass<JLInstSimplify> X("jl-inst-simplify",
                                      "JL instruction simplification");

JLInstSimplifyNewPM::Result
JLInstSimplifyNewPM::run(llvm::Function &F,
                         llvm::FunctionAnalysisManager &FAM) {
  bool changed = false;
  changed = jlInstSimplify(F, FAM.getResult<TargetLibraryAnalysis>(F),
                           FAM.getResult<AAManager>(F),
                           FAM.getResult<LoopAnalysis>(F));
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey JLInstSimplifyNewPM::Key;
