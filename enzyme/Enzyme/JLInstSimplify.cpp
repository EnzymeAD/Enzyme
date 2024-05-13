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

#include "llvm/Support/Debug.h"

#include "llvm/Analysis/TargetLibraryInfo.h"

#include "JLInstSimplify.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "jl-inst-simplify"
namespace {

bool jlInstSimplify(llvm::Function &F, TargetLibraryInfo &TLI,
                    llvm::AAResults &AA, llvm::LoopInfo &LI) {
  bool changed = false;

  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto cmp = dyn_cast<ICmpInst>(&I)) {
        if (cmp->use_empty())
          continue;
        auto lhs = getBaseObject(cmp->getOperand(0), /*offsetAllowed*/ false);
        auto rhs = getBaseObject(cmp->getOperand(1), /*offsetAllowed*/ false);
        if (lhs == rhs) {
          cmp->replaceAllUsesWith(cmp->isTrueWhenEqual()
                                      ? ConstantInt::getTrue(F.getContext())
                                      : ConstantInt::getFalse(F.getContext()));
          changed = true;
          continue;
        }
        if ((isNoAlias(lhs) && (isNoAlias(rhs) || isa<Argument>(rhs))) ||
            (isNoAlias(rhs) && isa<Argument>(lhs))) {
          cmp->replaceAllUsesWith(cmp->isTrueWhenEqual()
                                      ? ConstantInt::getFalse(F.getContext())
                                      : ConstantInt::getTrue(F.getContext()));
          changed = true;
          continue;
        }
        auto llhs = dyn_cast<LoadInst>(lhs);
        auto lrhs = dyn_cast<LoadInst>(rhs);
        if (llhs && lrhs) {
          auto lhsv =
              getBaseObject(llhs->getOperand(0), /*offsetAllowed*/ false);
          auto rhsv =
              getBaseObject(lrhs->getOperand(0), /*offsetAllowed*/ false);
          if ((isNoAlias(lhsv) && (isNoAlias(rhsv) || isa<Argument>(rhsv))) ||
              (isNoAlias(rhsv) && isa<Argument>(lhsv))) {
            bool legal = false;
            for (int i = 0; i < 2; i++) {
              Value *start = (i == 0) ? lhsv : rhsv;
              Instruction *starti = dyn_cast<Instruction>(start);
              if (!starti) {
                assert(isa<Argument>(starti));
                starti = &cast<Argument>(starti)
                              ->getParent()
                              ->getEntryBlock()
                              .front();
              }

              bool overwritten = false;
              allInstructionsBetween(
                  LI, starti, cmp, [&](Instruction *I) -> bool {
                    if (!I->mayWriteToMemory())
                      return /*earlyBreak*/ false;

                    for (auto LI : {llhs, lrhs})
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
              cmp->replaceAllUsesWith(
                  cmp->isTrueWhenEqual()
                      ? ConstantInt::getFalse(F.getContext())
                      : ConstantInt::getTrue(F.getContext()));
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

extern "C" FunctionPass *createJLInstSimplifyPass() {
  return new JLInstSimplify();
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
