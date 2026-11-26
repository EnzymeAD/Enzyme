//===- PreprocessUtils.h - Stateless preprocessing steps -------*- C++ -*-===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Steps of Enzyme's function preprocessing that touch no Enzyme state from
// LLVM 16 on, inline so that a pass plugin built against Enzyme's headers can
// call them without linking Enzyme.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_PREPROCESS_UTILS_H
#define ENZYME_PREPROCESS_UTILS_H

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"

#if LLVM_VERSION_MAJOR < 16
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Support/CommandLine.h"

extern "C" {
extern llvm::cl::opt<bool> EnzymeAggressiveAA;
}
#endif

static inline llvm::PassBuilder makePassBuilder(llvm::LLVMContext &Ctx) {
#if LLVM_VERSION_MAJOR >= 24
  return llvm::PassBuilder(Ctx);
#else
  return llvm::PassBuilder();
#endif
}

/// Registers the analyses of preprocessing, with `Ctx` as the PassBuilder's.
static inline void registerPreprocessAnalyses(
    llvm::LoopAnalysisManager &LAM, llvm::FunctionAnalysisManager &FAM,
    llvm::ModuleAnalysisManager &MAM, llvm::LLVMContext &Ctx) {
  using namespace llvm;
  // Explicitly chose AA passes that are stateless
  // and will not be invalidated
  FAM.registerPass([] { return TypeBasedAA(); });
  FAM.registerPass([] { return BasicAA(); });
  MAM.registerPass([] { return GlobalsAA(); });
  // CallGraphAnalysis required for GlobalsAA
  MAM.registerPass([] { return CallGraphAnalysis(); });

  FAM.registerPass([] { return ScopedNoAliasAA(); });

  // SCEVAA causes some breakage/segfaults
  // disable for now, consider enabling in future
  // FAM.registerPass([] { return SCEVAA(); });

#if LLVM_VERSION_MAJOR < 16
  if (EnzymeAggressiveAA)
    FAM.registerPass([] { return CFLSteensAA(); });
#endif

  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });
  FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });

  FAM.registerPass([] {
    auto AM = AAManager();
    AM.registerFunctionAnalysis<BasicAA>();
    AM.registerFunctionAnalysis<TypeBasedAA>();
    AM.registerModuleAnalysis<GlobalsAA>();
    AM.registerFunctionAnalysis<ScopedNoAliasAA>();

    // broken for different reasons
    // AM.registerFunctionAnalysis<SCEVAA>();

#if LLVM_VERSION_MAJOR < 16
    if (EnzymeAggressiveAA)
      AM.registerFunctionAnalysis<CFLSteensAA>();
#endif

    return AM;
  });

  PassBuilder PB = makePassBuilder(Ctx);
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
}

static inline void RemoveRedundantPHI(llvm::Function *F,
                                      llvm::FunctionAnalysisManager &FAM) {
  using namespace llvm;
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  for (BasicBlock &BB : *F) {
    for (BasicBlock::iterator II = BB.begin(); isa<PHINode>(II);) {
      PHINode *PN = cast<PHINode>(II);
      ++II;
      SmallPtrSet<Value *, 2> vals;
      SmallPtrSet<PHINode *, 2> done;
      SmallVector<PHINode *, 2> todo = {PN};
      while (todo.size() > 0) {
        PHINode *N = todo.back();
        todo.pop_back();
        if (done.count(N))
          continue;
        done.insert(N);
        if (vals.size() == 0 && todo.size() == 0 && PN != N &&
            DT.dominates(N, PN)) {
          vals.insert(N);
          break;
        }
        for (auto &v : N->incoming_values()) {
          if (isa<UndefValue>(v))
            continue;
          if (auto NN = dyn_cast<PHINode>(v)) {
            todo.push_back(NN);
            continue;
          }
          vals.insert(v);
          if (vals.size() > 1)
            break;
        }
        if (vals.size() > 1)
          break;
      }
      if (vals.size() == 1) {
        auto V = *vals.begin();
        if (!isa<Instruction>(V) || DT.dominates(cast<Instruction>(V), PN)) {
          PN->replaceAllUsesWith(V);
          PN->eraseFromParent();
        }
      }
    }
  }
}

#endif // ENZYME_PREPROCESS_UTILS_H
