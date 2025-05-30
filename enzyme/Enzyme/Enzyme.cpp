//===- Enzyme.cpp - Automatic Differentiation Transformation Pass  -------===//
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
// This file contains Enzyme, a transformation pass that takes replaces calls
// to function calls to *__enzyme_autodiff* with a call to the derivative of
// the function passed as the first argument.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>
#include <memory>

#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <optional>
#if LLVM_VERSION_MAJOR <= 16
#include "llvm/ADT/Optional.h"
#endif
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-reactant-intrinsic"


class ReactantBase {
public:
  ReactantBase(bool PostOpt) {
  }

  bool run(Module &M) {
    bool changed = true;

    for (Function &F : make_early_inc_range(M)) {
      if (!F.empty()) continue;
      if (F.getName() == "cudaMalloc") {
        auto entry = BasicBlock::Create(F.getContext(), "entry", &F);
        IRBuilder<> B(entry);

        auto entry = new BasicBlock()
        F.ad
      }
    }

    return changed;
  }
};

class ReactantOldPM : public ReactantBase, public ModulePass {
public:
  static char ID;
  EnzymeOldPM(bool PostOpt = false) : ReactantBase(PostOpt), ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();

    // AU.addRequiredID(LCSSAID);

    // LoopInfo is required to ensure that all loops have preheaders
    // AU.addRequired<LoopInfoWrapperPass>();

    // AU.addRequiredID(llvm::LoopSimplifyID);//<LoopSimplifyWrapperPass>();
  }
  bool runOnModule(Module &M) override { return run(M); }
};

} // namespace

char ReactantOldPM::ID = 0;

static RegisterPass<ReactantOldPM> X("enzyme", "Enzyme Pass");

ModulePass *createReactantPass(bool PostOpt) { return new EnzymeOldPM(PostOpt); }

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddReactantPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createReactantPass(/*PostOpt*/ false));
}

#include "llvm/Passes/PassPlugin.h"

class ReactantNewPM final : public ReactantBase,
                          public AnalysisInfoMixin<ReactantNewPM> {
  friend struct llvm::AnalysisInfoMixin<ReactantNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  ReactantNewPM(bool PostOpt = false) : ReactantBase(PostOpt) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    return ReactantBase::run(M) ? PreservedAnalyses::none()
                              : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

#undef DEBUG_TYPE
AnalysisKey ReactantNewPM::Key;

#include "ActivityAnalysisPrinter.h"
#include "JLInstSimplify.h"
#include "PreserveNVVM.h"
#include "TypeAnalysis/TypeAnalysisPrinter.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/CrossDSOCFI.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/GlobalSplit.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SROA.h"
// #include "llvm/Transforms/IPO/MemProfContextDisambiguation.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#if LLVM_VERSION_MAJOR >= 17
#include "llvm/Transforms/Utils/MoveAutoInit.h"
#endif
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopFlatten.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"

void augmentPassBuilder(llvm::PassBuilder &PB) {
#if LLVM_VERSION_MAJOR >= 20
  auto loadPass = [prePass](ModulePassManager &MPM, OptimizationLevel Level,
                            ThinOrFullLTOPhase)
#else
  auto loadPass = [prePass](ModulePassManager &MPM, OptimizationLevel Level)
#endif
  {
    MPM.addPass(ReactantNewPM());
  };

  // TODO need for perf reasons to move Enzyme pass to the pre vectorization.
  PB.registerOptimizerEarlyEPCallback(loadPass);

  auto loadLTO = [preLTOPass, loadPass](ModulePassManager &MPM,
                                        OptimizationLevel Level) {
#if LLVM_VERSION_MAJOR >= 20
    loadPass(MPM, Level, ThinOrFullLTOPhase::None);
#else
    loadPass(MPM, Level);
#endif
  };
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadLTO);
}

extern "C" void registerReactantAndPassPipeline(llvm::PassBuilder &PB,
                                              bool augment = false) {
}

extern "C" void registerReactant(llvm::PassBuilder &PB) {

  PB.registerPipelineParsingCallback(
      [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
         llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "reactant") {
          MPM.addPass(ReactantNewPM());
          return true;
        }
        return false;
      });
  registerReactantAndPassPipeline(PB, /*augment*/ false);
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ReactantNewPM", "v0.1", registerReactant};
}
