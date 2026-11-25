//===- Utils.cpp - IR helpers of the Poseidon pass ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

using namespace llvm;

namespace poseidon {

unsigned inlineCallsWhile(Function &F, function_ref<bool(CallBase &)> pick) {
  constexpr unsigned kMaxInlineRounds = 8;
  unsigned inlined = 0;
  for (unsigned round = 0; round < kMaxInlineRounds; ++round) {
    SmallVector<CallBase *, 4> todo;
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (pick(*CB))
          todo.push_back(CB);
    if (todo.empty())
      break;
    for (CallBase *CB : todo) {
      InlineFunctionInfo IFI;
      if (InlineFunction(*CB, IFI).isSuccess())
        ++inlined;
    }
  }
  return inlined;
}

void simplifyFunction(Function &F, PassBuilder &PB, OptimizationLevel Level,
                      function_ref<void()> afterPass) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  // The compile's pass instrumentation (printing, bisection, timing) stays
  // out of these runs; registered first, this analysis is the one kept.
  PassInstrumentationCallbacks PIC;
  if (afterPass) {
    PIC.registerAfterPassCallback(
        [&](StringRef, auto, const PreservedAnalyses &) { afterPass(); });
    PIC.registerAfterPassInvalidatedCallback(
        [&](StringRef, const PreservedAnalyses &) { afterPass(); });
  }
  LAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });
  CGAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });
  MAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  if (verifyFunction(F, &llvm::errs())) {
    F.print(llvm::errs());
    reportFatalInternalError("Poseidon: a rewritten function does not verify");
  }

  FunctionPassManager FPM =
      PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
  (void)FPM.run(F, FAM);
}

PassBuilder makePassBuilder(LLVMContext &C) {
#if LLVM_VERSION_MAJOR >= 24
  return PassBuilder(C);
#else
  return PassBuilder();
#endif
}

void simplifyFunction(Function &F, OptimizationLevel Level) {
  PassBuilder PB = makePassBuilder(F.getContext());
  simplifyFunction(F, PB, Level);
}

SCEVExpander makeSCEVExpander(ScalarEvolution &SE, const char *name) {
#if LLVM_VERSION_MAJOR >= 22 && !defined(ENZYME_LLVM22_PRERELEASE)
  return SCEVExpander(SE, name);
#else
  return SCEVExpander(SE, SE.getDataLayout(), name);
#endif
}

std::string shortDouble(double v) {
  std::string s;
  raw_string_ostream(s) << format("%g", v);
  return s;
}

std::string exactDouble(double v) {
  std::string s;
  raw_string_ostream(s) << format("%.17g", v);
  return s;
}

bool callsFunctionNamed(const Function &F, StringRef name) {
  return any_of(instructions(F), [&](const Instruction &I) {
    auto *CI = dyn_cast<CallInst>(&I);
    const Function *callee = CI ? CI->getCalledFunction() : nullptr;
    return callee && callee->getName() == name;
  });
}

std::string displayName(const Function &F) {
  return demangle(F.getName().str());
}

void topoSort(const SetVector<Instruction *> &insts,
              SmallVectorImpl<Instruction *> &instsSorted) {
  SmallPtrSet<Instruction *, 8> visited;
  SmallPtrSet<Instruction *, 8> onStack;

  std::function<void(Instruction *)> dfsVisit = [&](Instruction *I) {
    if (!visited.insert(I).second)
      return;
    onStack.insert(I);

    for (Value *op : fpOperands(*I)) {
      if (auto *oI = dyn_cast<Instruction>(op)) {
        if (insts.contains(oI)) {
          assert(!onStack.count(oI) &&
                 "a subgraph's instructions depend on each other in a cycle");
          dfsVisit(oI);
        }
      }
    }

    onStack.erase(I);
    instsSorted.push_back(I);
  };

  for (Instruction *I : insts)
    dfsVisit(I);
}

} // namespace poseidon
