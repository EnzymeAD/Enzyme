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
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>

using namespace llvm;

namespace poseidon {

void diagnose(const Function &F, const DiagnosticLocation &Loc,
              const Twine &Message, DiagnosticSeverity Severity) {
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F, "Poseidon: " + Message.str(), Loc, Severity));
}

void reportUserError(const Function &F, const DiagnosticLocation &Loc,
                     const Twine &Message) {
  diagnose(F, Loc, Message, DS_Error);
  // What LLVM's own handler does for a fatal error that is not a crash, short
  // of the crash recovery an in-process clang would turn it into.
  sys::RunInterruptHandlers();
  exit(1);
}

void reportUserError(const Instruction &I, const Twine &Message) {
  reportUserError(*I.getFunction(), I.getDebugLoc(), Message);
}

void reportUserError(LLVMContext &Ctx, const Twine &Message) {
  Ctx.diagnose(DiagnosticInfoGeneric("Poseidon: " + Message, DS_Error));
  sys::RunInterruptHandlers();
  exit(1);
}

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
    report_fatal_error("Poseidon intermediate function failed verification");
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

double stringToDouble(const std::string &str) {
  return std::strtod(str.c_str(), nullptr);
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

std::string displayName(const Function &F) {
  return demangle(F.getName().str());
}

// A Herbie literal is an exact rational whose numerator or denominator alone
// can leave double's range (1e-302 arrives over a 303-digit denominator), so
// the quotient is formed exactly and rounded once.
double literalToDouble(const std::string &str) {
  size_t div = str.find('/');
  if (div == std::string::npos)
    return stringToDouble(str);
  mpfr_prec_t exact = 4 * static_cast<mpfr_prec_t>(str.size()) + 64;
  mpfr_t num, den, quot;
  mpfr_init2(num, exact);
  mpfr_init2(den, exact);
  mpfr_init2(quot, 128);
  mpfr_set_str(num, str.substr(0, div).c_str(), 10, MPFR_RNDN);
  mpfr_set_str(den, str.substr(div + 1).c_str(), 10, MPFR_RNDN);
  mpfr_div(quot, num, den, MPFR_RNDN);
  double result = mpfr_get_d(quot, MPFR_RNDN);
  mpfr_clear(num);
  mpfr_clear(den);
  mpfr_clear(quot);
  return result;
}

void topoSort(const SetVector<Instruction *> &insts,
              SmallVectorImpl<Instruction *> &instsSorted) {
  SmallPtrSet<Instruction *, 8> visited;
  SmallPtrSet<Instruction *, 8> onStack;

  std::function<void(Instruction *)> dfsVisit = [&](Instruction *I) {
    if (visited.count(I))
      return;
    visited.insert(I);
    onStack.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
    for (auto &op : operands) {
      if (auto *oI = dyn_cast<Instruction>(op)) {
        if (insts.contains(oI)) {
          if (onStack.count(oI)) {
            report_fatal_error(
                "topoSort: Cycle detected in instruction dependencies!");
          }
          dfsVisit(oI);
        }
      }
    }

    onStack.erase(I);
    instsSorted.push_back(I);
  };

  for (auto *I : insts) {
    if (!visited.count(I)) {
      dfsVisit(I);
    }
  }
}

void reverseTopoSort(const SetVector<Instruction *> &insts,
                     SmallVectorImpl<Instruction *> &instsSorted) {
  topoSort(insts, instsSorted);
  std::reverse(instsSorted.begin(), instsSorted.end());
}

void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                      SmallPtrSetImpl<Instruction *> &exprInsts,
                      SmallPtrSetImpl<Value *> &visited) {
  if (!V || inputs.contains(V) || visited.contains(V)) {
    return;
  }

  visited.insert(V);

  if (auto *I = dyn_cast<Instruction>(V)) {
    exprInsts.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();

    for (auto &op : operands) {
      collectExprInsts(op, inputs, exprInsts, visited);
    }
  }
}

} // namespace poseidon
