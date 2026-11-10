//=- Utils.cpp - Utility functions for Poseidon optimization pass ----------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for the Poseidon floating-point
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <mpfr.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "llvm/Support/Debug.h"

#include "Core/Options.h"
#include "Herbie.h"
#include "Targets/Target.h"
#include "Types.h"
#include "Utils.h"

#define DEBUG_TYPE "poseidon"

using namespace llvm;

namespace poseidon {

void simplifyFunction(Function &F, OptimizationLevel Level) {

#if LLVM_VERSION_MAJOR >= 24
  PassBuilder PB(F.getContext());
#else
  PassBuilder PB;
#endif
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  if (verifyFunction(F, &llvm::errs())) {
    F.print(llvm::errs());
    llvm_unreachable("Poseidon intermediate function failed verification");
  }

  FunctionPassManager FPM =
      PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
  (void)FPM.run(F, FAM);
}

const std::unordered_set<std::string> &libmFuncs() {
  static const std::unordered_set<std::string> LibmFuncs = {
      "sin",      "cos",   "tan",   "asin",     "acos",  "atan",   "atan2",
      "sinh",     "cosh",  "tanh",  "asinh",    "acosh", "atanh",  "exp",
      "log",      "sqrt",  "cbrt",  "pow",      "powi",  "fabs",   "fma",
      "hypot",    "expm1", "log1p", "ceil",     "floor", "erf",    "exp2",
      "lgamma",   "log10", "log2",  "rint",     "round", "tgamma", "trunc",
      "copysign", "fdim",  "fmod",  "remainder"};
  return LibmFuncs;
}

double stringToDouble(const std::string &str) {
  char *end;
  errno = 0;
  double result = std::strtod(str.c_str(), &end);

  if (errno == ERANGE) {
    if (result == HUGE_VAL) {
      result = std::numeric_limits<double>::infinity();
    } else if (result == -HUGE_VAL) {
      result = -std::numeric_limits<double>::infinity();
    }
  }

  return result; // Denormalized values are fine
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
      if (isa<Instruction>(op)) {
        Instruction *oI = cast<Instruction>(op);
        if (insts.contains(oI)) {
          if (onStack.count(oI)) {
            llvm_unreachable(
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

void getUniqueArgs(const std::string &expr, SmallSet<std::string, 8> &args) {
  std::regex argPattern("v\\d+");

  std::sregex_iterator begin(expr.begin(), expr.end(), argPattern);
  std::sregex_iterator end;

  while (begin != end) {
    args.insert(begin->str());
    ++begin;
  }
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
