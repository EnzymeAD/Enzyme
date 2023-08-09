//===- OptBlas.cpp - Rewrite BLAS calls for better performance.  -------===//
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
// This file contains code to handle this new blas optimization pass.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"

#include <map>

#include "OptBlas.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "blas-opt"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#endif

bool optimizeBlas(bool Begin, Function &F) {
  bool changed = false;
  return changed;
}

namespace {

class OptimizeBlas final : public ModulePass {
  // class OptimizeBlas final : public PassInfoMixin<OptimizeBlas> {
public:
  static char ID;
  // bool Begin;
  OptimizeBlas() : ModulePass(ID) {}
  OptimizeBlas(char &pid) : ModulePass(pid) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {}
  bool runOnModule(Module &M) override { return optimizeFncsWithBlas(M); }
};

} // namespace

char OptimizeBlas::ID = 0;

static RegisterPass<OptimizeBlas> X("blas-opt", "Optimize Blas Pass");

ModulePass *createOptimizeBlasPass(bool Begin) {
  char pid = 0;
  return new OptimizeBlas(pid);
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddOptimizeBlasPass(LLVMPassManagerRef PM, uint8_t Begin) {
  unwrap(PM)->add(createOptimizeBlasPass((bool)Begin));
}

bool cmp_or_set(llvm::CallInst *CI, std::vector<llvm::Value *> values) {
  // first run trough to see if the already set args match.
  // second run if they do and then we set the nullptr.
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] == nullptr) {
      continue;
    }
    if (CI->getArgOperand(i) != values[i])
      return false;
  }
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] == nullptr) {
      values[i] = CI->getArgOperand(i);
    }
  }
  return true;
}

#include "BlasOpts.inc"

bool optimizeFncsWithBlas(llvm::Module &M) {

  using namespace llvm;

  errs() << "asdf\n";
  Function *F = M.getFunction("f");
  if (F) {
    errs() << "Found function: " << F->getName() << "\n";
  } else {
    return false;
  }

  optfirst(F, M);

  return true;
}

OptimizeBlasNewPM::Result
OptimizeBlasNewPM::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  llvm::errs() << "fooBar\n";
  bool changed = optimizeFncsWithBlas(M);
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey OptimizeBlasNewPM::Key;
