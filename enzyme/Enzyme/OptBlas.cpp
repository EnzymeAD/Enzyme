//===- PreserveNVVM.cpp - Mark NVVM attributes for preservation.  -------===//
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
// This file contains createPreserveNVVM, a transformation pass that marks
// calls to __nv_* functions, marking them as noinline as implementing the llvm
// intrinsic.
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
public:
  static char ID;
  // bool Begin;
  // OptimizeBlas(bool Begin = true) : ModulePass(ID), Begin(Begin) {}

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

llvm::AnalysisKey OptimizeBlasNewPM::Key;

bool optimizeFncsWithBlas(llvm::Module &M) {

  using namespace llvm;

  Function *F = M.getFunction("f");
  if (F) {
    errs() << "Found function: " << F->getName() << "\n";
  } else {
    return false;
  }

  BasicBlock *bb = &F->getEntryBlock();

  bool firstGer = true;

  Value *m = nullptr;
  Value *n = nullptr;
  Value *p = nullptr;

  Value *x = nullptr;
  Value *y = nullptr;
  Value *v = nullptr;
  Value *w = nullptr;

  Value *incx = nullptr;
  Value *incy = nullptr;
  Value *incv = nullptr;
  Value *incw = nullptr;

  Value *A = nullptr;
  Value *B = nullptr;
  Value *C = nullptr;

  Value *ldc = nullptr;

  Value *alpha = nullptr;
  Value *beta = nullptr;

  // create a vector of calls I will delete after the loop
  std::vector<CallInst *> callsToDelete;

  // Store insertion point for new instructions
  Instruction *insertionPoint = nullptr;

  for (auto &I : *bb) {
    if (auto *call = dyn_cast<CallInst>(&I)) {
      auto name = call->getCalledFunction()->getName();
      if (name.contains("dger_")) {
        errs() << "Found dger_ call\n";
        if (firstGer) {
          m = call->getArgOperand(0);
          n = call->getArgOperand(1);
          alpha = call->getArgOperand(2);
          x = call->getArgOperand(3);
          incx = call->getArgOperand(4);
          y = call->getArgOperand(5);
          incy = call->getArgOperand(6);
          A = call->getArgOperand(7);
          firstGer = false;
          callsToDelete.push_back(call);
          insertionPoint = call;
        } else {
          assert(n == call->getArgOperand(0));

          p = call->getArgOperand(1);
          beta = call->getArgOperand(2);
          v = call->getArgOperand(3);
          incv = call->getArgOperand(4);
          w = call->getArgOperand(5);
          incw = call->getArgOperand(6);
          B = call->getArgOperand(7);
          callsToDelete.push_back(call);
        }
      }
      if (name.contains("dgemm_")) {
        errs() << "Found dgemm_ call\n";
        assert(call->getArgOperand(2) == m);
        assert(call->getArgOperand(3) == n);
        assert(call->getArgOperand(4) == p);
        assert(call->getArgOperand(5) == alpha);
        assert(call->getArgOperand(6) == A);
        assert(call->getArgOperand(8) == B);
        assert(call->getArgOperand(10) == beta);
        C = call->getArgOperand(11);
        ldc = call->getArgOperand(12);
        callsToDelete.push_back(call);
      }
    }
  }
  if (callsToDelete.size() == 0) {
    return false;
  }
  insertionPoint = callsToDelete[0]->getPrevNode();
  for (auto call : callsToDelete) {
    call->eraseFromParent();
  }

  bb->getTerminator()->eraseFromParent();
  FunctionType *FTDot =
      FunctionType::get(Type::getDoubleTy(M.getContext()),
                        {m->getType(), y->getType(), incy->getType(),
                         v->getType(), incv->getType()},
                        false);
  std::string dot_name = "ddot_64_";
  Function *FDot =
      cast<Function>(M.getOrInsertFunction(dot_name, FTDot).getCallee());

  Function *FGer = M.getFunction("dger_64_");
  assert(FGer);

  // bb->setInsertPoint(insertionPoint);


  IRBuilder<> B1(bb);
  Value *dotRet = B1.CreateCall(FDot, {m, y, incy, v, incv});
  Value *alphaDotRet = B1.CreateFMul(alpha, dotRet);
  Value *alphabeta = B1.CreateFMul(alphaDotRet, beta);
  B1.CreateCall(FGer, {m, n, alphabeta, x, incx, w, incw, C, ldc});

  B1.CreateRetVoid();

  return true;
}
