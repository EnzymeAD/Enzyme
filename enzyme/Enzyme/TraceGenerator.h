//===- TraceGenerator.h - Trace sample statements and calls  --------------===//
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
// This file contains an instruction visitor that generates probabilistic
// programming traces for call sites and sample statements.
//
//===----------------------------------------------------------------------===//

#ifndef TraceGenerator_h
#define TraceGenerator_h

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"

#include "EnzymeLogic.h"
#include "TraceUtils.h"
#include "Utils.h"

class TraceGenerator final : public llvm::InstVisitor<TraceGenerator> {
private:
  EnzymeLogic &Logic;
  TraceUtils *const tutils;
  ProbProgMode mode = tutils->mode;
  bool autodiff;
  llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH> &originalToNewFn;
  const llvm::SmallPtrSetImpl<llvm::Function *> &generativeFunctions;
  const llvm::StringSet<> &activeRandomVariables;

public:
  TraceGenerator(
      EnzymeLogic &Logic, TraceUtils *tutils, bool autodiff,
      llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH>
          &originalToNewFn,
      const llvm::SmallPtrSetImpl<llvm::Function *> &generativeFunctions,
      const llvm::StringSet<> &activeRandomVariables);

  void visitFunction(llvm::Function &F);

  void handleSampleCall(llvm::CallInst &call, llvm::CallInst *new_call);

  void handleObserveCall(llvm::CallInst &call, llvm::CallInst *new_call);

  void handleArbitraryCall(llvm::CallInst &call, llvm::CallInst *new_call);

  void visitCallInst(llvm::CallInst &call);

  void visitReturnInst(llvm::ReturnInst &ret);
};

#endif /* TraceGenerator_h */
