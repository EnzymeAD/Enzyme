//===- TraceUtils.h - Utilites for interacting with traces  ---------------===//
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
// This file contains utilities for interacting with probabilistic programming
// traces using the probabilistic programming
// trace interface
//
//===----------------------------------------------------------------------===//

#ifndef TraceUtils_h
#define TraceUtils_h

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"

#include "TraceInterface.h"
#include "Utils.h"

class TraceUtils {

private:
  llvm::Value *trace;
  llvm::Value *observations;
  llvm::Value *likelihood;

public:
  TraceInterface *interface;
  ProbProgMode mode;
  llvm::Function *newFunc;
  llvm::SmallPtrSet<llvm::Function *, 4> sampleFunctions;
  llvm::SmallPtrSet<llvm::Function *, 4> observeFunctions;

  constexpr static const char TraceParameterAttribute[] = "enzyme_trace";
  constexpr static const char ObservationsParameterAttribute[] =
      "enzyme_observations";
  constexpr static const char LikelihoodParameterAttribute[] =
      "enzyme_likelihood";

public:
  TraceUtils(ProbProgMode mode,
             const llvm::SmallPtrSetImpl<llvm::Function *> &sampleFunctions,
             const llvm::SmallPtrSetImpl<llvm::Function *> &observeFunctions,
             llvm::Function *newFunc, llvm::Argument *trace,
             llvm::Argument *observations, llvm::Argument *likelihood,
             TraceInterface *interface);

  static TraceUtils *
  FromClone(ProbProgMode mode,
            const llvm::SmallPtrSetImpl<llvm::Function *> &sampleFunctions,
            const llvm::SmallPtrSetImpl<llvm::Function *> &observeFunctions,
            TraceInterface *interface, llvm::Function *oldFunc,
            llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH>
                &originalToNewFn);

  ~TraceUtils();

private:
  static std::pair<llvm::Value *, llvm::Constant *>
  ValueToVoidPtrAndSize(llvm::IRBuilder<> &Builder, llvm::Value *val,
                        llvm::Type *size_type);

public:
  TraceInterface *getTraceInterface();

  llvm::Value *getTrace();

  llvm::Value *getObservations();

  llvm::Value *getLikelihood();

  llvm::CallInst *CreateTrace(llvm::IRBuilder<> &Builder,
                              const llvm::Twine &Name = "trace");

  llvm::CallInst *FreeTrace(llvm::IRBuilder<> &Builder);

  llvm::CallInst *InsertChoice(llvm::IRBuilder<> &Builder, llvm::Value *address,
                               llvm::Value *score, llvm::Value *choice);

  llvm::CallInst *InsertCall(llvm::IRBuilder<> &Builder, llvm::Value *address,
                             llvm::Value *subtrace);

  llvm::CallInst *InsertArgument(llvm::IRBuilder<> &Builder, llvm::Value *name,
                                 llvm::Value *argument);

  llvm::CallInst *InsertReturn(llvm::IRBuilder<> &Builder, llvm::Value *ret);

  llvm::CallInst *InsertFunction(llvm::IRBuilder<> &Builder,
                                 llvm::Function *function);

  static llvm::CallInst *
  InsertChoiceGradient(llvm::IRBuilder<> &Builder,
                       llvm::FunctionType *interface_type,
                       llvm::Value *interface_function, llvm::Value *address,
                       llvm::Value *choice, llvm::Value *trace);

  static llvm::CallInst *
  InsertArgumentGradient(llvm::IRBuilder<> &Builder,
                         llvm::FunctionType *interface_type,
                         llvm::Value *interface_function, llvm::Value *name,
                         llvm::Value *argument, llvm::Value *trace);

  llvm::CallInst *GetTrace(llvm::IRBuilder<> &Builder, llvm::Value *address,
                           const llvm::Twine &Name = "");

  llvm::Instruction *GetChoice(llvm::IRBuilder<> &Builder, llvm::Value *address,
                               llvm::Type *choiceType,
                               const llvm::Twine &Name = "");

  llvm::Instruction *HasChoice(llvm::IRBuilder<> &Builder, llvm::Value *address,
                               const llvm::Twine &Name = "");

  llvm::Instruction *HasCall(llvm::IRBuilder<> &Builder, llvm::Value *address,
                             const llvm::Twine &Name = "");

  llvm::Instruction *
  SampleOrCondition(llvm::IRBuilder<> &Builder, llvm::Function *sample_fn,
                    llvm::ArrayRef<llvm::Value *> sample_args,
                    llvm::Value *address, const llvm::Twine &Name = "");

  llvm::CallInst *CreateOutlinedFunction(
      llvm::IRBuilder<> &Builder,
      llvm::function_ref<void(llvm::IRBuilder<> &, TraceUtils *,
                              llvm::ArrayRef<llvm::Value *>)>
          Outlined,
      llvm::Type *RetTy, llvm::ArrayRef<llvm::Value *> Arguments,
      bool needsLikelihood = true, const llvm::Twine &Name = "");

  bool isSampleCall(llvm::CallInst *call);

  bool isObserveCall(llvm::CallInst *call);
};

#endif /* TraceUtils_h */
