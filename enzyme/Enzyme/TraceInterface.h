//===- TraceInterface.h - Interact with probabilistic programming traces
//---===//
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
// This file contains an abstraction for static and dynamic implementations of
// the probabilistic programming interface.
//
//===----------------------------------------------------------------------===//---------------------------------------------------------------------===//

#ifndef TraceInterface_h
#define TraceInterface_h

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

class TraceInterface {
private:
  llvm::LLVMContext &C;

public:
  TraceInterface(llvm::LLVMContext &C);

  virtual ~TraceInterface() = default;

public:
  // implemented by enzyme
  virtual llvm::Function *getSampleFunction() = 0;
  static constexpr const char sampleFunctionName[] = "__enzyme_sample";

  // user implemented
  virtual llvm::Value *getTrace(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *getChoice(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *getLikelihood(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertCall(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertChoice(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *newTrace(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *freeTrace(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *hasCall(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *hasChoice(llvm::IRBuilder<> &Builder) = 0;

public:
  static llvm::IntegerType *sizeType(llvm::LLVMContext &C);
  static llvm::Type *stringType(llvm::LLVMContext &C);

public:
  llvm::FunctionType *getTraceTy();
  llvm::FunctionType *getChoiceTy();
  llvm::FunctionType *getLikelihoodTy();
  llvm::FunctionType *insertCallTy();
  llvm::FunctionType *insertChoiceTy();
  llvm::FunctionType *newTraceTy();
  llvm::FunctionType *freeTraceTy();
  llvm::FunctionType *hasCallTy();
  llvm::FunctionType *hasChoiceTy();

  static llvm::FunctionType *getTraceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *getChoiceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *getLikelihoodTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertCallTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertChoiceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *newTraceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *freeTraceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *hasCallTy(llvm::LLVMContext &C);
  static llvm::FunctionType *hasChoiceTy(llvm::LLVMContext &C);
};

class StaticTraceInterface final : public TraceInterface {
private:
  llvm::Function *sampleFunction = nullptr;
  // user implemented
  llvm::Function *getTraceFunction = nullptr;
  llvm::Function *getChoiceFunction = nullptr;
  llvm::Function *getLikelihoodFunction = nullptr;
  llvm::Function *insertCallFunction = nullptr;
  llvm::Function *insertChoiceFunction = nullptr;
  llvm::Function *newTraceFunction = nullptr;
  llvm::Function *freeTraceFunction = nullptr;
  llvm::Function *hasCallFunction = nullptr;
  llvm::Function *hasChoiceFunction = nullptr;

public:
  StaticTraceInterface(llvm::Module *M);

  ~StaticTraceInterface() = default;

public:
  // implemented by enzyme
  llvm::Function *getSampleFunction();

  // user implemented
  llvm::Value *getTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *getChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *getLikelihood(llvm::IRBuilder<> &Builder);
  llvm::Value *insertCall(llvm::IRBuilder<> &Builder);
  llvm::Value *insertChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *newTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *freeTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *hasCall(llvm::IRBuilder<> &Builder);
  llvm::Value *hasChoice(llvm::IRBuilder<> &Builder);
};

class DynamicTraceInterface final : public TraceInterface {
private:
  llvm::Function *sampleFunction = nullptr;

private:
  llvm::GlobalVariable *getTraceFunction = nullptr;
  llvm::GlobalVariable *getChoiceFunction = nullptr;
  llvm::GlobalVariable *getLikelihoodFunction = nullptr;
  llvm::GlobalVariable *insertCallFunction = nullptr;
  llvm::GlobalVariable *insertChoiceFunction = nullptr;
  llvm::GlobalVariable *newTraceFunction = nullptr;
  llvm::GlobalVariable *freeTraceFunction = nullptr;
  llvm::GlobalVariable *hasCallFunction = nullptr;
  llvm::GlobalVariable *hasChoiceFunction = nullptr;

public:
  DynamicTraceInterface(llvm::Value *dynamicInterface, llvm::Function *F);

  ~DynamicTraceInterface() = default;

private:
  llvm::GlobalVariable *MaterializeGetTrace(llvm::IRBuilder<> &Builder,
                                            llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeGetChoice(llvm::IRBuilder<> &Builder,
                                             llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeGetLikelihood(llvm::IRBuilder<> &Builder,
                                                 llvm::Value *,
                                                 llvm::Module &M);
  llvm::GlobalVariable *MaterializeInsertCall(llvm::IRBuilder<> &Builder,
                                              llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeInsertChoice(llvm::IRBuilder<> &Builder,
                                                llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeNewTrace(llvm::IRBuilder<> &Builder,
                                            llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeFreeTrace(llvm::IRBuilder<> &Builder,
                                             llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeHasCall(llvm::IRBuilder<> &Builder,
                                           llvm::Value *, llvm::Module &M);
  llvm::GlobalVariable *MaterializeHasChoice(llvm::IRBuilder<> &Builder,
                                             llvm::Value *, llvm::Module &M);

public:
  // implemented by enzyme
  llvm::Function *getSampleFunction();

  // user implemented
  llvm::Value *getTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *getChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *getLikelihood(llvm::IRBuilder<> &Builder);
  llvm::Value *insertCall(llvm::IRBuilder<> &Builder);
  llvm::Value *insertChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *newTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *freeTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *hasCall(llvm::IRBuilder<> &Builder);
  llvm::Value *hasChoice(llvm::IRBuilder<> &Builder);
};

#endif
