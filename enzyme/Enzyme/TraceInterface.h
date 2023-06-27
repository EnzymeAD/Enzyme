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
  // user implemented
  virtual llvm::Value *getTrace(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *getChoice(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertCall(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertChoice(llvm::IRBuilder<> &Builder) = 0;

  virtual llvm::Value *insertArgument(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertReturn(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertFunction(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertChoiceGradient(llvm::IRBuilder<> &Builder) = 0;
  virtual llvm::Value *insertArgumentGradient(llvm::IRBuilder<> &Builder) = 0;

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
  llvm::FunctionType *insertCallTy();
  llvm::FunctionType *insertChoiceTy();

  llvm::FunctionType *insertArgumentTy();
  llvm::FunctionType *insertReturnTy();
  llvm::FunctionType *insertFunctionTy();
  llvm::FunctionType *insertChoiceGradientTy();
  llvm::FunctionType *insertArgumentGradientTy();

  llvm::FunctionType *newTraceTy();
  llvm::FunctionType *freeTraceTy();
  llvm::FunctionType *hasCallTy();
  llvm::FunctionType *hasChoiceTy();

  static llvm::FunctionType *getTraceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *getChoiceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertCallTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertChoiceTy(llvm::LLVMContext &C);

  static llvm::FunctionType *insertArgumentTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertReturnTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertFunctionTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertChoiceGradientTy(llvm::LLVMContext &C);
  static llvm::FunctionType *insertArgumentGradientTy(llvm::LLVMContext &C);

  static llvm::FunctionType *newTraceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *freeTraceTy(llvm::LLVMContext &C);
  static llvm::FunctionType *hasCallTy(llvm::LLVMContext &C);
  static llvm::FunctionType *hasChoiceTy(llvm::LLVMContext &C);
};

class StaticTraceInterface final : public TraceInterface {
private:
  llvm::Function *getTraceFunction = nullptr;
  llvm::Function *getChoiceFunction = nullptr;
  llvm::Function *insertCallFunction = nullptr;
  llvm::Function *insertChoiceFunction = nullptr;
  llvm::Function *insertArgumentFunction = nullptr;
  llvm::Function *insertReturnFunction = nullptr;
  llvm::Function *insertFunctionFunction = nullptr;
  llvm::Function *insertChoiceGradientFunction = nullptr;
  llvm::Function *insertArgumentGradientFunction = nullptr;
  llvm::Function *newTraceFunction = nullptr;
  llvm::Function *freeTraceFunction = nullptr;
  llvm::Function *hasCallFunction = nullptr;
  llvm::Function *hasChoiceFunction = nullptr;

public:
  StaticTraceInterface(llvm::Module *M);

  StaticTraceInterface(llvm::LLVMContext &C, llvm::Function *getTraceFunction,
                       llvm::Function *getChoiceFunction,
                       llvm::Function *insertCallFunction,
                       llvm::Function *insertChoiceFunction,
                       llvm::Function *insertArgumentFunction,
                       llvm::Function *insertReturnFunction,
                       llvm::Function *insertFunctionFunction,
                       llvm::Function *insertChoiceGradientFunction,
                       llvm::Function *insertArgumentGradientFunction,
                       llvm::Function *newTraceFunction,
                       llvm::Function *freeTraceFunction,
                       llvm::Function *hasCallFunction,
                       llvm::Function *hasChoiceFunction);

  ~StaticTraceInterface() = default;

public:
  // user implemented
  llvm::Value *getTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *getChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *insertCall(llvm::IRBuilder<> &Builder);
  llvm::Value *insertChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *insertArgument(llvm::IRBuilder<> &Builder);
  llvm::Value *insertReturn(llvm::IRBuilder<> &Builder);
  llvm::Value *insertFunction(llvm::IRBuilder<> &Builder);
  llvm::Value *insertChoiceGradient(llvm::IRBuilder<> &Builder);
  llvm::Value *insertArgumentGradient(llvm::IRBuilder<> &Builder);
  llvm::Value *newTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *freeTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *hasCall(llvm::IRBuilder<> &Builder);
  llvm::Value *hasChoice(llvm::IRBuilder<> &Builder);
};

class DynamicTraceInterface final : public TraceInterface {
private:
  llvm::Function *getTraceFunction;
  llvm::Function *getChoiceFunction;
  llvm::Function *insertCallFunction;
  llvm::Function *insertChoiceFunction;
  llvm::Function *insertArgumentFunction;
  llvm::Function *insertReturnFunction;
  llvm::Function *insertFunctionFunction;
  llvm::Function *insertChoiceGradientFunction;
  llvm::Function *insertArgumentGradientFunction;
  llvm::Function *newTraceFunction;
  llvm::Function *freeTraceFunction;
  llvm::Function *hasCallFunction;
  llvm::Function *hasChoiceFunction;

public:
  DynamicTraceInterface(llvm::Value *dynamicInterface, llvm::Function *F);

  ~DynamicTraceInterface() = default;

private:
  llvm::Function *MaterializeInterfaceFunction(llvm::IRBuilder<> &Builder,
                                               llvm::Value *,
                                               llvm::FunctionType *,
                                               unsigned index, llvm::Module &M,
                                               const llvm::Twine &Name = "");

public:
  // user implemented
  llvm::Value *getTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *getChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *insertCall(llvm::IRBuilder<> &Builder);
  llvm::Value *insertChoice(llvm::IRBuilder<> &Builder);
  llvm::Value *insertArgument(llvm::IRBuilder<> &Builder);
  llvm::Value *insertReturn(llvm::IRBuilder<> &Builder);
  llvm::Value *insertFunction(llvm::IRBuilder<> &Builder);
  llvm::Value *insertChoiceGradient(llvm::IRBuilder<> &Builder);
  llvm::Value *insertArgumentGradient(llvm::IRBuilder<> &Builder);
  llvm::Value *newTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *freeTrace(llvm::IRBuilder<> &Builder);
  llvm::Value *hasCall(llvm::IRBuilder<> &Builder);
  llvm::Value *hasChoice(llvm::IRBuilder<> &Builder);
};

#endif
