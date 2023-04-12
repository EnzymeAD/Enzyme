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
//===----------------------------------------------------------------------===//

#include "TraceInterface.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

using namespace llvm;

TraceInterface::TraceInterface(LLVMContext &C) : C(C){};

IntegerType *TraceInterface::sizeType(LLVMContext &C) {
  return IntegerType::getInt64Ty(C);
}

Type *TraceInterface::stringType(LLVMContext &C) {
  return IntegerType::getInt8PtrTy(C);
}

FunctionType *TraceInterface::getTraceTy() { return getTraceTy(C); }
FunctionType *TraceInterface::getChoiceTy() { return getChoiceTy(C); }
FunctionType *TraceInterface::getLikelihoodTy() { return getLikelihoodTy(C); }
FunctionType *TraceInterface::insertCallTy() { return insertCallTy(C); }
FunctionType *TraceInterface::insertChoiceTy() { return insertChoiceTy(C); }
FunctionType *TraceInterface::newTraceTy() { return newTraceTy(C); }
FunctionType *TraceInterface::freeTraceTy() { return freeTraceTy(C); }
FunctionType *TraceInterface::hasCallTy() { return hasCallTy(C); }
FunctionType *TraceInterface::hasChoiceTy() { return hasChoiceTy(C); }

FunctionType *TraceInterface::getTraceTy(LLVMContext &C) {
  return FunctionType::get(PointerType::getInt8PtrTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C)},
                           false);
}

FunctionType *TraceInterface::getChoiceTy(LLVMContext &C) {
  return FunctionType::get(sizeType(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C), sizeType(C)},
                           false);
}

FunctionType *TraceInterface::getLikelihoodTy(LLVMContext &C) {
  return FunctionType::get(Type::getDoubleTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C)},
                           false);
}

FunctionType *TraceInterface::insertCallTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C)},
                           false);
}

FunctionType *TraceInterface::insertChoiceTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            Type::getDoubleTy(C), PointerType::getInt8PtrTy(C),
                            sizeType(C)},
                           false);
}

FunctionType *TraceInterface::newTraceTy(LLVMContext &C) {
  return FunctionType::get(PointerType::getInt8PtrTy(C), {}, false);
}

FunctionType *TraceInterface::freeTraceTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C)},
                           false);
}

FunctionType *TraceInterface::hasCallTy(LLVMContext &C) {
  return FunctionType::get(
      Type::getInt1Ty(C), {PointerType::getInt8PtrTy(C), stringType(C)}, false);
}

FunctionType *TraceInterface::hasChoiceTy(LLVMContext &C) {
  return FunctionType::get(
      Type::getInt1Ty(C), {PointerType::getInt8PtrTy(C), stringType(C)}, false);
}

StaticTraceInterface::StaticTraceInterface(Module *M)
    : TraceInterface(M->getContext()) {
  for (auto &&F : M->functions()) {
    if (F.isIntrinsic())
      continue;
    if (F.getName().contains("__enzyme_newtrace")) {
      assert(F.getFunctionType() == newTraceTy());
      newTraceFunction = &F;
    } else if (F.getName().contains("__enzyme_freetrace")) {
      assert(F.getFunctionType() == freeTraceTy());
      freeTraceFunction = &F;
    } else if (F.getName().contains("__enzyme_get_trace")) {
      assert(F.getFunctionType() == getTraceTy());
      getTraceFunction = &F;
    } else if (F.getName().contains("__enzyme_get_choice")) {
      assert(F.getFunctionType() == getChoiceTy());
      getChoiceFunction = &F;
    } else if (F.getName().contains("__enzyme_get_likelihood")) {
      assert(F.getFunctionType() == getLikelihoodTy());
      getLikelihoodFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_call")) {
      assert(F.getFunctionType() == insertCallTy());
      insertCallFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_choice")) {
      assert(F.getFunctionType() == insertChoiceTy());
      insertChoiceFunction = &F;
    } else if (F.getName().contains("__enzyme_has_call")) {
      assert(F.getFunctionType() == hasCallTy());
      hasCallFunction = &F;
    } else if (F.getName().contains("__enzyme_has_choice")) {
      assert(F.getFunctionType() == hasChoiceTy());
      hasChoiceFunction = &F;
    } else if (F.getName().contains(sampleFunctionName)) {
      assert(F.getFunctionType()->getNumParams() >= 3);
      sampleFunction = &F;
    }
  }

  newTraceFunction->addFnAttr("enzyme_notypeanalysis");
  freeTraceFunction->addFnAttr("enzyme_notypeanalysis");
  getTraceFunction->addFnAttr("enzyme_notypeanalysis");
  getChoiceFunction->addFnAttr("enzyme_notypeanalysis");
  getLikelihoodFunction->addFnAttr("enzyme_notypeanalysis");
  insertCallFunction->addFnAttr("enzyme_notypeanalysis");
  insertChoiceFunction->addFnAttr("enzyme_notypeanalysis");
  hasCallFunction->addFnAttr("enzyme_notypeanalysis");
  hasChoiceFunction->addFnAttr("enzyme_notypeanalysis");
  sampleFunction->addFnAttr("enzyme_notypeanalysis");

  newTraceFunction->addFnAttr("enzyme_inactive");
  freeTraceFunction->addFnAttr("enzyme_inactive");
  getTraceFunction->addFnAttr("enzyme_inactive");
  getChoiceFunction->addFnAttr("enzyme_inactive");
  getLikelihoodFunction->addFnAttr("enzyme_inactive");
  insertCallFunction->addFnAttr("enzyme_inactive");
  insertChoiceFunction->addFnAttr("enzyme_inactive");
  hasCallFunction->addFnAttr("enzyme_inactive");
  hasChoiceFunction->addFnAttr("enzyme_inactive");
  sampleFunction->addFnAttr("enzyme_inactive");

  assert(newTraceFunction);
  assert(freeTraceFunction);
  assert(getTraceFunction);
  assert(getChoiceFunction);
  assert(getLikelihoodFunction);
  assert(insertCallFunction);
  assert(insertChoiceFunction);
  assert(hasCallFunction);
  assert(hasChoiceFunction);
  assert(sampleFunction);
}

// implemented by enzyme
Function *StaticTraceInterface::getSampleFunction() { return sampleFunction; }

// user implemented
Value *StaticTraceInterface::getTrace(IRBuilder<> &Builder) {
  return getTraceFunction;
}
Value *StaticTraceInterface::getChoice(IRBuilder<> &Builder) {
  return getChoiceFunction;
}
Value *StaticTraceInterface::getLikelihood(IRBuilder<> &Builder) {
  return getLikelihoodFunction;
}
Value *StaticTraceInterface::insertCall(IRBuilder<> &Builder) {
  return insertCallFunction;
}
Value *StaticTraceInterface::insertChoice(IRBuilder<> &Builder) {
  return insertChoiceFunction;
}
Value *StaticTraceInterface::newTrace(IRBuilder<> &Builder) {
  return newTraceFunction;
}
Value *StaticTraceInterface::freeTrace(IRBuilder<> &Builder) {
  return freeTraceFunction;
}
Value *StaticTraceInterface::hasCall(IRBuilder<> &Builder) {
  return hasCallFunction;
}
Value *StaticTraceInterface::hasChoice(IRBuilder<> &Builder) {
  return hasChoiceFunction;
}

DynamicTraceInterface::DynamicTraceInterface(Value *dynamicInterface,
                                             Function *F)
    : TraceInterface(F->getContext()) {
  for (auto &&interface_func : F->getParent()->functions()) {
    if (interface_func.getName().contains(TraceInterface::sampleFunctionName)) {
      assert(interface_func.getFunctionType()->getNumParams() >= 3);
      sampleFunction = &interface_func;
    }
  }

  assert(sampleFunction);
  assert(dynamicInterface);

  auto &M = *F->getParent();
  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbg());

  getTraceFunction = MaterializeGetTrace(Builder, dynamicInterface, M);
  getChoiceFunction = MaterializeGetChoice(Builder, dynamicInterface, M);
  getLikelihoodFunction =
      MaterializeGetLikelihood(Builder, dynamicInterface, M);
  insertCallFunction = MaterializeInsertCall(Builder, dynamicInterface, M);
  insertChoiceFunction = MaterializeInsertChoice(Builder, dynamicInterface, M);
  newTraceFunction = MaterializeNewTrace(Builder, dynamicInterface, M);
  freeTraceFunction = MaterializeFreeTrace(Builder, dynamicInterface, M);
  hasCallFunction = MaterializeHasCall(Builder, dynamicInterface, M);
  hasChoiceFunction = MaterializeHasChoice(Builder, dynamicInterface, M);

  assert(getTraceFunction);
  assert(getChoiceFunction);
  assert(getLikelihoodFunction);
  assert(insertCallFunction);
  assert(insertChoiceFunction);
  assert(newTraceFunction);
  assert(freeTraceFunction);
  assert(hasCallFunction);
  assert(hasChoiceFunction);
}

GlobalVariable *
DynamicTraceInterface::MaterializeGetTrace(IRBuilder<> &Builder,
                                           Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(0));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(getTraceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "get_trace");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "get_trace");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *DynamicTraceInterface::MaterializeGetChoice(
    IRBuilder<> &Builder, Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(1));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(getChoiceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "get_choice");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "get_choice");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *DynamicTraceInterface::MaterializeGetLikelihood(
    IRBuilder<> &Builder, Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(8));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(getChoiceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "get_likelihood");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "get_likelihood");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *DynamicTraceInterface::MaterializeInsertCall(
    IRBuilder<> &Builder, Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(8));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(insertCallTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "get_likelihood");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "get_likelihood");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *DynamicTraceInterface::MaterializeInsertChoice(
    IRBuilder<> &Builder, Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(3));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(insertChoiceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "insert_choice");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "insert_choice");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *
DynamicTraceInterface::MaterializeNewTrace(IRBuilder<> &Builder,
                                           Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(4));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(newTraceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "new_trace");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "new_trace");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *DynamicTraceInterface::MaterializeFreeTrace(
    IRBuilder<> &Builder, Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(5));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(freeTraceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "free_trace");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "free_trace");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *
DynamicTraceInterface::MaterializeHasCall(IRBuilder<> &Builder,
                                          Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(6));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(hasCallTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "has_call");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "has_call");
  Builder.CreateStore(cast, global);

  return global;
}

GlobalVariable *DynamicTraceInterface::MaterializeHasChoice(
    IRBuilder<> &Builder, Value *dynamicInterface, Module &M) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(7));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(hasChoiceTy(), load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty, "has_choice");

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), "has_choice");
  Builder.CreateStore(cast, global);

  return global;
}

Function *DynamicTraceInterface::getSampleFunction() { return sampleFunction; }

// user implemented
Value *DynamicTraceInterface::getTrace(IRBuilder<> &Builder) {
  return Builder.CreateLoad(getTraceFunction->getValueType(), getTraceFunction,
                            "get_trace");
}

Value *DynamicTraceInterface::getChoice(IRBuilder<> &Builder) {
  return Builder.CreateLoad(getChoiceFunction->getValueType(),
                            getChoiceFunction, "get_choice");
}

Value *DynamicTraceInterface::getLikelihood(IRBuilder<> &Builder) {
  return Builder.CreateLoad(getLikelihoodFunction->getValueType(),
                            getLikelihoodFunction, "get_likelihood");
}

Value *DynamicTraceInterface::insertCall(IRBuilder<> &Builder) {
  return Builder.CreateLoad(insertCallFunction->getValueType(),
                            insertCallFunction, "insert_call");
}

Value *DynamicTraceInterface::insertChoice(IRBuilder<> &Builder) {
  return Builder.CreateLoad(insertChoiceFunction->getValueType(),
                            insertChoiceFunction, "insert_choice");
}

Value *DynamicTraceInterface::newTrace(IRBuilder<> &Builder) {
  return Builder.CreateLoad(newTraceFunction->getValueType(), newTraceFunction,
                            "new_trace");
}

Value *DynamicTraceInterface::freeTrace(IRBuilder<> &Builder) {
  return Builder.CreateLoad(freeTraceFunction->getValueType(),
                            freeTraceFunction, "free_trace");
}

Value *DynamicTraceInterface::hasCall(IRBuilder<> &Builder) {
  return Builder.CreateLoad(hasCallFunction->getValueType(), hasCallFunction,
                            "has_call");
}

Value *DynamicTraceInterface::hasChoice(IRBuilder<> &Builder) {
  return Builder.CreateLoad(hasChoiceFunction->getValueType(),
                            hasChoiceFunction, "has_choice");
}
