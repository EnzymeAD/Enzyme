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

#include "Utils.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

using namespace llvm;

TraceInterface::TraceInterface(LLVMContext &C) : C(C){};

PointerType *traceType(LLVMContext &C) {
  return getDefaultAnonymousTapeType(C);
}

Type *addressType(LLVMContext &C) { return PointerType::getInt8PtrTy(C); }

IntegerType *TraceInterface::sizeType(LLVMContext &C) {
  return IntegerType::getInt64Ty(C);
}

Type *TraceInterface::stringType(LLVMContext &C) {
  return IntegerType::getInt8PtrTy(C);
}

FunctionType *TraceInterface::getTraceTy() { return getTraceTy(C); }
FunctionType *TraceInterface::getChoiceTy() { return getChoiceTy(C); }
FunctionType *TraceInterface::insertCallTy() { return insertCallTy(C); }
FunctionType *TraceInterface::insertChoiceTy() { return insertChoiceTy(C); }
FunctionType *TraceInterface::insertArgumentTy() { return insertArgumentTy(C); }
FunctionType *TraceInterface::insertReturnTy() { return insertReturnTy(C); }
FunctionType *TraceInterface::insertFunctionTy() { return insertFunctionTy(C); }
FunctionType *TraceInterface::insertChoiceGradientTy() {
  return insertChoiceGradientTy(C);
}
FunctionType *TraceInterface::insertArgumentGradientTy() {
  return insertArgumentGradientTy(C);
}
FunctionType *TraceInterface::newTraceTy() { return newTraceTy(C); }
FunctionType *TraceInterface::freeTraceTy() { return freeTraceTy(C); }
FunctionType *TraceInterface::hasCallTy() { return hasCallTy(C); }
FunctionType *TraceInterface::hasChoiceTy() { return hasChoiceTy(C); }

FunctionType *TraceInterface::getTraceTy(LLVMContext &C) {
  return FunctionType::get(traceType(C), {traceType(C), stringType(C)}, false);
}

FunctionType *TraceInterface::getChoiceTy(LLVMContext &C) {
  return FunctionType::get(
      sizeType(C), {traceType(C), stringType(C), addressType(C), sizeType(C)},
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

FunctionType *TraceInterface::insertArgumentTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C), sizeType(C)},
                           false);
}

FunctionType *TraceInterface::insertReturnTy(LLVMContext &C) {
  return FunctionType::get(
      Type::getVoidTy(C),
      {PointerType::getInt8PtrTy(C), PointerType::getInt8PtrTy(C), sizeType(C)},
      false);
}

FunctionType *TraceInterface::insertFunctionTy(LLVMContext &C) {
  return FunctionType::get(
      Type::getVoidTy(C),
      {PointerType::getInt8PtrTy(C), PointerType::getInt8PtrTy(C)}, false);
}

FunctionType *TraceInterface::insertChoiceGradientTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C), sizeType(C)},
                           false);
}

FunctionType *TraceInterface::insertArgumentGradientTy(LLVMContext &C) {
  return FunctionType::get(Type::getVoidTy(C),
                           {PointerType::getInt8PtrTy(C), stringType(C),
                            PointerType::getInt8PtrTy(C), sizeType(C)},
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
    } else if (F.getName().contains("__enzyme_insert_call")) {
      assert(F.getFunctionType() == insertCallTy());
      insertCallFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_choice")) {
      assert(F.getFunctionType() == insertChoiceTy());
      insertChoiceFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_argument")) {
      assert(F.getFunctionType() == insertArgumentTy());
      insertArgumentFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_return")) {
      assert(F.getFunctionType() == insertReturnTy());
      insertReturnFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_function")) {
      assert(F.getFunctionType() == insertFunctionTy());
      insertFunctionFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_gradient_choice")) {
      assert(F.getFunctionType() == insertChoiceGradientTy());
      insertChoiceGradientFunction = &F;
    } else if (F.getName().contains("__enzyme_insert_gradient_argument")) {
      assert(F.getFunctionType() == insertArgumentGradientTy());
      insertArgumentGradientFunction = &F;
    } else if (F.getName().contains("__enzyme_has_call")) {
      assert(F.getFunctionType() == hasCallTy());
      hasCallFunction = &F;
    } else if (F.getName().contains("__enzyme_has_choice")) {
      assert(F.getFunctionType() == hasChoiceTy());
      hasChoiceFunction = &F;
    }
  }

  assert(newTraceFunction);
  assert(freeTraceFunction);
  assert(getTraceFunction);
  assert(getChoiceFunction);
  assert(insertCallFunction);
  assert(insertChoiceFunction);

  assert(insertArgumentFunction);
  assert(insertReturnFunction);
  assert(insertFunctionFunction);

  assert(insertChoiceGradientFunction);
  assert(insertArgumentGradientFunction);

  assert(hasCallFunction);
  assert(hasChoiceFunction);

  newTraceFunction->addFnAttr("enzyme_notypeanalysis");
  freeTraceFunction->addFnAttr("enzyme_notypeanalysis");
  getTraceFunction->addFnAttr("enzyme_notypeanalysis");
  getChoiceFunction->addFnAttr("enzyme_notypeanalysis");
  insertCallFunction->addFnAttr("enzyme_notypeanalysis");
  insertChoiceFunction->addFnAttr("enzyme_notypeanalysis");
  insertArgumentFunction->addFnAttr("enzyme_notypeanalysis");
  insertReturnFunction->addFnAttr("enzyme_notypeanalysis");
  insertFunctionFunction->addFnAttr("enzyme_notypeanalysis");
  insertChoiceGradientFunction->addFnAttr("enzyme_notypeanalysis");
  insertArgumentGradientFunction->addFnAttr("enzyme_notypeanalysis");
  hasCallFunction->addFnAttr("enzyme_notypeanalysis");
  hasChoiceFunction->addFnAttr("enzyme_notypeanalysis");

  newTraceFunction->addFnAttr("enzyme_inactive");
  freeTraceFunction->addFnAttr("enzyme_inactive");
  getTraceFunction->addFnAttr("enzyme_inactive");
  getChoiceFunction->addFnAttr("enzyme_inactive");
  insertCallFunction->addFnAttr("enzyme_inactive");
  insertChoiceFunction->addFnAttr("enzyme_inactive");
  insertArgumentFunction->addFnAttr("enzyme_inactive");
  insertReturnFunction->addFnAttr("enzyme_inactive");
  insertFunctionFunction->addFnAttr("enzyme_inactive");
  insertChoiceGradientFunction->addFnAttr("enzyme_inactive");
  insertArgumentGradientFunction->addFnAttr("enzyme_inactive");
  hasCallFunction->addFnAttr("enzyme_inactive");
  hasChoiceFunction->addFnAttr("enzyme_inactive");

  newTraceFunction->addFnAttr(Attribute::NoFree);
  getTraceFunction->addFnAttr(Attribute::NoFree);
  getChoiceFunction->addFnAttr(Attribute::NoFree);
  insertCallFunction->addFnAttr(Attribute::NoFree);
  insertChoiceFunction->addFnAttr(Attribute::NoFree);
  insertArgumentFunction->addFnAttr(Attribute::NoFree);
  insertReturnFunction->addFnAttr(Attribute::NoFree);
  insertFunctionFunction->addFnAttr(Attribute::NoFree);
  insertChoiceGradientFunction->addFnAttr(Attribute::NoFree);
  insertArgumentGradientFunction->addFnAttr(Attribute::NoFree);
  hasCallFunction->addFnAttr(Attribute::NoFree);
  hasChoiceFunction->addFnAttr(Attribute::NoFree);
}

StaticTraceInterface::StaticTraceInterface(
    LLVMContext &C, Function *getTraceFunction, Function *getChoiceFunction,
    Function *insertCallFunction, Function *insertChoiceFunction,
    Function *insertArgumentFunction, Function *insertReturnFunction,
    Function *insertFunctionFunction, Function *insertChoiceGradientFunction,
    Function *insertArgumentGradientFunction, Function *newTraceFunction,
    Function *freeTraceFunction, Function *hasCallFunction,
    Function *hasChoiceFunction)
    : TraceInterface(C), getTraceFunction(getTraceFunction),
      getChoiceFunction(getChoiceFunction),
      insertCallFunction(insertCallFunction),
      insertChoiceFunction(insertChoiceFunction),
      insertArgumentFunction(insertArgumentFunction),
      insertReturnFunction(insertReturnFunction),
      insertFunctionFunction(insertFunctionFunction),
      insertChoiceGradientFunction(insertChoiceGradientFunction),
      insertArgumentGradientFunction(insertArgumentGradientFunction),
      newTraceFunction(newTraceFunction), freeTraceFunction(freeTraceFunction),
      hasCallFunction(hasCallFunction), hasChoiceFunction(hasChoiceFunction){};

// user implemented
Value *StaticTraceInterface::getTrace(IRBuilder<> &Builder) {
  return getTraceFunction;
}
Value *StaticTraceInterface::getChoice(IRBuilder<> &Builder) {
  return getChoiceFunction;
}
Value *StaticTraceInterface::insertCall(IRBuilder<> &Builder) {
  return insertCallFunction;
}
Value *StaticTraceInterface::insertChoice(IRBuilder<> &Builder) {
  return insertChoiceFunction;
}
Value *StaticTraceInterface::insertArgument(IRBuilder<> &Builder) {
  return insertArgumentFunction;
}
Value *StaticTraceInterface::insertReturn(IRBuilder<> &Builder) {
  return insertReturnFunction;
}
Value *StaticTraceInterface::insertFunction(IRBuilder<> &Builder) {
  return insertFunctionFunction;
}
Value *StaticTraceInterface::insertChoiceGradient(IRBuilder<> &Builder) {
  return insertChoiceGradientFunction;
}
Value *StaticTraceInterface::insertArgumentGradient(IRBuilder<> &Builder) {
  return insertArgumentGradientFunction;
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
  assert(dynamicInterface);

  auto &M = *F->getParent();
  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbg());

  getTraceFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, getTraceTy(), 0, M, "get_trace");
  getChoiceFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, getChoiceTy(), 1, M, "get_choice");
  insertCallFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertCallTy(), 2, M, "insert_call");
  insertChoiceFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertChoiceTy(), 3, M, "insert_choice");
  insertArgumentFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertArgumentTy(), 4, M, "insert_argument");
  insertReturnFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertReturnTy(), 5, M, "insert_return");
  insertFunctionFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertFunctionTy(), 6, M, "insert_function");
  insertChoiceGradientFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertChoiceGradientTy(), 7, M,
      "insert_choice_gradient");
  insertArgumentGradientFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, insertArgumentGradientTy(), 8, M,
      "insert_argument_gradient");
  newTraceFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, newTraceTy(), 9, M, "new_trace");
  freeTraceFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, freeTraceTy(), 10, M, "free_trace");
  hasCallFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, hasCallTy(), 11, M, "has_call");
  hasChoiceFunction = MaterializeInterfaceFunction(
      Builder, dynamicInterface, hasChoiceTy(), 12, M, "has_choice");

  assert(newTraceFunction);
  assert(freeTraceFunction);
  assert(getTraceFunction);
  assert(getChoiceFunction);
  assert(insertCallFunction);
  assert(insertChoiceFunction);

  assert(insertArgumentFunction);
  assert(insertReturnFunction);
  assert(insertFunctionFunction);

  assert(insertChoiceGradientFunction);
  assert(insertArgumentGradientFunction);

  assert(hasCallFunction);
  assert(hasChoiceFunction);
}

Function *DynamicTraceInterface::MaterializeInterfaceFunction(
    IRBuilder<> &Builder, Value *dynamicInterface, FunctionType *FTy,
    unsigned index, Module &M, const Twine &Name) {
  auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(), dynamicInterface,
                                       Builder.getInt32(index));
  auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
  auto pty = PointerType::get(FTy, load->getPointerAddressSpace());
  auto cast = Builder.CreatePointerCast(load, pty);

  auto global =
      new GlobalVariable(M, pty, false, GlobalVariable::PrivateLinkage,
                         ConstantPointerNull::get(pty), Name + "_ptr");
  Builder.CreateStore(cast, global);

  Function *F = Function::Create(FTy, Function::PrivateLinkage, Name, M);
  F->addFnAttr(Attribute::AlwaysInline);
  BasicBlock *Entry = BasicBlock::Create(M.getContext(), "entry", F);

  IRBuilder<> WrapperBuilder(Entry);

  auto ToWrap = WrapperBuilder.CreateLoad(pty, global, Name);
  auto Args = SmallVector<Value *, 4>(make_pointer_range(F->args()));
  auto Call = WrapperBuilder.CreateCall(FTy, ToWrap, Args);

  if (!FTy->getReturnType()->isVoidTy()) {
    WrapperBuilder.CreateRet(Call);
  } else {
    WrapperBuilder.CreateRetVoid();
  }

  return F;
}

// user implemented
Value *DynamicTraceInterface::getTrace(IRBuilder<> &Builder) {
  return getTraceFunction;
}

Value *DynamicTraceInterface::getChoice(IRBuilder<> &Builder) {
  return getChoiceFunction;
}

Value *DynamicTraceInterface::insertCall(IRBuilder<> &Builder) {
  return insertCallFunction;
}

Value *DynamicTraceInterface::insertChoice(IRBuilder<> &Builder) {
  return insertChoiceFunction;
}

Value *DynamicTraceInterface::insertArgument(IRBuilder<> &Builder) {
  return insertArgumentFunction;
}

Value *DynamicTraceInterface::insertReturn(IRBuilder<> &Builder) {
  return insertReturnFunction;
}

Value *DynamicTraceInterface::insertFunction(IRBuilder<> &Builder) {
  return insertFunctionFunction;
}

Value *DynamicTraceInterface::insertChoiceGradient(IRBuilder<> &Builder) {
  return insertChoiceGradientFunction;
}

Value *DynamicTraceInterface::insertArgumentGradient(IRBuilder<> &Builder) {
  return insertArgumentGradientFunction;
}

Value *DynamicTraceInterface::newTrace(IRBuilder<> &Builder) {
  return newTraceFunction;
}

Value *DynamicTraceInterface::freeTrace(IRBuilder<> &Builder) {
  return freeTraceFunction;
}

Value *DynamicTraceInterface::hasCall(IRBuilder<> &Builder) {
  return hasCallFunction;
}

Value *DynamicTraceInterface::hasChoice(IRBuilder<> &Builder) {
  return hasChoiceFunction;
}
