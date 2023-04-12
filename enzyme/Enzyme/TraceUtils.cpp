//===- TraceUtils.cpp - Utilites for interacting with traces  ------------===//
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

#include "TraceUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "TraceInterface.h"

using namespace llvm;

TraceUtils::TraceUtils(ProbProgMode mode, Function *newFunc, Argument *trace,
                       Argument *observations, TraceInterface *interface)
    : trace(trace), observations(observations), interface(interface),
      mode(mode), newFunc(newFunc){};

TraceUtils *TraceUtils::FromFunctionSignature(Function *newFunc,
                                              TraceInterface *interface) {
  assert(interface);

  Argument *trace = nullptr;
  Argument *observations = nullptr;

  auto attributes = newFunc->getAttributes();
  for (int i = 0; i < newFunc->getFunctionType()->getNumParams(); ++i) {
    if (attributes.hasParamAttr(i, TraceParameterAttribute)) {
      trace = newFunc->arg_begin() + i;
    } else if (attributes.hasParamAttr(i, ObservationsParameterAttribute)) {
      observations = newFunc->arg_begin() + i;
    }
  }

  if (!trace)
    return nullptr;

  ProbProgMode mode =
      observations ? ProbProgMode::Condition : ProbProgMode::Trace;

  return new TraceUtils(mode, newFunc, trace, observations, interface);
}

TraceUtils *TraceUtils::FromClone(ProbProgMode mode, TraceInterface *interface,
                                  Function *oldFunc,
                                  ValueToValueMapTy &originalToNewFn) {
  assert(interface);

  auto &Context = oldFunc->getContext();

  FunctionType *orig_FTy = oldFunc->getFunctionType();
  Type *traceType = TraceInterface::getTraceTy(Context)->getReturnType();

  SmallVector<Type *, 4> params;

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    params.push_back(orig_FTy->getParamType(i));
  }

  if (mode == ProbProgMode::Condition)
    params.push_back(traceType);

  params.push_back(traceType);

  Type *RetTy = oldFunc->getReturnType();
  FunctionType *FTy = FunctionType::get(RetTy, params, oldFunc->isVarArg());

  Twine Name = (mode == ProbProgMode::Condition ? "condition_" : "trace_") +
               Twine(oldFunc->getName());

  Function *newFunc = Function::Create(
      FTy, Function::LinkageTypes::InternalLinkage, Name, oldFunc->getParent());

  auto DestArg = newFunc->arg_begin();
  auto SrcArg = oldFunc->arg_begin();

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    Argument *arg = SrcArg;
    originalToNewFn[arg] = DestArg;
    DestArg->setName(arg->getName());
    DestArg++;
    SrcArg++;
  }

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(newFunc, oldFunc, originalToNewFn,
                    CloneFunctionChangeType::LocalChangesOnly, Returns, "",
                    nullptr);
#else
  CloneFunctionInto(newFunc, oldFunc, originalToNewFn, true, Returns, "",
                    nullptr);
#endif

  newFunc->setLinkage(Function::LinkageTypes::InternalLinkage);

  Argument *trace = nullptr;
  Argument *observations = nullptr;

  if (mode == ProbProgMode::Condition) {
    auto arg = newFunc->arg_end() - 2;
    observations = arg;
    arg->setName("observations");
    arg->addAttr(Attribute::get(Context, ObservationsParameterAttribute));
  }

  auto arg = newFunc->arg_end() - 1;
  trace = arg;
  arg->setName("trace");
  arg->addAttr(Attribute::get(Context, TraceParameterAttribute));

  return new TraceUtils(mode, newFunc, trace, observations, interface);
};

TraceUtils *TraceUtils::CreateEmpty(ProbProgMode mode,
                                    TraceInterface *interface,
                                    FunctionType *orig_FTy, Module *M,
                                    const Twine &Name) {
  assert(interface);

  LLVMContext &Context = M->getContext();
  Type *traceType = TraceInterface::getTraceTy(Context)->getReturnType();

  SmallVector<Type *, 4> params;

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    params.push_back(orig_FTy->getParamType(i));
  }

  if (mode == ProbProgMode::Condition)
    params.push_back(traceType);

  params.push_back(traceType);

  Type *RetTy = orig_FTy->getReturnType();
  FunctionType *FTy = FunctionType::get(RetTy, params, orig_FTy->isVarArg());

  Twine prefixed_name =
      (mode == ProbProgMode::Condition ? "condition_" : "trace_") + Name;

  Function *newFunc = Function::Create(
      FTy, Function::LinkageTypes::InternalLinkage, prefixed_name, M);
  BasicBlock *entry = BasicBlock::Create(Context);
  entry->insertInto(newFunc);

  Argument *trace = nullptr;
  Argument *observations = nullptr;

  if (mode == ProbProgMode::Condition) {
    auto arg = newFunc->arg_end() - 2;
    observations = arg;
    arg->setName("observations");
    arg->addAttr(Attribute::get(Context, ObservationsParameterAttribute));
  }

  auto arg = newFunc->arg_end() - 1;
  trace = arg;
  arg->setName("trace");
  arg->addAttr(Attribute::get(Context, TraceParameterAttribute));

  return new TraceUtils(mode, newFunc, trace, observations, interface);
}

TraceUtils::~TraceUtils() = default;

TraceInterface *TraceUtils::getTraceInterface() { return interface; }

Value *TraceUtils::getTrace() { return trace; }

Value *TraceUtils::getObservations() { return observations; }

CallInst *TraceUtils::CreateTrace(IRBuilder<> &Builder, const Twine &Name) {
  return Builder.CreateCall(interface->newTraceTy(),
                            interface->newTrace(Builder), {}, Name);
}

CallInst *TraceUtils::InsertChoice(IRBuilder<> &Builder,
                                   FunctionType *interface_type,
                                   Value *interface_function, Value *address,
                                   Value *score, Value *choice, Value *trace) {
  Type *size_type = interface_type->getParamType(4);
  auto choicesize = choice->getType()->getPrimitiveSizeInBits();

  Value *retval;
  if (choice->getType()->isPointerTy()) {
    retval = Builder.CreatePointerCast(choice, Builder.getInt8PtrTy());
  } else {
    auto M = Builder.GetInsertBlock()->getModule();
    auto &DL = M->getDataLayout();
    auto pointersize = DL.getPointerSizeInBits();
    if (choicesize <= pointersize) {
      auto cast = Builder.CreateBitCast(
          choice, IntegerType::get(M->getContext(), choicesize));
      cast = choicesize == pointersize
                 ? cast
                 : Builder.CreateZExt(cast, Builder.getIntPtrTy(DL));
      retval = Builder.CreateIntToPtr(cast, Builder.getInt8PtrTy());
    } else {
      IRBuilder<> AllocaBuilder(Builder.GetInsertBlock()
                                    ->getParent()
                                    ->getEntryBlock()
                                    .getFirstNonPHIOrDbgOrLifetime());
      auto alloca = AllocaBuilder.CreateAlloca(choice->getType(), nullptr,
                                               choice->getName() + ".ptr");
      Builder.CreateStore(choice, alloca);
      retval = alloca;
    }
  }

  Value *args[] = {trace, address, score, retval,
                   ConstantInt::get(size_type, choicesize / 8)};

  auto call = Builder.CreateCall(interface_type, interface_function, args);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

CallInst *TraceUtils::InsertChoice(IRBuilder<> &Builder, Value *address,
                                   Value *score, Value *choice) {
  return TraceUtils::InsertChoice(Builder, interface->insertChoiceTy(),
                                  interface->insertChoice(Builder), address,
                                  score, choice, trace);
}

CallInst *TraceUtils::InsertCall(IRBuilder<> &Builder, Value *address,
                                 Value *subtrace) {
  Value *args[] = {trace, address, subtrace};

  auto call = Builder.CreateCall(interface->insertCallTy(),
                                 interface->insertCall(Builder), args);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

CallInst *TraceUtils::GetTrace(IRBuilder<> &Builder, Value *address,
                               const Twine &Name) {
  assert(address->getType()->isPointerTy());

  Value *args[] = {observations, address};

  auto call = Builder.CreateCall(interface->getTraceTy(),
                                 interface->getTrace(Builder), args, Name);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

Instruction *TraceUtils::GetChoice(llvm::IRBuilder<> &Builder,
                                   llvm::FunctionType *interface_type,
                                   llvm::Value *interface_function,
                                   llvm::Value *address, llvm::Type *choiceType,
                                   llvm::Value *trace,
                                   const llvm::Twine &Name) {
  IRBuilder<> AllocaBuilder(Builder.GetInsertBlock()
                                ->getParent()
                                ->getEntryBlock()
                                .getFirstNonPHIOrDbgOrLifetime());
  AllocaInst *store_dest =
      AllocaBuilder.CreateAlloca(choiceType, nullptr, Name + ".ptr");
  auto preallocated_size = choiceType->getPrimitiveSizeInBits() / 8;
  Type *size_type = interface_type->getParamType(3);

  Value *args[] = {
      trace, address,
      Builder.CreatePointerCast(store_dest, Builder.getInt8PtrTy()),
      ConstantInt::get(size_type, preallocated_size)};

  auto call = Builder.CreateCall(interface_type, interface_function, args,
                                 Name + ".size");

#if LLVM_VERSION_MAJOR >= 14
  call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call->getContext(), "enzyme_inactive"));
#else
  call->addAttribute(AttributeList::FunctionIndex,
                     Attribute::get(call->getContext(), "enzyme_inactive"));
#endif
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return Builder.CreateLoad(choiceType, store_dest, "from.trace." + Name);
}

Instruction *TraceUtils::GetChoice(IRBuilder<> &Builder, Value *address,
                                   Type *choiceType, const Twine &Name) {
  return TraceUtils::GetChoice(Builder, interface->getChoiceTy(),
                               interface->getChoice(Builder), address,
                               choiceType, observations, Name);
}

Instruction *TraceUtils::GetLikelihood(IRBuilder<> &Builder,
                                       FunctionType *interface_type,
                                       Value *interface_function,
                                       Value *address, Value *trace,
                                       const llvm::Twine &Name) {
  Value *args[] = {trace, address};

  return Builder.CreateCall(interface_type, interface_function, args, Name);
}

Instruction *TraceUtils::GetLikelihood(llvm::IRBuilder<> &Builder,
                                       llvm::Value *address,
                                       const llvm::Twine &Name) {
  return TraceUtils::GetLikelihood(Builder, interface->getLikelihoodTy(),
                                   interface->getLikelihood(Builder), address,
                                   observations, Name);
}

Instruction *TraceUtils::HasChoice(IRBuilder<> &Builder, Value *address,
                                   const Twine &Name) {
  Value *args[]{observations, address};

  auto call = Builder.CreateCall(interface->hasChoiceTy(),
                                 interface->hasChoice(Builder), args, Name);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

Instruction *TraceUtils::HasCall(IRBuilder<> &Builder, Value *address,
                                 const Twine &Name) {
  Value *args[]{observations, address};

  auto call = Builder.CreateCall(interface->hasCallTy(),
                                 interface->hasCall(Builder), args, Name);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}
