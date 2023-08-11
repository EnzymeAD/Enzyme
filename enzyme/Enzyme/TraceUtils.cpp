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

TraceUtils::TraceUtils(ProbProgMode mode,
                       const SmallPtrSetImpl<Function *> &sampleFunctions,
                       const SmallPtrSetImpl<Function *> &observeFunctions,
                       Function *newFunc, Argument *trace,
                       Argument *observations, Argument *likelihood,
                       TraceInterface *interface)
    : trace(trace), observations(observations), likelihood(likelihood),
      interface(interface), mode(mode), newFunc(newFunc),
      sampleFunctions(sampleFunctions.begin(), sampleFunctions.end()),
      observeFunctions(observeFunctions.begin(), observeFunctions.end()){};

TraceUtils *
TraceUtils::FromClone(ProbProgMode mode,
                      const SmallPtrSetImpl<Function *> &sampleFunctions,
                      const SmallPtrSetImpl<Function *> &observeFunctions,
                      TraceInterface *interface, Function *oldFunc,
                      ValueToValueMapTy &originalToNewFn) {
  auto &Context = oldFunc->getContext();
  FunctionType *orig_FTy = oldFunc->getFunctionType();
  SmallVector<Type *, 4> params;

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    params.push_back(orig_FTy->getParamType(i));
  }

  Type *likelihood_acc_type = PointerType::getDoublePtrTy(Context);
  params.push_back(likelihood_acc_type);

  if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition) {
    Type *traceType = interface->getTraceTy()->getReturnType();

    if (mode == ProbProgMode::Condition)
      params.push_back(traceType);

    params.push_back(traceType);
  }

  Type *RetTy = oldFunc->getReturnType();
  FunctionType *FTy = FunctionType::get(RetTy, params, oldFunc->isVarArg());

  const char *mode_str;
  switch (mode) {
  case ProbProgMode::Likelihood:
    mode_str = "likelihood";
    break;
  case ProbProgMode::Trace:
    mode_str = "trace";
    break;
  case ProbProgMode::Condition:
    mode_str = "condition";
    break;
  }

  Function *newFunc = Function::Create(
      FTy, Function::LinkageTypes::InternalLinkage,
      Twine(mode_str) + "_" + oldFunc->getName(), oldFunc->getParent());

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
  Argument *likelihood = nullptr;

  auto arg = newFunc->arg_end();

  if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition) {
    arg -= 1;
    trace = arg;
    arg->setName("trace");
    arg->addAttr(Attribute::get(Context, TraceParameterAttribute));
  }

  if (mode == ProbProgMode::Condition) {
    arg -= 1;
    observations = arg;
    arg->setName("observations");
    arg->addAttr(Attribute::get(Context, ObservationsParameterAttribute));
  }

  arg -= 1;
  likelihood = arg;
  arg->setName("likelihood");
  arg->addAttr(Attribute::get(Context, LikelihoodParameterAttribute));

  return new TraceUtils(mode, sampleFunctions, observeFunctions, newFunc, trace,
                        observations, likelihood, interface);
};

TraceUtils::~TraceUtils() = default;

TraceInterface *TraceUtils::getTraceInterface() { return interface; }

Value *TraceUtils::getTrace() { return trace; }

Value *TraceUtils::getObservations() { return observations; }

Value *TraceUtils::getLikelihood() { return likelihood; }

std::pair<Value *, Constant *>
TraceUtils::ValueToVoidPtrAndSize(IRBuilder<> &Builder, Value *val,
                                  Type *size_type) {
  auto valsize = val->getType()->getPrimitiveSizeInBits();

  if (val->getType()->isPointerTy()) {
    Value *retval = Builder.CreatePointerCast(val, Builder.getInt8PtrTy());
    return {retval, ConstantInt::get(size_type, valsize / 8)};
  }

  auto M = Builder.GetInsertBlock()->getModule();
  auto &DL = M->getDataLayout();
  auto pointersize = DL.getPointerSizeInBits();

  if (valsize <= pointersize) {
    auto cast =
        Builder.CreateBitCast(val, IntegerType::get(M->getContext(), valsize));
    if (valsize != pointersize)
      cast = Builder.CreateZExt(cast, Builder.getIntPtrTy(DL));

    Value *retval = Builder.CreateIntToPtr(cast, Builder.getInt8PtrTy());
    return {retval, ConstantInt::get(size_type, valsize / 8)};
  } else {
    auto insertPoint = Builder.GetInsertBlock()
                           ->getParent()
                           ->getEntryBlock()
                           .getFirstNonPHIOrDbgOrLifetime();
    IRBuilder<> AllocaBuilder(insertPoint);
    auto alloca = AllocaBuilder.CreateAlloca(val->getType(), nullptr,
                                             val->getName() + ".ptr");
    Builder.CreateStore(val, alloca);
    return {alloca, ConstantInt::get(size_type, valsize / 8)};
  }
}

CallInst *TraceUtils::CreateTrace(IRBuilder<> &Builder, const Twine &Name) {
  auto call = Builder.CreateCall(interface->newTraceTy(),
                                 interface->newTrace(Builder), {}, Name);
#if LLVM_VERSION_MAJOR >= 14
  call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call->getContext(), "enzyme_newtrace"));
#else
  call->addAttribute(AttributeList::FunctionIndex,
                     Attribute::get(call->getContext(), "enzyme_newtrace"));

#endif
  return call;
}

CallInst *TraceUtils::FreeTrace(IRBuilder<> &Builder) {
  auto call = Builder.CreateCall(interface->freeTraceTy(),
                                 interface->freeTrace(Builder), {trace});
#if LLVM_VERSION_MAJOR >= 14
  call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call->getContext(), "enzyme_freetrace"));
#else
  call->addAttribute(AttributeList::FunctionIndex,
                     Attribute::get(call->getContext(), "enzyme_freetrace"));

#endif
  return call;
}

CallInst *TraceUtils::InsertChoice(IRBuilder<> &Builder, Value *address,
                                   Value *score, Value *choice) {
  Type *size_type = interface->insertChoiceTy()->getParamType(4);
  auto &&[retval, sizeval] = ValueToVoidPtrAndSize(Builder, choice, size_type);

  Value *args[] = {trace, address, score, retval, sizeval};

  auto call = Builder.CreateCall(interface->insertChoiceTy(),
                                 interface->insertChoice(Builder), args);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

CallInst *TraceUtils::InsertCall(IRBuilder<> &Builder, Value *address,
                                 Value *subtrace) {
  Value *args[] = {trace, address, subtrace};

  auto call = Builder.CreateCall(interface->insertCallTy(),
                                 interface->insertCall(Builder), args);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
#if LLVM_VERSION_MAJOR >= 14
  call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call->getContext(), "enzyme_insert_call"));
#else
  call->addAttribute(AttributeList::FunctionIndex,
                     Attribute::get(call->getContext(), "enzyme_insert_call"));

#endif
  return call;
}

CallInst *TraceUtils::InsertArgument(IRBuilder<> &Builder, Value *name,
                                     Value *argument) {
  Type *size_type = interface->insertArgumentTy()->getParamType(3);
  auto &&[retval, sizeval] =
      ValueToVoidPtrAndSize(Builder, argument, size_type);

  Value *args[] = {trace, name, retval, sizeval};

  auto call = Builder.CreateCall(interface->insertArgumentTy(),
                                 interface->insertArgument(Builder), args);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

CallInst *TraceUtils::InsertReturn(IRBuilder<> &Builder, Value *val) {
  Type *size_type = interface->insertReturnTy()->getParamType(2);
  auto &&[retval, sizeval] = ValueToVoidPtrAndSize(Builder, val, size_type);

  Value *args[] = {trace, retval, sizeval};

  auto call = Builder.CreateCall(interface->insertReturnTy(),
                                 interface->insertReturn(Builder), args);
  return call;
}

CallInst *TraceUtils::InsertFunction(IRBuilder<> &Builder, Function *function) {
  assert(!function->isIntrinsic());
  auto FunctionPtr = Builder.CreateBitCast(function, Builder.getInt8PtrTy());

  Value *args[] = {trace, FunctionPtr};

  auto call = Builder.CreateCall(interface->insertFunctionTy(),
                                 interface->insertFunction(Builder), args);
  return call;
}

CallInst *TraceUtils::InsertChoiceGradient(IRBuilder<> &Builder,
                                           FunctionType *interface_type,
                                           Value *interface_function,
                                           Value *address, Value *choice,
                                           Value *trace) {
  Type *size_type = interface_type->getParamType(3);
  auto &&[retval, sizeval] = ValueToVoidPtrAndSize(Builder, choice, size_type);

  Value *args[] = {trace, address, retval, sizeval};

  auto call = Builder.CreateCall(interface_type, interface_function, args);
  call->addParamAttr(1, Attribute::ReadOnly);
  call->addParamAttr(1, Attribute::NoCapture);
  return call;
}

CallInst *TraceUtils::InsertArgumentGradient(IRBuilder<> &Builder,
                                             FunctionType *interface_type,
                                             Value *interface_function,
                                             Value *name, Value *argument,
                                             Value *trace) {
  Type *size_type = interface_type->getParamType(3);
  auto &&[retval, sizeval] =
      ValueToVoidPtrAndSize(Builder, argument, size_type);

  Value *args[] = {trace, name, retval, sizeval};

  auto call = Builder.CreateCall(interface_type, interface_function, args);
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

Instruction *TraceUtils::GetChoice(IRBuilder<> &Builder, Value *address,
                                   Type *choiceType, const Twine &Name) {
  IRBuilder<> AllocaBuilder(Builder.GetInsertBlock()
                                ->getParent()
                                ->getEntryBlock()
                                .getFirstNonPHIOrDbgOrLifetime());
  AllocaInst *store_dest =
      AllocaBuilder.CreateAlloca(choiceType, nullptr, Name + ".ptr");
  auto preallocated_size = choiceType->getPrimitiveSizeInBits() / 8;
  Type *size_type = interface->getChoiceTy()->getParamType(3);

  Value *args[] = {
      observations, address,
      Builder.CreatePointerCast(store_dest, Builder.getInt8PtrTy()),
      ConstantInt::get(size_type, preallocated_size)};

  auto call =
      Builder.CreateCall(interface->getChoiceTy(),
                         interface->getChoice(Builder), args, Name + ".size");

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

Instruction *TraceUtils::SampleOrCondition(IRBuilder<> &Builder,
                                           Function *sample_fn,
                                           ArrayRef<Value *> sample_args,
                                           Value *address, const Twine &Name) {
  auto &Context = Builder.getContext();
  auto parent_fn = Builder.GetInsertBlock()->getParent();

  switch (mode) {
  case ProbProgMode::Likelihood:
  case ProbProgMode::Trace: {
    auto sample_call = Builder.CreateCall(sample_fn->getFunctionType(),
                                          sample_fn, sample_args);
    return sample_call;
  }
  case ProbProgMode::Condition: {
    Instruction *hasChoice = HasChoice(Builder, address, "has.choice." + Name);

    Value *ThenChoice, *ElseChoice;
    BasicBlock *ThenBlock = BasicBlock::Create(
        Context, "condition." + Name + ".with.trace", parent_fn);
    BasicBlock *ElseBlock = BasicBlock::Create(
        Context, "condition." + Name + ".without.trace", parent_fn);
    BasicBlock *EndBlock = BasicBlock::Create(Context, "end", parent_fn);

    Builder.CreateCondBr(hasChoice, ThenBlock, ElseBlock);
    Builder.SetInsertPoint(ThenBlock);
    ThenChoice = GetChoice(Builder, address, sample_fn->getReturnType(), Name);
    Builder.CreateBr(EndBlock);

    Builder.SetInsertPoint(ElseBlock);
    ElseChoice = Builder.CreateCall(sample_fn->getFunctionType(), sample_fn,
                                    sample_args, "sample." + Name);
    Builder.CreateBr(EndBlock);

    Builder.SetInsertPoint(EndBlock);
    auto phi = Builder.CreatePHI(sample_fn->getReturnType(), 2);
    phi->addIncoming(ThenChoice, ThenBlock);
    phi->addIncoming(ElseChoice, ElseBlock);

    return phi;
  }
  }
}

CallInst *TraceUtils::CreateOutlinedFunction(
    IRBuilder<> &Builder,
    function_ref<void(IRBuilder<> &, TraceUtils *, ArrayRef<Value *>)> Outlined,
    Type *RetTy, ArrayRef<Value *> Arguments, bool needsLikelihood,
    const Twine &Name) {
  SmallVector<Type *, 4> Tys;
  SmallVector<Value *, 4> Vals;
  Module *M = Builder.GetInsertBlock()->getModule();

  for (auto Arg : Arguments) {
    Vals.push_back(Arg);
    Tys.push_back(Arg->getType());
  }

  if (needsLikelihood) {
    Vals.push_back(likelihood);
    Tys.push_back(likelihood->getType());
  }

  if (mode == ProbProgMode::Condition) {
    Vals.push_back(observations);
    Tys.push_back(observations->getType());
  }

  if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition) {
    Vals.push_back(trace);
    Tys.push_back(trace->getType());
  }

  FunctionType *FTy = FunctionType::get(RetTy, Tys, false);
  Function *F =
      Function::Create(FTy, Function::LinkageTypes::InternalLinkage, Name, M);
  F->addFnAttr(Attribute::AlwaysInline);

  auto Entry = BasicBlock::Create(M->getContext(), "entry", F);

  auto ArgRange = make_pointer_range(
      make_range(F->arg_begin(), F->arg_begin() + Arguments.size()));
  SmallVector<Value *, 4> Rets(ArgRange);

  auto idx = F->arg_begin() + Arguments.size();

  Argument *likelihood_arg = nullptr;
  if (needsLikelihood)
    likelihood_arg = idx++;

  Argument *observations_arg = nullptr;
  if (mode == ProbProgMode::Condition)
    observations_arg = idx++;

  Argument *trace_arg = nullptr;
  if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition)
    trace_arg = idx++;

  TraceUtils OutlineTutils =
      TraceUtils(mode, sampleFunctions, observeFunctions, F, trace_arg,
                 observations_arg, likelihood_arg, interface);
  IRBuilder<> OutlineBuilder(Entry);
  Outlined(OutlineBuilder, &OutlineTutils, Rets);

  return Builder.CreateCall(FTy, F, Vals);
}

bool TraceUtils::isSampleCall(CallInst *call) {
  auto F = getFunctionFromCall(call);
  return sampleFunctions.count(F);
}

bool TraceUtils::isObserveCall(CallInst *call) {
  auto F = getFunctionFromCall(call);
  return observeFunctions.count(F);
}
