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

#include "TraceGenerator.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#if LLVM_VERSION_MAJOR >= 8
#include "llvm/IR/InstrTypes.h"
#endif
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "FunctionUtils.h"
#include "TraceInterface.h"
#include "TraceUtils.h"
#include "Utils.h"

using namespace llvm;

TraceGenerator::TraceGenerator(
    EnzymeLogic &Logic, TraceUtils *tutils, bool autodiff,
    ValueMap<const Value *, WeakTrackingVH> &originalToNewFn,
    SmallPtrSetImpl<Function *> &generativeFunctions)
    : Logic(Logic), tutils(tutils), autodiff(autodiff),
      originalToNewFn(originalToNewFn),
      generativeFunctions(generativeFunctions) {
  assert(tutils);
};

void TraceGenerator::visitFunction(Function &F) {
  auto fn = tutils->newFunc;
  auto entry = fn->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();

  while (isa<AllocaInst>(entry) && entry->getNextNode()) {
    entry = entry->getNextNode();
  }

  IRBuilder<> Builder(entry);

  tutils->InsertFunction(Builder, tutils->newFunc);

  auto attributes = fn->getAttributes();
  for (size_t i = 0; i < fn->getFunctionType()->getNumParams(); ++i) {
    if (!attributes.hasParamAttr(i, TraceUtils::TraceParameterAttribute) &&
        !attributes.hasParamAttr(i, TraceUtils::ObservationsParameterAttribute))
      tutils->InsertArgument(Builder, fn->arg_begin() + i);
  }
}

void TraceGenerator::handleSampleCall(CallInst &call, CallInst *new_call) {
  SmallVector<Value *, 4> Args;
  SmallVector<Type *, 4> Tys;
  for (auto &arg : make_range(new_call->arg_begin() + 2, new_call->arg_end())) {
    Args.push_back(arg);
    Tys.push_back(arg->getType());
  }

  auto FT = FunctionType::get(call.getType(), Tys, false);

  TraceUtils *outline_tutils = TraceUtils::CreateEmpty(
      mode, tutils->interface, FT, call.getModule(), call.getName());
  Function *outline = outline_tutils->newFunc;
  Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
  Function *likelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));

  {
    // call outlined function
    IRBuilder<> Builder(new_call);

    if (mode == ProbProgMode::Condition)
      Args.push_back(tutils->getObservations());

    Args.push_back(tutils->getTrace());

    outline->addFnAttr(Attribute::AlwaysInline);
    auto outline_call =
        Builder.CreateCall(outline->getFunctionType(), outline, Args);
#if LLVM_VERSION_MAJOR >= 14
    outline_call->addAttributeAtIndex(
        AttributeList::FunctionIndex,
        Attribute::get(outline_call->getContext(), "enzyme_sample"));
    outline_call->addAttributeAtIndex(
        AttributeList::FunctionIndex,
        Attribute::get(outline_call->getContext(), "enzyme_active"));
#else
    outline_call->addAttribute(
        AttributeList::FunctionIndex,
        Attribute::get(outline_call->getContext(), "enzyme_sample"));
    outline_call->addAttribute(
        AttributeList::FunctionIndex,
        Attribute::get(outline_call->getContext(), "enzyme_active"));
#endif

    outline_call->takeName(new_call);
    new_call->replaceAllUsesWith(outline_call);
    new_call->eraseFromParent();

    auto density_fn = ValueAsMetadata::get(likelihoodfn);
    auto density_node = MDNode::get(outline_call->getContext(), {density_fn});

    outline_call->setMetadata("enzyme_pdf", density_node);

    if (autodiff) {
      auto likelihood_getter =
          ValueAsMetadata::get(tutils->interface->getLikelihood(Builder));
      auto likelihood_node =
          MDNode::get(outline_call->getContext(), {likelihood_getter});

      auto choice_getter =
          ValueAsMetadata::get(tutils->interface->getChoice(Builder));
      auto choice_node1 =
          MDNode::get(outline_call->getContext(), {choice_getter});

      auto choice_setter =
          ValueAsMetadata::get(tutils->interface->insertChoice(Builder));
      auto choice_node2 =
          MDNode::get(outline_call->getContext(), {choice_setter});

      outline_call->setMetadata("enzyme_likelihood_getter", likelihood_node);
      outline_call->setMetadata("enzyme_choice_getter", choice_node1);
      outline_call->setMetadata("enzyme_choice_setter", choice_node2);
    }
  }

  IRBuilder<> OutlineBuilder(&outline->getEntryBlock());

  Value *address = outline->arg_begin();

  SmallVector<Value *, 2> sample_args;
  for (unsigned i = 1; i <= samplefn->getFunctionType()->getNumParams(); ++i) {
    sample_args.push_back(outline->arg_begin() + i);
  }

  Instruction *choice;
  switch (mode) {
  case ProbProgMode::Trace: {
    auto sample_call = OutlineBuilder.CreateCall(samplefn->getFunctionType(),
                                                 samplefn, sample_args);
    choice = sample_call;

    break;
  }
  case ProbProgMode::Condition: {
    Instruction *hasChoice = outline_tutils->HasChoice(
        OutlineBuilder, address, "has.choice." + call.getName());

    Value *ThenChoice, *ElseChoice;
    BasicBlock *ThenBlock = BasicBlock::Create(call.getContext());
    BasicBlock *ElseBlock = BasicBlock::Create(call.getContext());
    BasicBlock *EndBlock = BasicBlock::Create(call.getContext());
    ThenBlock->insertInto(outline);
    ThenBlock->setName("condition." + call.getName() + ".with.trace");
    ElseBlock->insertInto(outline);
    ElseBlock->setName("condition." + call.getName() + ".without.trace");
    EndBlock->insertInto(outline);
    EndBlock->setName("end");

    OutlineBuilder.CreateCondBr(hasChoice, ThenBlock, ElseBlock);
    OutlineBuilder.SetInsertPoint(ThenBlock);
    ThenChoice = outline_tutils->GetChoice(
        OutlineBuilder, address, samplefn->getFunctionType()->getReturnType(),
        call.getName());
    OutlineBuilder.CreateBr(EndBlock);

    OutlineBuilder.SetInsertPoint(ElseBlock);
    ElseChoice =
        OutlineBuilder.CreateCall(samplefn->getFunctionType(), samplefn,
                                  sample_args, "sample." + call.getName());
    OutlineBuilder.CreateBr(EndBlock);

    OutlineBuilder.SetInsertPoint(EndBlock);
    auto phi = OutlineBuilder.CreatePHI(call.getType(), 2);
    phi->addIncoming(ThenChoice, ThenBlock);
    phi->addIncoming(ElseChoice, ElseBlock);
    choice = phi;
    break;
  }
  }

  SmallVector<Value *, 3> likelihood_args = SmallVector(sample_args);
  likelihood_args.push_back(choice);
  auto score = OutlineBuilder.CreateCall(likelihoodfn->getFunctionType(),
                                         likelihoodfn, likelihood_args,
                                         "likelihood." + call.getName());
  outline_tutils->InsertChoice(OutlineBuilder, address, score, choice);

  OutlineBuilder.CreateRet(choice);

  delete outline_tutils;
}

void TraceGenerator::handleArbitraryCall(CallInst &call, CallInst *new_call) {
  IRBuilder<> Builder(new_call);
  auto str = call.getName() + "." + call.getCalledFunction()->getName();
  auto address = Builder.CreateGlobalStringPtr(str.str());

  SmallVector<Value *, 2> args;
  for (auto it = new_call->arg_begin(); it != new_call->arg_end(); it++) {
    args.push_back(*it);
  }

  Function *called = getFunctionFromCall(&call);
  assert(called);

  Function *samplefn = Logic.CreateTrace(called, generativeFunctions, mode,
                                         autodiff, tutils->interface);

  auto trace = tutils->CreateTrace(Builder);

  Instruction *tracecall;
  switch (mode) {
  case ProbProgMode::Trace: {
    SmallVector<Value *, 2> args_and_trace = SmallVector(args);
    args_and_trace.push_back(trace);
    tracecall =
        Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                           args_and_trace, "trace." + called->getName());
    break;
  }
  case ProbProgMode::Condition: {
    Instruction *hasCall =
        tutils->HasCall(Builder, address, "has.call." + call.getName());
#if LLVM_VERSION_MAJOR >= 8
    Instruction *ThenTerm, *ElseTerm;
#else
    TerminatorInst *ThenTerm, *ElseTerm;
#endif
    Value *ElseTracecall, *ThenTracecall;
    SplitBlockAndInsertIfThenElse(hasCall, new_call, &ThenTerm, &ElseTerm);

    new_call->getParent()->setName(hasCall->getParent()->getName() + ".cntd");

    Builder.SetInsertPoint(ThenTerm);
    {
      ThenTerm->getParent()->setName("condition." + call.getName() +
                                     ".with.trace");
      SmallVector<Value *, 2> args_and_cond = SmallVector(args);
      auto observations =
          tutils->GetTrace(Builder, address, called->getName() + ".subtrace");
      args_and_cond.push_back(observations);
      args_and_cond.push_back(trace);
      ThenTracecall =
          Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                             args_and_cond, "condition." + called->getName());
    }

    Builder.SetInsertPoint(ElseTerm);
    {
      ElseTerm->getParent()->setName("condition." + call.getName() +
                                     ".without.trace");
      SmallVector<Value *, 2> args_and_null = SmallVector(args);
      auto observations = ConstantPointerNull::get(cast<PointerType>(
          tutils->getTraceInterface()->newTraceTy()->getReturnType()));
      args_and_null.push_back(observations);
      args_and_null.push_back(trace);
      ElseTracecall =
          Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                             args_and_null, "trace." + called->getName());
    }

    Builder.SetInsertPoint(new_call);
    auto phi = Builder.CreatePHI(samplefn->getFunctionType()->getReturnType(),
                                 2, call.getName());
    phi->addIncoming(ThenTracecall, ThenTerm->getParent());
    phi->addIncoming(ElseTracecall, ElseTerm->getParent());
    tracecall = phi;
  }
  }

  tutils->InsertCall(Builder, address, trace);

  tracecall->takeName(new_call);
  new_call->replaceAllUsesWith(tracecall);
  new_call->eraseFromParent();
}

void TraceGenerator::visitCallInst(CallInst &call) {

  if (!generativeFunctions.count(call.getCalledFunction()))
    return;

  CallInst *new_call = dyn_cast<CallInst>(originalToNewFn[&call]);

  if (call.getCalledFunction() ==
      tutils->getTraceInterface()->getSampleFunction()) {
    handleSampleCall(call, new_call);
  } else {
    handleArbitraryCall(call, new_call);
  }
}

void TraceGenerator::visitReturnInst(ReturnInst &ret) {

  if (!ret.getReturnValue())
    return;

  ReturnInst *new_ret = dyn_cast<ReturnInst>(originalToNewFn[&ret]);

  IRBuilder<> Builder(new_ret);
  tutils->InsertReturn(Builder, new_ret->getReturnValue());
}
