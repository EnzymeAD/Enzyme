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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
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
    bool shouldSkipParam =
        attributes.hasParamAttr(i, TraceUtils::TraceParameterAttribute) ||
        attributes.hasParamAttr(i,
                                TraceUtils::ObservationsParameterAttribute) ||
        attributes.hasParamAttr(i, TraceUtils::LikelihoodParameterAttribute);
    if (shouldSkipParam)
      continue;

    auto arg = fn->arg_begin() + i;
    auto name = Builder.CreateGlobalStringPtr(arg->getName());

    auto Outlined = [](IRBuilder<> &OutlineBuilder, TraceUtils *OutlineTutils,
                       ArrayRef<Argument *> Arguments) {
      OutlineTutils->InsertArgument(OutlineBuilder, Arguments[0], Arguments[1]);
      OutlineBuilder.CreateRetVoid();
    };

    auto call = tutils->CreateOutlinedFunction(
        Builder, Outlined, Builder.getVoidTy(), {name, arg}, false,
        "outline_insert_argument_" + arg->getName());

#if LLVM_VERSION_MAJOR >= 14
    call->addAttributeAtIndex(
        AttributeList::FunctionIndex,
        Attribute::get(F.getContext(), "enzyme_insert_argument"));
    call->addAttributeAtIndex(AttributeList::FunctionIndex,
                              Attribute::get(F.getContext(), "enzyme_active"));
#else
    call->addAttribute(
        AttributeList::FunctionIndex,
        Attribute::get(F.getContext(), "enzyme_insert_argument"));
    call->addAttribute(AttributeList::FunctionIndex,
                       Attribute::get(F.getContext(), "enzyme_active"));
#endif
    if (autodiff) {
      auto gradient_setter = ValueAsMetadata::get(
          tutils->interface->insertChoiceGradient(Builder));
      auto gradient_setter_node =
          MDNode::get(F.getContext(), {gradient_setter});

      call->setMetadata("enzyme_gradient_setter", gradient_setter_node);
    }
  }
}

void TraceGenerator::handleSampleCall(CallInst &call, CallInst *new_call) {
  // create outlined sample function
  SmallVector<Value *, 4> Args(
      make_range(new_call->arg_begin() + 2, new_call->arg_end()));

  Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
  Function *likelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));

  IRBuilder<> Builder(new_call);

  auto OutlinedSample = [Name = call.getName(),
                         samplefn](IRBuilder<> &OutlineBuilder,
                                   TraceUtils *OutlineTutils,
                                   ArrayRef<Argument *> Arguments) {
    SmallVector<Value *, 2> SampleArgs(
        make_range(Arguments.begin() + 1, Arguments.end()));

    auto choice = OutlineTutils->SampleOrCondition(
        OutlineBuilder, samplefn, SampleArgs, Arguments[0], Name);
    OutlineBuilder.CreateRet(choice);
  };

  std::string mode_str =
      mode == ProbProgMode::Condition ? "condition" : "sample";

  auto sample_call = tutils->CreateOutlinedFunction(
      Builder, OutlinedSample,
      tutils->getTraceInterface()->insertChoiceTy()->getParamType(2), Args,
      false, mode_str + "_" + call.getName());

#if LLVM_VERSION_MAJOR >= 14
  sample_call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_sample"));
  sample_call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_active"));
#else
  sample_call->addAttribute(AttributeList::FunctionIndex,
                            Attribute::get(call.getContext(), "enzyme_sample"));
  sample_call->addAttribute(AttributeList::FunctionIndex,
                            Attribute::get(call.getContext(), "enzyme_active"));
#endif

  if (autodiff) {
    auto gradient_setter =
        ValueAsMetadata::get(tutils->interface->insertChoiceGradient(Builder));
    auto gradient_setter_node =
        MDNode::get(call.getContext(), {gradient_setter});

    sample_call->setMetadata("enzyme_gradient_setter", gradient_setter_node);
  }

  // calculate and accumulate log likelihood

  SmallVector<Value *, 3> LikelihoodArgs(
      make_range(Args.begin() + 1, Args.end()));
  LikelihoodArgs.push_back(sample_call);

  auto score =
      Builder.CreateCall(likelihoodfn->getFunctionType(), likelihoodfn,
                         LikelihoodArgs, "likelihood." + call.getName());

  auto log_prob_sum = Builder.CreateLoad(
      Builder.getDoubleTy(), tutils->getLikelihood(), "log_prob_sum");
  auto acc = Builder.CreateFAdd(log_prob_sum, score);
  Builder.CreateStore(acc, tutils->getLikelihood());

  // create outlined trace function

  Value *trace_args[] = {new_call->getArgOperand(2), score, sample_call};

  auto OutlinedTrace = [](IRBuilder<> &OutlineBuilder,
                          TraceUtils *OutlineTutils,
                          ArrayRef<Argument *> Arguments) {
    OutlineTutils->InsertChoice(OutlineBuilder, Arguments[0], Arguments[1],
                                Arguments[2]);
    OutlineBuilder.CreateRetVoid();
  };

  auto trace_call = tutils->CreateOutlinedFunction(
      Builder, OutlinedTrace, Builder.getVoidTy(), trace_args, false,
      "trace_" + call.getName());

#if LLVM_VERSION_MAJOR >= 14
  trace_call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_inactive"));
  trace_call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_notypeanalysis"));
#else
  trace_call->addAttribute(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_inactive"));
  trace_call->addAttribute(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_notypeanalysis"));
#endif

  sample_call->takeName(new_call);
  new_call->replaceAllUsesWith(sample_call);
  new_call->eraseFromParent();
}

void TraceGenerator::handleArbitraryCall(CallInst &call, CallInst *new_call) {
  IRBuilder<> Builder(new_call);
  auto str = call.getName() + "." + call.getCalledFunction()->getName();
  auto address = Builder.CreateGlobalStringPtr(str.str());

  SmallVector<Value *, 2> args;
  for (auto it = new_call->arg_begin(); it != new_call->arg_end(); it++) {
    args.push_back(*it);
  }

  args.push_back(tutils->getLikelihood());

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
    Instruction *ThenTerm, *ElseTerm;
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
