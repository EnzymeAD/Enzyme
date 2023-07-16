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

#include "llvm/Analysis/ValueTracking.h"
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
    const SmallPtrSetImpl<Function *> &generativeFunctions,
    const StringSet<> &activeRandomVariables)
    : Logic(Logic), tutils(tutils), autodiff(autodiff),
      originalToNewFn(originalToNewFn),
      generativeFunctions(generativeFunctions),
      activeRandomVariables(activeRandomVariables) {
  assert(tutils);
};

void TraceGenerator::visitFunction(Function &F) {
  if (mode == ProbProgMode::Likelihood)
    return;

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
                       ArrayRef<Value *> Arguments) {
      OutlineTutils->InsertArgument(OutlineBuilder, Arguments[0], Arguments[1]);
      OutlineBuilder.CreateRetVoid();
    };

    auto call = tutils->CreateOutlinedFunction(
        Builder, Outlined, Builder.getVoidTy(), {name, arg}, false,
        "outline_insert_argument");

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
          tutils->interface->insertArgumentGradient(Builder));
      auto gradient_setter_node =
          MDNode::get(F.getContext(), {gradient_setter});

      call->setMetadata("enzyme_gradient_setter", gradient_setter_node);
    }
  }
}

void TraceGenerator::handleObserveCall(CallInst &call, CallInst *new_call) {
  IRBuilder<> Builder(new_call);

  SmallVector<Value *, 4> Args(
      make_range(new_call->arg_begin() + 2, new_call->arg_end()));

  Value *observed = new_call->getArgOperand(0);
  Function *likelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));
  Value *address = new_call->getArgOperand(2);

  StringRef const_address;
  bool is_address_const = getConstantStringInfo(address, const_address);
  bool is_random_var_active =
      activeRandomVariables.empty() ||
      (is_address_const && activeRandomVariables.count(const_address));
  Attribute activity_attribute = Attribute::get(
      call.getContext(),
      is_random_var_active ? "enzyme_active" : "enzyme_inactive_val");

  // calculate and accumulate log likelihood
  Args.push_back(observed);

  auto score = Builder.CreateCall(likelihoodfn->getFunctionType(), likelihoodfn,
                                  ArrayRef<Value *>(Args).slice(1),
                                  "likelihood." + call.getName());

#if LLVM_VERSION_MAJOR >= 14
  score->addAttributeAtIndex(AttributeList::FunctionIndex, activity_attribute);
#else
  score->addAttribute(AttributeList::FunctionIndex, activity_attribute);
#endif

  auto log_prob_sum = Builder.CreateLoad(
      Builder.getDoubleTy(), tutils->getLikelihood(), "log_prob_sum");
  auto acc = Builder.CreateFAdd(log_prob_sum, score);
  Builder.CreateStore(acc, tutils->getLikelihood());

  // create outlined trace function
  if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition) {
    Value *trace_args[] = {address, score, observed};

    auto OutlinedTrace = [](IRBuilder<> &OutlineBuilder,
                            TraceUtils *OutlineTutils,
                            ArrayRef<Value *> Arguments) {
      OutlineTutils->InsertChoice(OutlineBuilder, Arguments[0], Arguments[1],
                                  Arguments[2]);
      OutlineBuilder.CreateRetVoid();
    };

    auto trace_call = tutils->CreateOutlinedFunction(
        Builder, OutlinedTrace, Builder.getVoidTy(), trace_args, false,
        "outline_insert_choice");

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
  }

  if (!call.getType()->isVoidTy()) {
    observed->takeName(new_call);
    new_call->replaceAllUsesWith(observed);
  }
  new_call->eraseFromParent();
}

void TraceGenerator::handleSampleCall(CallInst &call, CallInst *new_call) {
  // create outlined sample function
  SmallVector<Value *, 4> Args(
      make_range(new_call->arg_begin() + 2, new_call->arg_end()));

  Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
  Function *likelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));
  Value *address = new_call->getArgOperand(2);

  IRBuilder<> Builder(new_call);

  auto OutlinedSample = [samplefn](IRBuilder<> &OutlineBuilder,
                                   TraceUtils *OutlineTutils,
                                   ArrayRef<Value *> Arguments) {
    auto choice = OutlineTutils->SampleOrCondition(
        OutlineBuilder, samplefn, Arguments.slice(1), Arguments[0],
        samplefn->getName());
    OutlineBuilder.CreateRet(choice);
  };

  const char *mode_str;
  switch (mode) {
  case ProbProgMode::Likelihood:
  case ProbProgMode::Trace:
    mode_str = "sample";
    break;
  case ProbProgMode::Condition:
    mode_str = "condition";
    break;
  }

  auto sample_call = tutils->CreateOutlinedFunction(
      Builder, OutlinedSample, samplefn->getReturnType(), Args, false,
      Twine(mode_str) + "_" + samplefn->getName());

  StringRef const_address;
  bool is_address_const = getConstantStringInfo(address, const_address);
  bool is_random_var_active =
      activeRandomVariables.empty() ||
      (is_address_const && activeRandomVariables.count(const_address));
  Attribute activity_attribute = Attribute::get(
      call.getContext(),
      is_random_var_active ? "enzyme_active" : "enzyme_inactive_val");

#if LLVM_VERSION_MAJOR >= 14
  sample_call->addAttributeAtIndex(
      AttributeList::FunctionIndex,
      Attribute::get(call.getContext(), "enzyme_sample"));
  sample_call->addAttributeAtIndex(AttributeList::FunctionIndex,
                                   activity_attribute);
#else
  sample_call->addAttribute(AttributeList::FunctionIndex,
                            Attribute::get(call.getContext(), "enzyme_sample"));
  sample_call->addAttribute(AttributeList::FunctionIndex, activity_attribute);
#endif

  if (autodiff &&
      (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition)) {
    auto gradient_setter =
        ValueAsMetadata::get(tutils->interface->insertChoiceGradient(Builder));
    auto gradient_setter_node =
        MDNode::get(call.getContext(), {gradient_setter});

    sample_call->setMetadata("enzyme_gradient_setter", gradient_setter_node);
  }

  // calculate and accumulate log likelihood
  Args.push_back(sample_call);

  auto score = Builder.CreateCall(likelihoodfn->getFunctionType(), likelihoodfn,
                                  ArrayRef<Value *>(Args).slice(1),
                                  "likelihood." + call.getName());

#if LLVM_VERSION_MAJOR >= 14
  score->addAttributeAtIndex(AttributeList::FunctionIndex, activity_attribute);
#else
  score->addAttribute(AttributeList::FunctionIndex, activity_attribute);
#endif

  auto log_prob_sum = Builder.CreateLoad(
      Builder.getDoubleTy(), tutils->getLikelihood(), "log_prob_sum");
  auto acc = Builder.CreateFAdd(log_prob_sum, score);
  Builder.CreateStore(acc, tutils->getLikelihood());

  // create outlined trace function

  if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition) {
    Value *trace_args[] = {address, score, sample_call};

    auto OutlinedTrace = [](IRBuilder<> &OutlineBuilder,
                            TraceUtils *OutlineTutils,
                            ArrayRef<Value *> Arguments) {
      OutlineTutils->InsertChoice(OutlineBuilder, Arguments[0], Arguments[1],
                                  Arguments[2]);
      OutlineBuilder.CreateRetVoid();
    };

    auto trace_call = tutils->CreateOutlinedFunction(
        Builder, OutlinedTrace, Builder.getVoidTy(), trace_args, false,
        "outline_insert_choice");

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
  }

  sample_call->takeName(new_call);
  new_call->replaceAllUsesWith(sample_call);
  new_call->eraseFromParent();
}

void TraceGenerator::handleArbitraryCall(CallInst &call, CallInst *new_call) {
  IRBuilder<> Builder(new_call);

  SmallVector<Value *, 2> args;
  for (auto it = new_call->arg_begin(); it != new_call->arg_end(); it++) {
    args.push_back(*it);
  }

  Function *called = getFunctionFromCall(&call);
  assert(called);

  Function *samplefn = Logic.CreateTrace(
      called, tutils->sampleFunctions, tutils->observeFunctions,
      activeRandomVariables, mode, autodiff, tutils->interface);

  Instruction *replacement;
  switch (mode) {
  case ProbProgMode::Likelihood: {
    SmallVector<Value *, 2> args_and_likelihood = SmallVector(args);
    args_and_likelihood.push_back(tutils->getLikelihood());
    replacement =
        Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                           args_and_likelihood, "eval." + called->getName());
    break;
  }
  case ProbProgMode::Trace: {
    auto trace = tutils->CreateTrace(Builder);
    auto address = Builder.CreateGlobalStringPtr(
        (call.getName() + "." + called->getName()).str());

    SmallVector<Value *, 2> args_and_trace = SmallVector(args);
    args_and_trace.push_back(tutils->getLikelihood());
    args_and_trace.push_back(trace);
    replacement =
        Builder.CreateCall(samplefn->getFunctionType(), samplefn,
                           args_and_trace, "trace." + called->getName());

    tutils->InsertCall(Builder, address, trace);
    break;
  }
  case ProbProgMode::Condition: {
    auto trace = tutils->CreateTrace(Builder);
    auto address = Builder.CreateGlobalStringPtr(
        (call.getName() + "." + called->getName()).str());

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
      args_and_cond.push_back(tutils->getLikelihood());
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
      args_and_null.push_back(tutils->getLikelihood());
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
    replacement = phi;

    tutils->InsertCall(Builder, address, trace);
    break;
  }
  }

  replacement->takeName(new_call);
  new_call->replaceAllUsesWith(replacement);
  new_call->eraseFromParent();
}

void TraceGenerator::visitCallInst(CallInst &call) {
  auto fn = getFunctionFromCall(&call);

  if (!generativeFunctions.count(fn))
    return;

  CallInst *new_call = dyn_cast<CallInst>(originalToNewFn[&call]);

  if (tutils->isSampleCall(&call)) {
    handleSampleCall(call, new_call);
  } else if (tutils->isObserveCall(&call)) {
    handleObserveCall(call, new_call);
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
