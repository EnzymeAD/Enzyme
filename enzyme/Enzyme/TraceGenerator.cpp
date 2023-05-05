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
  // create outlined sample function
  SmallVector<Value *, 4> Args;
  SmallVector<Type *, 4> Tys;
  for (auto &arg : make_range(new_call->arg_begin() + 2, new_call->arg_end())) {
    Args.push_back(arg);
    Tys.push_back(arg->getType());
  }

  auto orig_FTy = FunctionType::get(call.getType(), Tys, false);

  Type *traceType =
      TraceInterface::getTraceTy(call.getContext())->getReturnType();

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
      (mode == ProbProgMode::Condition ? "condition_" : "sample_") +
      call.getName();

  Function *outlinedSample =
      Function::Create(FTy, Function::LinkageTypes::InternalLinkage,
                       prefixed_name, call.getModule());
  BasicBlock *entry = BasicBlock::Create(call.getContext());
  entry->insertInto(outlinedSample);

  Argument *trace = nullptr;
  Argument *observations = nullptr;

  auto arg = outlinedSample->arg_end() - 1;
  trace = arg;
  arg->setName("trace");
  arg->addAttr(
      Attribute::get(call.getContext(), TraceUtils::TraceParameterAttribute));

  if (mode == ProbProgMode::Condition) {
    auto arg = outlinedSample->arg_end() - 2;
    observations = arg;
    arg->setName("observations");
    arg->addAttr(Attribute::get(call.getContext(),
                                TraceUtils::ObservationsParameterAttribute));
  }

  Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
  Function *likelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));

  IRBuilder<> Builder(new_call);
  // call outlined sample function

  if (mode == ProbProgMode::Condition)
    Args.push_back(tutils->getObservations());

  Args.push_back(tutils->getTrace());

  outlinedSample->addFnAttr(Attribute::AlwaysInline);
  auto sample_call = Builder.CreateCall(outlinedSample->getFunctionType(),
                                        outlinedSample, Args);
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

  // build outlined sample function

  IRBuilder<> OutlineBuilder(&outlinedSample->getEntryBlock());

  Value *address = outlinedSample->arg_begin();

  SmallVector<Value *, 2> sample_args;
  for (unsigned i = 1; i <= samplefn->getFunctionType()->getNumParams(); ++i) {
    sample_args.push_back(outlinedSample->arg_begin() + i);
  }

  auto choice =
      tutils->SampleOrCondition(OutlineBuilder, samplefn, sample_args, trace,
                                observations, address, call.getName());

  OutlineBuilder.CreateRet(choice);

  // calculate and accumulate log likelihood

  SmallVector<Value *, 3> likelihood_args;
  for (auto &&arg : make_range(Args.begin() + 1, Args.end() - 1)) {
    likelihood_args.push_back(arg);
  }
  likelihood_args.push_back(sample_call);
  auto score =
      Builder.CreateCall(likelihoodfn->getFunctionType(), likelihoodfn,
                         likelihood_args, "likelihood." + call.getName());

  auto log_prob_sum = Builder.CreateLoad(
      Builder.getDoubleTy(), tutils->getLikelihood(), "log_prob_sum");
  auto acc = Builder.CreateFAdd(log_prob_sum, score);
  Builder.CreateStore(acc, tutils->getLikelihood());

  // create outlined trace function
  Type *trace_params[] = {trace->getType(), address->getType(),
                          score->getType(), choice->getType()};
  auto trace_FTy = FunctionType::get(Type::getVoidTy(call.getContext()),
                                     trace_params, false);

  Function *outlinedTrace =
      Function::Create(trace_FTy, Function::LinkageTypes::InternalLinkage,
                       "trace_" + call.getName(), call.getModule());
  BasicBlock *trace_entry = BasicBlock::Create(call.getContext());
  trace_entry->insertInto(outlinedTrace);

  // call outlined trace function

  Value *trace_args[] = {tutils->getTrace(), new_call->getArgOperand(2), score,
                         sample_call};

  outlinedTrace->addFnAttr(Attribute::AlwaysInline);
  outlinedTrace->addFnAttr("enzyme_notypeanalysis");
  outlinedTrace->addFnAttr("enzyme_inactive");
  auto trace_call = Builder.CreateCall(outlinedTrace->getFunctionType(),
                                       outlinedTrace, trace_args);

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

  // build outlined trace function

  IRBuilder<> TraceBuilder(trace_entry);
  trace_entry->setName("entry");

  TraceUtils::InsertChoice(
      TraceBuilder, tutils->getTraceInterface()->insertChoiceTy(),
      tutils->getTraceInterface()->insertChoice(TraceBuilder),
      outlinedTrace->arg_begin() + 1, outlinedTrace->arg_begin() + 2,
      outlinedTrace->arg_begin() + 3, outlinedTrace->arg_begin() + 0);

  TraceBuilder.CreateRetVoid();

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
