#ifndef TraceGenerator_h
#define TraceGenerator_h

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Value.h"

#include "EnzymeLogic.h"
#include "FunctionUtils.h"
#include "GradientUtils.h"

using namespace llvm;

class TraceGenerator : public llvm::InstVisitor<TraceGenerator> {
private:
  EnzymeLogic &Logic;
  GradientUtils *const gutils;
  TraceUtils *const tutils;
  ProbProgMode mode = tutils->mode;
  Value *conditioning_trace = tutils->conditioning_trace;

public:
  TraceGenerator(EnzymeLogic &Logic, GradientUtils *const gutils, TraceUtils *const tutils) : Logic(Logic), gutils(gutils), tutils(tutils) {};

  void visitCallInst(llvm::CallInst &call) {
        
    if (!tutils->generativeFunctions.contains(call.getCalledFunction()))
      return;
    
    CallInst* new_call = dyn_cast<CallInst>(tutils->originalToNewFn[&call]);
    IRBuilder<> Builder(new_call);
    
    if (call.getCalledFunction() == tutils->getTraceInterface().sample) {
      Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
      Function *loglikelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));
      Value *address = new_call->getArgOperand(2);
      
      SmallVector<Value*, 2> sample_args;
      for (auto it = new_call->arg_begin() + 3; it != new_call->arg_end(); it++) {
        sample_args.push_back(*it);
      }
      
      Value *choice;
      switch (mode) {
        case ProbProgMode::Trace:
          choice = Builder.CreateCall(samplefn->getFunctionType(), samplefn, sample_args, "sample." + call.getName());
          break;
        case ProbProgMode::Replay:
          choice = tutils->GetChoice(Builder, address);
      }
          
      SmallVector<Value*, 3> loglikelihood_args;
      loglikelihood_args.insert(loglikelihood_args.begin(), sample_args.begin(), sample_args.end());
      loglikelihood_args.push_back(choice);
      auto score = Builder.CreateCall(loglikelihoodfn->getFunctionType(), loglikelihoodfn, loglikelihood_args, "likelihood." + call.getName());
      
      tutils->InsertChoice(Builder, address, score, choice);
      
      new_call->replaceAllUsesWith(choice);
      new_call->eraseFromParent();
    } else {
      
      // do it in tutils initalizer ...
      Function *samplefn;
      switch (mode) {
        case ProbProgMode::Trace:
          samplefn = Logic.CreateTrace(call.getCalledFunction(), tutils->getTraceInterface(), tutils->generativeFunctions);
          break;
        case ProbProgMode::Replay:
          auto conditioning_trace = tutils->GetTrace(Builder, address);
          samplefn = Logic.CreateTrace(call.getCalledFunction(), tutils->getTraceInterface(), tutils->generativeFunctions);
          break;
      }
      
      SmallVector<Value*, 2> args;
      for (auto it = new_call->arg_begin(); it != new_call->arg_end(); it++) {
        args.push_back(*it);
      }
      
      CallInst *tracecall = Builder.CreateCall(samplefn->getFunctionType(), samplefn, args, call.getName());
      Value *ret = Builder.CreateExtractValue(tracecall, {0});
      Value *subtrace = Builder.CreateExtractValue(tracecall, {1});
      
      auto address = Builder.CreateGlobalStringPtr(tracecall->getCalledFunction()->getName());
      auto score = ConstantFP::getNullValue(Builder.getDoubleTy());
      auto noise = ConstantFP::getNullValue(Builder.getDoubleTy());
      tutils->InsertCall(Builder, address, subtrace, score, noise);
      
      new_call->replaceAllUsesWith(ret);
      new_call->eraseFromParent();
    }
  }
  
  };


#endif /* TraceGenerator_h */
