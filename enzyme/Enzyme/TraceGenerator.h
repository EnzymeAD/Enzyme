#ifndef TraceGenerator_h
#define TraceGenerator_h

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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
//  Value *conditioning_trace = tutils->conditioning_trace;

public:
  TraceGenerator(EnzymeLogic &Logic, GradientUtils *const gutils, TraceUtils *const tutils) : Logic(Logic), gutils(gutils), tutils(tutils) {};

  void visitCallInst(llvm::CallInst &call) {
    
    if (!tutils->generativeFunctions.contains(call.getCalledFunction()))
      return;
    
    CallInst* new_call = dyn_cast<CallInst>(tutils->originalToNewFn[&call]);
    IRBuilder<> Builder(new_call);
    
    if (call.getCalledFunction() == tutils->getTraceInterface()->getSampleFunction()) {
      Function *samplefn = GetFunctionFromValue(new_call->getArgOperand(0));
      Function *loglikelihoodfn = GetFunctionFromValue(new_call->getArgOperand(1));
      Value *address = new_call->getArgOperand(2);
      
      SmallVector<Value*, 2> sample_args;
      for (auto it = new_call->arg_begin() + 3; it != new_call->arg_end(); it++) {
        sample_args.push_back(*it);
      }
      
      Value *choice;
      switch (mode) {
        case ProbProgMode::Trace: {
          choice = Builder.CreateCall(samplefn->getFunctionType(), samplefn, sample_args, "sample." + call.getName());
          break;
        }
        case ProbProgMode::Condition: {
          Value *hasChoice = tutils->HasChoice(Builder, address);
          Instruction *ThenTerm, *ElseTerm;
          Value *ThenChoice, *ElseChoice;
          SplitBlockAndInsertIfThenElse(hasChoice, new_call, &ThenTerm, &ElseTerm);
          
          Builder.SetInsertPoint(ThenTerm); {
            ThenChoice = tutils->GetChoice(Builder, address, samplefn->getFunctionType()->getReturnType());
          }
          
          Builder.SetInsertPoint(ElseTerm); {
            ElseChoice = Builder.CreateCall(samplefn->getFunctionType(), samplefn, sample_args, "sample." + call.getName());
          }
          
          Builder.SetInsertPoint(new_call);
          auto phi = Builder.CreatePHI(new_call->getType(), 2);
          phi->addIncoming(ThenChoice, ThenTerm->getParent());
          phi->addIncoming(ElseChoice, ElseTerm->getParent());
          choice = phi;
          break;
        }
      }
      
      SmallVector<Value*, 3> loglikelihood_args = SmallVector(sample_args);
      loglikelihood_args.push_back(choice);
      auto score = Builder.CreateCall(loglikelihoodfn->getFunctionType(), loglikelihoodfn, loglikelihood_args, "likelihood." + call.getName());
      tutils->InsertChoice(Builder, address, score, choice);
      
      new_call->replaceAllUsesWith(choice);
      new_call->eraseFromParent();
    } else {
      auto address = Builder.CreateGlobalStringPtr(call.getCalledFunction()->getName());
      
      SmallVector<Value*, 2> args;
      for (auto it = new_call->arg_begin(); it != new_call->arg_end(); it++) {
        args.push_back(*it);
      }
      
      if (tutils->hasDynamicTraceInterface())
        args.push_back(tutils->getDynamicTraceInterface());
      
      Function *samplefn = Logic.CreateTrace(call.getCalledFunction(), tutils->generativeFunctions, tutils->mode, tutils->hasDynamicTraceInterface());

      Value *tracecall;
      switch (mode) {
        case ProbProgMode::Trace: {
          tracecall = Builder.CreateCall(samplefn->getFunctionType(), samplefn, args, call.getName());
          break;
        }
        case ProbProgMode::Condition: {
          Value *hasCall = tutils->HasCall(Builder, address);
          Instruction *ThenTerm, *ElseTerm;
          Value *ElseTracecall, *ThenTracecall;
          SplitBlockAndInsertIfThenElse(hasCall, new_call, &ThenTerm, &ElseTerm);
          
          Builder.SetInsertPoint(ThenTerm); {
            SmallVector<Value*, 2> args_and_cond = SmallVector(args);
            auto trace = tutils->GetTrace(Builder, address);
            args_and_cond.push_back(trace);
            ThenTracecall = Builder.CreateCall(samplefn->getFunctionType(), samplefn, args_and_cond, call.getName());
          }

          Builder.SetInsertPoint(ElseTerm); {
            SmallVector<Value*, 2> args_and_null = SmallVector(args);
            auto trace = ConstantPointerNull::get(cast<PointerType>(tutils->getTraceInterface()->newTraceTy()->getReturnType()));
            args_and_null.push_back(trace);
            ElseTracecall = Builder.CreateCall(samplefn->getFunctionType(), samplefn, args_and_null, call.getName());
          }

          Builder.SetInsertPoint(new_call);
          auto phi = Builder.CreatePHI(samplefn->getFunctionType()->getReturnType(), 2);
          phi->addIncoming(ThenTracecall, ThenTerm->getParent());
          phi->addIncoming(ElseTracecall, ElseTerm->getParent());
          tracecall = phi;
        }
      }
     
      Value *ret = Builder.CreateExtractValue(tracecall, {0});
      Value *subtrace = Builder.CreateExtractValue(tracecall, {1});
      
      tutils->InsertCall(Builder, address, subtrace);
      
      new_call->replaceAllUsesWith(ret);
      new_call->eraseFromParent();
    }
  }
  
  };


#endif /* TraceGenerator_h */
