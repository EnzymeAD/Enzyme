#ifndef TraceUtils_h
#define TraceUtils_h

#include <map>
#include <deque>

#include "Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Triple.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/User.h"
#include "llvm/IR/AbstractCallSite.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"


using namespace llvm;

struct TraceInterface {
  // user implemented
  Function *getTrace;
  Function *getChoice;
  Function *insertCall;
  Function *insertChoice;
  Function *newTrace;
  Function *freeTrace;
  // implemented by enzyme
  Function *sample;
};


class TraceUtils {

private:
  TraceInterface interface;
  CallInst *trace;

public:
  ProbProgMode mode;
  Function *newFunc;
  Function *oldFunc;
  Value *conditioning_trace;

public:
  ValueToValueMapTy originalToNewFn;
  SmallPtrSetImpl<Function*> &generativeFunctions;

public:
  TraceUtils(ProbProgMode mode, TraceInterface interface, Function *newFunc, Function *oldFunc, ValueToValueMapTy vmap, SmallPtrSetImpl<Function*> &generativeFunctions, Value *conditioning_trace) : mode(mode), interface(interface), conditioning_trace(conditioning_trace), newFunc(newFunc), oldFunc(oldFunc), generativeFunctions(generativeFunctions) {
    originalToNewFn.insert(vmap.begin(), vmap.end());
    originalToNewFn.getMDMap() = vmap.getMDMap();
  }
  
  TraceUtils(ProbProgMode mode, TraceInterface interface, Function* F, SmallPtrSetImpl<Function*> &generativeFunctions, Value *conditioning_trace) : mode(mode), interface(interface), conditioning_trace(conditioning_trace), oldFunc(F), generativeFunctions(generativeFunctions) {
    FunctionType *orig_FTy = oldFunc->getFunctionType();
    SmallVector<Type *, 4> params;

    for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
        params.push_back(orig_FTy->getParamType(i));
    }
    
    Type *NewTy = StructType::get(oldFunc->getContext(), {oldFunc->getReturnType(), interface.newTrace->getReturnType()});

    FunctionType *FTy = FunctionType::get(NewTy, params, oldFunc->isVarArg());
    newFunc = Function::Create(FTy, oldFunc->getLinkage(),
                         "simulate_" + oldFunc->getName(), oldFunc->getParent());

    newFunc->setLinkage(Function::LinkageTypes::InternalLinkage);
    
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
    CloneFunctionInto(newFunc, oldFunc, originalToNewFn, true, Returns, "", nullptr);
  #endif
    
    newFunc->setLinkage(Function::LinkageTypes::InternalLinkage);
    
    // Create trace for current function
    
    IRBuilder<> Builder(newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    trace = CreateTrace(Builder);
    DILocation *L = DILocation::get(newFunc->getContext(), 0, 0, newFunc->getSubprogram());
    trace->setDebugLoc(DebugLoc(L));
    
    // Replace returns with ret trace
    
    SmallVector<ReturnInst*, 3> toReplace;
    for (auto&& BB: *newFunc) {
      for (auto&& Inst: BB) {
        if (auto Ret = dyn_cast<ReturnInst>(&Inst)) {
          toReplace.push_back(Ret);
        }
      }
    }
    
    for (auto Ret: toReplace) {
      IRBuilder<> Builder(Ret);
      Value *retvals[2] = {Ret->getReturnValue(), trace};
      Builder.CreateAggregateRet(retvals, 2);
      Ret->eraseFromParent();
    }
  };

public:
    TraceInterface getTraceInterface() {
      return interface;
    }
  
    Value *getTrace() {
      return trace;
    }
  
  CallInst* CreateTrace(IRBuilder<> &Builder) {
    auto trace = Builder.CreateCall(interface.newTrace->getFunctionType(), interface.newTrace);
    trace->setName("trace");
    return trace;
  }

  CallInst* InsertChoice(IRBuilder<> &Builder, Value* address, Value *score, Value *choice) {
    Value *args[] = {
        trace,
        address,
        score,
        choice
    };
    auto call = Builder.CreateCall(interface.insertChoice->getFunctionType(), interface.insertChoice, args);
    return call;
    }

  CallInst* InsertCall(IRBuilder<> &Builder, Value *address, Value *subtrace, Value *score, Value *noise) {
    Value *args[] = {
      trace,
      address,
      subtrace,
      score,
      noise
    };
    auto call = Builder.CreateCall(interface.insertCall->getFunctionType(), interface.insertCall, args);
    return call;
    }
  
  CallInst *GetTrace(IRBuilder<> &Builder, Value *address) {
    Value *args[] = {
      conditioning_trace,
      address
    };
    auto call = Builder.CreateCall(interface.getTrace->getFunctionType(), interface.getTrace, args);
    return call;
  }
  
  CallInst *GetChoice(IRBuilder<> &Builder, Value *address) {
    Value *args[] = {
      conditioning_trace,
      address
    };
                                                                                  //TODO getChoice
    auto call = Builder.CreateCall(interface.getTrace->getFunctionType(), interface.getTrace, args);
    return call;
  }
};

#endif /* TraceUtils_h */
