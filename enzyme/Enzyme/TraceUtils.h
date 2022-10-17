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

class TraceInterface {
  
public:
  // implemented by enzyme
  virtual Function* getSampleFunction() = 0;

  // user implemented
  virtual Value *getTrace(IRBuilder<> &Builder) = 0;
  virtual FunctionType *getTraceTy() = 0;
  
  virtual Value *getChoice(IRBuilder<> &Builder) = 0;
  virtual FunctionType *getChoiceTy() = 0;
  
  virtual Value *insertCall(IRBuilder<> &Builder) = 0;
  virtual FunctionType *insertCallTy() = 0;
  
  virtual Value *insertChoice(IRBuilder<> &Builder) = 0;
  virtual FunctionType *insertChoiceTy() = 0;
  
  virtual Value *newTrace(IRBuilder<> &Builder) = 0;
  virtual FunctionType *newTraceTy() = 0;
  
  virtual Value *freeTrace(IRBuilder<> &Builder) = 0;
  virtual FunctionType *freeTraceTy() = 0;
  
  virtual Value *hasCall(IRBuilder<> &Builder) = 0;
  virtual FunctionType *hasCallTy() = 0;
  
  virtual Value *hasChoice(IRBuilder<> &Builder) = 0;
  virtual FunctionType *hasChoiceTy() = 0;
  
  virtual ~TraceInterface() = default;
};

class StaticTraceInterface final: public TraceInterface {
private:
  Function *sampleFunction;
  
  // user implemented
  Function *getTraceFunction = nullptr;
  Function *getChoiceFunction = nullptr;
  Function *insertCallFunction = nullptr;
  Function *insertChoiceFunction = nullptr;
  Function *newTraceFunction = nullptr;
  Function *freeTraceFunction = nullptr;
  Function *hasCallFunction = nullptr;
  Function *hasChoiceFunction = nullptr;
  
public:
  StaticTraceInterface(Module *M) {
    for (auto&& F: M->functions()) {
      if (F.getName().contains("__enzyme_newtrace")) {
        assert(F.getFunctionType()->getNumParams() == 0);
        newTraceFunction = &F;
      } else if (F.getName().contains("__enzyme_freetrace")) {
        assert(F.getFunctionType()->getNumParams() == 1);
        freeTraceFunction = &F;
      } else if (F.getName().contains("__enzyme_get_trace")) {
        assert(F.getFunctionType()->getNumParams() == 2);
        getTraceFunction = &F;
      } else if (F.getName().contains("__enzyme_get_choice")) {
        assert(F.getFunctionType()->getNumParams() == 4);
        getChoiceFunction = &F;
      } else if (F.getName().contains("__enzyme_insert_call")) {
        assert(F.getFunctionType()->getNumParams() == 3);
        insertCallFunction = &F;
      } else if (F.getName().contains("__enzyme_insert_choice")) {
        assert(F.getFunctionType()->getNumParams() == 5);
        insertChoiceFunction = &F;
      } else if (F.getName().contains("__enzyme_has_call")) {
        assert(F.getFunctionType()->getNumParams() == 2);
        hasCallFunction = &F;
      } else if (F.getName().contains("__enzyme_has_choice")) {
        assert(F.getFunctionType()->getNumParams() == 2);
        hasChoiceFunction = &F;
      } else if (F.getName().contains("__enzyme_sample")) {
        assert(F.getFunctionType()->getNumParams() >= 3);
        sampleFunction = &F;
      }
    }
    
    assert(newTraceFunction != nullptr &&
           freeTraceFunction != nullptr &&
           getTraceFunction != nullptr &&
           getChoiceFunction != nullptr &&
           insertCallFunction != nullptr &&
           insertChoiceFunction != nullptr &&
           hasCallFunction != nullptr &&
           hasChoiceFunction != nullptr &&
           sampleFunction != nullptr);
  }
  
  ~StaticTraceInterface() = default;

  
public:
  // implemented by enzyme
  Function* getSampleFunction() { return sampleFunction; }

  // user implemented
  Value *getTrace(IRBuilder<> &Builder) { return getTraceFunction; }
  FunctionType *getTraceTy() { return getTraceFunction->getFunctionType(); }
  
  Value *getChoice(IRBuilder<> &Builder) { return getChoiceFunction; }
  FunctionType *getChoiceTy() { return getChoiceFunction->getFunctionType(); }
  
  Value *insertCall(IRBuilder<> &Builder) { return insertCallFunction; }
  FunctionType *insertCallTy() { return insertCallFunction->getFunctionType(); }
  
  Value *insertChoice(IRBuilder<> &Builder) { return insertChoiceFunction; }
  FunctionType *insertChoiceTy() { return insertChoiceFunction->getFunctionType(); }
  
  Value *newTrace(IRBuilder<> &Builder) { return newTraceFunction; }
  FunctionType *newTraceTy() { return newTraceFunction->getFunctionType(); }
  
  Value *freeTrace(IRBuilder<> &Builder) { return freeTraceFunction; }
  FunctionType *freeTraceTy() { return freeTraceFunction->getFunctionType(); }
  
  Value *hasCall(IRBuilder<> &Builder) { return hasCallFunction; }
  FunctionType *hasCallTy() { return hasCallFunction->getFunctionType(); }
  
  Value *hasChoice(IRBuilder<> &Builder) { return hasChoiceFunction; }
  FunctionType *hasChoiceTy() { return hasChoiceFunction->getFunctionType(); }
};


class DynamicTraceInterface final: public TraceInterface {
private:
  Function *sampleFunction;
  Value *dynamicInterface;

  LLVMContext &C;
  
public:
  DynamicTraceInterface(Function *sampleFunction, Value *dynamicInterface) : sampleFunction(sampleFunction), dynamicInterface(dynamicInterface), C(sampleFunction->getContext()) {}
  
  ~DynamicTraceInterface() = default;

  
private:
  IntegerType *sizeType() { return IntegerType::getInt64Ty(C); }
  Type *stringType() { return IntegerType::getInt8PtrTy(C); }
  
public:
  // implemented by enzyme
  Function* getSampleFunction() { return sampleFunction; }

  // user implemented
  Value *getTrace(IRBuilder<> &Builder) {
    Builder.getContext();
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(0));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(getTraceTy()));
  }
  FunctionType *getTraceTy() { return FunctionType::get(PointerType::getInt8PtrTy(C), {PointerType::getInt8PtrTy(C), stringType() }, false); }
  
  Value *getChoice(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(1));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(getChoiceTy()));
  }
  FunctionType *getChoiceTy() { return FunctionType::get(sizeType(), {PointerType::getInt8PtrTy(C), stringType(), PointerType::getInt8PtrTy(C), sizeType()}, false); }
  
  Value *insertCall(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(2));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(insertCallTy()));
  }
  FunctionType *insertCallTy() { return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C), stringType(), PointerType::getInt8PtrTy(C)}, false); }
  
  Value *insertChoice(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(3));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(insertChoiceTy()));
  }
  FunctionType *insertChoiceTy() { return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C), stringType(), Type::getDoubleTy(C), PointerType::getInt8PtrTy(C), sizeType()}, false); }
  
  Value *newTrace(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(4));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(newTraceTy()));
  }
  FunctionType *newTraceTy() { return FunctionType::get(PointerType::getInt8PtrTy(C), {}, false); }
  
  Value *freeTrace(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(5));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(freeTraceTy()));
    
  }
  FunctionType *freeTraceTy() { return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C)}, false); }
  
  Value *hasCall(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(6));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(hasCallTy()));
  }
  FunctionType *hasCallTy() { return FunctionType::get(Type::getInt1Ty(C), {PointerType::getInt8PtrTy(C), stringType()}, false); }
  
  Value *hasChoice(IRBuilder<> &Builder) {
    auto ptr = Builder.CreateInBoundsGEP(PointerType::getInt8PtrTy(C), dynamicInterface, Builder.getInt32(7));
    auto load = Builder.CreateLoad(PointerType::getInt8PtrTy(C), ptr);
    return Builder.CreatePointerCast(load, PointerType::getUnqual(hasChoiceTy()));
  }
  FunctionType *hasChoiceTy() { return FunctionType::get(Type::getInt1Ty(C), {PointerType::getInt8PtrTy(C), stringType()}, false); }
};


class TraceUtils {

private:
  TraceInterface *interface = nullptr;
  Value *dynamic_interface = nullptr;
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
  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, Function *newFunc, Function *oldFunc, ValueToValueMapTy vmap, SmallPtrSetImpl<Function*> &generativeFunctions) : mode(mode), newFunc(newFunc), oldFunc(oldFunc), generativeFunctions(generativeFunctions) {
    originalToNewFn.insert(vmap.begin(), vmap.end());
    originalToNewFn.getMDMap() = vmap.getMDMap();
  }
  
  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, Function* F, SmallPtrSetImpl<Function*> &generativeFunctions) : mode(mode), oldFunc(F), generativeFunctions(generativeFunctions) {
    
    if (!has_dynamic_interface) {
      interface = new StaticTraceInterface(F->getParent());
    }
    
    FunctionType *orig_FTy = oldFunc->getFunctionType();
    Type *traceType = !has_dynamic_interface ? interface->newTraceTy()->getReturnType() : PointerType::getInt8PtrTy(oldFunc->getContext());
    SmallVector<Type *, 4> params;

    for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
        params.push_back(orig_FTy->getParamType(i));
    }
    
    if (has_dynamic_interface)
      params.push_back(PointerType::getUnqual(PointerType::getInt8PtrTy(oldFunc->getContext())));
    
    if (mode == ProbProgMode::Condition)
      params.push_back(traceType);
    
    Type *NewTy = StructType::get(oldFunc->getContext(), {oldFunc->getReturnType(), traceType});

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
    
    if (has_dynamic_interface) {
      Function *sample = nullptr;
      for (auto&& interface_func: F->getParent()->functions()) {
        if (interface_func.getName().contains("__enzyme_sample")) {
          assert(interface_func.getFunctionType()->getNumParams() >= 3);
          sample = &interface_func;
        }
      }
      dynamic_interface = newFunc->getArg(newFunc->getFunctionType()->getNumParams() - (1 + (mode == ProbProgMode::Condition)));
      interface = new DynamicTraceInterface(sample, dynamic_interface);
    }
    
    // Create trace for current function
    
    IRBuilder<> Builder(newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    trace = CreateTrace(Builder);
    
    conditioning_trace = newFunc->getArg(newFunc->getFunctionType()->getNumParams() - 1);
    
    if (newFunc->getSubprogram()) {
      DILocation *L = DILocation::get(newFunc->getContext(), 0, 0, newFunc->getSubprogram());
      trace->setDebugLoc(DebugLoc(L));
    }
    
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
  TraceInterface* getTraceInterface() {
    return interface;
  }
  
  Value *getDynamicTraceInterface() {
    return dynamic_interface;
  }
  
  bool hasDynamicTraceInterface() {
    return dynamic_interface != nullptr;
  }
  
    Value *getTrace() {
      return trace;
    }
  
  CallInst* CreateTrace(IRBuilder<> &Builder) {
    auto trace = Builder.CreateCall(interface->newTraceTy(), interface->newTrace(Builder));
    trace->setName("trace");
    return trace;
  }

  CallInst* InsertChoice(IRBuilder<> &Builder, Value* address, Value *score, Value *choice) {
    auto size = choice->getType()->getPrimitiveSizeInBits() / 8;
    Type *size_type = interface->getChoiceTy()->getParamType(3);

    auto M = interface->getSampleFunction()->getParent();
    auto &DL = M->getDataLayout();
    auto pointersize = DL.getPointerSizeInBits();
    
    Value *retval;
    if (choice->getType()->isPointerTy()) {
      retval = Builder.CreatePointerCast(choice, Builder.getInt8PtrTy());
    } else {
      auto alloca = Builder.CreateAlloca(choice->getType());
      Builder.CreateStore(choice, alloca);
      bool fitsInPointer = choice->getType()->getPrimitiveSizeInBits() == pointersize;
      if (fitsInPointer) {
        auto dblptr = PointerType::get(Builder.getInt8PtrTy(), DL.getAllocaAddrSpace());
        retval = Builder.CreateLoad(Builder.getInt8PtrTy(), Builder.CreatePointerCast(alloca, dblptr));
      } else {
        retval = alloca;
      }
    }
    
    Value *args[] = {
        trace,
        address,
        score,
        retval,
        ConstantInt::get(size_type, size)
    };
    
    auto call = Builder.CreateCall(interface->insertChoiceTy(), interface->insertChoice(Builder), args);
    return call;
    }

  CallInst* InsertCall(IRBuilder<> &Builder, Value *address, Value *subtrace) {
    Value *args[] = {
      trace,
      address,
      subtrace
    };
    
    auto call = Builder.CreateCall(interface->insertCallTy(), interface->insertCall(Builder), args);
    return call;
    }
  
  CallInst *GetTrace(IRBuilder<> &Builder, Value *address) {
    assert(address->getType()->isPointerTy());
    
    Value *args[] = {
      conditioning_trace,
      address
    };
    
    auto call = Builder.CreateCall(interface->getTraceTy(), interface->getTrace(Builder), args);
    return call;
  }
  
  Value *GetChoice(IRBuilder<> &Builder, Value *address, Type *choiceType) {
    AllocaInst *store_dest = Builder.CreateAlloca(choiceType);
    auto preallocated_size = choiceType->getPrimitiveSizeInBits() / 8;
    Type *size_type = interface->getChoiceTy()->getParamType(3);
    
    Value *args[] = {
      conditioning_trace,
      address,
      Builder.CreatePointerCast(store_dest, Builder.getInt8PtrTy()),
      ConstantInt::get(size_type, preallocated_size)
    };

    Builder.CreateCall(interface->getChoiceTy(), interface->getChoice(Builder), args);
    return Builder.CreateLoad(choiceType, store_dest);
  }
  
  Value *HasChoice(IRBuilder<> &Builder, Value *address) {
    Value *args[] {
      conditioning_trace,
      address
    };
    
    return Builder.CreateCall(interface->hasChoiceTy(), interface->hasChoice(Builder), args);
  }
  
  Value *HasCall(IRBuilder<> &Builder, Value *address) {
    Value *args[] {
      conditioning_trace,
      address
    };
    
    return Builder.CreateCall(interface->hasCallTy(), interface->hasCall(Builder), args);
  }
  
  ~TraceUtils() {
    delete interface;
  }
  
};

#endif /* TraceUtils_h */
