#ifndef TraceUtils_h
#define TraceUtils_h

#include <deque>

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

using namespace llvm;

class TraceInterface {

public:
  // implemented by enzyme
  virtual Function *getSampleFunction() = 0;

  // user implemented
  virtual Value *getTrace() = 0;
  virtual FunctionType *getTraceTy() = 0;

  virtual Value *getChoice() = 0;
  virtual FunctionType *getChoiceTy() = 0;

  virtual Value *insertCall() = 0;
  virtual FunctionType *insertCallTy() = 0;

  virtual Value *insertChoice() = 0;
  virtual FunctionType *insertChoiceTy() = 0;

  virtual Value *newTrace() = 0;
  virtual FunctionType *newTraceTy() = 0;

  virtual Value *freeTrace() = 0;
  virtual FunctionType *freeTraceTy() = 0;

  virtual Value *hasCall() = 0;
  virtual FunctionType *hasCallTy() = 0;

  virtual Value *hasChoice() = 0;
  virtual FunctionType *hasChoiceTy() = 0;

  virtual ~TraceInterface() = default;
};

class StaticTraceInterface final : public TraceInterface {
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
    for (auto &&F : M->functions()) {
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

    assert(newTraceFunction != nullptr && freeTraceFunction != nullptr &&
           getTraceFunction != nullptr && getChoiceFunction != nullptr &&
           insertCallFunction != nullptr && insertChoiceFunction != nullptr &&
           hasCallFunction != nullptr && hasChoiceFunction != nullptr &&
           sampleFunction != nullptr);
  }

  ~StaticTraceInterface() = default;

public:
  // implemented by enzyme
  Function *getSampleFunction() { return sampleFunction; }

  // user implemented
  Value *getTrace() { return getTraceFunction; }
  FunctionType *getTraceTy() { return getTraceFunction->getFunctionType(); }

  Value *getChoice() { return getChoiceFunction; }
  FunctionType *getChoiceTy() { return getChoiceFunction->getFunctionType(); }

  Value *insertCall() { return insertCallFunction; }
  FunctionType *insertCallTy() { return insertCallFunction->getFunctionType(); }

  Value *insertChoice() { return insertChoiceFunction; }
  FunctionType *insertChoiceTy() {
    return insertChoiceFunction->getFunctionType();
  }

  Value *newTrace() { return newTraceFunction; }
  FunctionType *newTraceTy() { return newTraceFunction->getFunctionType(); }

  Value *freeTrace() { return freeTraceFunction; }
  FunctionType *freeTraceTy() { return freeTraceFunction->getFunctionType(); }

  Value *hasCall() { return hasCallFunction; }
  FunctionType *hasCallTy() { return hasCallFunction->getFunctionType(); }

  Value *hasChoice() { return hasChoiceFunction; }
  FunctionType *hasChoiceTy() { return hasChoiceFunction->getFunctionType(); }
};

class DynamicTraceInterface final : public TraceInterface {
private:
  Function *sampleFunction;
  Value *dynamicInterface;
  Function *F;

  LLVMContext &C;

private:
  Value *getTraceFunction = nullptr;
  Value *getChoiceFunction = nullptr;
  Value *insertCallFunction = nullptr;
  Value *insertChoiceFunction = nullptr;
  Value *newTraceFunction = nullptr;
  Value *freeTraceFunction = nullptr;
  Value *hasCallFunction = nullptr;
  Value *hasChoiceFunction = nullptr;

public:
  DynamicTraceInterface(Function *sampleFunction, Value *dynamicInterface,
                        Function *F)
      : sampleFunction(sampleFunction), dynamicInterface(dynamicInterface),
        F(F), C(sampleFunction->getContext()) {}

  ~DynamicTraceInterface() = default;

private:
  IntegerType *sizeType() { return IntegerType::getInt64Ty(C); }
  Type *stringType() { return IntegerType::getInt8PtrTy(C); }

public:
  // implemented by enzyme
  Function *getSampleFunction() { return sampleFunction; }

  // user implemented
  Value *getTrace() {
    if (getTraceFunction)
      return getTraceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());

    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(0));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return getTraceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(getTraceTy()), "get_trace");
  }

  FunctionType *getTraceTy() {
    return FunctionType::get(PointerType::getInt8PtrTy(C),
                             {PointerType::getInt8PtrTy(C), stringType()},
                             false);
  }

  Value *getChoice() {
    if (getChoiceFunction)
      return getChoiceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(1));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return getChoiceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(getChoiceTy()), "get_choice");
  }

  FunctionType *getChoiceTy() {
    return FunctionType::get(sizeType(),
                             {PointerType::getInt8PtrTy(C), stringType(),
                              PointerType::getInt8PtrTy(C), sizeType()},
                             false);
  }

  Value *insertCall() {
    if (insertCallFunction)
      return insertCallFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(2));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return insertCallFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(insertCallTy()), "insert_call");
  }

  FunctionType *insertCallTy() {
    return FunctionType::get(Type::getVoidTy(C),
                             {PointerType::getInt8PtrTy(C), stringType(),
                              PointerType::getInt8PtrTy(C)},
                             false);
  }

  Value *insertChoice() {
    if (insertChoiceFunction)
      return insertChoiceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(3));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return insertChoiceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(insertChoiceTy()), "insert_choice");
  }

  FunctionType *insertChoiceTy() {
    return FunctionType::get(Type::getVoidTy(C),
                             {PointerType::getInt8PtrTy(C), stringType(),
                              Type::getDoubleTy(C),
                              PointerType::getInt8PtrTy(C), sizeType()},
                             false);
  }

  Value *newTrace() {
    if (newTraceFunction)
      return newTraceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(4));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return newTraceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(newTraceTy()), "new_trace");
  }

  FunctionType *newTraceTy() {
    return FunctionType::get(PointerType::getInt8PtrTy(C), {}, false);
  }

  Value *freeTrace() {
    if (freeTraceFunction)
      return freeTraceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(5));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return freeTraceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(freeTraceTy()), "free_trace");
  }

  FunctionType *freeTraceTy() {
    return FunctionType::get(Type::getVoidTy(C), {PointerType::getInt8PtrTy(C)},
                             false);
  }

  Value *hasCall() {
    if (hasCallFunction)
      return hasCallFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(6));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return hasCallFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(hasCallTy()), "has_call");
  }

  FunctionType *hasCallTy() {
    return FunctionType::get(Type::getInt1Ty(C),
                             {PointerType::getInt8PtrTy(C), stringType()},
                             false);
  }

  Value *hasChoice() {
    if (hasChoiceFunction)
      return hasChoiceFunction;

    IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    auto ptr = Builder.CreateInBoundsGEP(Builder.getInt8PtrTy(),
                                         dynamicInterface, Builder.getInt32(7));
    auto load = Builder.CreateLoad(Builder.getInt8PtrTy(), ptr);
    return hasChoiceFunction = Builder.CreatePointerCast(
               load, PointerType::getUnqual(hasChoiceTy()), "has_choice");
  }

  FunctionType *hasChoiceTy() {
    return FunctionType::get(Type::getInt1Ty(C),
                             {PointerType::getInt8PtrTy(C), stringType()},
                             false);
  }
};

class TraceUtils {

private:
  TraceInterface *interface = nullptr;
  Value *dynamic_interface = nullptr;
  Instruction *trace;
  Value *observations = nullptr;

public:
  ProbProgMode mode;
  Function *newFunc;
  Function *oldFunc;

public:
  ValueToValueMapTy originalToNewFn;
  SmallPtrSetImpl<Function *> &generativeFunctions;

public:
  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, Function *newFunc,
             Function *oldFunc, ValueToValueMapTy vmap,
             SmallPtrSetImpl<Function *> &generativeFunctions)
      : mode(mode), newFunc(newFunc), oldFunc(oldFunc),
        generativeFunctions(generativeFunctions) {
    originalToNewFn.insert(vmap.begin(), vmap.end());
    originalToNewFn.getMDMap() = vmap.getMDMap();
  }

  TraceUtils(ProbProgMode mode, bool has_dynamic_interface, Function *F,
             SmallPtrSetImpl<Function *> &generativeFunctions)
      : mode(mode), oldFunc(F), generativeFunctions(generativeFunctions) {
    auto &Context = oldFunc->getContext();

    if (!has_dynamic_interface) {
      interface = new StaticTraceInterface(F->getParent());
    }

    FunctionType *orig_FTy = oldFunc->getFunctionType();
    Type *traceType = !has_dynamic_interface
                          ? interface->newTraceTy()->getReturnType()
                          : PointerType::getInt8PtrTy(Context);
    SmallVector<Type *, 4> params;

    for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
      params.push_back(orig_FTy->getParamType(i));
    }

    if (has_dynamic_interface)
      params.push_back(
          PointerType::getUnqual(PointerType::getInt8PtrTy(Context)));

    if (mode == ProbProgMode::Condition)
      params.push_back(traceType);

    Type *RetTy = traceType;
    if (!oldFunc->getReturnType()->isVoidTy())
      RetTy = StructType::get(Context, {oldFunc->getReturnType(), traceType});

    FunctionType *FTy = FunctionType::get(RetTy, params, oldFunc->isVarArg());

    Twine Name = (mode == ProbProgMode::Condition ? "condition_" : "trace_") +
                 Twine(oldFunc->getName());

    newFunc = Function::Create(FTy, Function::LinkageTypes::InternalLinkage,
                               Name, oldFunc->getParent());

    auto DestArg = newFunc->arg_begin();
    auto SrcArg = oldFunc->arg_begin();

    for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
      Argument *arg = SrcArg;
      originalToNewFn[arg] = DestArg;
      DestArg->setName(arg->getName());
      DestArg++;
      SrcArg++;
    }

    if (has_dynamic_interface) {
      auto arg = newFunc->arg_end() - (1 + (mode == ProbProgMode::Condition));
      dynamic_interface = arg;
      arg->setName("interface");
      arg->addAttr(Attribute::ReadOnly);
      arg->addAttr(Attribute::NoCapture);
    }

    if (mode == ProbProgMode::Condition) {
      auto arg = newFunc->arg_end() - 1;
      observations = arg;
      arg->setName("observations");
      if (oldFunc->getReturnType()->isVoidTy())
        arg->addAttr(Attribute::Returned);
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

    if (has_dynamic_interface) {
      Function *sample = nullptr;
      for (auto &&interface_func : F->getParent()->functions()) {
        if (interface_func.getName().contains("__enzyme_sample")) {
          assert(interface_func.getFunctionType()->getNumParams() >= 3);
          sample = &interface_func;
        }
      }
      interface = new DynamicTraceInterface(sample, dynamic_interface, newFunc);
    }

    // Create trace for current function

    IRBuilder<> Builder(
        newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    Builder.SetCurrentDebugLocation(oldFunc->getEntryBlock()
                                        .getFirstNonPHIOrDbgOrLifetime()
                                        ->getDebugLoc());

    trace = CreateTrace(Builder);

    // Replace returns with ret trace

    SmallVector<ReturnInst *, 3> toReplace;
    for (auto &&BB : *newFunc) {
      for (auto &&Inst : BB) {
        if (auto Ret = dyn_cast<ReturnInst>(&Inst)) {
          toReplace.push_back(Ret);
        }
      }
    }

    for (auto Ret : toReplace) {
      IRBuilder<> Builder(Ret);
      if (Ret->getReturnValue()) {
        Value *retvals[2] = {Ret->getReturnValue(), trace};
        Builder.CreateAggregateRet(retvals, 2);
      } else {
        Builder.CreateRet(trace);
      }
      Ret->eraseFromParent();
    }
  };

  ~TraceUtils() { delete interface; }

public:
  TraceInterface *getTraceInterface() { return interface; }

  Value *getDynamicTraceInterface() { return dynamic_interface; }

  bool hasDynamicTraceInterface() { return dynamic_interface != nullptr; }

  Value *getTrace() { return trace; }

  CallInst *CreateTrace(IRBuilder<> &Builder, const Twine &Name = "trace") {
    return Builder.CreateCall(interface->newTraceTy(), interface->newTrace(),
                              {}, Name);
  }

  CallInst *InsertChoice(IRBuilder<> &Builder, Value *address, Value *score,
                         Value *choice) {
    auto size = choice->getType()->getPrimitiveSizeInBits() / 8;
    Type *size_type = interface->getChoiceTy()->getParamType(3);

    auto M = interface->getSampleFunction()->getParent();
    auto &DL = M->getDataLayout();
    auto pointersize = DL.getPointerSizeInBits();

    Value *retval;
    if (choice->getType()->isPointerTy()) {
      retval = Builder.CreatePointerCast(choice, Builder.getInt8PtrTy());
    } else {
      IRBuilder<> AllocaBuilder(
          newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
      auto alloca = AllocaBuilder.CreateAlloca(choice->getType(), nullptr,
                                               choice->getName() + ".ptr");
      Builder.CreateStore(choice, alloca);
      bool fitsInPointer =
          choice->getType()->getPrimitiveSizeInBits() == pointersize;
      if (fitsInPointer) {
        auto dblptr =
            PointerType::get(Builder.getInt8PtrTy(), DL.getAllocaAddrSpace());
        retval = Builder.CreateLoad(Builder.getInt8PtrTy(),
                                    Builder.CreatePointerCast(alloca, dblptr));
      } else {
        retval = alloca;
      }
    }

    Value *args[] = {trace, address, score, retval,
                     ConstantInt::get(size_type, size)};

    auto call = Builder.CreateCall(interface->insertChoiceTy(),
                                   interface->insertChoice(), args);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  CallInst *InsertCall(IRBuilder<> &Builder, Value *address, Value *subtrace) {
    Value *args[] = {trace, address, subtrace};

    auto call = Builder.CreateCall(interface->insertCallTy(),
                                   interface->insertCall(), args);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  CallInst *GetTrace(IRBuilder<> &Builder, Value *address,
                     const Twine &Name = "") {
    assert(address->getType()->isPointerTy());

    Value *args[] = {observations, address};

    auto call = Builder.CreateCall(interface->getTraceTy(),
                                   interface->getTrace(), args, Name);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  Instruction *GetChoice(IRBuilder<> &Builder, Value *address, Type *choiceType,
                         const Twine &Name = "") {
    IRBuilder<> AllocaBuilder(
        newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
    AllocaInst *store_dest =
        AllocaBuilder.CreateAlloca(choiceType, nullptr, Name + ".ptr");
    auto preallocated_size = choiceType->getPrimitiveSizeInBits() / 8;
    Type *size_type = interface->getChoiceTy()->getParamType(3);

    Value *args[] = {
        observations, address,
        Builder.CreatePointerCast(store_dest, Builder.getInt8PtrTy()),
        ConstantInt::get(size_type, preallocated_size)};

    auto call = Builder.CreateCall(
        interface->getChoiceTy(), interface->getChoice(), args, Name + ".size");
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return Builder.CreateLoad(choiceType, store_dest, "from.trace." + Name);
  }

  Instruction *HasChoice(IRBuilder<> &Builder, Value *address,
                         const Twine &Name = "") {
    Value *args[]{observations, address};

    auto call = Builder.CreateCall(interface->hasChoiceTy(),
                                   interface->hasChoice(), args, Name);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }

  Instruction *HasCall(IRBuilder<> &Builder, Value *address,
                       const Twine &Name = "") {
    Value *args[]{observations, address};

    auto call = Builder.CreateCall(interface->hasCallTy(), interface->hasCall(),
                                   args, Name);
    call->addParamAttr(1, Attribute::ReadOnly);
    call->addParamAttr(1, Attribute::NoCapture);
    return call;
  }
};

#endif /* TraceUtils_h */
