//===- CApi.cpp - Enzyme API exported to C for external use -----------===//
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
// This file defines various utility functions of Enzyme for access via C
//
//===----------------------------------------------------------------------===//
#include "CApi.h"
#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "DiffeGradientUtils.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"
#if LLVM_VERSION_MAJOR >= 16
#include "llvm/Analysis/TargetLibraryInfo.h"
#else
#include "SCEV/TargetLibraryInfo.h"
#endif
#include "TraceInterface.h"

// #include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/Attributor.h"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#define getAttribute getAttributeAtIndex
#define hasAttribute hasAttributeAtIndex
#endif

using namespace llvm;

TargetLibraryInfo eunwrap(LLVMTargetLibraryInfoRef P) {
  return TargetLibraryInfo(*reinterpret_cast<TargetLibraryInfoImpl *>(P));
}

EnzymeLogic &eunwrap(EnzymeLogicRef LR) { return *(EnzymeLogic *)LR; }

TraceInterface *eunwrap(EnzymeTraceInterfaceRef Ref) {
  return (TraceInterface *)Ref;
}

TypeAnalysis &eunwrap(EnzymeTypeAnalysisRef TAR) {
  return *(TypeAnalysis *)TAR;
}
AugmentedReturn *eunwrap(EnzymeAugmentedReturnPtr ARP) {
  return (AugmentedReturn *)ARP;
}
EnzymeAugmentedReturnPtr ewrap(const AugmentedReturn &AR) {
  return (EnzymeAugmentedReturnPtr)(&AR);
}

ConcreteType eunwrap(CConcreteType CDT, llvm::LLVMContext &ctx) {
  switch (CDT) {
  case DT_Anything:
    return BaseType::Anything;
  case DT_Integer:
    return BaseType::Integer;
  case DT_Pointer:
    return BaseType::Pointer;
  case DT_Half:
    return ConcreteType(llvm::Type::getHalfTy(ctx));
  case DT_Float:
    return ConcreteType(llvm::Type::getFloatTy(ctx));
  case DT_Double:
    return ConcreteType(llvm::Type::getDoubleTy(ctx));
  case DT_Unknown:
    return BaseType::Unknown;
  }
  llvm_unreachable("Unknown concrete type to unwrap");
}

std::vector<int> eunwrap(IntList IL) {
  std::vector<int> v;
  for (size_t i = 0; i < IL.size; i++) {
    v.push_back((int)IL.data[i]);
  }
  return v;
}
std::set<int64_t> eunwrap64(IntList IL) {
  std::set<int64_t> v;
  for (size_t i = 0; i < IL.size; i++) {
    v.insert((int64_t)IL.data[i]);
  }
  return v;
}
TypeTree eunwrap(CTypeTreeRef CTT) { return *(TypeTree *)CTT; }

CConcreteType ewrap(const ConcreteType &CT) {
  if (auto flt = CT.isFloat()) {
    if (flt->isHalfTy())
      return DT_Half;
    if (flt->isFloatTy())
      return DT_Float;
    if (flt->isDoubleTy())
      return DT_Double;
  } else {
    switch (CT.SubTypeEnum) {
    case BaseType::Integer:
      return DT_Integer;
    case BaseType::Pointer:
      return DT_Pointer;
    case BaseType::Anything:
      return DT_Anything;
    case BaseType::Unknown:
      return DT_Unknown;
    case BaseType::Float:
      llvm_unreachable("Illegal conversion of concretetype");
    }
  }
  llvm_unreachable("Illegal conversion of concretetype");
}

IntList ewrap(const std::vector<int> &offsets) {
  IntList IL;
  IL.size = offsets.size();
  IL.data = new int64_t[IL.size];
  for (size_t i = 0; i < offsets.size(); i++) {
    IL.data[i] = offsets[i];
  }
  return IL;
}

CTypeTreeRef ewrap(const TypeTree &TT) {
  return (CTypeTreeRef)(new TypeTree(TT));
}

FnTypeInfo eunwrap(CFnTypeInfo CTI, llvm::Function *F) {
  FnTypeInfo FTI(F);
  // auto &ctx = F->getContext();
  FTI.Return = eunwrap(CTI.Return);

  size_t argnum = 0;
  for (auto &arg : F->args()) {
    FTI.Arguments[&arg] = eunwrap(CTI.Arguments[argnum]);
    FTI.KnownValues[&arg] = eunwrap64(CTI.KnownValues[argnum]);
    argnum++;
  }
  return FTI;
}

extern "C" {

void EnzymeSetCLBool(void *ptr, uint8_t val) {
  auto cl = (llvm::cl::opt<bool> *)ptr;
  cl->setValue((bool)val);
}

uint8_t EnzymeGetCLBool(void *ptr) {
  auto cl = (llvm::cl::opt<bool> *)ptr;
  return (uint8_t)(bool)cl->getValue();
}

void EnzymeSetCLInteger(void *ptr, int64_t val) {
  auto cl = (llvm::cl::opt<int> *)ptr;
  cl->setValue((int)val);
}

int64_t EnzymeGetCLInteger(void *ptr) {
  auto cl = (llvm::cl::opt<int> *)ptr;
  return (int64_t)cl->getValue();
}

EnzymeLogicRef CreateEnzymeLogic(uint8_t PostOpt) {
  return (EnzymeLogicRef)(new EnzymeLogic((bool)PostOpt));
}

EnzymeTraceInterfaceRef FindEnzymeStaticTraceInterface(LLVMModuleRef M) {
  return (EnzymeTraceInterfaceRef)(new StaticTraceInterface(unwrap(M)));
}

EnzymeTraceInterfaceRef CreateEnzymeStaticTraceInterface(
    LLVMContextRef C, LLVMValueRef getTraceFunction,
    LLVMValueRef getChoiceFunction, LLVMValueRef insertCallFunction,
    LLVMValueRef insertChoiceFunction, LLVMValueRef insertArgumentFunction,
    LLVMValueRef insertReturnFunction, LLVMValueRef insertFunctionFunction,
    LLVMValueRef insertChoiceGradientFunction,
    LLVMValueRef insertArgumentGradientFunction, LLVMValueRef newTraceFunction,
    LLVMValueRef freeTraceFunction, LLVMValueRef hasCallFunction,
    LLVMValueRef hasChoiceFunction) {
  return (EnzymeTraceInterfaceRef)(new StaticTraceInterface(
      *unwrap(C), cast<Function>(unwrap(getTraceFunction)),
      cast<Function>(unwrap(getChoiceFunction)),
      cast<Function>(unwrap(insertCallFunction)),
      cast<Function>(unwrap(insertChoiceFunction)),
      cast<Function>(unwrap(insertArgumentFunction)),
      cast<Function>(unwrap(insertReturnFunction)),
      cast<Function>(unwrap(insertFunctionFunction)),
      cast<Function>(unwrap(insertChoiceGradientFunction)),
      cast<Function>(unwrap(insertArgumentGradientFunction)),
      cast<Function>(unwrap(newTraceFunction)),
      cast<Function>(unwrap(freeTraceFunction)),
      cast<Function>(unwrap(hasCallFunction)),
      cast<Function>(unwrap(hasChoiceFunction))));
};

EnzymeTraceInterfaceRef
CreateEnzymeDynamicTraceInterface(LLVMValueRef interface, LLVMValueRef F) {
  return (EnzymeTraceInterfaceRef)(new DynamicTraceInterface(
      unwrap(interface), cast<Function>(unwrap(F))));
}

void ClearEnzymeLogic(EnzymeLogicRef Ref) { eunwrap(Ref).clear(); }

void EnzymeLogicErasePreprocessedFunctions(EnzymeLogicRef Ref) {
  auto &Logic = eunwrap(Ref);
  for (const auto &pair : Logic.PPC.cache)
    pair.second->eraseFromParent();
}

void FreeEnzymeLogic(EnzymeLogicRef Ref) { delete (EnzymeLogic *)Ref; }

void FreeTraceInterface(EnzymeTraceInterfaceRef Ref) {
  delete (TraceInterface *)Ref;
}

EnzymeTypeAnalysisRef CreateTypeAnalysis(EnzymeLogicRef Log,
                                         char **customRuleNames,
                                         CustomRuleType *customRules,
                                         size_t numRules) {
  TypeAnalysis *TA = new TypeAnalysis(((EnzymeLogic *)Log)->PPC.FAM);
  for (size_t i = 0; i < numRules; i++) {
    CustomRuleType rule = customRules[i];
    TA->CustomRules[customRuleNames[i]] =
        [=](int direction, TypeTree &returnTree, ArrayRef<TypeTree> argTrees,
            ArrayRef<std::set<int64_t>> knownValues, CallInst *call,
            TypeAnalyzer *TA) -> uint8_t {
      CTypeTreeRef creturnTree = (CTypeTreeRef)(&returnTree);
      CTypeTreeRef *cargs = new CTypeTreeRef[argTrees.size()];
      IntList *kvs = new IntList[argTrees.size()];
      for (size_t i = 0; i < argTrees.size(); ++i) {
        cargs[i] = (CTypeTreeRef)(&(argTrees[i]));
        kvs[i].size = knownValues[i].size();
        kvs[i].data = new int64_t[kvs[i].size];
        size_t j = 0;
        for (auto val : knownValues[i]) {
          kvs[i].data[j] = val;
          j++;
        }
      }
      uint8_t result = rule(direction, creturnTree, cargs, kvs, argTrees.size(),
                            wrap(call), TA);
      delete[] cargs;
      for (size_t i = 0; i < argTrees.size(); ++i) {
        delete[] kvs[i].data;
      }
      delete[] kvs;
      return result;
    };
  }
  return (EnzymeTypeAnalysisRef)TA;
}

void ClearTypeAnalysis(EnzymeTypeAnalysisRef TAR) { eunwrap(TAR).clear(); }

void FreeTypeAnalysis(EnzymeTypeAnalysisRef TAR) {
  TypeAnalysis *TA = (TypeAnalysis *)TAR;
  delete TA;
}

void *EnzymeAnalyzeTypes(EnzymeTypeAnalysisRef TAR, CFnTypeInfo CTI,
                         LLVMValueRef F) {
  FnTypeInfo FTI(eunwrap(CTI, cast<Function>(unwrap(F))));
  return (void *)&((TypeAnalysis *)TAR)->analyzeFunction(FTI).analyzer;
}

void *EnzymeGradientUtilsTypeAnalyzer(GradientUtils *G) {
  return (void *)&G->TR.analyzer;
}

void EnzymeGradientUtilsErase(GradientUtils *G, LLVMValueRef I) {
  return G->erase(cast<Instruction>(unwrap(I)));
}
void EnzymeGradientUtilsEraseWithPlaceholder(GradientUtils *G, LLVMValueRef I,
                                             uint8_t erase) {
  return G->eraseWithPlaceholder(cast<Instruction>(unwrap(I)),
                                 "_replacementABI", erase != 0);
}

void EnzymeGradientUtilsReplaceAWithB(GradientUtils *G, LLVMValueRef A,
                                      LLVMValueRef B) {
  return G->replaceAWithB(unwrap(A), unwrap(B));
}

void EnzymeRegisterAllocationHandler(char *Name, CustomShadowAlloc AHandle,
                                     CustomShadowFree FHandle) {
  shadowHandlers[Name] = [=](IRBuilder<> &B, CallInst *CI,
                             ArrayRef<Value *> Args,
                             GradientUtils *gutils) -> llvm::Value * {
    SmallVector<LLVMValueRef, 3> refs;
    for (auto a : Args)
      refs.push_back(wrap(a));
    return unwrap(
        AHandle(wrap(&B), wrap(CI), Args.size(), refs.data(), gutils));
  };
  shadowErasers[Name] = [=](IRBuilder<> &B, Value *ToFree) -> llvm::CallInst * {
    return cast_or_null<CallInst>(unwrap(FHandle(wrap(&B), wrap(ToFree))));
  };
}

void EnzymeRegisterCallHandler(char *Name,
                               CustomAugmentedFunctionForward FwdHandle,
                               CustomFunctionReverse RevHandle) {
  auto &pair = customCallHandlers[Name];
  pair.first = [=](IRBuilder<> &B, CallInst *CI, GradientUtils &gutils,
                   Value *&normalReturn, Value *&shadowReturn,
                   Value *&tape) -> bool {
    LLVMValueRef normalR = wrap(normalReturn);
    LLVMValueRef shadowR = wrap(shadowReturn);
    LLVMValueRef tapeR = wrap(tape);
    uint8_t noMod =
        FwdHandle(wrap(&B), wrap(CI), &gutils, &normalR, &shadowR, &tapeR);
    normalReturn = unwrap(normalR);
    shadowReturn = unwrap(shadowR);
    tape = unwrap(tapeR);
    return noMod != 0;
  };
  pair.second = [=](IRBuilder<> &B, CallInst *CI, DiffeGradientUtils &gutils,
                    Value *tape) {
    RevHandle(wrap(&B), wrap(CI), &gutils, wrap(tape));
  };
}

void EnzymeRegisterFwdCallHandler(char *Name, CustomFunctionForward FwdHandle) {
  auto &pair = customFwdCallHandlers[Name];
  pair = [=](IRBuilder<> &B, CallInst *CI, GradientUtils &gutils,
             Value *&normalReturn, Value *&shadowReturn) -> bool {
    LLVMValueRef normalR = wrap(normalReturn);
    LLVMValueRef shadowR = wrap(shadowReturn);
    uint8_t noMod = FwdHandle(wrap(&B), wrap(CI), &gutils, &normalR, &shadowR);
    normalReturn = unwrap(normalR);
    shadowReturn = unwrap(shadowR);
    return noMod != 0;
  };
}

uint64_t EnzymeGradientUtilsGetWidth(GradientUtils *gutils) {
  return gutils->getWidth();
}

LLVMTypeRef EnzymeGradientUtilsGetShadowType(GradientUtils *gutils,
                                             LLVMTypeRef T) {
  return wrap(gutils->getShadowType(unwrap(T)));
}

LLVMTypeRef EnzymeGetShadowType(uint64_t width, LLVMTypeRef T) {
  return wrap(GradientUtils::getShadowType(unwrap(T), width));
}

LLVMValueRef EnzymeGradientUtilsNewFromOriginal(GradientUtils *gutils,
                                                LLVMValueRef val) {
  return wrap(gutils->getNewFromOriginal(unwrap(val)));
}

CDerivativeMode EnzymeGradientUtilsGetMode(GradientUtils *gutils) {
  return (CDerivativeMode)gutils->mode;
}

CDIFFE_TYPE
EnzymeGradientUtilsGetDiffeType(GradientUtils *G, LLVMValueRef oval,
                                uint8_t foreignFunction) {
  return (CDIFFE_TYPE)(G->getDiffeType(unwrap(oval), foreignFunction != 0));
}

CDIFFE_TYPE
EnzymeGradientUtilsGetReturnDiffeType(GradientUtils *G, LLVMValueRef oval,
                                      uint8_t *needsPrimal,
                                      uint8_t *needsShadow) {
  bool needsPrimalB;
  bool needsShadowB;
  auto res = (CDIFFE_TYPE)(G->getReturnDiffeType(unwrap(oval), &needsPrimalB,
                                                 &needsShadowB));
  if (needsPrimal)
    *needsPrimal = needsPrimalB;
  if (needsShadow)
    *needsShadow = needsShadowB;
  return res;
}

void EnzymeGradientUtilsSetDebugLocFromOriginal(GradientUtils *gutils,
                                                LLVMValueRef val,
                                                LLVMValueRef orig) {
  return cast<Instruction>(unwrap(val))
      ->setDebugLoc(gutils->getNewFromOriginal(
          cast<Instruction>(unwrap(orig))->getDebugLoc()));
}

LLVMValueRef EnzymeGradientUtilsLookup(GradientUtils *gutils, LLVMValueRef val,
                                       LLVMBuilderRef B) {
  return wrap(gutils->lookupM(unwrap(val), *unwrap(B)));
}

LLVMValueRef EnzymeGradientUtilsInvertPointer(GradientUtils *gutils,
                                              LLVMValueRef val,
                                              LLVMBuilderRef B) {
  return wrap(gutils->invertPointerM(unwrap(val), *unwrap(B)));
}

LLVMValueRef EnzymeGradientUtilsDiffe(DiffeGradientUtils *gutils,
                                      LLVMValueRef val, LLVMBuilderRef B) {
  return wrap(gutils->diffe(unwrap(val), *unwrap(B)));
}

void EnzymeGradientUtilsAddToDiffe(DiffeGradientUtils *gutils, LLVMValueRef val,
                                   LLVMValueRef diffe, LLVMBuilderRef B,
                                   LLVMTypeRef T) {
  gutils->addToDiffe(unwrap(val), unwrap(diffe), *unwrap(B), unwrap(T));
}

void EnzymeGradientUtilsAddToInvertedPointerDiffe(
    DiffeGradientUtils *gutils, LLVMValueRef orig, LLVMValueRef origVal,
    LLVMTypeRef addingType, unsigned start, unsigned size, LLVMValueRef origptr,
    LLVMValueRef dif, LLVMBuilderRef BuilderM, unsigned align,
    LLVMValueRef mask) {
#if LLVM_VERSION_MAJOR >= 10
  MaybeAlign align2;
  if (align)
    align2 = MaybeAlign(align);
#else
  auto align2 = align;
#endif
  auto inst = cast_or_null<Instruction>(unwrap(orig));
  gutils->addToInvertedPtrDiffe(inst, unwrap(origVal), unwrap(addingType),
                                start, size, unwrap(origptr), unwrap(dif),
                                *unwrap(BuilderM), align2, unwrap(mask));
}

void EnzymeGradientUtilsAddToInvertedPointerDiffeTT(
    DiffeGradientUtils *gutils, LLVMValueRef orig, LLVMValueRef origVal,
    CTypeTreeRef vd, unsigned LoadSize, LLVMValueRef origptr,
    LLVMValueRef prediff, LLVMBuilderRef BuilderM, unsigned align,
    LLVMValueRef premask) {
#if LLVM_VERSION_MAJOR >= 10
  MaybeAlign align2;
  if (align)
    align2 = MaybeAlign(align);
#else
  auto align2 = align;
#endif
  auto inst = cast_or_null<Instruction>(unwrap(orig));
  gutils->addToInvertedPtrDiffe(inst, unwrap(origVal), *(TypeTree *)vd,
                                LoadSize, unwrap(origptr), unwrap(prediff),
                                *unwrap(BuilderM), align2, unwrap(premask));
}

void EnzymeGradientUtilsSetDiffe(DiffeGradientUtils *gutils, LLVMValueRef val,
                                 LLVMValueRef diffe, LLVMBuilderRef B) {
  gutils->setDiffe(unwrap(val), unwrap(diffe), *unwrap(B));
}

uint8_t EnzymeGradientUtilsIsConstantValue(GradientUtils *gutils,
                                           LLVMValueRef val) {
  return gutils->isConstantValue(unwrap(val));
}

uint8_t EnzymeGradientUtilsIsConstantInstruction(GradientUtils *gutils,
                                                 LLVMValueRef val) {
  return gutils->isConstantInstruction(cast<Instruction>(unwrap(val)));
}

LLVMBasicBlockRef EnzymeGradientUtilsAllocationBlock(GradientUtils *gutils) {
  return wrap(gutils->inversionAllocs);
}

void EnzymeGradientUtilsGetUncacheableArgs(GradientUtils *gutils,
                                           LLVMValueRef orig, uint8_t *data,
                                           uint64_t size) {
  if (gutils->mode == DerivativeMode::ForwardMode)
    return;

  CallInst *call = cast<CallInst>(unwrap(orig));

  auto found = gutils->overwritten_args_map_ptr->find(call);
  assert(found != gutils->overwritten_args_map_ptr->end());

  const std::vector<bool> &overwritten_args = found->second;

  if (size != overwritten_args.size()) {
    llvm::errs() << " orig: " << *call << "\n";
    llvm::errs() << " size: " << size
                 << " overwritten_args.size(): " << overwritten_args.size()
                 << "\n";
  }
  assert(size == overwritten_args.size());
  for (uint64_t i = 0; i < size; i++) {
    data[i] = overwritten_args[i];
  }
}

CTypeTreeRef EnzymeGradientUtilsAllocAndGetTypeTree(GradientUtils *gutils,
                                                    LLVMValueRef val) {
  auto v = unwrap(val);
  TypeTree TT = gutils->TR.query(v);
  TypeTree *pTT = new TypeTree(TT);
  return (CTypeTreeRef)pTT;
}

void EnzymeGradientUtilsDumpTypeResults(GradientUtils *gutils) {
  gutils->TR.dump();
}

void EnzymeGradientUtilsSubTransferHelper(
    GradientUtils *gutils, CDerivativeMode mode, LLVMTypeRef secretty,
    uint64_t intrinsic, uint64_t dstAlign, uint64_t srcAlign, uint64_t offset,
    uint8_t dstConstant, LLVMValueRef shadow_dst, uint8_t srcConstant,
    LLVMValueRef shadow_src, LLVMValueRef length, LLVMValueRef isVolatile,
    LLVMValueRef MTI, uint8_t allowForward, uint8_t shadowsLookedUp) {
  auto orig = unwrap(MTI);
  assert(orig);
  SubTransferHelper(gutils, (DerivativeMode)mode, unwrap(secretty),
                    (Intrinsic::ID)intrinsic, (unsigned)dstAlign,
                    (unsigned)srcAlign, (unsigned)offset, (bool)dstConstant,
                    unwrap(shadow_dst), (bool)srcConstant, unwrap(shadow_src),
                    unwrap(length), unwrap(isVolatile), cast<CallInst>(orig),
                    (bool)allowForward, (bool)shadowsLookedUp);
}

LLVMValueRef EnzymeCreateForwardDiff(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, CDerivativeMode mode,
    uint8_t freeMemory, unsigned width, LLVMTypeRef additionalArg,
    CFnTypeInfo typeInfo, uint8_t *_overwritten_args,
    size_t overwritten_args_size, EnzymeAugmentedReturnPtr augmented) {
  SmallVector<DIFFE_TYPE, 4> nconstant_args((DIFFE_TYPE *)constant_args,
                                            (DIFFE_TYPE *)constant_args +
                                                constant_args_size);
  std::vector<bool> overwritten_args;
  assert(overwritten_args_size == cast<Function>(unwrap(todiff))->arg_size());
  for (uint64_t i = 0; i < overwritten_args_size; i++) {
    overwritten_args.push_back(_overwritten_args[i]);
  }
  return wrap(eunwrap(Logic).CreateForwardDiff(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA), returnValue, (DerivativeMode)mode, freeMemory, width,
      unwrap(additionalArg), eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      overwritten_args, eunwrap(augmented)));
}
LLVMValueRef EnzymeCreatePrimalAndGradient(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, uint8_t dretUsed,
    CDerivativeMode mode, unsigned width, uint8_t freeMemory,
    LLVMTypeRef additionalArg, uint8_t forceAnonymousTape, CFnTypeInfo typeInfo,
    uint8_t *_overwritten_args, size_t overwritten_args_size,
    EnzymeAugmentedReturnPtr augmented, uint8_t AtomicAdd) {
  std::vector<DIFFE_TYPE> nconstant_args((DIFFE_TYPE *)constant_args,
                                         (DIFFE_TYPE *)constant_args +
                                             constant_args_size);
  std::vector<bool> overwritten_args;
  assert(overwritten_args_size == cast<Function>(unwrap(todiff))->arg_size());
  for (uint64_t i = 0; i < overwritten_args_size; i++) {
    overwritten_args.push_back(_overwritten_args[i]);
  }
  return wrap(eunwrap(Logic).CreatePrimalAndGradient(
      (ReverseCacheKey){
          .todiff = cast<Function>(unwrap(todiff)),
          .retType = (DIFFE_TYPE)retType,
          .constant_args = nconstant_args,
          .overwritten_args = overwritten_args,
          .returnUsed = (bool)returnValue,
          .shadowReturnUsed = (bool)dretUsed,
          .mode = (DerivativeMode)mode,
          .width = width,
          .freeMemory = (bool)freeMemory,
          .AtomicAdd = (bool)AtomicAdd,
          .additionalType = unwrap(additionalArg),
          .forceAnonymousTape = (bool)forceAnonymousTape,
          .typeInfo = eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      },
      eunwrap(TA), eunwrap(augmented)));
}
EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    EnzymeLogicRef Logic, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnUsed, uint8_t shadowReturnUsed,
    CFnTypeInfo typeInfo, uint8_t *_overwritten_args,
    size_t overwritten_args_size, uint8_t forceAnonymousTape, unsigned width,
    uint8_t AtomicAdd) {

  SmallVector<DIFFE_TYPE, 4> nconstant_args((DIFFE_TYPE *)constant_args,
                                            (DIFFE_TYPE *)constant_args +
                                                constant_args_size);
  std::vector<bool> overwritten_args;
  assert(overwritten_args_size == cast<Function>(unwrap(todiff))->arg_size());
  for (uint64_t i = 0; i < overwritten_args_size; i++) {
    overwritten_args.push_back(_overwritten_args[i]);
  }
  return ewrap(eunwrap(Logic).CreateAugmentedPrimal(
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA), returnUsed, shadowReturnUsed,
      eunwrap(typeInfo, cast<Function>(unwrap(todiff))), overwritten_args,
      forceAnonymousTape, width, AtomicAdd));
}

LLVMValueRef EnzymeCreateTrace(
    EnzymeLogicRef Logic, LLVMValueRef totrace, LLVMValueRef *sample_functions,
    size_t sample_functions_size, LLVMValueRef *observe_functions,
    size_t observe_functions_size, const char *active_random_variables[],
    size_t active_random_variables_size, CProbProgMode mode, uint8_t autodiff,
    EnzymeTraceInterfaceRef interface) {

  SmallPtrSet<Function *, 4> SampleFunctions;
  for (size_t i = 0; i < sample_functions_size; i++) {
    SampleFunctions.insert(cast<Function>(unwrap(sample_functions[i])));
  }

  SmallPtrSet<Function *, 4> ObserveFunctions;
  for (size_t i = 0; i < observe_functions_size; i++) {
    ObserveFunctions.insert(cast<Function>(unwrap(observe_functions[i])));
  }

  StringSet<> ActiveRandomVariables;
  for (size_t i = 0; i < active_random_variables_size; i++) {
    ActiveRandomVariables.insert(active_random_variables[i]);
  }

  return wrap(eunwrap(Logic).CreateTrace(
      cast<Function>(unwrap(totrace)), SampleFunctions, ObserveFunctions,
      ActiveRandomVariables, (ProbProgMode)mode, (bool)autodiff,
      eunwrap(interface)));
}

LLVMValueRef
EnzymeExtractFunctionFromAugmentation(EnzymeAugmentedReturnPtr ret) {
  auto AR = (AugmentedReturn *)ret;
  return wrap(AR->fn);
}

LLVMTypeRef
EnzymeExtractUnderlyingTapeTypeFromAugmentation(EnzymeAugmentedReturnPtr ret) {
  auto AR = (AugmentedReturn *)ret;
  return wrap(AR->tapeType);
}

LLVMTypeRef
EnzymeExtractTapeTypeFromAugmentation(EnzymeAugmentedReturnPtr ret) {
  auto AR = (AugmentedReturn *)ret;
  auto found = AR->returns.find(AugmentedStruct::Tape);
  if (found == AR->returns.end()) {
    return wrap((Type *)nullptr);
  }
  if (found->second == -1) {
    return wrap(AR->fn->getReturnType());
  }
  return wrap(
      cast<StructType>(AR->fn->getReturnType())->getTypeAtIndex(found->second));
}
void EnzymeExtractReturnInfo(EnzymeAugmentedReturnPtr ret, int64_t *data,
                             uint8_t *existed, size_t len) {
  assert(len == 3);
  auto AR = (AugmentedReturn *)ret;
  AugmentedStruct todo[] = {AugmentedStruct::Tape, AugmentedStruct::Return,
                            AugmentedStruct::DifferentialReturn};
  for (size_t i = 0; i < len; i++) {
    auto found = AR->returns.find(todo[i]);
    if (found != AR->returns.end()) {
      existed[i] = true;
      data[i] = (int64_t)found->second;
    } else {
      existed[i] = false;
    }
  }
}

CTypeTreeRef EnzymeNewTypeTree() { return (CTypeTreeRef)(new TypeTree()); }
CTypeTreeRef EnzymeNewTypeTreeCT(CConcreteType CT, LLVMContextRef ctx) {
  return (CTypeTreeRef)(new TypeTree(eunwrap(CT, *unwrap(ctx))));
}
CTypeTreeRef EnzymeNewTypeTreeTR(CTypeTreeRef CTR) {
  return (CTypeTreeRef)(new TypeTree(*(TypeTree *)(CTR)));
}
void EnzymeFreeTypeTree(CTypeTreeRef CTT) { delete (TypeTree *)CTT; }
uint8_t EnzymeSetTypeTree(CTypeTreeRef dst, CTypeTreeRef src) {
  return *(TypeTree *)dst = *(TypeTree *)src;
}
uint8_t EnzymeMergeTypeTree(CTypeTreeRef dst, CTypeTreeRef src) {
  return ((TypeTree *)dst)->orIn(*(TypeTree *)src, /*PointerIntSame*/ false);
}
uint8_t EnzymeCheckedMergeTypeTree(CTypeTreeRef dst, CTypeTreeRef src,
                                   uint8_t *legalP) {
  bool legal = true;
  bool res =
      ((TypeTree *)dst)
          ->checkedOrIn(*(TypeTree *)src, /*PointerIntSame*/ false, legal);
  *legalP = legal;
  return res;
}

void EnzymeTypeTreeOnlyEq(CTypeTreeRef CTT, int64_t x) {
  // TODO only inst
  *(TypeTree *)CTT = ((TypeTree *)CTT)->Only(x, nullptr);
}
void EnzymeTypeTreeData0Eq(CTypeTreeRef CTT) {
  *(TypeTree *)CTT = ((TypeTree *)CTT)->Data0();
}

void EnzymeTypeTreeLookupEq(CTypeTreeRef CTT, int64_t size, const char *dl) {
  *(TypeTree *)CTT = ((TypeTree *)CTT)->Lookup(size, DataLayout(dl));
}
void EnzymeTypeTreeCanonicalizeInPlace(CTypeTreeRef CTT, int64_t size,
                                       const char *dl) {
  ((TypeTree *)CTT)->CanonicalizeInPlace(size, DataLayout(dl));
}

CConcreteType EnzymeTypeTreeInner0(CTypeTreeRef CTT) {
  return ewrap(((TypeTree *)CTT)->Inner0());
}

void EnzymeTypeTreeShiftIndiciesEq(CTypeTreeRef CTT, const char *datalayout,
                                   int64_t offset, int64_t maxSize,
                                   uint64_t addOffset) {
  DataLayout DL(datalayout);
  *(TypeTree *)CTT =
      ((TypeTree *)CTT)->ShiftIndices(DL, offset, maxSize, addOffset);
}
const char *EnzymeTypeTreeToString(CTypeTreeRef src) {
  std::string tmp = ((TypeTree *)src)->str();
  char *cstr = new char[tmp.length() + 1];
  std::strcpy(cstr, tmp.c_str());

  return cstr;
}

// TODO deprecated
void EnzymeTypeTreeToStringFree(const char *cstr) { delete[] cstr; }

const char *EnzymeTypeAnalyzerToString(void *src) {
  auto TA = (TypeAnalyzer *)src;
  std::string str;
  raw_string_ostream ss(str);
  TA->dump(ss);
  ss.str();
  char *cstr = new char[str.length() + 1];
  std::strcpy(cstr, str.c_str());
  return cstr;
}

const char *EnzymeGradientUtilsInvertedPointersToString(GradientUtils *gutils,
                                                        void *src) {
  std::string str;
  raw_string_ostream ss(str);
  for (auto z : gutils->invertedPointers) {
    ss << "available inversion for " << *z.first << " of " << *z.second << "\n";
  }
  ss.str();
  char *cstr = new char[str.length() + 1];
  std::strcpy(cstr, str.c_str());
  return cstr;
}

LLVMValueRef EnzymeGradientUtilsCallWithInvertedBundles(
    GradientUtils *gutils, LLVMValueRef func, LLVMTypeRef funcTy,
    LLVMValueRef *args_vr, uint64_t args_size, LLVMValueRef orig_vr,
    CValueType *valTys, uint64_t valTys_size, LLVMBuilderRef B,
    uint8_t lookup) {
  auto orig = cast<CallInst>(unwrap(orig_vr));

  ArrayRef<ValueType> ar((ValueType *)valTys, valTys_size);

  IRBuilder<> &BR = *unwrap(B);

  auto Defs = gutils->getInvertedBundles(orig, ar, BR, lookup != 0);

  SmallVector<Value *, 1> args;
  for (size_t i = 0; i < args_size; i++) {
    args.push_back(unwrap(args_vr[i]));
  }

  auto callval = unwrap(func);

  auto res =
      BR.CreateCall(cast<FunctionType>(unwrap(funcTy)), callval, args, Defs);
  return wrap(res);
}

void EnzymeStringFree(const char *cstr) { delete[] cstr; }

void EnzymeMoveBefore(LLVMValueRef inst1, LLVMValueRef inst2,
                      LLVMBuilderRef B) {
  Instruction *I1 = cast<Instruction>(unwrap(inst1));
  Instruction *I2 = cast<Instruction>(unwrap(inst2));
  if (I1 != I2) {
    if (B != nullptr) {
      IRBuilder<> &BR = *unwrap(B);
      if (I1->getIterator() == BR.GetInsertPoint()) {
        if (I2->getNextNode() == nullptr)
          BR.SetInsertPoint(I1->getParent());
        else
          BR.SetInsertPoint(I1->getNextNode());
      }
    }
    I1->moveBefore(I2);
  }
}

void EnzymeSetMustCache(LLVMValueRef inst1) {
  Instruction *I1 = cast<Instruction>(unwrap(inst1));
  I1->setMetadata("enzyme_mustcache", MDNode::get(I1->getContext(), {}));
}

uint8_t EnzymeHasFromStack(LLVMValueRef inst1) {
  Instruction *I1 = cast<Instruction>(unwrap(inst1));
  return hasMetadata(I1, "enzyme_fromstack") != 0;
}

void EnzymeCloneFunctionDISubprogramInto(LLVMValueRef NF, LLVMValueRef F) {
  auto &OldFunc = *cast<Function>(unwrap(F));
  auto &NewFunc = *cast<Function>(unwrap(NF));
  auto OldSP = OldFunc.getSubprogram();
  if (!OldSP)
    return;
  DIBuilder DIB(*OldFunc.getParent(), /*AllowUnresolved=*/false,
                OldSP->getUnit());
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  DISubprogram::DISPFlags SPFlags = DISubprogram::SPFlagDefinition |
                                    DISubprogram::SPFlagOptimized |
                                    DISubprogram::SPFlagLocalToUnit;
  auto NewSP = DIB.createFunction(
      OldSP->getUnit(), NewFunc.getName(), NewFunc.getName(), OldSP->getFile(),
      /*LineNo=*/0, SPType, /*ScopeLine=*/0, DINode::FlagZero, SPFlags);
  NewFunc.setSubprogram(NewSP);
  DIB.finalizeSubprogram(NewSP);
  return;
}

void EnzymeReplaceFunctionImplementation(LLVMModuleRef M) {
  ReplaceFunctionImplementation(*unwrap(M));
}

#if LLVM_VERSION_MAJOR >= 15

static bool runAttributorOnFunctions(InformationCache &InfoCache,
                                     SetVector<Function *> &Functions,
                                     AnalysisGetter &AG,
                                     CallGraphUpdater &CGUpdater,
                                     bool DeleteFns, bool IsModulePass) {
  if (Functions.empty())
    return false;

  // Create an Attributor and initially empty information cache that is filled
  // while we identify default attribute opportunities.
  AttributorConfig AC(CGUpdater);
  AC.RewriteSignatures = false;
  AC.IsModulePass = IsModulePass;
  AC.DeleteFns = DeleteFns;
  Attributor A(Functions, InfoCache, AC);

  for (Function *F : Functions) {
    // Populate the Attributor with abstract attribute opportunities in the
    // function and the information cache with IR information.
    A.identifyDefaultAbstractAttributes(*F);
  }

  ChangeStatus Changed = A.run();

  return Changed == ChangeStatus::CHANGED;
}
struct MyAttributorLegacyPass : public ModulePass {
  static char ID;

  MyAttributorLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    AnalysisGetter AG;
    SetVector<Function *> Functions;
    for (Function &F : M)
      Functions.insert(&F);

    CallGraphUpdater CGUpdater;
    BumpPtrAllocator Allocator;
    InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ nullptr);
    return runAttributorOnFunctions(InfoCache, Functions, AG, CGUpdater,
                                    /* DeleteFns*/ true,
                                    /* IsModulePass */ true);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // FIXME: Think about passes we will preserve and add them here.
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};
extern "C++" char MyAttributorLegacyPass::ID = 0;
void EnzymeAddAttributorLegacyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(new MyAttributorLegacyPass());
}
#else
void EnzymeAddAttributorLegacyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAttributorLegacyPass());
}
#endif

LLVMMetadataRef EnzymeMakeNonConstTBAA(LLVMMetadataRef MD) {
  auto M = cast<MDNode>(unwrap(MD));
  if (M->getNumOperands() != 4)
    return MD;
  auto CAM = dyn_cast<ConstantAsMetadata>(M->getOperand(3));
  if (!CAM)
    return MD;
  if (!CAM->getValue()->isOneValue())
    return MD;
  SmallVector<Metadata *, 4> MDs;
  for (auto &M : M->operands())
    MDs.push_back(M);
  MDs[3] =
      ConstantAsMetadata::get(ConstantInt::get(CAM->getValue()->getType(), 0));
  return wrap(MDNode::get(M->getContext(), MDs));
}
void EnzymeCopyMetadata(LLVMValueRef inst1, LLVMValueRef inst2) {
  cast<Instruction>(unwrap(inst1))
      ->copyMetadata(*cast<Instruction>(unwrap(inst2)));
}
LLVMMetadataRef EnzymeAnonymousAliasScopeDomain(const char *str,
                                                LLVMContextRef ctx) {
  MDBuilder MDB(*unwrap(ctx));
  MDNode *scope = MDB.createAnonymousAliasScopeDomain(str);
  return wrap(scope);
}
LLVMMetadataRef EnzymeAnonymousAliasScope(LLVMMetadataRef domain,
                                          const char *str) {
  auto dom = cast<MDNode>(unwrap(domain));
  MDBuilder MDB(dom->getContext());
  MDNode *scope = MDB.createAnonymousAliasScope(dom, str);
  return wrap(scope);
}
uint8_t EnzymeLowerSparsification(LLVMValueRef F, uint8_t replaceAll) {
  return LowerSparsification(cast<Function>(unwrap(F)), replaceAll != 0);
}

void EnzymeAttributeKnownFunctions(LLVMValueRef FC) {
  attributeKnownFunctions(*cast<Function>(unwrap(FC)));
}

void EnzymeSetCalledFunction(LLVMValueRef C_CI, LLVMValueRef C_F,
                             uint64_t *argrem, uint64_t num_argrem) {
  auto CI = cast<CallInst>(unwrap(C_CI));
  auto F = cast<Function>(unwrap(C_F));
  auto Attrs = CI->getAttributes();
  AttributeList NewAttrs;

  if (CI->getType() == F->getReturnType()) {
    for (auto attr : Attrs.getAttributes(AttributeList::ReturnIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::ReturnIndex, attr);
  }
  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  size_t argremsz = 0;
  size_t nexti = 0;
  SmallVector<Value *, 1> vals;
#if LLVM_VERSION_MAJOR >= 14
  for (size_t i = 0, end = CI->arg_size(); i < end; i++)
#else
  for (size_t i = 0, end = CI->getNumArgOperands(); i < end; i++)
#endif
  {
    if (argremsz < num_argrem) {
      if (i == argrem[argremsz]) {
        argremsz++;
        continue;
      }
    }
    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i))
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
    vals.push_back(CI->getArgOperand(i));
    nexti++;
  }
  assert(argremsz == num_argrem);

  IRBuilder<> B(CI);
  SmallVector<OperandBundleDef, 1> Bundles;
  for (unsigned I = 0, E = CI->getNumOperandBundles(); I != E; ++I)
    Bundles.emplace_back(CI->getOperandBundleAt(I));
  auto NC = B.CreateCall(F, vals, Bundles);
  NC->setAttributes(NewAttrs);
  NC->copyMetadata(*CI);

  if (CI->getType() == F->getReturnType())
    CI->replaceAllUsesWith(NC);

  if (!NC->getType()->isVoidTy())
    NC->takeName(CI);
  NC->setCallingConv(CI->getCallingConv());
  CI->eraseFromParent();
}

// clones a function to now miss the return or args
LLVMValueRef EnzymeCloneFunctionWithoutReturnOrArgs(LLVMValueRef FC,
                                                    uint8_t keepReturnU,
                                                    uint64_t *argrem,
                                                    uint64_t num_argrem) {
  auto F = cast<Function>(unwrap(FC));
  auto FT = F->getFunctionType();
  bool keepReturn = keepReturnU != 0;

  size_t argremsz = 0;
  size_t nexti = 0;
  SmallVector<Type *, 1> types;
  auto Attrs = F->getAttributes();
  AttributeList NewAttrs;
  for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {
    if (argremsz < num_argrem) {
      if (i == argrem[argremsz]) {
        argremsz++;
        continue;
      }
    }
    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i))
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
    types.push_back(F->getFunctionType()->getParamType(i));
    nexti++;
  }
  if (keepReturn) {
    for (auto attr : Attrs.getAttributes(AttributeList::ReturnIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::ReturnIndex, attr);
  }
  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  FunctionType *FTy = FunctionType::get(
      keepReturn ? F->getReturnType() : Type::getVoidTy(F->getContext()), types,
      FT->isVarArg());

  // Create the new function
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  // Loop over the arguments, copying the names of the mapped arguments over...
  nexti = 0;
  argremsz = 0;
  Function::arg_iterator DestI = NewF->arg_begin();
  for (const Argument &I : F->args()) {
    if (argremsz < num_argrem) {
      if (I.getArgNo() == argrem[argremsz]) {
        VMap[&I] = UndefValue::get(I.getType());
        argremsz++;
        continue;
      }
    }
    DestI->setName(I.getName()); // Copy the name over...
    VMap[&I] = &*DestI++;        // Add mapping to VMap
  }

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
#endif

  if (!keepReturn) {
    for (auto &B : *NewF) {
      if (auto RI = dyn_cast<ReturnInst>(B.getTerminator())) {
        IRBuilder<> B(RI);
        auto NRI = B.CreateRetVoid();
        NRI->copyMetadata(*RI);
        RI->eraseFromParent();
      }
    }
  }
  NewF->setAttributes(NewAttrs);
  if (!keepReturn)
    for (auto &Arg : NewF->args())
      Arg.removeAttr(Attribute::Returned);
  SmallVector<std::pair<unsigned, MDNode *>, 1> MD;
  F->getAllMetadata(MD);
  for (auto pair : MD)
    if (pair.first != LLVMContext::MD_dbg)
      NewF->addMetadata(pair.first, *pair.second);
  NewF->takeName(F);
  NewF->setCallingConv(F->getCallingConv());
  if (!keepReturn)
    NewF->addFnAttr("enzyme_retremove", "");

  if (num_argrem) {
    SmallVector<uint64_t, 1> previdx;
    if (Attrs.hasAttribute(AttributeList::FunctionIndex, "enzyme_parmremove")) {
      auto attr =
          Attrs.getAttribute(AttributeList::FunctionIndex, "enzyme_parmremove");
      auto prevstr = attr.getValueAsString();
      SmallVector<StringRef, 1> sub;
      prevstr.split(sub, ",");
      for (auto s : sub) {
        uint64_t ival;
        bool b = s.getAsInteger(10, ival);
        assert(!b);
        previdx.push_back(ival);
      }
    }
    SmallVector<uint64_t, 1> nextidx;
    for (size_t i = 0; i < num_argrem; i++) {
      auto val = argrem[i];
      nextidx.push_back(val);
    }

    size_t prevcnt = 0;
    size_t nextcnt = 0;
    SmallVector<uint64_t, 1> out;
    while (prevcnt < previdx.size() && nextcnt < nextidx.size()) {
      if (previdx[prevcnt] <= nextidx[nextcnt] + prevcnt) {
        out.push_back(previdx[prevcnt]);
        prevcnt++;
      } else {
        out.push_back(nextidx[nextcnt] + prevcnt);
        nextcnt++;
      }
    }
    while (prevcnt < previdx.size()) {
      out.push_back(previdx[prevcnt]);
      prevcnt++;
    }
    while (nextcnt < nextidx.size()) {
      out.push_back(nextidx[nextcnt] + prevcnt);
      nextcnt++;
    }

    std::string remstr;
    for (auto arg : out) {
      if (remstr.size())
        remstr += ",";
      remstr += std::to_string(arg);
    }

    NewF->addFnAttr("enzyme_parmremove", remstr);
  }
  return wrap(NewF);
}
LLVMTypeRef EnzymeAllocaType(LLVMValueRef V) {
  return wrap(cast<AllocaInst>(unwrap(V))->getAllocatedType());
}
}

static size_t num_rooting(llvm::Type *T, llvm::Function *F) {
  CountTrackedPointers tracked(T);
  if (tracked.derived) {
    llvm::errs() << *F << "\n";
    llvm::errs() << "Invalid Derived Type: " << *T << "\n";
  }
  assert(!tracked.derived);
  if (tracked.count != 0 && !tracked.all)
    return tracked.count;
  return 0;
}

extern "C" {
#if LLVM_VERSION_MAJOR >= 10
void EnzymeFixupJuliaCallingConvention(LLVMValueRef F_C) {
  auto F = cast<Function>(unwrap(F_C));
  if (F->empty())
    return;
  auto RT = F->getReturnType();
  std::set<size_t> srets;
  std::set<size_t> enzyme_srets;
  std::set<size_t> enzyme_srets_v;
  std::set<size_t> rroots;
  std::set<size_t> rroots_v;

  auto FT = F->getFunctionType();
  auto Attrs = F->getAttributes();
  for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           Attribute::StructRet))
      srets.insert(i);
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret"))
      enzyme_srets.insert(i);
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret_v"))
      enzyme_srets_v.insert(i);
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots"))
      rroots.insert(i);
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots_v"))
      rroots_v.insert(i);
  }
  // Regular julia function, needing no intervention
  if (srets.size() == 1) {
    assert(*srets.begin() == 0);
    assert(enzyme_srets.size() == 0);
    assert(enzyme_srets_v.size() == 0);
    assert(rroots_v.size() == 0);
    if (rroots.size()) {
      assert(rroots.size() == 1);
      assert(*rroots.begin() == 1);
    }
    return;
  }
  // No sret/rooting, no intervention needed.
  if (srets.size() == 0 && enzyme_srets.size() == 0 &&
      enzyme_srets_v.size() == 0 && rroots.size() == 0 &&
      rroots_v.size() == 0) {
    return;
  }

  assert(srets.size() == 0);

  SmallVector<Type *, 1> Types;
  if (!RT->isVoidTy()) {
    Types.push_back(RT);
  }

  for (auto idx : enzyme_srets) {
    llvm::Type *T = nullptr;
#if LLVM_VERSION_MAJOR >= 18
    llvm_unreachable("Unhandled");
    // T = F->getParamAttribute(idx, Attribute::AttrKind::ElementType)
    //        .getValueAsType();
#else
    T = FT->getParamType(idx)->getPointerElementType();
#endif
    Types.push_back(T);
  }
  for (auto idx : enzyme_srets_v) {
    llvm::Type *T = nullptr;
    auto AT = cast<ArrayType>(FT->getParamType(idx));
#if LLVM_VERSION_MAJOR >= 18
    llvm_unreachable("Unhandled");
    // T = F->getParamAttribute(idx, Attribute::AttrKind::ElementType)
    //         .getValueAsType();
#else
    T = AT->getElementType()->getPointerElementType();
#endif
    for (size_t i = 0; i < AT->getNumElements(); i++)
      Types.push_back(T);
  }

  StructType *ST =
      Types.size() <= 1 ? nullptr : StructType::get(F->getContext(), Types);
  Type *sretTy = nullptr;
  if (Types.size())
    sretTy = Types.size() == 1 ? Types[0] : ST;
  size_t numRooting = sretTy ? num_rooting(sretTy, F) : 0;

  auto T_jlvalue = StructType::get(F->getContext(), {});
  auto T_prjlvalue = PointerType::get(T_jlvalue, AddressSpace::Tracked);
  ArrayType *roots_AT =
      numRooting ? ArrayType::get(T_prjlvalue, numRooting) : nullptr;

  AttributeList NewAttrs;
  SmallVector<Type *, 1> types;
  size_t nexti = 0;
  if (sretTy) {
    types.push_back(PointerType::getUnqual(sretTy));
    NewAttrs = NewAttrs.addAttribute(
        F->getContext(), AttributeList::FirstArgIndex + nexti,
        Attribute::get(F->getContext(), Attribute::StructRet, sretTy));
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     Attribute::NoAlias);
    nexti++;
  }
  if (roots_AT) {
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     "enzymejl_returnRoots");
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     Attribute::NoAlias);
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FirstArgIndex + nexti,
                                     Attribute::WriteOnly);
    types.push_back(PointerType::getUnqual(roots_AT));
    nexti++;
  }
  for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {
    if (enzyme_srets.count(i) || enzyme_srets_v.count(i) || rroots.count(i) ||
        rroots_v.count(i))
      continue;

    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i))
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
    types.push_back(F->getFunctionType()->getParamType(i));
    nexti++;
  }
  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  FunctionType *FTy = FunctionType::get(Type::getVoidTy(F->getContext()), types,
                                        FT->isVarArg());

  // Create the new function
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();
  Argument *sret = nullptr;
  if (sretTy) {
    sret = &*DestI;
    DestI++;
  }
  Argument *roots = nullptr;
  if (roots_AT) {
    roots = &*DestI;
    DestI++;
  }
  // To handle the deleted args, it needs to be replaced by a non-arg operand.
  // This map contains the temporary phi nodes corresponding
  //

  std::map<size_t, PHINode *> delArgMap;
  for (Argument &I : F->args()) {
    auto i = I.getArgNo();
    if (enzyme_srets.count(i) || enzyme_srets_v.count(i) || rroots.count(i) ||
        rroots_v.count(i)) {
      VMap[&I] = delArgMap[i] = PHINode::Create(I.getType(), 0);
      continue;
    }
    assert(DestI != NewF->arg_end());
    DestI->setName(I.getName()); // Copy the name over...
    VMap[&I] = &*DestI++;        // Add mapping to VMap
  }

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
#endif

  SmallVector<CallInst *, 1> callers;
  for (auto U : F->users()) {
    auto CI = dyn_cast<CallInst>(U);
    assert(CI);
    assert(CI->getCalledFunction() == F);
    callers.push_back(CI);
  }

  size_t curOffset = 0;

  std::function<size_t(IRBuilder<> &, Value *, size_t)> recur =
      [&](IRBuilder<> &B, Value *V, size_t offset) -> size_t {
    auto T = V->getType();
    if (CountTrackedPointers(T).count == 0)
      return offset;
    if (roots_AT == nullptr)
      return offset;
    if (isa<PointerType>(T)) {
      if (isSpecialPtr(T)) {
        if (!roots_AT) {
          llvm::errs() << *V << " \n";
          llvm::errs() << *cast<Instruction>(V)->getParent()->getParent()
                       << " \n";
        }
        assert(roots_AT);
        assert(roots);
        auto gep = B.CreateConstInBoundsGEP2_32(roots_AT, roots, 0, offset);
        if (T != T_prjlvalue)
          V = B.CreatePointerCast(V, T_prjlvalue);
        B.CreateStore(V, gep);
        offset++;
      }
      return offset;
    } else if (auto ST = dyn_cast<StructType>(T)) {
      for (size_t i = 0; i < ST->getNumElements(); i++) {
        offset = recur(B, GradientUtils::extractMeta(B, V, i), offset);
      }
      return offset;
    } else if (auto AT = dyn_cast<ArrayType>(T)) {
      for (size_t i = 0; i < AT->getNumElements(); i++) {
        offset = recur(B, GradientUtils::extractMeta(B, V, i), offset);
      }
      return offset;
    } else if (auto VT = dyn_cast<VectorType>(T)) {
#if LLVM_VERSION_MAJOR >= 12
      size_t count = VT->getElementCount().getKnownMinValue();
#else
      size_t count = VT->getNumElements();
#endif
      for (size_t i = 0; i < count; i++) {
        offset = recur(B, B.CreateExtractElement(V, {i}), offset);
      }
      return offset;
    }
    return offset;
  };

  size_t sretCount = 0;
  if (!RT->isVoidTy()) {
    for (auto &RT : Returns) {
      IRBuilder<> B(RT);
      Value *gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, 0) : sret;
      Value *rval = RT->getReturnValue();
      B.CreateStore(rval, gep);
      recur(B, rval, 0);
      auto NR = B.CreateRetVoid();
      RT->eraseFromParent();
      RT = NR;
    }
    if (roots_AT)
      curOffset = CountTrackedPointers(RT).count;
    sretCount++;
  }

  for (auto i : enzyme_srets) {
    auto arg = delArgMap[i];
    assert(arg);
    SmallVector<Instruction *, 1> uses;
    SmallVector<unsigned, 1> op;
    for (auto &U : arg->uses()) {
      auto I = cast<Instruction>(U.getUser());
      uses.push_back(I);
      op.push_back(U.getOperandNo());
    }
    IRBuilder<> EB(&NewF->getEntryBlock().front());
    auto gep =
        ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount) : sret;
    for (size_t i = 0; i < uses.size(); i++) {
      uses[i]->setOperand(op[i], gep);
    }
    for (auto &RT : Returns) {
      IRBuilder<> B(RT);
      auto val = B.CreateLoad(Types[sretCount], gep);
      recur(B, val, curOffset);
    }
    if (roots_AT)
      curOffset += CountTrackedPointers(Types[sretCount]).count;
    sretCount++;
    delete arg;
  }
  for (auto i : enzyme_srets_v) {
    auto AT = cast<ArrayType>(FT->getParamType(i));
    auto arg = delArgMap[i];
    assert(arg);
    SmallVector<Instruction *, 1> uses;
    SmallVector<unsigned, 1> op;
    for (auto &U : arg->uses()) {
      auto I = cast<Instruction>(U.getUser());
      uses.push_back(I);
      op.push_back(U.getOperandNo());
    }
    IRBuilder<> EB(&NewF->getEntryBlock().front());
    Value *val = UndefValue::get(AT);
    for (size_t j = 0; j < AT->getNumElements(); j++) {
      auto gep =
          ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount + j) : sret;
      val = EB.CreateInsertValue(val, gep, j);
    }
    for (size_t i = 0; i < uses.size(); i++) {
      uses[i]->setOperand(op[i], val);
    }
    for (auto &RT : Returns) {
      IRBuilder<> B(RT);
      for (size_t j = 0; j < AT->getNumElements(); j++) {
        Value *em = GradientUtils::extractMeta(B, val, j);
        em = B.CreateLoad(Types[sretCount + j], em);
        recur(B, em, curOffset);
      }
    }
    if (roots_AT)
      curOffset +=
          CountTrackedPointers(Types[sretCount]).count * AT->getNumElements();
    sretCount += AT->getNumElements();
    delete arg;
  }

  for (auto i : rroots) {
    auto arg = delArgMap[i];
    assert(arg);
    llvm::Type *T = nullptr;
#if LLVM_VERSION_MAJOR >= 18
    llvm_unreachable("Unhandled");
    // T = F->getParamAttribute(i, Attribute::AttrKind::ElementType)
    //        .getValueAsType();
#else
    T = FT->getParamType(i)->getPointerElementType();
#endif
    IRBuilder<> EB(&NewF->getEntryBlock().front());
    arg->replaceAllUsesWith(EB.CreateAlloca(T));
    delete arg;
  }
  for (auto i : rroots_v) {
    auto arg = delArgMap[i];
    assert(arg);
    auto AT = cast<ArrayType>(FT->getParamType(i));
    llvm::Type *T = nullptr;
#if LLVM_VERSION_MAJOR >= 18
    llvm_unreachable("Unhandled");
    // T = F->getParamAttribute(i, Attribute::AttrKind::ElementType)
    //        .getValueAsType();
#else
    T = AT->getElementType()->getPointerElementType();
#endif
    IRBuilder<> EB(&NewF->getEntryBlock().front());
    Value *val = UndefValue::get(AT);
    for (size_t j = 0; j < AT->getNumElements(); j++) {
      val = EB.CreateInsertValue(val, EB.CreateAlloca(T), j);
    }
    arg->replaceAllUsesWith(val);
    delete arg;
  }
  assert(curOffset == numRooting);
  assert(sretCount == Types.size());

  for (auto CI : callers) {
    auto Attrs = CI->getAttributes();
    AttributeList NewAttrs;
    IRBuilder<> B(CI);
    IRBuilder<> EB(&CI->getParent()->getParent()->getEntryBlock().front());
    SmallVector<Value *, 1> vals;
    size_t nexti = 0;
    Value *sret = nullptr;
    if (sretTy) {
      sret = EB.CreateAlloca(sretTy);
      vals.push_back(sret);
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti,
          Attribute::get(F->getContext(), Attribute::StructRet, sretTy));
      nexti++;
    }
    AllocaInst *roots = nullptr;
    if (roots_AT) {
      roots = EB.CreateAlloca(roots_AT);
      vals.push_back(roots);
      NewAttrs = NewAttrs.addAttribute(

          F->getContext(), AttributeList::FirstArgIndex + nexti,
          "enzymejl_returnRoots");
      nexti++;
    }

    for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::FunctionIndex, attr);

    SmallVector<Value *, 1> sret_vals;
    SmallVector<Value *, 1> sretv_vals;
#if LLVM_VERSION_MAJOR >= 14
    for (size_t i = 0, end = CI->arg_size(); i < end; i++)
#else
    for (size_t i = 0, end = CI->getNumArgOperands(); i < end; i++)
#endif
    {
      if (rroots.count(i) || rroots_v.count(i)) {
        continue;
      }
      if (enzyme_srets.count(i)) {
        sret_vals.push_back(CI->getArgOperand(i));
        continue;
      }
      if (enzyme_srets_v.count(i)) {
        sretv_vals.push_back(CI->getArgOperand(i));
        continue;
      }

      for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i))
        NewAttrs = NewAttrs.addAttribute(
            F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
      vals.push_back(CI->getArgOperand(i));
      nexti++;
    }

    sretCount = 0;
    if (!RT->isVoidTy()) {
      sretCount++;
    }

    for (Value *ptr : sret_vals) {
      auto gep =
          ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount) : sret;
      auto ld = B.CreateLoad(Types[sretCount], ptr);
      B.CreateStore(ld, gep);
      sretCount++;
    }
    for (Value *ptr_v : sretv_vals) {
      auto AT = cast<ArrayType>(ptr_v->getType());
      for (size_t j = 0; j < AT->getNumElements(); j++) {
        auto gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount + j)
                      : sret;
        auto ptr = GradientUtils::extractMeta(B, ptr_v, j);
        auto ld = B.CreateLoad(Types[sretCount], ptr);
        B.CreateStore(ld, gep);
      }
      sretCount += AT->getNumElements();
    }

    SmallVector<OperandBundleDef, 1> Bundles;
    for (unsigned I = 0, E = CI->getNumOperandBundles(); I != E; ++I)
      Bundles.emplace_back(CI->getOperandBundleAt(I));
    auto NC = B.CreateCall(NewF, vals, Bundles);
    NC->setAttributes(NewAttrs);

    SmallVector<std::pair<unsigned, MDNode *>, 4> TheMDs;
    CI->getAllMetadataOtherThanDebugLoc(TheMDs);
    SmallVector<unsigned, 1> toCopy;
    for (auto pair : TheMDs)
      if (pair.first != LLVMContext::MD_range) {
        toCopy.push_back(pair.first);
      }
    if (!toCopy.empty())
      NC->copyMetadata(*CI, toCopy);
    NC->setDebugLoc(CI->getDebugLoc());

    sretCount = 0;
    if (!RT->isVoidTy()) {
      auto gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, 0) : sret;
      auto ld = B.CreateLoad(RT, gep);
      if (auto MD = CI->getMetadata(LLVMContext::MD_range))
        ld->setMetadata(LLVMContext::MD_range, MD);
      ld->takeName(CI);
      CI->replaceAllUsesWith(ld);
      sretCount++;
    }

    for (auto ptr : sret_vals) {
      auto gep =
          ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount) : sret;
      auto ld = B.CreateLoad(Types[sretCount], gep);
      auto SI = B.CreateStore(ld, ptr);
      PostCacheStore(SI, B);
      sretCount++;
    }
    for (auto ptr_v : sretv_vals) {
      auto AT = cast<ArrayType>(ptr_v->getType());
      for (size_t j = 0; j < AT->getNumElements(); j++) {
        auto gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount + j)
                      : sret;
        auto ptr = GradientUtils::extractMeta(B, ptr_v, j);
        auto ld = B.CreateLoad(Types[sretCount], gep);
        auto SI = B.CreateStore(ld, ptr);
        PostCacheStore(SI, B);
      }
      sretCount += AT->getNumElements();
    }

    NC->setCallingConv(CI->getCallingConv());
    CI->eraseFromParent();
  }
  NewF->setAttributes(NewAttrs);
  SmallVector<std::pair<unsigned, MDNode *>, 1> MD;
  F->getAllMetadata(MD);
  for (auto pair : MD)
    if (pair.first != LLVMContext::MD_dbg)
      NewF->addMetadata(pair.first, *pair.second);
  NewF->takeName(F);
  NewF->setCallingConv(F->getCallingConv());
  F->eraseFromParent();
}
#endif

LLVMValueRef EnzymeBuildExtractValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                     unsigned *Index, unsigned Size,
                                     const char *Name) {
  return wrap(unwrap(B)->CreateExtractValue(
      unwrap(AggVal), ArrayRef<unsigned>(Index, Size), Name));
}

LLVMValueRef EnzymeBuildInsertValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                    LLVMValueRef EltVal, unsigned *Index,
                                    unsigned Size, const char *Name) {
  return wrap(unwrap(B)->CreateInsertValue(
      unwrap(AggVal), unwrap(EltVal), ArrayRef<unsigned>(Index, Size), Name));
}
}
