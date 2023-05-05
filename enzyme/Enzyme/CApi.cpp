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
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "DiffeGradientUtils.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"
#include "SCEV/TargetLibraryInfo.h"
#include "TraceInterface.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#if LLVM_VERSION_MAJOR >= 9
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/Attributor.h"
#endif

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
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

EnzymeTraceInterfaceRef CreateEnzymeStaticTraceInterface(LLVMModuleRef M) {
  return (EnzymeTraceInterfaceRef)(new StaticTraceInterface(unwrap(M)));
}

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

void EnzymeRegisterAllocationHandler(char *Name, CustomShadowAlloc AHandle,
                                     CustomShadowFree FHandle) {
  shadowHandlers[std::string(Name)] =
      [=](IRBuilder<> &B, CallInst *CI, ArrayRef<Value *> Args,
          GradientUtils *gutils) -> llvm::Value * {
    SmallVector<LLVMValueRef, 3> refs;
    for (auto a : Args)
      refs.push_back(wrap(a));
    return unwrap(
        AHandle(wrap(&B), wrap(CI), Args.size(), refs.data(), gutils));
  };
  shadowErasers[std::string(Name)] = [=](IRBuilder<> &B,
                                         Value *ToFree) -> llvm::CallInst * {
    return cast_or_null<CallInst>(unwrap(FHandle(wrap(&B), wrap(ToFree))));
  };
}

void EnzymeRegisterCallHandler(char *Name,
                               CustomAugmentedFunctionForward FwdHandle,
                               CustomFunctionReverse RevHandle) {
  auto &pair = customCallHandlers[std::string(Name)];
  pair.first = [=](IRBuilder<> &B, CallInst *CI, GradientUtils &gutils,
                   Value *&normalReturn, Value *&shadowReturn, Value *&tape) {
    LLVMValueRef normalR = wrap(normalReturn);
    LLVMValueRef shadowR = wrap(shadowReturn);
    LLVMValueRef tapeR = wrap(tape);
    FwdHandle(wrap(&B), wrap(CI), &gutils, &normalR, &shadowR, &tapeR);
    normalReturn = unwrap(normalR);
    shadowReturn = unwrap(shadowR);
    tape = unwrap(tapeR);
  };
  pair.second = [=](IRBuilder<> &B, CallInst *CI, DiffeGradientUtils &gutils,
                    Value *tape) {
    RevHandle(wrap(&B), wrap(CI), &gutils, wrap(tape));
  };
}

void EnzymeRegisterFwdCallHandler(char *Name, CustomFunctionForward FwdHandle) {
  auto &pair = customFwdCallHandlers[std::string(Name)];
  pair = [=](IRBuilder<> &B, CallInst *CI, GradientUtils &gutils,
             Value *&normalReturn, Value *&shadowReturn) {
    LLVMValueRef normalR = wrap(normalReturn);
    LLVMValueRef shadowR = wrap(shadowReturn);
    FwdHandle(wrap(&B), wrap(CI), &gutils, &normalR, &shadowR);
    normalReturn = unwrap(normalR);
    shadowReturn = unwrap(shadowR);
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
  auto res = (CDIFFE_TYPE)(G->getReturnDiffeType(cast<CallInst>(unwrap(oval)),
                                                 &needsPrimalB, &needsShadowB));
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
    DiffeGradientUtils *gutils, LLVMValueRef orig, LLVMTypeRef addingType,
    unsigned start, unsigned size, LLVMValueRef origptr, LLVMValueRef dif,
    LLVMBuilderRef BuilderM, unsigned align, LLVMValueRef mask) {
#if LLVM_VERSION_MAJOR >= 10
  MaybeAlign align2;
  if (align)
    align2 = MaybeAlign(align);
#else
  auto align2 = align;
#endif
  gutils->addToInvertedPtrDiffe(
      cast_or_null<Instruction>(unwrap(orig)), unwrap(addingType), start, size,
      unwrap(origptr), unwrap(dif), *unwrap(BuilderM), align2, unwrap(mask));
}

void EnzymeGradientUtilsAddToInvertedPointerDiffeTT(
    DiffeGradientUtils *gutils, LLVMValueRef orig, CTypeTreeRef vd,
    unsigned LoadSize, LLVMValueRef origptr, LLVMValueRef prediff,
    LLVMBuilderRef BuilderM, unsigned align, LLVMValueRef premask) {
#if LLVM_VERSION_MAJOR >= 10
  MaybeAlign align2;
  if (align)
    align2 = MaybeAlign(align);
#else
  auto align2 = align;
#endif
  gutils->addToInvertedPtrDiffe(cast_or_null<Instruction>(unwrap(orig)),
                                *(TypeTree *)vd, LoadSize, unwrap(origptr),
                                unwrap(prediff), *unwrap(BuilderM), align2,
                                unwrap(premask));
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

LLVMValueRef CreateTrace(EnzymeLogicRef Logic, LLVMValueRef totrace,
                         LLVMValueRef *generative_functions,
                         size_t generative_functions_size, CProbProgMode mode,
                         uint8_t autodiff, EnzymeTraceInterfaceRef interface) {

  llvm::SmallPtrSet<Function *, 4> GenerativeFunctions;
  for (uint64_t i = 0; i < generative_functions_size; i++) {
    GenerativeFunctions.insert(cast<Function>(unwrap(generative_functions[i])));
  }

  return wrap(eunwrap(Logic).CreateTrace(
      cast<Function>(unwrap(totrace)), GenerativeFunctions, (ProbProgMode)mode,
      (bool)autodiff, eunwrap(interface)));
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
    GradientUtils *gutils, LLVMValueRef func, LLVMValueRef *args_vr,
    uint64_t args_size, LLVMValueRef orig_vr, CValueType *valTys,
    uint64_t valTys_size, LLVMBuilderRef B, uint8_t lookup) {
  auto orig = cast<CallInst>(unwrap(orig_vr));

  ArrayRef<ValueType> ar((ValueType *)valTys, valTys_size);

  IRBuilder<> &BR = *unwrap(B);

  auto Defs = gutils->getInvertedBundles(orig, ar, BR, lookup != 0);

  SmallVector<Value *, 1> args;
  for (size_t i = 0; i < args_size; i++) {
    args.push_back(unwrap(args_vr[i]));
  }

  auto callval = unwrap(func);

#if LLVM_VERSION_MAJOR > 7
  auto res = BR.CreateCall(
      cast<FunctionType>(callval->getType()->getPointerElementType()), callval,
      args, Defs);
#else
  auto res = BR.CreateCall(callval, args, Defs);
#endif
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

#if LLVM_VERSION_MAJOR >= 8
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
#endif

void EnzymeReplaceFunctionImplementation(LLVMModuleRef M) {
  ReplaceFunctionImplementation(*unwrap(M));
}

#if LLVM_VERSION_MAJOR >= 9
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
#if LLVM_VERSION_MAJOR >= 8
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), F->getParent());
#else
  Function *NewF =
      Function::Create(FTy, F->getLinkage(), F->getName(), F->getParent());
#endif

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
  for (auto pair : MD)
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
      size_t cnt = 0;
      for (auto idx : previdx) {
        if (idx <= val + cnt)
          cnt++;
      }
      nextidx.push_back(val);
    }

    size_t prevcnt = 0;
    size_t nextcnt = 0;
    SmallVector<uint64_t, 1> out;
    while (prevcnt < previdx.size() && nextcnt < nextidx.size()) {
      if (previdx[prevcnt] < nextidx[nextcnt]) {
        out.push_back(previdx[prevcnt]);
        prevcnt++;
      } else {
        out.push_back(nextidx[nextcnt]);
        nextcnt++;
      }
    }
    while (prevcnt < previdx.size()) {
      out.push_back(previdx[prevcnt]);
      prevcnt++;
    }
    while (nextcnt < nextidx.size()) {
      out.push_back(nextidx[nextcnt]);
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
}
