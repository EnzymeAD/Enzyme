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
#include "DifferentialUseAnalysis.h"
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

#define addAttribute addAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#define getAttribute getAttributeAtIndex
#define hasAttribute hasAttributeAtIndex

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
  case DT_X86_FP80:
    return ConcreteType(llvm::Type::getX86_FP80Ty(ctx));
  case DT_BFloat16:
    return ConcreteType(llvm::Type::getBFloatTy(ctx));
  case DT_FP128:
    return ConcreteType(llvm::Type::getFP128Ty(ctx));
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
    if (flt->isX86_FP80Ty())
      return DT_X86_FP80;
    if (flt->isBFloatTy())
      return DT_BFloat16;
    if (flt->isFP128Ty())
      return DT_FP128;
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

void EnzymeSetCLString(void *ptr, const char *val) {
  if (auto *clopt = static_cast<cl::opt<std::string> *>(ptr))
    clopt->setValue(val);
}

EnzymeLogicRef CreateEnzymeLogic(uint8_t PostOpt) {
  return (EnzymeLogicRef)(new EnzymeLogic((bool)PostOpt));
}

void EnzymeLogicSetExternalContext(EnzymeLogicRef Ref, void *ExternalContext) {
  eunwrap(Ref).ExternalContext = ExternalContext;
}

void *EnzymeLogicGetExternalContext(EnzymeLogicRef Ref) {
  return eunwrap(Ref).ExternalContext;
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
  EnzymeLogic &Logic = eunwrap(Log);
  TypeAnalysis *TA = new TypeAnalysis(Logic);
  for (size_t i = 0; i < numRules; i++) {
    CustomRuleType rule = customRules[i];
    TA->CustomRules[customRuleNames[i]] =
        [=](int direction, TypeTree &returnTree, ArrayRef<TypeTree> argTrees,
            ArrayRef<std::set<int64_t>> knownValues, CallBase *call,
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

EnzymeLogicRef EnzymeTypeAnalysisGetLogic(EnzymeTypeAnalysisRef TAR) {
  return (EnzymeLogicRef) & ((TypeAnalysis *)TAR)->Logic;
}

void *EnzymeAnalyzeTypes(EnzymeTypeAnalysisRef TAR, CFnTypeInfo CTI,
                         LLVMValueRef F) {
  FnTypeInfo FTI(eunwrap(CTI, cast<Function>(unwrap(F))));
  return (void *)((TypeAnalysis *)TAR)->analyzeFunction(FTI).analyzer;
}

void *EnzymeGradientUtilsTypeAnalyzer(GradientUtils *G) {
  return (void *)&G->TR.analyzer;
}

EnzymeTypeAnalysisRef EnzymeGetTypeAnalysisFromTypeAnalyzer(void *TAR) {
  return (EnzymeTypeAnalysisRef) & ((TypeAnalyzer *)TAR)->interprocedural;
}

void EnzymeGradientUtilsErase(GradientUtils *G, LLVMValueRef I) {
  return G->erase(cast<Instruction>(unwrap(I)));
}
void EnzymeGradientUtilsEraseWithPlaceholder(GradientUtils *G, LLVMValueRef I,
                                             LLVMValueRef orig, uint8_t erase) {
  return G->eraseWithPlaceholder(cast<Instruction>(unwrap(I)),
                                 cast<Instruction>(unwrap(orig)),
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
  if (FHandle)
    shadowErasers[Name] = [=](IRBuilder<> &B,
                              Value *ToFree) -> llvm::CallInst * {
      return cast_or_null<CallInst>(unwrap(FHandle(wrap(&B), wrap(ToFree))));
    };
}

void EnzymeRegisterCallHandler(const char *Name,
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

void EnzymeRegisterDiffUseCallHandler(char *Name,
                                      CustomFunctionDiffUse Handle) {
  auto &pair = customDiffUseHandlers[Name];
  pair = [=](const CallInst *CI, const GradientUtils *gutils, const Value *arg,
             bool isshadow, DerivativeMode mode, bool &useDefault) -> bool {
    uint8_t useDefaultC = 0;
    uint8_t noMod = Handle(wrap(CI), gutils, wrap(arg), isshadow,
                           (CDerivativeMode)(mode), &useDefaultC);
    useDefault = useDefaultC != 0;
    return noMod != 0;
  };
}

uint8_t EnzymeGradientUtilsGetRuntimeActivity(GradientUtils *gutils) {
  return gutils->runtimeActivity;
}

void *EnzymeGradientUtilsGetExternalContext(GradientUtils *gutils) {
  return gutils->Logic.ExternalContext;
}

uint8_t EnzymeGradientUtilsGetStrongZero(GradientUtils *gutils) {
  return gutils->strongZero;
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
                                      uint8_t *needsShadow,
                                      CDerivativeMode mode) {
  bool needsPrimalB;
  bool needsShadowB;
  auto res = (CDIFFE_TYPE)(G->getReturnDiffeType(
      unwrap(oval), &needsPrimalB, &needsShadowB, (DerivativeMode)mode));
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

LLVMValueRef EnzymeInsertValue(LLVMBuilderRef B, LLVMValueRef val,
                               LLVMValueRef val2, unsigned *sz, int64_t length,
                               const char *name) {
  return wrap(unwrap(B)->CreateInsertValue(
      unwrap(val), unwrap(val2), ArrayRef<unsigned>(sz, sz + length), name));
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
  MaybeAlign align2;
  if (align)
    align2 = MaybeAlign(align);
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
  MaybeAlign align2;
  if (align)
    align2 = MaybeAlign(align);
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

uint8_t EnzymeGradientUtilsGetUncacheableArgs(GradientUtils *gutils,
                                              LLVMValueRef orig, uint8_t *data,
                                              uint64_t size) {
  if (gutils->mode == DerivativeMode::ForwardMode ||
      gutils->mode == DerivativeMode::ForwardModeError)
    return 0;

  if (!gutils->overwritten_args_map_ptr)
    return 0;

  CallInst *call = cast<CallInst>(unwrap(orig));

  assert(gutils->overwritten_args_map_ptr);
  auto found = gutils->overwritten_args_map_ptr->find(call);
  if (found == gutils->overwritten_args_map_ptr->end()) {
    llvm::errs() << " oldFunc " << *gutils->oldFunc << "\n";
    for (auto &pair : *gutils->overwritten_args_map_ptr) {
      llvm::errs() << " + " << *pair.first << "\n";
    }
    llvm::errs() << " could not find call orig in overwritten_args_map_ptr "
                 << *call << "\n";
  }
  assert(found != gutils->overwritten_args_map_ptr->end());

  const std::vector<bool> &overwritten_args = found->second.second;

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
  return 1;
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

LLVMBasicBlockRef EnzymeGradientUtilsAddReverseBlock(GradientUtils *gutils,
                                                     LLVMBasicBlockRef block,
                                                     const char *name,
                                                     uint8_t forkCache,
                                                     uint8_t push) {
  return wrap(gutils->addReverseBlock(cast<BasicBlock>(unwrap(block)), name,
                                      forkCache, push));
}

void EnzymeGradientUtilsSetReverseBlock(GradientUtils *gutils,
                                        LLVMBasicBlockRef block) {
  auto endBlock = cast<BasicBlock>(unwrap(block));
  auto found = gutils->reverseBlockToPrimal.find(endBlock);
  assert(found != gutils->reverseBlockToPrimal.end());
  auto &vec = gutils->reverseBlocks[found->second];
  assert(vec.size());
  vec.push_back(endBlock);
}

LLVMValueRef EnzymeCreateForwardDiff(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, EnzymeTypeAnalysisRef TA, uint8_t returnValue,
    CDerivativeMode mode, uint8_t freeMemory, uint8_t runtimeActivity,
    uint8_t strongZero, unsigned width, LLVMTypeRef additionalArg,
    CFnTypeInfo typeInfo, uint8_t subsequent_calls_may_write,
    uint8_t *_overwritten_args, size_t overwritten_args_size,
    EnzymeAugmentedReturnPtr augmented) {
  SmallVector<DIFFE_TYPE, 4> nconstant_args((DIFFE_TYPE *)constant_args,
                                            (DIFFE_TYPE *)constant_args +
                                                constant_args_size);
  std::vector<bool> overwritten_args;
  assert(overwritten_args_size == cast<Function>(unwrap(todiff))->arg_size());
  for (uint64_t i = 0; i < overwritten_args_size; i++) {
    overwritten_args.push_back(_overwritten_args[i]);
  }
  return wrap(eunwrap(Logic).CreateForwardDiff(
      RequestContext(cast_or_null<Instruction>(unwrap(request_req)),
                     unwrap(request_ip)),
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA), returnValue, (DerivativeMode)mode, freeMemory,
      runtimeActivity, strongZero, width, unwrap(additionalArg),
      eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      subsequent_calls_may_write, overwritten_args, eunwrap(augmented)));
}
LLVMValueRef EnzymeCreatePrimalAndGradient(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, EnzymeTypeAnalysisRef TA, uint8_t returnValue,
    uint8_t dretUsed, CDerivativeMode mode, uint8_t runtimeActivity,
    uint8_t strongZero, unsigned width, uint8_t freeMemory,
    LLVMTypeRef additionalArg, uint8_t forceAnonymousTape, CFnTypeInfo typeInfo,
    uint8_t subsequent_calls_may_write, uint8_t *_overwritten_args,
    size_t overwritten_args_size, EnzymeAugmentedReturnPtr augmented,
    uint8_t AtomicAdd) {
  std::vector<DIFFE_TYPE> nconstant_args((DIFFE_TYPE *)constant_args,
                                         (DIFFE_TYPE *)constant_args +
                                             constant_args_size);
  std::vector<bool> overwritten_args;
  assert(overwritten_args_size == cast<Function>(unwrap(todiff))->arg_size());
  for (uint64_t i = 0; i < overwritten_args_size; i++) {
    overwritten_args.push_back(_overwritten_args[i]);
  }
  return wrap(eunwrap(Logic).CreatePrimalAndGradient(
      RequestContext(cast_or_null<Instruction>(unwrap(request_req)),
                     unwrap(request_ip)),
      (ReverseCacheKey){
          .todiff = cast<Function>(unwrap(todiff)),
          .retType = (DIFFE_TYPE)retType,
          .constant_args = nconstant_args,
          .subsequent_calls_may_write = (bool)subsequent_calls_may_write,
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
          .runtimeActivity = (bool)runtimeActivity,
          .strongZero = (bool)strongZero},
      eunwrap(TA), eunwrap(augmented)));
}
EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, EnzymeTypeAnalysisRef TA, uint8_t returnUsed,
    uint8_t shadowReturnUsed, CFnTypeInfo typeInfo,
    uint8_t subsequent_calls_may_write, uint8_t *_overwritten_args,
    size_t overwritten_args_size, uint8_t forceAnonymousTape,
    uint8_t runtimeActivity, uint8_t strongZero, unsigned width,
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
      RequestContext(cast_or_null<Instruction>(unwrap(request_req)),
                     unwrap(request_ip)),
      cast<Function>(unwrap(todiff)), (DIFFE_TYPE)retType, nconstant_args,
      eunwrap(TA), returnUsed, shadowReturnUsed,
      eunwrap(typeInfo, cast<Function>(unwrap(todiff))),
      subsequent_calls_may_write, overwritten_args, forceAnonymousTape,
      runtimeActivity, strongZero, width, AtomicAdd));
}

LLVMValueRef EnzymeCreateBatch(EnzymeLogicRef Logic, LLVMValueRef request_req,
                               LLVMBuilderRef request_ip, LLVMValueRef tobatch,
                               unsigned width, CBATCH_TYPE *arg_types,
                               size_t arg_types_size, CBATCH_TYPE retType) {

  return wrap(eunwrap(Logic).CreateBatch(
      RequestContext(cast_or_null<Instruction>(unwrap(request_req)),
                     unwrap(request_ip)),
      cast<Function>(unwrap(tobatch)), width,
      ArrayRef<BATCH_TYPE>((BATCH_TYPE *)arg_types,
                           (BATCH_TYPE *)arg_types + arg_types_size),
      (BATCH_TYPE)retType));
}

LLVMValueRef EnzymeCreateTrace(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef totrace, LLVMValueRef *sample_functions,
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
      RequestContext(cast_or_null<Instruction>(unwrap(request_req)),
                     unwrap(request_ip)),
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

static MDNode *extractMDNode(MetadataAsValue *MAV) {
  Metadata *MD = MAV->getMetadata();
  assert((isa<MDNode>(MD) || isa<ConstantAsMetadata>(MD)) &&
         "Expected a metadata node or a canonicalized constant");

  if (MDNode *N = dyn_cast<MDNode>(MD))
    return N;

  return MDNode::get(MAV->getContext(), MD);
}

CTypeTreeRef EnzymeTypeTreeFromMD(LLVMValueRef Val) {
  TypeTree *Ret = new TypeTree();
  MDNode *N = Val ? extractMDNode(unwrap<MetadataAsValue>(Val)) : nullptr;
  Ret->insertFromMD(N);
  return (CTypeTreeRef)N;
}

LLVMValueRef EnzymeTypeTreeToMD(CTypeTreeRef CTR, LLVMContextRef ctx) {
  auto MD = ((TypeTree *)CTR)->toMD(*unwrap(ctx));
  return wrap(MetadataAsValue::get(MD->getContext(), MD));
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
void EnzymeTypeTreeInsertEq(CTypeTreeRef CTT, const int64_t *indices,
                            size_t len, CConcreteType ct, LLVMContextRef ctx) {
  std::vector<int> seq;
  for (size_t i = 0; i < len; i++) {
    seq.push_back(indices[i]);
  }
  ((TypeTree *)CTT)->insert(seq, eunwrap(ct, *unwrap(ctx)));
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

void EnzymeSetStringMD(LLVMValueRef Inst, const char *Kind, LLVMValueRef Val) {
  MDNode *N = Val ? extractMDNode(unwrap<MetadataAsValue>(Val)) : nullptr;
  Value *V = unwrap(Inst);
  if (auto I = dyn_cast<Instruction>(V))
    I->setMetadata(Kind, N);
  else
    cast<GlobalVariable>(V)->setMetadata(Kind, N);
}

LLVMValueRef EnzymeGetStringMD(LLVMValueRef Inst, const char *Kind) {
  auto *I = unwrap<Instruction>(Inst);
  assert(I && "Expected instruction");
  if (auto *MD = I->getMetadata(Kind))
    return wrap(MetadataAsValue::get(I->getContext(), MD));
  return nullptr;
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

void EnzymeDetectReadonlyOrThrow(LLVMModuleRef M) {
  DetectReadonlyOrThrow(*unwrap(M));
}

void EnzymeDumpModuleRef(LLVMModuleRef M) {
  llvm::errs() << *unwrap(M) << "\n";
}

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

extern "C" void RunAttributorOnModule(LLVMModuleRef M0) {
  auto &M = *unwrap(M0);
  AnalysisGetter AG;
  SetVector<Function *> Functions;
  for (Function &F : M)
    Functions.insert(&F);

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ nullptr);
  runAttributorOnFunctions(InfoCache, Functions, AG, CGUpdater,
                           /* DeleteFns*/ true,
                           /* IsModulePass */ true);
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
  for (size_t i = 0, end = CI->arg_size(); i < end; i++) {
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
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);

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
        (void)b;
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
LLVMValueRef EnzymeComputeByteOffsetOfGEP(LLVMBuilderRef B_r, LLVMValueRef V_r,
                                          LLVMTypeRef T_r) {
  IRBuilder<> &B = *unwrap(B_r);
  auto T = cast<IntegerType>(unwrap(T_r));
  auto width = T->getBitWidth();
  auto uw = unwrap(V_r);
  GEPOperator *gep = isa<GetElementPtrInst>(uw)
                         ? cast<GEPOperator>(cast<GetElementPtrInst>(uw))
                         : cast<GEPOperator>(cast<ConstantExpr>(uw));
  auto &DL = B.GetInsertBlock()->getParent()->getParent()->getDataLayout();

#if LLVM_VERSION_MAJOR >= 20
  SmallMapVector<Value *, APInt, 4> VariableOffsets;
#else
  MapVector<Value *, APInt> VariableOffsets;
#endif
  APInt Offset(width, 0);
  bool success = collectOffset(gep, DL, width, VariableOffsets, Offset);
  (void)success;
  assert(success);
  Value *start = ConstantInt::get(T, Offset);
  for (auto &pair : VariableOffsets)
    start = B.CreateAdd(
        start, B.CreateMul(pair.first, ConstantInt::get(T, pair.second)));
  return wrap(start);
}
}

extern "C" {

void EnzymeFixupBatchedJuliaCallingConvention(LLVMValueRef F_C) {
  auto F = cast<Function>(unwrap(F_C));
  if (F->empty())
    return;
  auto RT = F->getReturnType();
  auto FT = F->getFunctionType();
  auto Attrs = F->getAttributes();

  AttributeList NewAttrs;
  SmallVector<Type *, 1> types;
  SmallSet<size_t, 1> changed;
  for (auto pair : llvm::enumerate(FT->params())) {
    auto T = pair.value();
    auto i = pair.index();
    bool sretv = false;
    StringRef value;
    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i)) {
      if (attr.isStringAttribute() &&
          attr.getKindAsString() == "enzyme_sret_v") {
        sretv = true;
        value = attr.getValueAsString();
      } else {
        NewAttrs = NewAttrs.addAttribute(
            F->getContext(), AttributeList::FirstArgIndex + types.size(), attr);
      }
    }
    if (auto AT = dyn_cast<ArrayType>(T)) {
      if (auto PT = dyn_cast<PointerType>(AT->getElementType())) {
        auto AS = PT->getAddressSpace();
        if (AS == 11 || AS == 12 || AS == 13 || sretv) {
          for (unsigned i = 0; i < AT->getNumElements(); i++) {
            if (sretv) {
              NewAttrs = NewAttrs.addAttribute(
                  F->getContext(), AttributeList::FirstArgIndex + types.size(),
                  Attribute::get(F->getContext(), "enzyme_sret", value));
            }
            types.push_back(PT);
          }
          changed.insert(i);
          continue;
        }
      }
    }
    types.push_back(T);
  }
  if (changed.size() == 0)
    return;

  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  for (auto attr : Attrs.getAttributes(AttributeList::ReturnIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::ReturnIndex, attr);

  FunctionType *FTy =
      FunctionType::get(FT->getReturnType(), types, FT->isVarArg());

  // Create the new function
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();

  // To handle the deleted args, it needs to be replaced by a non-arg operand.
  // This map contains the temporary phi nodes corresponding
  SmallVector<Instruction *, 1> toInsert;
  for (Argument &I : F->args()) {
    auto T = I.getType();
    if (auto AT = dyn_cast<ArrayType>(T)) {
      if (changed.count(I.getArgNo())) {
        Value *V = UndefValue::get(T);
        for (unsigned i = 0; i < AT->getNumElements(); i++) {
          DestI->setName(I.getName() + "." +
                         std::to_string(i)); // Copy the name over...
          unsigned idx[1] = {i};
          auto IV = InsertValueInst::Create(V, (llvm::Value *)&*DestI++, idx);
          toInsert.push_back(IV);
          V = IV;
        }
        VMap[&I] = V;
        continue;
      }
    }
    DestI->setName(I.getName()); // Copy the name over...
    VMap[&I] = &*DestI++;        // Add mapping to VMap
  }

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);

  {
    IRBuilder<> EB(&*NewF->getEntryBlock().begin());
    for (auto I : toInsert)
      EB.Insert(I);
  }

  SmallVector<CallInst *, 1> callers;
  for (auto U : F->users()) {
    auto CI = dyn_cast<CallInst>(U);
    assert(CI);
    assert(CI->getCalledFunction() == F);
    callers.push_back(CI);
  }

  for (auto CI : callers) {
    auto Attrs = CI->getAttributes();
    AttributeList NewAttrs;
    IRBuilder<> B(CI);

    for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::FunctionIndex, attr);

    for (auto attr : Attrs.getAttributes(AttributeList::ReturnIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::ReturnIndex, attr);

    SmallVector<Value *, 1> vals;
    for (size_t j = 0, end = CI->arg_size(); j < end; j++) {

      auto T = CI->getArgOperand(j)->getType();
      if (auto AT = dyn_cast<ArrayType>(T)) {
        if (isa<PointerType>(AT->getElementType())) {
          if (changed.count(j)) {
            bool sretv = false;
            for (auto attr :
                 Attrs.getAttributes(AttributeList::FirstArgIndex + j)) {
              if (attr.isStringAttribute() &&
                  attr.getKindAsString() == "enzyme_sret_v") {
                sretv = true;
              }
            }
            for (unsigned i = 0; i < AT->getNumElements(); i++) {
              if (sretv)
                NewAttrs = NewAttrs.addAttribute(
                    F->getContext(), AttributeList::FirstArgIndex + vals.size(),
                    Attribute::get(F->getContext(), "enzyme_sret"));
              vals.push_back(
                  GradientUtils::extractMeta(B, CI->getArgOperand(j), i));
            }
            continue;
          }
        }
      }

      for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + j)) {
        if (attr.isStringAttribute() &&
            attr.getKindAsString() == "enzyme_sret_v") {
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + vals.size(),
              Attribute::get(F->getContext(), "enzyme_sret"));
        } else {
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + vals.size(),
              attr);
        }
      }

      vals.push_back(CI->getArgOperand(j));
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
      toCopy.push_back(pair.first);
    if (!toCopy.empty())
      NC->copyMetadata(*CI, toCopy);
    NC->setDebugLoc(CI->getDebugLoc());

    if (!RT->isVoidTy()) {
      NC->takeName(CI);
      CI->replaceAllUsesWith(NC);
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

bool needsReRooting(llvm::Argument *arg, bool is_v, bool &anyJLStore) {

  llvm::Type *SRetType;

  auto Attrs = arg->getParent()->getAttributes();

  if (!is_v)
    SRetType = convertSRetTypeFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + arg->getArgNo(),
                          "enzyme_sret")
            .getValueAsString());
  else
    SRetType = convertSRetTypeFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + arg->getArgNo(),
                          "enzyme_sret_v")
            .getValueAsString());

  CountTrackedPointers tracked(SRetType);
  if (tracked.count == 0) {
    return false;
  }

  bool hasReturnRootingAfterArg = false;
  for (size_t i = arg->getArgNo() + 1; i < arg->getParent()->arg_size(); i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots") ||
        Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots_v")) {
      hasReturnRootingAfterArg = true;
      break;
    }
  }

  // If there is no returnRoots, we _must_ reroot the arg.
  if (!hasReturnRootingAfterArg) {
    return true;
  }

  SmallVector<std::tuple<bool, Value *>> todo = {{is_v, arg}};

  SmallVector<Value *> storedValues;

  bool legal = true;
  while (!todo.empty()) {
    auto curv = todo.pop_back_val();
    auto &&[local_isv, cur] = curv;
    for (auto &U : cur->uses()) {
      auto I = cast<Instruction>(U.getUser());

      if (is_v) {
        auto EVI = cast<ExtractValueInst>(cur);
        assert(EVI->getType()->isPointerTy());
        todo.emplace_back(!is_v, EVI);
        continue;
      }

      if (isPointerArithmeticInst(I)) {
        todo.emplace_back(is_v, I);
        continue;
      }
      if (isa<LoadInst>(I)) {
        continue;
      }
      if (auto SI = dyn_cast<StoreInst>(I)) {
        assert(SI->getValueOperand() != cur);

        if (CountTrackedPointers(SI->getValueOperand()->getType()).count == 0)
          continue;

        storedValues.push_back(SI->getValueOperand());
        anyJLStore = true;
        continue;
      }

      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "Unknown user of sret-like argument\n";
      CustomErrorHandler(ss.str().c_str(), wrap(I), ErrorType::GCRewrite,
                         wrap(cur), wrap(arg), nullptr);
      legal = false;
      anyJLStore = true;
      break;
    }
  }

  if (legal) {
    while (!storedValues.empty()) {
      auto sv = storedValues.pop_back_val();
      bool foundUse = false;
      for (auto &U : sv->uses()) {
        if (auto SI = dyn_cast<StoreInst>(U.getUser())) {
          if (SI->getValueOperand() == sv) {
            auto base = getBaseObject(SI->getPointerOperand());
            if (base == arg) {
              continue;
            }
            if (auto evi = dyn_cast<ExtractValueInst>(base)) {
              base = evi->getAggregateOperand();
            }
            if (auto arg2 = dyn_cast<Argument>(base)) {
              if (Attrs
                      .getAttribute(AttributeList::FirstArgIndex +
                                        arg2->getArgNo(),
                                    "enzymejl_returnRoots")
                      .isValid() ||
                  Attrs
                      .getAttribute(AttributeList::FirstArgIndex +
                                        arg2->getArgNo(),
                                    "enzymejl_returnRoots_v")
                      .isValid()) {
                foundUse = true;
                break;
              }
            }
          }
        }
      }
      if (!foundUse) {
        if (auto IVI = dyn_cast<InsertValueInst>(sv)) {
          CountTrackedPointers tracked(
              IVI->getInsertedValueOperand()->getType());
          if (tracked.count == 0) {
            storedValues.push_back(IVI->getAggregateOperand());
            continue;
          }
          if (isa<UndefValue>(IVI->getAggregateOperand()) ||
              isa<PoisonValue>(IVI->getAggregateOperand()) ||
              isa<ConstantAggregateZero>(IVI->getAggregateOperand())) {
            storedValues.push_back(IVI->getInsertedValueOperand());
            continue;
          }
        }
        if (!isa<PointerType>(sv->getType()) ||
            !isSpecialPtr(cast<PointerType>(sv->getType()))) {
          llvm::errs() << " sf: " << *arg->getParent() << "\n";
          llvm::errs() << "Pointer of wrong type: " << *sv << "\n";
          assert(0);
        }

        if (hasReturnRootingAfterArg) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << "Could not find use of stored value\n";
          CustomErrorHandler(ss.str().c_str(), wrap(sv), ErrorType::GCRewrite,
                             nullptr, wrap(arg), nullptr);
        }
        legal = false;
        break;
      }
    }
  }

#if LLVM_VERSION_MAJOR < 18
  // assert(legal);
#else
  assert(!legal);
#endif

  return !legal;
}

bool needsReReturning(llvm::Argument *arg, bool is_v) {

  llvm::Type *SRetType;

  auto Attrs = arg->getParent()->getAttributes();

  if (!is_v)
    SRetType = convertSRetTypeFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + arg->getArgNo(),
                          "enzymejl_returnRoots")
            .getValueAsString());
  else
    SRetType = convertSRetTypeFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + arg->getArgNo(),
                          "enzymejl_returnRoots_v")
            .getValueAsString());

  CountTrackedPointers tracked(SRetType);
  if (tracked.count == 0) {
    return false;
  }

  bool hasSRetBeforeArg = false;
  for (size_t i = 0; i < arg->getArgNo(); i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret") ||
        Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret_v")) {
      hasSRetBeforeArg = true;
      break;
    }
  }

  if (!hasSRetBeforeArg) {
    return true;
  }

  return false;
}

// TODO, for sret/sret_v check if it actually stores the jlvalue_t's into the
// sret If so, confirm that those values are saved elsewhere in a returnroot
void EnzymeFixupJuliaCallingConvention(LLVMValueRef F_C) {
  auto F = cast<Function>(unwrap(F_C));
  if (F->empty())
    return;
  auto RT = F->getReturnType();
  std::set<size_t> srets;
  std::set<size_t> enzyme_srets;
  std::set<size_t> enzyme_srets_v;

  std::set<size_t> reroot_enzyme_srets;
  std::set<size_t> reroot_enzyme_srets_v;

  std::set<size_t> rroots;
  std::set<size_t> rroots_v;

  std::set<size_t> reret_roots;
  std::set<size_t> reret_roots_v;

  auto FT = F->getFunctionType();
  auto Attrs = F->getAttributes();
  for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           Attribute::StructRet))
      srets.insert(i);
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret")) {
      bool anyJLStore = false;
      if (needsReRooting(F->getArg(i), false, anyJLStore)) {
        reroot_enzyme_srets.insert(i);
        enzyme_srets.insert(i);
      } else if (anyJLStore) {
        enzyme_srets.insert(i);
      }
    }
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i, "enzyme_sret_v")) {
      bool anyJLStore = false;
      if (needsReRooting(F->getArg(i), true, anyJLStore)) {
        reroot_enzyme_srets.insert(i);
        enzyme_srets_v.insert(i);
      } else if (anyJLStore) {
        enzyme_srets_v.insert(i);
      }
    }
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots")) {
      rroots.insert(i);
      if (needsReReturning(F->getArg(i), false)) {
        reret_roots.insert(i);
      }
    }
    if (Attrs.hasAttribute(AttributeList::FirstArgIndex + i,
                           "enzymejl_returnRoots_v")) {
      rroots_v.insert(i);
      if (needsReReturning(F->getArg(i), true)) {
        reret_roots_v.insert(i);
      }
    }
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
    llvm::Type *SRetType = F->getParamStructRetType(0);
    if (CountTrackedPointers(SRetType).count) {
      if (rroots.size())
        return;
    }
    F->addParamAttr(0, Attribute::get(F->getContext(), "enzyme_sret",
                                      convertSRetTypeToString(SRetType)));
    Attrs = F->getAttributes();
    srets.clear();
    bool anyJLStore = false;
    size_t i = 0;
    if (needsReRooting(F->getArg(i), false, anyJLStore)) {
      reroot_enzyme_srets.insert(i);
      enzyme_srets.insert(i);
    } else if (anyJLStore) {
      enzyme_srets.insert(i);
    }
  } else if (srets.size() == 0 && enzyme_srets.size() == 0 &&
             enzyme_srets_v.size() == 0 && rroots.size() == 0 &&
             rroots_v.size() == 0) {
    // No sret/rooting, no intervention needed.
    return;
  }

  assert(srets.size() == 0);

  SmallVector<Type *, 1> Types;
  if (!RT->isVoidTy()) {
    Types.push_back(RT);
  }

  auto T_jlvalue = StructType::get(F->getContext(), {});
  auto T_prjlvalue = PointerType::get(T_jlvalue, AddressSpace::Tracked);

  size_t numRooting = RT->isVoidTy() ? 0 : CountTrackedPointers(RT).count;

  for (auto idx : enzyme_srets) {
    llvm::Type *SRetType = convertSRetTypeFromString(
        Attrs.getAttribute(AttributeList::FirstArgIndex + idx, "enzyme_sret")
            .getValueAsString());
#if LLVM_VERSION_MAJOR < 17
    if (F->getContext().supportsTypedPointers()) {
      auto T = FT->getParamType(idx)->getPointerElementType();
      assert(T == SRetType);
    }
#endif
    Types.push_back(SRetType);
    if (reroot_enzyme_srets.count(idx)) {
      numRooting += CountTrackedPointers(SRetType).count;
    }
  }
  for (auto idx : rroots) {
    size_t count = convertRRootCountFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + idx,
                          "enzymejl_returnRoots")
            .getValueAsString());
    auto T = ArrayType::get(T_prjlvalue, count);
#if LLVM_VERSION_MAJOR < 17
    if (F->getContext().supportsTypedPointers()) {
      auto NT = FT->getParamType(idx)->getPointerElementType();
      assert(NT == T);
    }
#endif
    if (reret_roots.count(idx)) {
      Types.push_back(T);
    }
    numRooting += count;
  }
  for (auto idx : enzyme_srets_v) {
    llvm::Type *SRetType = convertSRetTypeFromString(
        Attrs.getAttribute(AttributeList::FirstArgIndex + idx, "enzyme_sret_v")
            .getValueAsString());
    auto AT = cast<ArrayType>(FT->getParamType(idx));
#if LLVM_VERSION_MAJOR < 17
    if (F->getContext().supportsTypedPointers()) {
      auto T = AT->getElementType()->getPointerElementType();
      assert(T == SRetType);
    }
#endif
    for (size_t i = 0; i < AT->getNumElements(); i++) {
      Types.push_back(SRetType);
      if (reroot_enzyme_srets_v.count(idx)) {
        numRooting += CountTrackedPointers(SRetType).count;
      }
    }
  }
  for (auto idx : rroots_v) {
    size_t count = convertRRootCountFromString(
        Attrs
            .getAttribute(AttributeList::FirstArgIndex + idx,
                          "enzymejl_returnRoots_v")
            .getValueAsString());
    auto AT = cast<ArrayType>(FT->getParamType(idx));
    auto T = ArrayType::get(T_prjlvalue, count);
#if LLVM_VERSION_MAJOR < 17
    if (F->getContext().supportsTypedPointers()) {
      auto NT = AT->getPointerElementType();
      assert(NT == T);
    }
#endif
    numRooting += AT->getNumElements() * count;
    if (reret_roots_v.count(idx)) {
      for (size_t i = 0; i < AT->getNumElements(); i++) {
        Types.push_back(T);
      }
    }
  }

  StructType *ST =
      Types.size() <= 1 ? nullptr : StructType::get(F->getContext(), Types);
  Type *sretTy = nullptr;
  if (Types.size())
    sretTy = Types.size() == 1 ? Types[0] : ST;

  ArrayType *roots_AT =
      numRooting ? ArrayType::get(T_prjlvalue, numRooting) : nullptr;

  if (sretTy) {
    CountTrackedPointers countF(sretTy);
    if (countF.all) {
      roots_AT = nullptr;
      numRooting = 0;
      reroot_enzyme_srets.clear();
      reroot_enzyme_srets_v.clear();
    } else if (countF.count) {
      if (!roots_AT) {
        llvm::errs() << " sretTy: " << *sretTy << "\n";
        llvm::errs() << " numRooting: " << numRooting << "\n";
        llvm::errs() << " tracked.count: " << countF.count << "\n";
      }
      assert(roots_AT);
      if (numRooting != countF.count) {
        llvm::errs() << " sretTy: " << *sretTy << "\n";
        llvm::errs() << " numRooting: " << numRooting << "\n";
        llvm::errs() << " tracked.count: " << countF.count << "\n";
      }
      assert(numRooting == countF.count);
    }
  }

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
    NewAttrs = NewAttrs.addAttribute(
        F->getContext(), AttributeList::FirstArgIndex + nexti,
        "enzymejl_returnRoots", std::to_string(numRooting));
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

    for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i)) {
      if (attr.isStringAttribute())
        if (attr.getKindAsString() == "enzyme_sret" ||
            attr.getKindAsString() == "enzyme_sret_v") {
          continue;
        }
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
    }
    types.push_back(F->getFunctionType()->getParamType(i));
    nexti++;
  }
  for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
    NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                     AttributeList::FunctionIndex, attr);

  FunctionType *FTy = FunctionType::get(Type::getVoidTy(F->getContext()), types,
                                        FT->isVarArg());

  // Create the new function
  auto &M = *F->getParent();
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                                    F->getName(), &M);

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
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);

  SmallVector<CallInst *, 1> callers;
  for (auto U : F->users()) {
    auto CI = dyn_cast<CallInst>(U);
    assert(CI);
    assert(CI->getCalledFunction() == F);
    callers.push_back(CI);
  }

  {
    size_t curOffset = 0;
    size_t sretCount = 0;
    if (!RT->isVoidTy()) {
      for (auto &RT : Returns) {
        IRBuilder<> B(RT);
        Value *gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, 0) : sret;
        Value *rval = RT->getReturnValue();
        B.CreateStore(rval, gep);

        if (roots) {
          moveSRetToFromRoots(B, rval->getType(), rval, roots_AT, roots,
                              /*rootOffset*/ 0,
                              SRetRootMovement::SRetValueToRootPointer);
        }

        auto NR = B.CreateRetVoid();
        RT->eraseFromParent();
        RT = NR;
      }
      if (roots_AT)
        curOffset = CountTrackedPointers(RT).count;
      sretCount++;
    }

    // TODO this must be re-ordered to interleave the sret/roots/etc args as
    // required.

    for (size_t i = 0, end = FT->getNumParams(); i < end; i++) {

      if (enzyme_srets.count(i)) {
        auto argFound = delArgMap.find(i);
        assert(argFound != delArgMap.end());
        auto arg = argFound->second;
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

        if (reroot_enzyme_srets.count(i)) {
          assert(roots_AT);
          for (auto &RT : Returns) {
            IRBuilder<> B(RT);
            moveSRetToFromRoots(B, Types[sretCount], gep, roots_AT, roots,
                                curOffset,
                                SRetRootMovement::SRetPointerToRootPointer);
          }
          curOffset += CountTrackedPointers(Types[sretCount]).count;
        }

        delete arg;

        sretCount++;
        continue;
      }

      if (rroots.count(i)) {

        size_t subCount = convertRRootCountFromString(
            Attrs
                .getAttribute(AttributeList::FirstArgIndex + i,
                              "enzymejl_returnRoots")
                .getValueAsString());

        auto argFound = delArgMap.find(i);
        assert(argFound != delArgMap.end());
        auto arg = argFound->second;
        assert(arg);
        SmallVector<Instruction *, 1> uses;
        SmallVector<unsigned, 1> op;
        for (auto &U : arg->uses()) {
          auto I = cast<Instruction>(U.getUser());
          uses.push_back(I);
          op.push_back(U.getOperandNo());
        }
        IRBuilder<> EB(&NewF->getEntryBlock().front());

        Value *gep = nullptr;
        if (roots_AT) {
          assert(roots);
          assert(roots_AT);

          gep = roots;
          if (curOffset != 0) {
            gep = EB.CreateConstInBoundsGEP2_32(roots_AT, roots, 0, curOffset);
          }
          if (subCount != numRooting) {
            gep = EB.CreatePointerCast(
                gep,
                PointerType::getUnqual(ArrayType::get(T_prjlvalue, subCount)));
          }
          curOffset += subCount;

        } else {
          assert(sret);
          gep =
              ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount) : sret;

          assert(reret_roots.count(i));

          sretCount++;
        }

        for (size_t i = 0; i < uses.size(); i++) {
          uses[i]->setOperand(op[i], gep);
        }

        delete arg;
        continue;
      }

      if (enzyme_srets_v.count(i)) {
        auto AT = cast<ArrayType>(FT->getParamType(i));

        auto argFound = delArgMap.find(i);
        assert(argFound != delArgMap.end());
        auto arg = argFound->second;
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
              ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount + j)
                 : sret;
          val = EB.CreateInsertValue(val, gep, j);
        }
        for (size_t i = 0; i < uses.size(); i++) {
          uses[i]->setOperand(op[i], val);
        }

        if (reroot_enzyme_srets_v.count(i)) {
          assert(roots_AT);
          auto numLocalRoots = CountTrackedPointers(Types[sretCount]).count;
          for (auto &RT : Returns) {
            IRBuilder<> B(RT);
            for (size_t j = 0; j < AT->getNumElements(); j++) {
              Value *em = GradientUtils::extractMeta(B, val, j);
              moveSRetToFromRoots(B, Types[sretCount + j], em, roots_AT, roots,
                                  curOffset + numLocalRoots * j,
                                  SRetRootMovement::SRetPointerToRootPointer);
            }
          }
          curOffset += numLocalRoots * AT->getNumElements();
        }

        delete arg;

        sretCount += AT->getNumElements();
        continue;
      }

      if (rroots_v.count(i)) {
        size_t subCount = convertRRootCountFromString(
            Attrs
                .getAttribute(AttributeList::FirstArgIndex + i,
                              "enzymejl_returnRoots_v")
                .getValueAsString());

        auto AT = cast<ArrayType>(FT->getParamType(i));

        auto argFound = delArgMap.find(i);
        assert(argFound != delArgMap.end());
        auto arg = argFound->second;
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
          Value *gep = nullptr;
          if (roots_AT) {
            assert(roots);

            gep = roots;
            if (curOffset != 0 || j != 0) {
              gep =
                  EB.CreateConstInBoundsGEP2_32(roots_AT, roots, 0, curOffset);
            }
            if (subCount != numRooting) {
              gep = EB.CreatePointerCast(
                  gep, PointerType::getUnqual(
                           ArrayType::get(T_prjlvalue, subCount)));
            }
            curOffset += subCount;
          } else {
            assert(reret_roots_v.count(i));
            assert(sret);
            gep = ST ? EB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount)
                     : sret;
            sretCount++;
          }
          val = EB.CreateInsertValue(val, gep, j);
        }
        for (size_t i = 0; i < uses.size(); i++) {
          uses[i]->setOperand(op[i], val);
        }
        delete arg;

        continue;
      }
    }

    assert(curOffset == numRooting);
    assert(sretCount == Types.size());
  }

  // TODO fix caller side
  for (auto CI : callers) {
    auto Attrs = CI->getAttributes();
    AttributeList NewAttrs;
    IRBuilder<> B(CI);
    IRBuilder<> EB(&CI->getParent()->getParent()->getEntryBlock().front());
    SmallVector<Value *, 1> vals;
    size_t nexti = 0;
    Value *sret = nullptr;
    if (sretTy) {
      sret = EB.CreateAlloca(sretTy, 0, "stack_sret");
      vals.push_back(sret);
      NewAttrs = NewAttrs.addAttribute(
          F->getContext(), AttributeList::FirstArgIndex + nexti,
          Attribute::get(F->getContext(), Attribute::StructRet, sretTy));
      nexti++;
    }
    AllocaInst *roots = nullptr;
    if (roots_AT) {
      roots = EB.CreateAlloca(roots_AT, 0, "stack_roots_AT");
      vals.push_back(roots);
      NewAttrs = NewAttrs.addAttribute(

          F->getContext(), AttributeList::FirstArgIndex + nexti,
          "enzymejl_returnRoots", std::to_string(numRooting));
      nexti++;
    }

    for (auto attr : Attrs.getAttributes(AttributeList::FunctionIndex))
      NewAttrs = NewAttrs.addAttribute(F->getContext(),
                                       AttributeList::FunctionIndex, attr);

    SmallVector<std::tuple<Value *, Value *, Type *>> preCallReplacements;
    SmallVector<std::tuple<Value *, Value *, Type *>> postCallReplacements;

    {
      size_t local_root_count =
          RT->isVoidTy() ? 0 : CountTrackedPointers(RT).count;
      size_t sretCount = 0;
      if (!RT->isVoidTy()) {
        sretCount++;
      }

      /// TODO continue from here down for external rewrites
      for (size_t i = 0, end = CI->arg_size(); i < end; i++) {

        if (enzyme_srets.count(i)) {

          auto val = CI->getArgOperand(i);

          IRBuilder<> AIB(cast<Instruction>(val));
          Value *gep = sret;
          if (ST) {
            gep = AIB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount);
          }

          if (auto AI = dyn_cast<AllocaInst>(getBaseObject(val, false))) {
            AI->replaceAllUsesWith(gep);
            AI->eraseFromParent();
          } else {
            assert(!isa<UndefValue>(val));
            assert(!isa<PoisonValue>(val));
            assert(!isa<ConstantPointerNull>(val));
            // TODO consider doing pre-emptive pre zero of the section?
            postCallReplacements.emplace_back(val, gep, Types[sretCount]);
            preCallReplacements.emplace_back(val, gep, Types[sretCount]);
          }

          if (reroot_enzyme_srets.count(i)) {
            local_root_count += CountTrackedPointers(Types[sretCount]).count;
          }

          sretCount++;
          continue;
        }

        if (enzyme_srets_v.count(i)) {
          auto VAT = cast<ArrayType>(CI->getArgOperand(i)->getType());

          if (reroot_enzyme_srets_v.count(i)) {
            local_root_count += CountTrackedPointers(Types[sretCount]).count *
                                VAT->getNumElements();
          }

          for (size_t j = 0; j < VAT->getNumElements(); j++) {

            IRBuilder<> AIB(
                cast<Instruction>(CI->getArgOperand(i))->getNextNode());
            auto val = GradientUtils::extractMeta(AIB, CI->getArgOperand(i), j);

            Value *gep = sret;
            if (ST) {
              gep = AIB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount);
            }
            if (auto AI = dyn_cast<AllocaInst>(getBaseObject(val, false))) {
              AI->replaceAllUsesWith(gep);
              AI->eraseFromParent();
            } else {
              assert(!isa<UndefValue>(val));
              assert(!isa<PoisonValue>(val));
              assert(!isa<ConstantPointerNull>(val));
              // TODO consider doing pre-emptive pre zero of the section?
              postCallReplacements.emplace_back(val, gep, Types[sretCount]);
              preCallReplacements.emplace_back(val, gep, Types[sretCount]);
            }

            sretCount++;
          }

          continue;
        }

        if (rroots.count(i)) {
          auto val = CI->getArgOperand(i);
          IRBuilder<> AIB(cast<Instruction>(val));

          size_t subCount = convertRRootCountFromString(
              Attrs
                  .getAttribute(AttributeList::FirstArgIndex + i,
                                "enzymejl_returnRoots")
                  .getValueAsString());

          Value *gep = nullptr;

          if (roots_AT) {
            assert(roots);
            gep = roots;
            if (local_root_count != 0) {
              gep = AIB.CreateConstInBoundsGEP2_32(roots_AT, roots, 0,
                                                   local_root_count);
            }

            if (subCount != numRooting) {
              gep = AIB.CreatePointerCast(
                  gep, PointerType::getUnqual(
                           ArrayType::get(T_prjlvalue, subCount)));
            }
            local_root_count += subCount;
          } else {
            assert(reret_roots.count(i));
            assert(sret);
            gep = sret;
            if (ST) {
              gep = AIB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount);
            }
            sretCount++;
          }

          if (auto AI = dyn_cast<AllocaInst>(getBaseObject(val, false))) {
            AI->replaceAllUsesWith(gep);
            AI->eraseFromParent();
          } else {
            assert(!isa<UndefValue>(val));
            assert(!isa<PoisonValue>(val));
            assert(!isa<ConstantPointerNull>(val));
            // TODO consider doing pre-emptive pre zero of the section?
            preCallReplacements.emplace_back(
                val, gep, ArrayType::get(T_prjlvalue, subCount));
            postCallReplacements.emplace_back(
                val, gep, ArrayType::get(T_prjlvalue, subCount));
          }
          continue;
        }

        if (rroots_v.count(i)) {

          size_t subCount = convertRRootCountFromString(
              Attrs
                  .getAttribute(AttributeList::FirstArgIndex + i,
                                "enzymejl_returnRoots")
                  .getValueAsString());

          auto VAT = dyn_cast<ArrayType>(CI->getArgOperand(i)->getType());
          for (size_t j = 0; j < VAT->getNumElements(); j++) {

            IRBuilder<> AIB(
                cast<Instruction>(CI->getArgOperand(i))->getNextNode());
            auto val = GradientUtils::extractMeta(EB, CI->getArgOperand(i), j);

            Value *gep = nullptr;

            if (roots_AT) {
              assert(roots);
              gep = roots;
              if (local_root_count != 0) {
                gep = AIB.CreateConstInBoundsGEP2_32(roots_AT, roots, 0,
                                                     local_root_count);
              }
              gep = AIB.CreatePointerCast(
                  gep, PointerType::getUnqual(
                           ArrayType::get(T_prjlvalue, subCount)));
              local_root_count += subCount;
            } else {
              assert(reret_roots_v.count(i));
              assert(sret);

              gep = sret;
              if (ST) {
                gep = AIB.CreateConstInBoundsGEP2_32(ST, sret, 0, sretCount);
              }
              sretCount++;
            }
            if (auto AI = dyn_cast<AllocaInst>(getBaseObject(val, false))) {
              AI->replaceAllUsesWith(gep);
              AI->eraseFromParent();
            } else {
              assert(!isa<UndefValue>(val));
              assert(!isa<PoisonValue>(val));
              assert(!isa<ConstantPointerNull>(val));
              // TODO consider doing pre-emptive pre zero of the section?
              preCallReplacements.emplace_back(
                  val, gep, ArrayType::get(T_prjlvalue, subCount));
              postCallReplacements.emplace_back(
                  val, gep, ArrayType::get(T_prjlvalue, subCount));
            }
          }
          continue;
        }

        for (auto attr : Attrs.getAttributes(AttributeList::FirstArgIndex + i))
          NewAttrs = NewAttrs.addAttribute(
              F->getContext(), AttributeList::FirstArgIndex + nexti, attr);
        vals.push_back(CI->getArgOperand(i));
        nexti++;
      }

      assert(sretCount == Types.size());
      assert(local_root_count == numRooting);
    }

    // Because we will += into the corresponding derivative sret, we need to
    // pass in the values that were actually there before the call
    // TODO we can optimize this further and avoid the copy in the primal and/or
    // forward mode as the copy is _only_ needed for the adjoint.
    for (auto &&[val, gep, ty] : preCallReplacements) {
      copyNonJLValueInto(B, ty, ty, gep, {}, ty, val, {}, /*shouldZero*/ true);
    }

    // Actually perform the call, copying over relevant information.
    SmallVector<OperandBundleDef, 1> Bundles;
    for (unsigned I = 0, E = CI->getNumOperandBundles(); I != E; ++I)
      Bundles.emplace_back(CI->getOperandBundleAt(I));

    if (!NewF->getFunctionType()->isVarArg() &&
        NewF->getFunctionType()->getNumParams() != vals.size()) {
      llvm::errs() << "NewF: " << *NewF << "\n";
      for (size_t i = 0; i < vals.size(); i++) {
        llvm::errs() << " Args[" << i << "] = " << *vals[i] << "\n";
      }
    }
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

    if (!RT->isVoidTy()) {
      auto gep = ST ? B.CreateConstInBoundsGEP2_32(ST, sret, 0, 0) : sret;
      auto ld = B.CreateLoad(RT, gep);
      if (auto MD = CI->getMetadata(LLVMContext::MD_range))
        ld->setMetadata(LLVMContext::MD_range, MD);
      ld->takeName(CI);
      Value *replacement = ld;

      // We don't need to override the jlvalue_t's with the rooted versions here
      // since we already stored the full value into the sret above.
      // if (fromRoots) {
      //  replacement = moveSRetToFromRoots(B, replacement->getType(),
      //  replacement, root_AT, root, /*rootOffset*/0,
      //  SRetRootMovement::RootPointerToSRetValue);
      //}

      CI->replaceAllUsesWith(replacement);
    }

    for (auto &&[val, gep, ty] : postCallReplacements) {
      auto ld = B.CreateLoad(ty, gep);
      auto SI = B.CreateStore(ld, val);
      PostCacheStore(SI, B);
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
