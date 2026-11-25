//===- CApi.h - Enzyme API exported to C for external use      -----------===//
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
// This file declares various utility functions of Enzyme for access via C
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_CAPI_H
#define ENZYME_CAPI_H

#include "llvm-c/Core.h"
#include "llvm-c/DataTypes.h"
// #include "llvm-c/Initialization.h"
#include "llvm-c/Target.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct EnzymeOpaqueTypeAnalysis;
typedef struct EnzymeOpaqueTypeAnalysis *EnzymeTypeAnalysisRef;

struct EnzymeOpaqueLogic;
typedef struct EnzymeOpaqueLogic *EnzymeLogicRef;

struct EnzymeOpaqueAugmentedReturn;
typedef struct EnzymeOpaqueAugmentedReturn *EnzymeAugmentedReturnPtr;

struct EnzymeOpaqueTraceInterface;
typedef struct EnzymeOpaqueTraceInterface *EnzymeTraceInterfaceRef;

struct IntList {
  int64_t *data;
  size_t size;
};

typedef enum {
  DT_Anything = 0,
  DT_Integer = 1,
  DT_Pointer = 2,
  DT_Half = 3,
  DT_Float = 4,
  DT_Double = 5,
  DT_Unknown = 6,
  DT_X86_FP80 = 7,
  DT_BFloat16 = 8,
  DT_FP128 = 9,
} CConcreteType;

struct CDataPair {
  struct IntList offsets;
  CConcreteType datatype;
};

/*
struct CTypeTree {
  struct CDataPair *data;
  size_t size;
};
*/

typedef enum {
  VT_None = 0,
  VT_Primal = 1,
  VT_Shadow = 2,
  VT_Both = VT_Primal | VT_Shadow,
} CValueType;

struct EnzymeTypeTree;
typedef struct EnzymeTypeTree *CTypeTreeRef;
CTypeTreeRef EnzymeNewTypeTree();
CTypeTreeRef EnzymeNewTypeTreeCT(CConcreteType, LLVMContextRef ctx);
CTypeTreeRef EnzymeNewTypeTreeTR(CTypeTreeRef);
void EnzymeFreeTypeTree(CTypeTreeRef CTT);
uint8_t EnzymeSetTypeTree(CTypeTreeRef dst, CTypeTreeRef src);
uint8_t EnzymeMergeTypeTree(CTypeTreeRef dst, CTypeTreeRef src);
uint8_t EnzymeCheckedMergeTypeTree(CTypeTreeRef dst, CTypeTreeRef src,
                                   uint8_t *legalP);
void EnzymeTypeTreeOnlyEq(CTypeTreeRef dst, int64_t x);
void EnzymeTypeTreeData0Eq(CTypeTreeRef dst);
void EnzymeTypeTreeLookupEq(CTypeTreeRef CTT, int64_t size, const char *dl);
void EnzymeTypeTreeCanonicalizeInPlace(CTypeTreeRef CTT, int64_t size,
                                       const char *dl);
CConcreteType EnzymeTypeTreeInner0(CTypeTreeRef CTT);
void EnzymeTypeTreeShiftIndiciesEq(CTypeTreeRef dst, const char *datalayout,
                                   int64_t offset, int64_t maxSize,
                                   uint64_t addOffset);
void EnzymeTypeTreeInsertEq(CTypeTreeRef dst, const int64_t *indices,
                            size_t len, CConcreteType ct, LLVMContextRef ctx);
const char *EnzymeTypeTreeToString(CTypeTreeRef src);
void EnzymeTypeTreeToStringFree(const char *cstr);
CTypeTreeRef EnzymeTypeTreeFromMD(LLVMValueRef Val);
LLVMValueRef EnzymeTypeTreeToMD(CTypeTreeRef CTR, LLVMContextRef ctx);

void EnzymeStringFree(const char *cstr);

void EnzymeSetCLBool(void *, uint8_t);
void EnzymeSetCLInteger(void *, int64_t);
void EnzymeSetCLString(void *, const char *);
uint8_t EnzymeGetCLBool(void *);
int64_t EnzymeGetCLInteger(void *);

struct CFnTypeInfo {
  /// Types of arguments, assumed of size len(Arguments)
  CTypeTreeRef *Arguments;

  /// Type of return
  CTypeTreeRef Return;

  /// The specific constant(s) known to represented by an argument, if constant
  // map is [arg number] => list
  struct IntList *KnownValues;
};

typedef enum {
  DFT_OUT_DIFF = 0,  // add differential to an output struct. Only for scalar
                     // values in ReverseMode variants.
  DFT_DUP_ARG = 1,   // duplicate the argument and store differential inside.
                     // For references, pointers, or integers in ReverseMode
                     // variants. For all types in ForwardMode variants.
  DFT_CONSTANT = 2,  // no differential. Usable everywhere.
  DFT_DUP_NONEED = 3 // duplicate this argument and store differential inside,
                     // but don't need the forward. Same as DUP_ARG otherwise.
} CDIFFE_TYPE;

typedef enum { BT_SCALAR = 0, BT_VECTOR = 1 } CBATCH_TYPE;

typedef enum {
  DEM_ForwardMode = 0,
  DEM_ReverseModePrimal = 1,
  DEM_ReverseModeGradient = 2,
  DEM_ReverseModeCombined = 3,
  DEM_ForwardModeSplit = 4,
  DEM_ForwardModeError = 5
} CDerivativeMode;

typedef enum {
  DEM_Trace = 0,
  DEM_Condition = 1,
} CProbProgMode;

typedef uint8_t (*CustomRuleType)(int /*direction*/, CTypeTreeRef /*return*/,
                                  CTypeTreeRef * /*args*/,
                                  struct IntList * /*knownValues*/,
                                  size_t /*numArgs*/, LLVMValueRef,
                                  void * /*TA*/);
EnzymeTypeAnalysisRef CreateTypeAnalysis(EnzymeLogicRef Log,
                                         char **customRuleNames,
                                         CustomRuleType *customRules,
                                         size_t numRules);
void ClearTypeAnalysis(EnzymeTypeAnalysisRef);
void FreeTypeAnalysis(EnzymeTypeAnalysisRef);

EnzymeLogicRef EnzymeTypeAnalysisGetLogic(EnzymeTypeAnalysisRef TAR);

// Type analyzers are passed around as opaque `void *`.
void *EnzymeAnalyzeTypes(EnzymeTypeAnalysisRef TAR, CFnTypeInfo CTI,
                         LLVMValueRef F);
EnzymeTypeAnalysisRef EnzymeGetTypeAnalysisFromTypeAnalyzer(void *TAR);
EnzymeLogicRef EnzymeTypeAnalyzerGetLogic(void *analyzer);
const char *EnzymeTypeAnalyzerToString(void *src);

EnzymeTraceInterfaceRef FindEnzymeStaticTraceInterface(LLVMModuleRef M);
EnzymeTraceInterfaceRef CreateEnzymeStaticTraceInterface(
    LLVMContextRef C, LLVMValueRef getTraceFunction,
    LLVMValueRef getChoiceFunction, LLVMValueRef insertCallFunction,
    LLVMValueRef insertChoiceFunction, LLVMValueRef insertArgumentFunction,
    LLVMValueRef insertReturnFunction, LLVMValueRef insertFunctionFunction,
    LLVMValueRef insertChoiceGradientFunction,
    LLVMValueRef insertArgumentGradientFunction, LLVMValueRef newTraceFunction,
    LLVMValueRef freeTraceFunction, LLVMValueRef hasCallFunction,
    LLVMValueRef hasChoiceFunction);
EnzymeTraceInterfaceRef
CreateEnzymeDynamicTraceInterface(LLVMValueRef interface, LLVMValueRef F);
void FreeTraceInterface(EnzymeTraceInterfaceRef Ref);
EnzymeLogicRef CreateEnzymeLogic(uint8_t PostOpt);
void ClearEnzymeLogic(EnzymeLogicRef);
void FreeEnzymeLogic(EnzymeLogicRef);
void EnzymeLogicErasePreprocessedFunctions(EnzymeLogicRef);
void EnzymeLogicSetExternalContext(EnzymeLogicRef, void *ExternalContext);
void *EnzymeLogicGetExternalContext(EnzymeLogicRef);

void EnzymeExtractReturnInfo(EnzymeAugmentedReturnPtr ret, int64_t *data,
                             uint8_t *existed, size_t len);

LLVMValueRef
EnzymeExtractFunctionFromAugmentation(EnzymeAugmentedReturnPtr ret);
LLVMTypeRef EnzymeExtractTapeTypeFromAugmentation(EnzymeAugmentedReturnPtr ret);
LLVMTypeRef
EnzymeExtractUnderlyingTapeTypeFromAugmentation(EnzymeAugmentedReturnPtr ret);

class GradientUtils;
class DiffeGradientUtils;

typedef LLVMValueRef (*CustomShadowAlloc)(LLVMBuilderRef, LLVMValueRef,
                                          size_t /*numArgs*/, LLVMValueRef *,
                                          GradientUtils *);
typedef LLVMValueRef (*CustomShadowFree)(LLVMBuilderRef, LLVMValueRef);

void EnzymeRegisterAllocationHandler(char *Name, CustomShadowAlloc AHandle,
                                     CustomShadowFree FHandle);

typedef uint8_t (*CustomFunctionForward)(LLVMBuilderRef, LLVMValueRef,
                                         GradientUtils *, LLVMValueRef *,
                                         LLVMValueRef *);
void EnzymeRegisterFwdCallHandler(char *Name, CustomFunctionForward FwdHandle);

typedef uint8_t (*CustomFunctionDiffUse)(LLVMValueRef, const GradientUtils *,
                                         LLVMValueRef, uint8_t, CDerivativeMode,
                                         uint8_t *);
void EnzymeRegisterDiffUseCallHandler(char *Name, CustomFunctionDiffUse Handle);

typedef uint8_t (*CustomAugmentedFunctionForward)(LLVMBuilderRef, LLVMValueRef,
                                                  GradientUtils *,
                                                  LLVMValueRef *,
                                                  LLVMValueRef *,
                                                  LLVMValueRef *);

typedef void (*CustomFunctionReverse)(LLVMBuilderRef, LLVMValueRef,
                                      DiffeGradientUtils *, LLVMValueRef);

LLVMValueRef EnzymeCreateForwardDiff(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, EnzymeTypeAnalysisRef TA, uint8_t returnValue,
    CDerivativeMode mode, uint8_t freeMemory, uint8_t runtimeActivity,
    uint8_t strongZero, unsigned width, LLVMTypeRef additionalArg,
    CFnTypeInfo typeInfo, uint8_t subsequent_calls_may_write,
    uint8_t *_overwritten_args, size_t overwritten_args_size,
    EnzymeAugmentedReturnPtr augmented);

LLVMValueRef EnzymeCreatePrimalAndGradient(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, EnzymeTypeAnalysisRef TA, uint8_t returnValue,
    uint8_t dretUsed, CDerivativeMode mode, uint8_t runtimeActivity,
    uint8_t strongZero, unsigned width, uint8_t freeMemory,
    LLVMTypeRef additionalArg, uint8_t forceAnonymousTape, CFnTypeInfo typeInfo,
    uint8_t subsequent_calls_may_write, uint8_t *_overwritten_args,
    size_t overwritten_args_size, EnzymeAugmentedReturnPtr augmented,
    uint8_t AtomicAdd);

EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef todiff, CDIFFE_TYPE retType, CDIFFE_TYPE *constant_args,
    size_t constant_args_size, EnzymeTypeAnalysisRef TA, uint8_t returnUsed,
    uint8_t shadowReturnUsed, CFnTypeInfo typeInfo,
    uint8_t subsequent_calls_may_write, uint8_t *_overwritten_args,
    size_t overwritten_args_size, uint8_t *_nowrite_shadows,
    size_t nowrite_shadows_size, uint8_t forceAnonymousTape,
    uint8_t runtimeActivity, uint8_t strongZero, unsigned width,
    uint8_t AtomicAdd);

LLVMValueRef EnzymeCreateBatch(EnzymeLogicRef Logic, LLVMValueRef request_req,
                               LLVMBuilderRef request_ip, LLVMValueRef tobatch,
                               unsigned width, CBATCH_TYPE *arg_types,
                               size_t arg_types_size, CBATCH_TYPE retType);

LLVMValueRef EnzymeCreateTrace(
    EnzymeLogicRef Logic, LLVMValueRef request_req, LLVMBuilderRef request_ip,
    LLVMValueRef totrace, LLVMValueRef *sample_functions,
    size_t sample_functions_size, LLVMValueRef *observe_functions,
    size_t observe_functions_size, const char *active_random_variables[],
    size_t active_random_variables_size, CProbProgMode mode, uint8_t autodiff,
    EnzymeTraceInterfaceRef interface);

void EnzymeRegisterCallHandler(const char *Name,
                               CustomAugmentedFunctionForward FwdHandle,
                               CustomFunctionReverse RevHandle);

void *EnzymeGradientUtilsGetExternalContext(GradientUtils *gutils);
EnzymeLogicRef EnzymeGradientUtilsGetLogic(GradientUtils *gutils);
void *EnzymeGradientUtilsTypeAnalyzer(GradientUtils *G);
CDerivativeMode EnzymeGradientUtilsGetMode(GradientUtils *gutils);
uint64_t EnzymeGradientUtilsGetWidth(GradientUtils *gutils);
uint8_t EnzymeGradientUtilsGetRuntimeActivity(GradientUtils *gutils);
uint8_t EnzymeGradientUtilsGetStrongZero(GradientUtils *gutils);
uint8_t EnzymeGradientUtilsGetAtomicAdd(GradientUtils *gutils);
LLVMTypeRef EnzymeGradientUtilsGetShadowType(GradientUtils *gutils,
                                             LLVMTypeRef T);
LLVMTypeRef EnzymeGetShadowType(uint64_t width, LLVMTypeRef T);

LLVMValueRef EnzymeGradientUtilsNewFromOriginal(GradientUtils *gutils,
                                                LLVMValueRef val);
void EnzymeReplaceOriginalToNew(GradientUtils *gutils, LLVMValueRef origC,
                                LLVMValueRef repC);
void EnzymeGradientUtilsSetDebugLocFromOriginal(GradientUtils *gutils,
                                                LLVMValueRef val,
                                                LLVMValueRef orig);
LLVMValueRef EnzymeGradientUtilsLookup(GradientUtils *gutils, LLVMValueRef val,
                                       LLVMBuilderRef builder);
LLVMValueRef EnzymeGradientUtilsInvertPointer(GradientUtils *gutils,
                                              LLVMValueRef val,
                                              LLVMBuilderRef builder);
void EnzymeGradientUtilsErase(GradientUtils *G, LLVMValueRef I);
void EnzymeGradientUtilsEraseWithPlaceholder(GradientUtils *G, LLVMValueRef I,
                                             LLVMValueRef orig, uint8_t erase);
void EnzymeGradientUtilsReplaceAWithB(GradientUtils *G, LLVMValueRef A,
                                      LLVMValueRef B);

CDIFFE_TYPE EnzymeGradientUtilsGetDiffeType(GradientUtils *G, LLVMValueRef oval,
                                            uint8_t foreignFunction);
CDIFFE_TYPE EnzymeGradientUtilsGetReturnDiffeType(GradientUtils *G,
                                                  LLVMValueRef oval,
                                                  uint8_t *needsPrimal,
                                                  uint8_t *needsShadow,
                                                  CDerivativeMode mode);
uint8_t EnzymeGradientUtilsGetAugmentedPrimalReturnUsed(GradientUtils *G,
                                                        LLVMValueRef oval);
uint8_t EnzymeGradientUtilsIsConstantValue(GradientUtils *gutils,
                                           LLVMValueRef val);
uint8_t EnzymeGradientUtilsIsConstantInstruction(GradientUtils *gutils,
                                                 LLVMValueRef val);
uint8_t EnzymeGradientUtilsGetUncacheableArgs(GradientUtils *gutils,
                                              LLVMValueRef orig, uint8_t *data,
                                              uint64_t size);
CTypeTreeRef EnzymeGradientUtilsAllocAndGetTypeTree(GradientUtils *gutils,
                                                    LLVMValueRef val);
void EnzymeGradientUtilsDumpTypeResults(GradientUtils *gutils);
const char *EnzymeGradientUtilsInvertedPointersToString(GradientUtils *gutils,
                                                        void *src);

LLVMBasicBlockRef EnzymeGradientUtilsAllocationBlock(GradientUtils *gutils);
LLVMBasicBlockRef EnzymeGradientUtilsAddReverseBlock(GradientUtils *gutils,
                                                     LLVMBasicBlockRef block,
                                                     const char *name,
                                                     uint8_t forkCache,
                                                     uint8_t push);
void EnzymeGradientUtilsSetReverseBlock(GradientUtils *gutils,
                                        LLVMBasicBlockRef block);

LLVMValueRef EnzymeGradientUtilsCallWithInvertedBundles(
    GradientUtils *gutils, LLVMValueRef func, LLVMTypeRef funcTy,
    LLVMValueRef *args_vr, uint64_t args_size, LLVMValueRef orig_vr,
    CValueType *valTys, uint64_t valTys_size, LLVMBuilderRef B, uint8_t lookup);
void EnzymeGradientUtilsSubTransferHelper(
    GradientUtils *gutils, CDerivativeMode mode, LLVMTypeRef secretty,
    uint64_t intrinsic, uint64_t dstAlign, uint64_t srcAlign, uint64_t offset,
    uint8_t dstConstant, LLVMValueRef shadow_dst, LLVMValueRef primal_dst,
    uint8_t srcConstant, LLVMValueRef shadow_src, LLVMValueRef primal_src,
    LLVMValueRef length, LLVMValueRef isVolatile, LLVMValueRef MTI,
    uint8_t allowForward, uint8_t shadowsLookedUp);

LLVMValueRef EnzymeGradientUtilsDiffe(DiffeGradientUtils *gutils,
                                      LLVMValueRef val, LLVMBuilderRef B);
void EnzymeGradientUtilsSetDiffe(DiffeGradientUtils *gutils, LLVMValueRef val,
                                 LLVMValueRef diffe, LLVMBuilderRef B);
void EnzymeGradientUtilsAddToDiffe(DiffeGradientUtils *gutils, LLVMValueRef val,
                                   LLVMValueRef diffe, LLVMBuilderRef B,
                                   LLVMTypeRef T);
void EnzymeGradientUtilsAddToInvertedPointerDiffe(
    DiffeGradientUtils *gutils, LLVMValueRef orig, LLVMValueRef origVal,
    LLVMTypeRef addingType, unsigned start, unsigned size, LLVMValueRef origptr,
    LLVMValueRef dif, LLVMBuilderRef BuilderM, unsigned align,
    LLVMValueRef mask);
void EnzymeGradientUtilsAddToInvertedPointerDiffeTT(
    DiffeGradientUtils *gutils, LLVMValueRef orig, LLVMValueRef origVal,
    CTypeTreeRef vd, unsigned LoadSize, LLVMValueRef origptr,
    LLVMValueRef prediff, LLVMBuilderRef BuilderM, unsigned align,
    LLVMValueRef premask);

// Callbacks that frontends may set to customize code generation.
typedef LLVMValueRef (*CustomAllocatorFn)(LLVMBuilderRef, LLVMTypeRef,
                                          LLVMValueRef /*Count*/,
                                          LLVMValueRef /*Align*/, uint8_t,
                                          LLVMValueRef *);
typedef LLVMValueRef (*CustomDeallocatorFn)(LLVMBuilderRef, LLVMValueRef);
typedef void (*CustomZeroFn)(LLVMBuilderRef, LLVMTypeRef, LLVMValueRef /*Ptr*/,
                             uint8_t);
typedef void (*CustomRuntimeInactiveErrorFn)(LLVMBuilderRef, LLVMValueRef,
                                             LLVMValueRef);
typedef LLVMValueRef *(*EnzymePostCacheStoreFn)(LLVMValueRef, LLVMBuilderRef,
                                                uint64_t *size);
typedef LLVMTypeRef (*EnzymeDefaultTapeTypeFn)(LLVMContextRef);
typedef LLVMValueRef (*EnzymeUndefinedValueForTypeFn)(LLVMModuleRef,
                                                      LLVMTypeRef, uint8_t);
typedef LLVMValueRef (*EnzymeSanitizeDerivativesFn)(LLVMValueRef,
                                                    LLVMValueRef toset,
                                                    LLVMBuilderRef,
                                                    LLVMValueRef);
typedef void (*EnzymeShadowAllocRewriteFn)(LLVMValueRef, void *, LLVMValueRef,
                                           uint64_t, LLVMValueRef, uint8_t);
typedef LLVMValueRef (*EnzymeFixupReturnFn)(LLVMBuilderRef, LLVMValueRef);

extern CustomAllocatorFn CustomAllocator;
extern CustomDeallocatorFn CustomDeallocator;
extern CustomZeroFn CustomZero;
extern CustomRuntimeInactiveErrorFn CustomRuntimeInactiveError;
extern EnzymePostCacheStoreFn EnzymePostCacheStore;
extern EnzymeDefaultTapeTypeFn EnzymeDefaultTapeType;
extern EnzymeUndefinedValueForTypeFn EnzymeUndefinedValueForType;
extern EnzymeSanitizeDerivativesFn EnzymeSanitizeDerivatives;
extern EnzymeShadowAllocRewriteFn EnzymeShadowAllocRewrite;
extern EnzymeFixupReturnFn EnzymeFixupReturn;

// Utilities on LLVM IR.
LLVMValueRef EnzymeInsertValue(LLVMBuilderRef B, LLVMValueRef val,
                               LLVMValueRef val2, unsigned *sz, int64_t length,
                               const char *name);
LLVMValueRef EnzymeBuildExtractValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                     unsigned *Index, unsigned Size,
                                     const char *Name);
LLVMValueRef EnzymeBuildInsertValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                    LLVMValueRef EltVal, unsigned *Index,
                                    unsigned Size, const char *Name);
LLVMValueRef EnzymeComputeByteOffsetOfGEP(LLVMBuilderRef B_r, LLVMValueRef V_r,
                                          LLVMTypeRef T_r);
LLVMTypeRef EnzymeAllocaType(LLVMValueRef V);
void EnzymeMoveBefore(LLVMValueRef inst1, LLVMValueRef inst2, LLVMBuilderRef B);
void EnzymeSetStringMD(LLVMValueRef Inst, const char *Kind, LLVMValueRef Val);
LLVMValueRef EnzymeGetStringMD(LLVMValueRef Inst, const char *Kind);
void EnzymeSetMustCache(LLVMValueRef inst1);
uint8_t EnzymeHasFromStack(LLVMValueRef inst1);
void EnzymeCopyMetadata(LLVMValueRef inst1, LLVMValueRef inst2);
void EnzymeCopyAlignment(LLVMValueRef inst1, LLVMValueRef inst2);
void EnzymeTakeName(LLVMValueRef inst1, LLVMValueRef inst2);
LLVMMetadataRef EnzymeMakeNonConstTBAA(LLVMMetadataRef MD);
LLVMMetadataRef EnzymeAnonymousAliasScopeDomain(const char *str,
                                                LLVMContextRef ctx);
LLVMMetadataRef EnzymeAnonymousAliasScope(LLVMMetadataRef domain,
                                          const char *str);
void EnzymeCloneFunctionDISubprogramInto(LLVMValueRef NF, LLVMValueRef F);
LLVMValueRef EnzymeCloneFunctionWithoutReturnOrArgs(LLVMValueRef FC,
                                                    uint8_t keepReturnU,
                                                    uint64_t *argrem,
                                                    uint64_t num_argrem);
void EnzymeSetCalledFunction(LLVMValueRef C_CI, LLVMValueRef C_F,
                             uint64_t *argrem, uint64_t num_argrem);
void EnzymeAttributeKnownFunctions(LLVMValueRef FC);
uint8_t EnzymeLowerSparsification(LLVMValueRef F, uint8_t replaceAll);
void EnzymeReplaceFunctionImplementation(LLVMModuleRef M);
void EnzymeDetectReadonlyOrThrow(LLVMModuleRef M);
void RunAttributorOnModule(LLVMModuleRef M0);
void EnzymeAddAttributorLegacyPass(LLVMPassManagerRef PM);
void AddPreserveNVVMPass(LLVMPassManagerRef PM, uint8_t Begin);

void EnzymeDumpModuleRef(LLVMModuleRef M);
void EnzymeDumpValueRef(LLVMValueRef M);
void EnzymeDumpTypeRef(LLVMTypeRef M);

#ifdef __cplusplus
}
#endif

#endif
