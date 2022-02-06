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
#include "llvm-c/Initialization.h"
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

struct EnzymeTypeTree;
typedef struct EnzymeTypeTree *CTypeTreeRef;
CTypeTreeRef EnzymeNewTypeTree();
CTypeTreeRef EnzymeNewTypeTreeCT(CConcreteType, LLVMContextRef ctx);
CTypeTreeRef EnzymeNewTypeTreeTR(CTypeTreeRef);
void EnzymeFreeTypeTree(CTypeTreeRef CTT);
uint8_t EnzymeSetTypeTree(CTypeTreeRef dst, CTypeTreeRef src);
uint8_t EnzymeMergeTypeTree(CTypeTreeRef dst, CTypeTreeRef src);
void EnzymeTypeTreeOnlyEq(CTypeTreeRef dst, int64_t x);
void EnzymeTypeTreeData0Eq(CTypeTreeRef dst);
void EnzymeTypeTreeShiftIndiciesEq(CTypeTreeRef dst, const char *datalayout,
                                   int64_t offset, int64_t maxSize,
                                   uint64_t addOffset);
const char *EnzymeTypeTreeToString(CTypeTreeRef src);
void EnzymeTypeTreeToStringFree(const char *cstr);

void EnzymeSetCLBool(void *, uint8_t);
void EnzymeSetCLInteger(void *, int64_t);

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
  DFT_OUT_DIFF = 0,  // add differential to an output struct
  DFT_DUP_ARG = 1,   // duplicate the argument and store differential inside
  DFT_CONSTANT = 2,  // no differential
  DFT_DUP_NONEED = 3 // duplicate this argument and store differential inside,
                     // but don't need the forward
} CDIFFE_TYPE;

typedef enum {
  DEM_ForwardMode = 0,
  DEM_ReverseModePrimal = 1,
  DEM_ReverseModeGradient = 2,
  DEM_ReverseModeCombined = 3,
  DEM_ForwardModeSplit = 4,
} CDerivativeMode;

/// Generates a new function based on the input, which uses forward-mode AD.
///
/// @param mode: DEM_ForwardMode and DEM_ForwardModeSplit
/// @param retType: DFT_DUP_ARG, DFT_CONSTANT
/// @param constant_args: pointing to combinations of DFT_DUP_ARG and
/// DFT_CONSTANT
/// @param width: integer n >= 1. The generated functions expects n additional
/// inputs for each input marked as DFT_DUP_ARG.
/// @param returnValue: Whether to return the primary return value.
/// The similar dret_used is not available here / implicitely set to true, as it
/// is the only place where we return gradient information.
/// DEM_ReverseModeCombined. Otherwise, should be set to the amount of
/// input params of `todiff`, which might change between the forward and the
/// reverse pass.
LLVMValueRef EnzymeCreateForwardDiff(
    EnzymeLogicRef, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, CDerivativeMode mode,
    unsigned width, LLVMTypeRef additionalArg, struct CFnTypeInfo typeInfo,
    uint8_t *_uncacheable_args, size_t uncacheable_args_size);

/// Generates a new function based on the input, which uses reverse-mode AD.
///
/// Based on the
/// @param retType: When returning f32/f64 we might use DFT_OUT_DIFF.
///      When returning anything else, one should use DFT_CONSTANT
/// @param width: currently only supporting width=1 here
/// @param mode: Should be any of DEM_ReverseModePrimal,
/// DEM_ReverseModeGradient, DEM_ReverseModeCombined.
/// @param augmented: Pass a nullptr, in case of TODO
/// @param AtomicAdd: Enables thread-safe accumulates for being called within
///      a parallel context.
/// @param uncacheable_args_size: Should be set to 0, as long as using
/// DEM_ReverseModeCombined. Otherwise, should be set to the amount of
/// input params of `todiff`, which might change between the forward and the
/// reverse pass.
LLVMValueRef EnzymeCreatePrimalAndGradient(
    EnzymeLogicRef, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnValue, uint8_t dretUsed,
    CDerivativeMode mode, unsigned width, LLVMTypeRef additionalArg,
    struct CFnTypeInfo typeInfo, uint8_t *_uncacheable_args,
    size_t uncacheable_args_size, EnzymeAugmentedReturnPtr augmented,
    uint8_t AtomicAdd);

/// Generates an augmented forward function based on the input.
///
/// Cached information will be stored on a Tape.
/// Will usually be used in combination with EnzymeCreatePrimalAndGradient for
/// reverse-mode-split AD.
/// @param retType: DFT_DUP_ARG, DFT_CONSTANT
/// @param constant_args: pointing to combinations of DFT_DUP_ARG and
/// DFT_CONSTANT
/// @param forceAnonymousTape: TODO
/// @param AtomicAdd: Enables thread-safe accumulates for being called within
///      a parallel context.
EnzymeAugmentedReturnPtr EnzymeCreateAugmentedPrimal(
    EnzymeLogicRef, LLVMValueRef todiff, CDIFFE_TYPE retType,
    CDIFFE_TYPE *constant_args, size_t constant_args_size,
    EnzymeTypeAnalysisRef TA, uint8_t returnUsed, struct CFnTypeInfo typeInfo,
    uint8_t *_uncacheable_args, size_t uncacheable_args_size,
    uint8_t forceAnonymousTape, uint8_t AtomicAdd);

typedef uint8_t (*CustomRuleType)(int /*direction*/, CTypeTreeRef /*return*/,
                                  CTypeTreeRef * /*args*/,
                                  struct IntList * /*knownValues*/,
                                  size_t /*numArgs*/, LLVMValueRef);

/// Creates a new TypeAnalysis, to be used by the three EnzymeCreate functions.
///
/// Usually the TypeAnalysisRef will be reused for a complete compilation
/// process.
EnzymeTypeAnalysisRef CreateTypeAnalysis(EnzymeLogicRef Log,
                                         char **customRuleNames,
                                         CustomRuleType *customRules,
                                         size_t numRules);
void ClearTypeAnalysis(EnzymeTypeAnalysisRef);
void FreeTypeAnalysis(EnzymeTypeAnalysisRef);

/// This will be used by enzyme to manage some internal settings.
///
/// Enzyme requires two optimization runs for best performance, one before AD
/// and one after. The second one can be applied automatically be setting
/// `PostOpt` to 1. Usually the LogicRef will be reused for a complete
/// compilation process.
/// @param PostOpt Should be set to 1, except for debug builds.
EnzymeLogicRef CreateEnzymeLogic(uint8_t PostOpt);
void ClearEnzymeLogic(EnzymeLogicRef);
void FreeEnzymeLogic(EnzymeLogicRef);

void EnzymeExtractReturnInfo(EnzymeAugmentedReturnPtr ret, int64_t *data,
                             uint8_t *existed, size_t len);

LLVMValueRef
EnzymeExtractFunctionFromAugmentation(EnzymeAugmentedReturnPtr ret);
LLVMTypeRef EnzymeExtractTapeTypeFromAugmentation(EnzymeAugmentedReturnPtr ret);

typedef LLVMValueRef (*CustomShadowAlloc)(LLVMBuilderRef, LLVMValueRef,
                                          size_t /*numArgs*/, LLVMValueRef *);
typedef LLVMValueRef (*CustomShadowFree)(LLVMBuilderRef, LLVMValueRef,
                                         LLVMValueRef);

void EnzymeRegisterAllocationHandler(char *Name, CustomShadowAlloc AHandle,
                                     CustomShadowFree FHandle);

class GradientUtils;
class DiffeGradientUtils;

typedef void (*CustomFunctionForward)(LLVMBuilderRef, LLVMValueRef,
                                      GradientUtils *, LLVMValueRef *,
                                      LLVMValueRef *);

typedef void (*CustomAugmentedFunctionForward)(LLVMBuilderRef, LLVMValueRef,
                                               GradientUtils *, LLVMValueRef *,
                                               LLVMValueRef *, LLVMValueRef *);

typedef void (*CustomFunctionReverse)(LLVMBuilderRef, LLVMValueRef,
                                      DiffeGradientUtils *, LLVMValueRef);

#ifdef __cplusplus
}
#endif

#endif
