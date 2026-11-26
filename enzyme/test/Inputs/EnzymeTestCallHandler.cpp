//===- EnzymeTestCallHandler.cpp - Custom call handlers for LIT tests -----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic plugin loaded by `opt` in LIT tests to register custom call handlers
// exercising `GradientUtils::cacheForReverse` when custom forward handlers
// return primal or shadow function arguments.
//
//===----------------------------------------------------------------------===//

#include "CApi.h"
#include "llvm-c/Core.h"

/// Custom augmented forward handler for `__enzyme_test_arg_shadow(ptr %p)`.
/// Forwards the shadow of `%p` as the call's shadow return, exercising
/// `CacheType::Shadow` when `%p` is an argument of the differentiated function.
static uint8_t forwardArgShadow(LLVMBuilderRef builder, LLVMValueRef call,
                                GradientUtils *gutils,
                                LLVMValueRef *normalReturn,
                                LLVMValueRef *shadowReturn,
                                LLVMValueRef *tape) {
  (void)normalReturn;
  (void)tape;
  if (shadowReturn) {
    LLVMValueRef ptrOperand = LLVMGetOperand(call, 0);
    *shadowReturn =
        EnzymeGradientUtilsInvertPointer(gutils, ptrOperand, builder);
  }
  return 1;
}

/// No-op custom reverse handler for calls whose derivatives propagate solely
/// through their primal or shadow return value.
static void reverseNoOp(LLVMBuilderRef builder, LLVMValueRef call,
                        DiffeGradientUtils *gutils, LLVMValueRef tape) {
  (void)builder;
  (void)call;
  (void)gutils;
  (void)tape;
}

/// Custom augmented forward handler for
/// `__enzyme_test_arg_tape(ptr %out, double %scale)`.
/// Passes the primal `%scale` operand via `*tape`, exercising `CacheType::Tape`
/// when `%scale` is an argument of the differentiated function.
static uint8_t forwardArgTape(LLVMBuilderRef builder, LLVMValueRef call,
                              GradientUtils *gutils, LLVMValueRef *normalReturn,
                              LLVMValueRef *shadowReturn, LLVMValueRef *tape) {
  (void)builder;
  (void)normalReturn;
  (void)shadowReturn;
  LLVMValueRef scaleOperand = LLVMGetOperand(call, 1);
  *tape = EnzymeGradientUtilsNewFromOriginal(gutils, scaleOperand);
  return 0;
}

/// Custom reverse handler for `__enzyme_test_arg_tape(ptr %out, double
/// %scale)`. Scales `*out'` in place by the rebuilt `tape` value (`*out' *=
/// scale`).
static void reverseArgTape(LLVMBuilderRef builder, LLVMValueRef call,
                           DiffeGradientUtils *gutils, LLVMValueRef tape) {
  auto *baseGUtils = reinterpret_cast<GradientUtils *>(gutils);
  LLVMValueRef outOperand = LLVMGetOperand(call, 0);
  LLVMValueRef shadowOut =
      EnzymeGradientUtilsInvertPointer(baseGUtils, outOperand, builder);
  shadowOut = EnzymeGradientUtilsLookup(baseGUtils, shadowOut, builder);
  LLVMTypeRef valTy = LLVMTypeOf(tape);
  LLVMValueRef prevGrad = LLVMBuildLoad2(builder, valTy, shadowOut, "d_out");
  LLVMValueRef scaledGrad =
      LLVMBuildFMul(builder, prevGrad, tape, "scaled_d_out");
  LLVMBuildStore(builder, scaledGrad, shadowOut);
}

/// Custom augmented forward handler for `__enzyme_test_arg_self(double %s)`.
/// Replaces `*normalReturn` with the primal `%s` operand, exercising
/// `CacheType::Self` replacement when `%s` is a function argument.
static uint8_t forwardArgSelf(LLVMBuilderRef builder, LLVMValueRef call,
                              GradientUtils *gutils, LLVMValueRef *normalReturn,
                              LLVMValueRef *shadowReturn, LLVMValueRef *tape) {
  (void)builder;
  (void)shadowReturn;
  (void)tape;
  if (normalReturn) {
    LLVMValueRef argOperand = LLVMGetOperand(call, 0);
    *normalReturn = EnzymeGradientUtilsNewFromOriginal(gutils, argOperand);
  }
  return 0;
}

namespace {
struct RegisterTestCallHandlers {
  RegisterTestCallHandlers() {
    EnzymeRegisterCallHandler("__enzyme_test_arg_shadow", forwardArgShadow,
                              reverseNoOp);
    EnzymeRegisterCallHandler("__enzyme_test_arg_tape", forwardArgTape,
                              reverseArgTape);
    EnzymeRegisterCallHandler("__enzyme_test_arg_self", forwardArgSelf,
                              reverseNoOp);
  }
};
} // namespace

static RegisterTestCallHandlers registerTestCallHandlers;
