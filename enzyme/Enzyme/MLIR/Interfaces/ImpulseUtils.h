//===- ImpulseUtils.h - Utilities for Impulse dialect passes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H
#define ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace impulse {

enum class ImpulseMode {
  Call = 0,
  Simulate = 1,
  Generate = 2,
  Regenerate = 3,
};

class ImpulseUtils {
public:
  FunctionOpInterface newFunc;

  ImpulseMode mode;
  FunctionOpInterface oldFunc;
  IRMapping originalToNewFn;

  Block *initializationBlock;

  ImpulseUtils(FunctionOpInterface newFunc_, FunctionOpInterface oldFunc_,
               IRMapping &originalToNewFn_, ImpulseMode mode_)
      : newFunc(newFunc_), mode(mode_), oldFunc(oldFunc_),
        originalToNewFn(originalToNewFn_),
        initializationBlock(&*(newFunc.getFunctionBody().begin())) {}

  static ImpulseUtils *CreateFromClone(FunctionOpInterface toeval,
                                       ImpulseMode mode,
                                       int64_t positionSize = -1,
                                       int64_t constraintSize = -1);
};

/// Creates a constant of `type` with every element equal to `value`.
Value emitSplatConstant(OpBuilder &builder, Location loc, RankedTensorType type,
                        double value);

/// Creates a rank-0 `i64` tensor constant.
Value emitI64ScalarConstant(OpBuilder &builder, Location loc, int64_t value);

/// Creates a rank-0 `i1` tensor constant.
Value emitI1ScalarConstant(OpBuilder &builder, Location loc, bool value);

/// Broadcasts the rank-0 tensor `scalar` to `type`.
Value emitBroadcastScalar(OpBuilder &builder, Location loc, Value scalar,
                          RankedTensorType type);

/// log(exp(lhs) + exp(rhs)), or lhs + rhs where lhs - rhs is NaN.
Value emitLogAddExp(OpBuilder &builder, Location loc, Value lhs, Value rhs);

/// Creates a matrix constant of `type` whose (i, j) entry is 1 where
/// `isOne(i, j)` holds and 0 elsewhere.
Value emitIndicatorMatrix(OpBuilder &builder, Location loc,
                          RankedTensorType type,
                          function_ref<bool(int64_t, int64_t)> isOne);

/// Clones the operations of `block` except its terminator at the insertion
/// point through `mapping`, and returns the terminator's operands mapped.
SmallVector<Value> cloneBlockBody(OpBuilder &builder, Block &block,
                                  IRMapping &mapping);

/// Attributes of an `enzyme.autodiff_region` with `numActiveArgs` active
/// arguments that differentiates its first result and passes its second
/// through as a constant, followed by the entries of `extra` when it is set.
SmallVector<NamedAttribute> buildGradientRegionAttrs(Builder &builder,
                                                     unsigned numActiveArgs,
                                                     DictionaryAttr extra);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H
