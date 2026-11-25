//===- ImpulseUtils.h - Utilities for Impulse dialect passes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H
#define ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H

#include "Dialect/Impulse/Impulse.h"
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

/// Clones the non-empty function `fn` into its symbol table as
/// `<name>.<mode>` with the signature the lowering in `mode` needs. A traced
/// mode returns a `1 x positionSize` f64 trace and an f64 weight before the
/// results of `fn`. Generate takes a `1 x constraintSize` constraint before
/// the inputs of `fn`, and regenerate takes the previous `1 x positionSize`
/// trace and, when `constraintSize` is not negative, such a constraint.
FunctionOpInterface cloneForMode(FunctionOpInterface fn, ImpulseMode mode,
                                 int64_t positionSize = -1,
                                 int64_t constraintSize = -1);

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

/// Creates a rank-0 constant `value` of the element type of `type` and
/// broadcasts it to `type`.
Value emitBroadcastConstant(OpBuilder &builder, Location loc,
                            RankedTensorType type, double value);

/// Creates a constant of the matrix type `type` whose (row, col) entry is
/// `entry(row, col)`.
Value emitMatrixConstant(OpBuilder &builder, Location loc,
                         RankedTensorType type,
                         function_ref<double(int64_t row, int64_t col)> entry);

/// Sums all elements of `tensor` to a rank-0 tensor by a dot product with
/// ones.
Value emitSumAll(OpBuilder &builder, Location loc, Value tensor);

/// log(exp(lhs) + exp(rhs)), or lhs + rhs where lhs - rhs is NaN.
Value emitLogAddExp(OpBuilder &builder, Location loc, Value lhs, Value rhs);

/// Elementwise log1p(exp(x)).
Value emitSoftplus(OpBuilder &builder, Location loc, Value x);

/// Elementwise log(exp(x) - 1), the inverse of emitSoftplus.
Value emitSoftplusInv(OpBuilder &builder, Location loc, Value x);

/// Draws `sampleType` values from `distribution` with parameters 0 and 1,
/// that is uniform on [0, 1) or standard normal, and advances `rng`.
Value emitRandomDraw(OpBuilder &builder, Location loc, Value &rng,
                     RankedTensorType sampleType, RngDistribution distribution);

/// Draws `sampleType` values uniform on [0, 1) and advances `rng`.
Value emitUniformDraw(OpBuilder &builder, Location loc, Value &rng,
                      RankedTensorType sampleType);

/// Reads a value of `type` from row 0 of the rank-2 `row`, starting at column
/// `column`.
Value emitRowSlice(OpBuilder &builder, Location loc, Value row, int64_t column,
                   RankedTensorType type);

/// Clones the operations of `block` except its terminator at the insertion
/// point through `mapping`, and returns the terminator's operands mapped.
SmallVector<Value> cloneBlockBody(OpBuilder &builder, Block &block,
                                  IRMapping &mapping);

/// Returns `value` passed through an `enzyme.dump` labeled `label` when
/// `debugDump` is set, else `value` itself.
Value conditionalDump(OpBuilder &builder, Location loc, Value value,
                      StringRef label, bool debugDump);

/// A scalar and its gradient with respect to each argument of the region that
/// computed it.
struct ValueAndGradient {
  Value value;
  SmallVector<Value> grads;
};

/// Emits an `enzyme.autodiff_region` over `args`, seeded with the rank-0
/// `seed`, whose body `body` builds from the region arguments the scalar to
/// differentiate, drawing from the RNG state it is given. Advances `rng` to
/// the RNG state the body leaves, which the region passes through as a
/// constant, and returns the scalar and its gradients. `extraAttrs`, when set,
/// are added to the attributes of the region.
ValueAndGradient
emitGradientRegion(OpBuilder &builder, Location loc, ValueRange args,
                   Value seed, Value &rng, DictionaryAttr extraAttrs,
                   function_ref<Value(ValueRange regionArgs, Value &rng)> body);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_IMPULSE_UTILS_H
