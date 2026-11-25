//===- DistBuilder.h - Emit distribution sampler/logpdf IR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds, out of impulse, enzyme.broadcast, arith and math ops, the sampler
// and the log density of an `#impulse.dist` attribute and the map of
// unconstrained values onto a support.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_DIST_BUILDER_H
#define ENZYME_MLIR_INTERFACES_DIST_BUILDER_H

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace impulse {

/// Maps `unconstrained` onto `support`; the identity for a null or REAL
/// support.
Value emitConstrain(OpBuilder &builder, Location loc, SupportAttr support,
                    Value unconstrained);
/// Inverse of emitConstrain.
Value emitUnconstrain(OpBuilder &builder, Location loc, SupportAttr support,
                      Value constrained);
/// Returns log|det J| of emitConstrain at `unconstrained`, summed to a rank-0
/// tensor.
Value emitConstrainLogAbsDetJacobian(OpBuilder &builder, Location loc,
                                     SupportAttr support, Value unconstrained);

/// Elementwise log1p(exp(x)).
Value emitSoftplus(OpBuilder &builder, Location loc, Value x);
/// Elementwise log(exp(x) - 1), the inverse of emitSoftplus.
Value emitSoftplusInv(OpBuilder &builder, Location loc, Value x);

/// Draws `sampleType` values uniform on [0, 1) and advances `rng`.
Value emitUniformDraw(OpBuilder &builder, Location loc, Value &rng,
                      RankedTensorType sampleType);

/// False for rejection-loop samplers, whose RNG stream cannot be replayed.
bool distSamplerEmittable(DistAttr dist);

/// Draws a `sampleType` sample of `dist` with parameters `distArgs` from
/// `rng`. Returns the advanced RNG state, then the sample.
SmallVector<Value> emitDistSampler(OpBuilder &builder, Location loc,
                                   DistAttr dist, Value rng,
                                   ValueRange distArgs,
                                   RankedTensorType sampleType);

/// Returns the log density of `observed` under `dist` with parameters
/// `distArgs`, summed to a rank-0 tensor.
Value emitDistLogpdf(OpBuilder &builder, Location loc, DistAttr dist,
                     Value observed, ValueRange distArgs);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_DIST_BUILDER_H
