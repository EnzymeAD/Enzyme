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

/// Draws a `sampleType` sample of `dist` with parameters `distArgs` and
/// advances `rng`. `dist` must have a loop-free sampler.
Value emitDistSampler(OpBuilder &builder, Location loc, DistAttr dist,
                      Value &rng, ValueRange distArgs,
                      RankedTensorType sampleType);

/// Returns the log density of `observed` under `dist` with parameters
/// `distArgs`, summed to a rank-0 tensor.
Value emitDistLogpdf(OpBuilder &builder, Location loc, DistAttr dist,
                     Value observed, ValueRange distArgs);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_DIST_BUILDER_H
