//===- TransformUtils.h - Constraint transforms for HMC --------* C++ -*-===//
//
// This file declares utility functions for constraint transforms for HMC
// inference.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_TRANSFORM_UTILS_H
#define ENZYME_MLIR_INTERFACES_TRANSFORM_UTILS_H

#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace enzyme {
namespace transforms {

/// Get the unconstrained size given a constrained size and support kind.
int64_t getUnconstrainedSize(int64_t constrainedSize, SupportKind kind);

/// Get the constrained size given an unconstrained size and support kind.
int64_t getConstrainedSize(int64_t unconstrainedSize, SupportKind kind);

/// Transform from constrained to unconstrained space.
Value unconstrain(OpBuilder &builder, Location loc, Value constrained,
                  SupportAttr support);

/// Transform from unconstrained to constrained space.
Value constrain(OpBuilder &builder, Location loc, Value unconstrained,
                SupportAttr support);

/// Compute log |det J| of the transform from unconstrained to constrained.
Value logAbsDetJacobian(OpBuilder &builder, Location loc, Value unconstrained,
                        SupportAttr support);

Value createLogit(OpBuilder &builder, Location loc, Value x);
Value createLogSigmoid(OpBuilder &builder, Location loc, Value x);
} // namespace transforms
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_TRANSFORM_UTILS_H
