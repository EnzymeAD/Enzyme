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

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace enzyme {
namespace transforms {

/// Unconstrained size for `support`; SIMPLEX maps m*K to m*(K-1).
int64_t getUnconstrainedSize(int64_t constrainedSize,
                             impulse::SupportAttr support);

/// Transform from constrained to unconstrained space.
Value unconstrain(OpBuilder &builder, Location loc, Value constrained,
                  impulse::SupportAttr support);

/// Transform from unconstrained to constrained space.
Value constrain(OpBuilder &builder, Location loc, Value unconstrained,
                impulse::SupportAttr support);

/// Compute log |det J| of the transform from unconstrained to constrained.
Value logAbsDetJacobian(OpBuilder &builder, Location loc, Value unconstrained,
                        impulse::SupportAttr support);
} // namespace transforms
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_TRANSFORM_UTILS_H
