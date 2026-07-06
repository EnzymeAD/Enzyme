//===- DistBuilder.h - Emit distribution sampler/logpdf IR ----*- C++ -*-===//
//
// Shared bijection/distribution library (F2 spike). A distribution is a
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

/// The base support pushed through each transform's codomain.
SupportAttr deriveSupportFromDist(MLIRContext *ctx, DistAttr dist);

/// biject_to: the transform chain mapping unconstrained space onto `support`.
SmallVector<XformAttr> bijectorFor(MLIRContext *ctx, SupportAttr support);

/// forward: unconstrained z -> constrained y.
Value emitBijectorForward(OpBuilder &builder, Location loc, XformAttr xform,
                          SupportAttr support, Value z);
Value emitBijectorInverse(OpBuilder &builder, Location loc, XformAttr xform,
                          SupportAttr support, Value y);
/// log|det d(forward)/dz| summed over the event, evaluated at unconstrained z.
Value emitBijectorLogAbsDetJac(OpBuilder &builder, Location loc,
                               XformAttr xform, SupportAttr support, Value z);

/// SIMPLEX maps m*(K-1) -> m*K.
int64_t bijectorForwardSize(int64_t unconstrainedSize, SupportAttr support);
int64_t bijectorInverseSize(int64_t constrainedSize, SupportAttr support);

bool isConstantJacobian(XformKind kind);

Value emitLogit(OpBuilder &builder, Location loc, Value x);
Value emitLogSigmoid(OpBuilder &builder, Location loc, Value x);

Value emitSoftplus(OpBuilder &builder, Location loc, Value x);
Value emitSoftplusInv(OpBuilder &builder, Location loc, Value x);

struct ScaledUnitLowerCholeskyResult {
  Value scaleTril;
  Value sumLogDiag;
};

/// NumPyro's ScaledUnitLowerCholeskyTransform; `xFlat` = [strict lower, diag].
ScaledUnitLowerCholeskyResult
emitScaledUnitLowerCholesky(OpBuilder &builder, Location loc, Value xFlat,
                            int64_t d, Type elemType);

/// STUDENT_T (draws a Gamma), and POISSON: their bitwise RNG stream cannot be
/// reproduced without replaying the traced `@trace while` consumption, so those
/// sites keep their traced sampler while the logpdf is still emitted.
bool distSamplerEmittable(DistAttr dist);

/// Returns {newRng, value}.
SmallVector<Value> emitDistSampler(OpBuilder &builder, Location loc,
                                   DistAttr dist, Value rng,
                                   ValueRange distArgs,
                                   RankedTensorType sampleType);

/// base_logpdf(inverse(y)) minus the forward chain's log|det J|, as a scalar.
Value emitDistLogpdf(OpBuilder &builder, Location loc, DistAttr dist,
                     Value observed, ValueRange distArgs);

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_DIST_BUILDER_H
