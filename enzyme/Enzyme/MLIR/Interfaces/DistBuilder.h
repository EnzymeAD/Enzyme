//===- DistBuilder.h - Emit distribution sampler/logpdf IR ----*- C++ -*-===//
//
// Builds the sampler, the log density and the unconstraining bijection of an
// `#impulse.dist` attribute out of impulse, enzyme.broadcast, arith and math
// ops.
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

/// The bijection from unconstrained space onto `support`, IDENTITY when
/// `support` is null.
XformAttr bijectorFor(MLIRContext *ctx, SupportAttr support);

/// forward: unconstrained z -> constrained y.
Value emitBijectorForward(OpBuilder &builder, Location loc, XformAttr xform,
                          SupportAttr support, Value z);
/// inverse: constrained y -> unconstrained z.
Value emitBijectorInverse(OpBuilder &builder, Location loc, XformAttr xform,
                          SupportAttr support, Value y);
/// log|det d(forward)/dz| summed over the event, evaluated at unconstrained z.
Value emitBijectorLogAbsDetJac(OpBuilder &builder, Location loc,
                               XformAttr xform, SupportAttr support, Value z);

/// True when log|det J| of `kind` does not depend on its input.
bool isConstantJacobian(XformKind kind);

/// Elementwise log(x / (1 - x)).
Value emitLogit(OpBuilder &builder, Location loc, Value x);
/// Elementwise log(sigmoid(x)), stable for large |x|.
Value emitLogSigmoid(OpBuilder &builder, Location loc, Value x);

/// Elementwise log1p(exp(x)).
Value emitSoftplus(OpBuilder &builder, Location loc, Value x);
/// Elementwise log(exp(x) - 1), the inverse of emitSoftplus.
Value emitSoftplusInv(OpBuilder &builder, Location loc, Value x);

struct ScaledUnitLowerCholeskyResult {
  Value scaleTril;
  Value sumLogDiag;
};

/// Builds the scale factor diag(softplus(d)) * L, with L unit lower triangular,
/// from `xFlat` = [strict lower entries of L in row-major order, d], and the
/// sum of log(softplus(d)).
ScaledUnitLowerCholeskyResult
emitScaledUnitLowerCholesky(OpBuilder &builder, Location loc, Value xFlat,
                            int64_t d, Type elemType);

/// Draws `sampleType` values uniform on [0, 1) and advances `rng`.
Value emitUniformDraw(OpBuilder &builder, Location loc, Value &rng,
                      RankedTensorType sampleType);

/// False for rejection-loop samplers, whose RNG stream cannot be replayed.
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
