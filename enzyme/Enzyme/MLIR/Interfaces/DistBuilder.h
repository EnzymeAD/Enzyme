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
/// Spike registry: REAL -> [identity], POSITIVE -> [exp].
SmallVector<XformAttr> bijectorFor(MLIRContext *ctx, SupportAttr support);

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
