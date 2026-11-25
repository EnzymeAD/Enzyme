//===- DistBuilder.cpp - Emit distribution sampler/logpdf IR --------------===//
//
// Each sampler and log density fixes its operation order and its RNG draws:
// reordering them changes the bits of the result.
//
//===----------------------------------------------------------------------===//

#include "Interfaces/DistBuilder.h"
#include "Dialect/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::impulse;

static SupportKind applyXformToSupport(XformAttr xform, SupportKind in) {
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return in;
  case XformKind::AFFINE:
    // loc + scale * [0, 1] is an interval with operand bounds.
    return in == SupportKind::UNIT_INTERVAL ? SupportKind::INTERVAL : in;
  case XformKind::EXP:
  case XformKind::SOFTPLUS:
    return SupportKind::POSITIVE;
  case XformKind::SIGMOID:
    return SupportKind::UNIT_INTERVAL;
  case XformKind::INTERVAL:
    return SupportKind::INTERVAL;
  case XformKind::GREATER_THAN:
    return SupportKind::GREATER_THAN;
  case XformKind::LESS_THAN:
    return SupportKind::LESS_THAN;
  case XformKind::STICK_BREAKING:
    return SupportKind::SIMPLEX;
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    return SupportKind::REAL;
  case XformKind::COMPOSE: {
    SupportKind kind = in;
    if (ArrayAttr components = xform.getComponents())
      for (Attribute a : components)
        kind = applyXformToSupport(cast<XformAttr>(a), kind);
    return kind;
  }
  }
  return in;
}

/// Support of the base distribution before its transform chain.
static SupportKind baseSupport(DistBaseKind base) {
  switch (base) {
  case DistBaseKind::STD_UNIFORM:
    return SupportKind::UNIT_INTERVAL;
  case DistBaseKind::EXPONENTIAL:
  case DistBaseKind::GAMMA:
  case DistBaseKind::HALF_NORMAL:
  case DistBaseKind::HALF_CAUCHY:
  case DistBaseKind::HALF_STUDENT_T:
    return SupportKind::POSITIVE;
  case DistBaseKind::DIRICHLET:
    return SupportKind::SIMPLEX;
  case DistBaseKind::STD_NORMAL:
  case DistBaseKind::STUDENT_T:
  case DistBaseKind::BERNOULLI:
  case DistBaseKind::POISSON:
  case DistBaseKind::DISCRETE_UNIFORM:
  case DistBaseKind::MVN:
  case DistBaseKind::GUMBEL:
  case DistBaseKind::CUSTOM:
    return SupportKind::REAL;
  }
  return SupportKind::REAL;
}

SupportAttr impulse::deriveSupportFromDist(MLIRContext *ctx, DistAttr dist) {
  SupportKind kind = baseSupport(dist.getBase());
  if (ArrayAttr transforms = dist.getTransforms())
    for (Attribute a : transforms)
      kind = applyXformToSupport(cast<XformAttr>(a), kind);
  return SupportAttr::get(ctx, kind, /*lower=*/FloatAttr(),
                          /*upper=*/FloatAttr(), /*event_size=*/IntegerAttr());
}

RankedTensorType impulse::deriveSampleValueType(DistAttr dist,
                                                TypeRange distArgTypes) {
  switch (dist.getBase()) {
  case DistBaseKind::MVN:
    if (distArgTypes.empty())
      return RankedTensorType();
    return dyn_cast<RankedTensorType>(distArgTypes.front());
  default:
    return RankedTensorType();
  }
}

bool impulse::distSamplerEmittable(DistAttr dist) {
  switch (dist.getBase()) {
  case DistBaseKind::GAMMA:
  case DistBaseKind::STUDENT_T:
  case DistBaseKind::HALF_STUDENT_T:
  case DistBaseKind::POISSON:
  case DistBaseKind::DIRICHLET:
    return false;
  default:
    return true;
  }
}

SmallVector<XformAttr> impulse::bijectorFor(MLIRContext *ctx,
                                            SupportAttr support) {
  XformKind kind = XformKind::IDENTITY;
  if (support) {
    switch (support.getKind()) {
    case SupportKind::REAL:
      kind = XformKind::IDENTITY;
      break;
    case SupportKind::POSITIVE:
      kind = XformKind::EXP;
      break;
    case SupportKind::UNIT_INTERVAL:
      kind = XformKind::SIGMOID;
      break;
    case SupportKind::INTERVAL:
      kind = XformKind::INTERVAL;
      break;
    case SupportKind::GREATER_THAN:
      kind = XformKind::GREATER_THAN;
      break;
    case SupportKind::LESS_THAN:
      kind = XformKind::LESS_THAN;
      break;
    case SupportKind::SIMPLEX:
      kind = XformKind::STICK_BREAKING;
      break;
    }
  }
  return {XformAttr::get(ctx, kind, /*components=*/ArrayAttr())};
}

Value impulse::emitLogit(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  Value oneConst = emitSplatConstant(builder, loc, xType, 1.0);
  auto oneMinusX = arith::SubFOp::create(builder, loc, oneConst, x);
  auto logX = math::LogOp::create(builder, loc, x);
  auto logOneMinusX = math::LogOp::create(builder, loc, oneMinusX);
  return arith::SubFOp::create(builder, loc, logX, logOneMinusX);
}

Value impulse::emitLogSigmoid(OpBuilder &builder, Location loc, Value x) {
  // -log_add_exp(-x, 0), stable for large |x|.
  auto negX = arith::NegFOp::create(builder, loc, x);
  Value zeroConst =
      emitSplatConstant(builder, loc, cast<RankedTensorType>(x.getType()), 0.0);
  auto softplusNegX = impulse::emitLogAddExp(builder, loc, negX, zeroConst);
  return arith::NegFOp::create(builder, loc, softplusNegX);
}

Value impulse::emitSoftplus(OpBuilder &builder, Location loc, Value x) {
  Value expX = math::ExpOp::create(builder, loc, x);
  return math::Log1pOp::create(builder, loc, expX);
}

Value impulse::emitSoftplusInv(OpBuilder &builder, Location loc, Value x) {
  Value expX = math::ExpOp::create(builder, loc, x);
  Value one =
      emitSplatConstant(builder, loc, cast<RankedTensorType>(x.getType()), 1.0);
  Value expXMinus1 = arith::SubFOp::create(builder, loc, expX, one);
  return math::LogOp::create(builder, loc, expXMinus1);
}

/// Creates a rank-0 constant of `elemType` holding `v`.
static Value scalarConst(OpBuilder &builder, Location loc, Type elemType,
                         double v) {
  return emitSplatConstant(builder, loc, RankedTensorType::get({}, elemType),
                           v);
}

/// `v` must hold one element unless it already has `sampleType`.
static Value broadcastToSample(OpBuilder &builder, Location loc, Value v,
                               RankedTensorType sampleType) {
  auto vt = cast<RankedTensorType>(v.getType());
  if (vt == sampleType)
    return v;
  Type elemType = sampleType.getElementType();
  Value scalar = v;
  if (vt.getRank() != 0)
    scalar = impulse::ReshapeOp::create(builder, loc,
                                        RankedTensorType::get({}, elemType), v);
  return emitBroadcastScalar(builder, loc, scalar, sampleType);
}

/// Sums the rank-1 `vector` to a rank-0 tensor by a dot product with ones.
static Value sumVector(OpBuilder &builder, Location loc, Value vector) {
  auto type = cast<RankedTensorType>(vector.getType());
  Value ones = emitSplatConstant(builder, loc, type, 1.0);
  return impulse::DotOp::create(
      builder, loc, RankedTensorType::get({}, type.getElementType()), vector,
      ones, builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({0}));
}

/// Sums `v` to a rank-0 tensor; a one-element `v` is reshaped, not summed.
static Value reduceSumToScalar(OpBuilder &builder, Location loc, Value v) {
  auto vt = cast<RankedTensorType>(v.getType());
  Type elemType = vt.getElementType();
  int64_t n = vt.getNumElements();
  if (n == 1)
    return impulse::ReshapeOp::create(builder, loc,
                                      RankedTensorType::get({}, elemType), v);
  Value flat = v;
  if (vt.getRank() != 1)
    flat = impulse::ReshapeOp::create(builder, loc,
                                      RankedTensorType::get({n}, elemType), v);
  return sumVector(builder, loc, flat);
}

/// Sums each row of the `numRows` x `numCols` matrix `v`.
static Value reduceLastAxis(OpBuilder &builder, Location loc, Value v,
                            int64_t numRows, int64_t numCols) {
  Type elemType = cast<RankedTensorType>(v.getType()).getElementType();
  auto vecType = RankedTensorType::get({numRows}, elemType);
  auto onesType = RankedTensorType::get({numCols}, elemType);
  Value ones = emitSplatConstant(builder, loc, onesType, 1.0);
  return impulse::DotOp::create(builder, loc, vecType, v, ones,
                                builder.getDenseI64ArrayAttr({1}),
                                builder.getDenseI64ArrayAttr({0}));
}

/// Creates a matrix constant of `type` whose (i, j) entry is 1 where
/// `isOne(i, j)` holds and 0 elsewhere.
static Value emitIndicatorMatrix(OpBuilder &builder, Location loc,
                                 RankedTensorType type,
                                 function_ref<bool(int64_t, int64_t)> isOne) {
  const llvm::fltSemantics &semantics =
      cast<FloatType>(type.getElementType()).getFloatSemantics();
  APFloat zero(semantics, 0);
  APFloat one(semantics, 1);
  SmallVector<APFloat> values;
  values.reserve(type.getNumElements());
  for (int64_t i = 0, rows = type.getDimSize(0); i < rows; ++i)
    for (int64_t j = 0, cols = type.getDimSize(1); j < cols; ++j)
      values.push_back(isOne(i, j) ? one : zero);
  return arith::ConstantOp::create(builder, loc, type,
                                   DenseElementsAttr::get(type, values));
}

namespace {
/// m contiguous simplex blocks of eventSize constrained (n = eventSize - 1
/// free) entries.
struct SimplexDims {
  int64_t eventSize;
  int64_t n;
  int64_t m;
};
} // namespace

/// Creates a constant of the matrix type `ty` whose every row is `rowVals`.
static Value matConst(OpBuilder &builder, Location loc, RankedTensorType ty,
                      ArrayRef<double> rowVals) {
  auto elemType = ty.getElementType();
  int64_t m = ty.getShape()[0], ncol = ty.getShape()[1];
  SmallVector<APFloat> vals;
  vals.reserve(m * ncol);
  for (int64_t r = 0; r < m; ++r)
    for (int64_t c = 0; c < ncol; ++c)
      vals.push_back(cast<FloatAttr>(builder.getFloatAttr(elemType, rowVals[c]))
                         .getValue());
  return arith::ConstantOp::create(builder, loc, ty,
                                   DenseElementsAttr::get(ty, vals));
}

/// The n x n matrix of ones above the diagonal, and on it unless `strict`.
static Value triOnes(OpBuilder &builder, Location loc, int64_t n, bool strict,
                     Type elemType) {
  return emitIndicatorMatrix(
      builder, loc, RankedTensorType::get({n, n}, elemType),
      [&](int64_t i, int64_t j) { return strict ? i < j : i <= j; });
}

/// Multiplies `mat` on the right by the triangular ones matrix `tri`, which
/// sums each row along its prefixes.
static Value prefixSum(OpBuilder &builder, Location loc, Value mat, Value tri) {
  return impulse::DotOp::create(builder, loc, mat.getType(), mat, tri,
                                builder.getDenseI64ArrayAttr({1}),
                                builder.getDenseI64ArrayAttr({0}));
}

/// The m x n matrix whose column c holds log(n - c), the stick-breaking offset.
static Value shiftMat(OpBuilder &builder, Location loc, const SimplexDims &d,
                      Type elemType) {
  SmallVector<double> rowVals(d.n);
  for (int64_t c = 0; c < d.n; ++c)
    rowVals[c] = std::log((double)(d.n - c));
  return matConst(builder, loc, RankedTensorType::get({d.m, d.n}, elemType),
                  rowVals);
}

/// Stick-breaking map from m * n unconstrained values to m simplex blocks.
static Value constrainSimplex(OpBuilder &builder, Location loc,
                              Value unconstrained, const SimplexDims &d) {
  auto elemType =
      cast<RankedTensorType>(unconstrained.getType()).getElementType();
  auto xMatType = RankedTensorType::get({d.m, d.n}, elemType);
  auto yBufType = RankedTensorType::get({d.m, d.eventSize}, elemType);
  auto colType = RankedTensorType::get({d.m, 1}, elemType);

  Value xmat =
      impulse::ReshapeOp::create(builder, loc, xMatType, unconstrained);
  Value xsh = arith::SubFOp::create(builder, loc, xmat,
                                    shiftMat(builder, loc, d, elemType));
  Value z = impulse::LogisticOp::create(builder, loc, xMatType, xsh);
  Value negXsh = arith::NegFOp::create(builder, loc, xsh);
  Value log1mZ = impulse::emitLogSigmoid(builder, loc, negXsh);

  Value logPexcl =
      prefixSum(builder, loc, log1mZ,
                triOnes(builder, loc, d.n, /*strict=*/true, elemType));
  Value pExcl = math::ExpOp::create(builder, loc, logPexcl);
  Value ybody = arith::MulFOp::create(builder, loc, z, pExcl);

  Value onesCol = emitSplatConstant(
      builder, loc, RankedTensorType::get({d.n, 1}, elemType), 1.0);
  Value total = impulse::DotOp::create(builder, loc, colType, log1mZ, onesCol,
                                       builder.getDenseI64ArrayAttr({1}),
                                       builder.getDenseI64ArrayAttr({0}));
  Value remainder = math::ExpOp::create(builder, loc, total);

  auto idx = [&](int64_t v) { return emitI64ScalarConstant(builder, loc, v); };
  Value ybuf = emitSplatConstant(builder, loc, yBufType, 0.0);
  ybuf = impulse::DynamicUpdateSliceOp::create(
      builder, loc, yBufType, ybuf, ybody, ValueRange{idx(0), idx(0)});
  ybuf = impulse::DynamicUpdateSliceOp::create(
      builder, loc, yBufType, ybuf, remainder,
      ValueRange{idx(0), idx(d.eventSize - 1)});

  return impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.eventSize}, elemType), ybuf);
}

/// Inverse of constrainSimplex.
static Value unconstrainSimplex(OpBuilder &builder, Location loc,
                                Value constrained, const SimplexDims &d) {
  auto elemType =
      cast<RankedTensorType>(constrained.getType()).getElementType();
  auto yMatType = RankedTensorType::get({d.m, d.eventSize}, elemType);
  auto xMatType = RankedTensorType::get({d.m, d.n}, elemType);

  Value ymat = impulse::ReshapeOp::create(builder, loc, yMatType, constrained);
  Value ycrop = impulse::SliceOp::create(
      builder, loc, xMatType, ymat, builder.getDenseI64ArrayAttr({0, 0}),
      builder.getDenseI64ArrayAttr({d.m, d.n}),
      builder.getDenseI64ArrayAttr({1, 1}));

  Value cumIncl =
      prefixSum(builder, loc, ycrop,
                triOnes(builder, loc, d.n, /*strict=*/false, elemType));
  Value z1m = arith::SubFOp::create(
      builder, loc, emitSplatConstant(builder, loc, xMatType, 1.0), cumIncl);
  Value ratio = arith::DivFOp::create(builder, loc, ycrop, z1m);
  Value logRatio = math::LogOp::create(builder, loc, ratio);
  Value x = arith::AddFOp::create(builder, loc, logRatio,
                                  shiftMat(builder, loc, d, elemType));

  return impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.n}, elemType), x);
}

/// log|det J| of constrainSimplex at `unconstrained`, summed over all blocks.
static Value logDetSimplex(OpBuilder &builder, Location loc,
                           Value unconstrained, const SimplexDims &d) {
  auto elemType =
      cast<RankedTensorType>(unconstrained.getType()).getElementType();
  auto xMatType = RankedTensorType::get({d.m, d.n}, elemType);

  Value xmat =
      impulse::ReshapeOp::create(builder, loc, xMatType, unconstrained);
  Value xsh = arith::SubFOp::create(builder, loc, xmat,
                                    shiftMat(builder, loc, d, elemType));
  Value logZ = impulse::emitLogSigmoid(builder, loc, xsh);
  Value negXsh = arith::NegFOp::create(builder, loc, xsh);
  Value log1mZ = impulse::emitLogSigmoid(builder, loc, negXsh);
  Value logPexcl =
      prefixSum(builder, loc, log1mZ,
                triOnes(builder, loc, d.n, /*strict=*/true, elemType));
  Value term = arith::AddFOp::create(builder, loc, logZ, log1mZ);
  term = arith::AddFOp::create(builder, loc, term, logPexcl);

  Value flat = impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.n}, elemType), term);
  return sumVector(builder, loc, flat);
}

/// Simplex block sizes of a constrained vector of `constrainedSize` entries.
static SimplexDims simplexDimsFromConstrained(SupportAttr support,
                                              int64_t constrainedSize) {
  int64_t eventSize = support.getEventSize().getInt();
  return {eventSize, eventSize - 1, constrainedSize / eventSize};
}

/// Simplex block sizes of an unconstrained vector of `unconstrainedSize`
/// entries.
static SimplexDims simplexDimsFromUnconstrained(SupportAttr support,
                                                int64_t unconstrainedSize) {
  int64_t eventSize = support.getEventSize().getInt();
  return {eventSize, eventSize - 1, unconstrainedSize / (eventSize - 1)};
}

/// Applies the forward map of `xform` to `z`. AFFINE reads its loc and scale
/// from `distArgs`, the bounded kinds read their bounds from `support`.
static Value emitXformForward(OpBuilder &builder, Location loc, XformAttr xform,
                              SupportAttr support, ValueRange distArgs,
                              Value z) {
  auto zType = cast<RankedTensorType>(z.getType());
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return z;
  case XformKind::EXP:
    return math::ExpOp::create(builder, loc, z);
  case XformKind::SOFTPLUS:
    return emitSoftplus(builder, loc, z);
  case XformKind::SIGMOID:
    return impulse::LogisticOp::create(builder, loc, zType, z);
  case XformKind::AFFINE: {
    Value locB = broadcastToSample(builder, loc, distArgs[0], zType);
    Value scaleB = broadcastToSample(builder, loc, distArgs[1], zType);
    Value scaled = arith::MulFOp::create(builder, loc, scaleB, z);
    return arith::AddFOp::create(builder, loc, locB, scaled);
  }
  case XformKind::INTERVAL: {
    double lower = support.getLowerBound().getValueAsDouble();
    double upper = support.getUpperBound().getValueAsDouble();
    Value lowerConst = emitSplatConstant(builder, loc, zType, lower);
    Value scaleConst = emitSplatConstant(builder, loc, zType, upper - lower);
    auto sigmoid = impulse::LogisticOp::create(builder, loc, zType, z);
    auto scaled = arith::MulFOp::create(builder, loc, scaleConst, sigmoid);
    return arith::AddFOp::create(builder, loc, lowerConst, scaled);
  }
  case XformKind::GREATER_THAN: {
    double lower = support.getLowerBound().getValueAsDouble();
    Value lowerConst = emitSplatConstant(builder, loc, zType, lower);
    auto expZ = math::ExpOp::create(builder, loc, z);
    return arith::AddFOp::create(builder, loc, lowerConst, expZ);
  }
  case XformKind::LESS_THAN: {
    double upper = support.getUpperBound().getValueAsDouble();
    Value upperConst = emitSplatConstant(builder, loc, zType, upper);
    auto expZ = math::ExpOp::create(builder, loc, z);
    return arith::SubFOp::create(builder, loc, upperConst, expZ);
  }
  case XformKind::STICK_BREAKING:
    return constrainSimplex(
        builder, loc, z,
        simplexDimsFromUnconstrained(support, zType.getNumElements()));
  case XformKind::COMPOSE:
    if (ArrayAttr components = xform.getComponents())
      for (Attribute component : components)
        z = emitXformForward(builder, loc, cast<XformAttr>(component), support,
                             distArgs, z);
    return z;
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    // Not elementwise; emitScaledUnitLowerCholesky builds it.
    return z;
  }
  llvm_unreachable("unknown XformKind");
}

/// Applies the inverse map of `xform` to `y`, reading the same operands as
/// emitXformForward.
static Value emitXformInverse(OpBuilder &builder, Location loc, XformAttr xform,
                              SupportAttr support, ValueRange distArgs,
                              Value y) {
  auto yType = cast<RankedTensorType>(y.getType());
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return y;
  case XformKind::EXP:
    return math::LogOp::create(builder, loc, y);
  case XformKind::SOFTPLUS:
    return emitSoftplusInv(builder, loc, y);
  case XformKind::SIGMOID:
    return emitLogit(builder, loc, y);
  case XformKind::AFFINE: {
    Value locB = broadcastToSample(builder, loc, distArgs[0], yType);
    Value scaleB = broadcastToSample(builder, loc, distArgs[1], yType);
    Value centered = arith::SubFOp::create(builder, loc, y, locB);
    return arith::DivFOp::create(builder, loc, centered, scaleB);
  }
  case XformKind::INTERVAL: {
    double lower = support.getLowerBound().getValueAsDouble();
    double upper = support.getUpperBound().getValueAsDouble();
    Value lowerConst = emitSplatConstant(builder, loc, yType, lower);
    Value scaleConst = emitSplatConstant(builder, loc, yType, upper - lower);
    auto shifted = arith::SubFOp::create(builder, loc, y, lowerConst);
    auto normalized = arith::DivFOp::create(builder, loc, shifted, scaleConst);
    return emitLogit(builder, loc, normalized);
  }
  case XformKind::GREATER_THAN: {
    double lower = support.getLowerBound().getValueAsDouble();
    Value lowerConst = emitSplatConstant(builder, loc, yType, lower);
    auto shifted = arith::SubFOp::create(builder, loc, y, lowerConst);
    return math::LogOp::create(builder, loc, shifted);
  }
  case XformKind::LESS_THAN: {
    double upper = support.getUpperBound().getValueAsDouble();
    Value upperConst = emitSplatConstant(builder, loc, yType, upper);
    auto shifted = arith::SubFOp::create(builder, loc, upperConst, y);
    return math::LogOp::create(builder, loc, shifted);
  }
  case XformKind::STICK_BREAKING:
    return unconstrainSimplex(
        builder, loc, y,
        simplexDimsFromConstrained(support, yType.getNumElements()));
  case XformKind::COMPOSE:
    if (ArrayAttr components = xform.getComponents())
      for (Attribute component : llvm::reverse(components.getValue()))
        y = emitXformInverse(builder, loc, cast<XformAttr>(component), support,
                             distArgs, y);
    return y;
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    return y;
  }
  llvm_unreachable("unknown XformKind");
}

/// Elementwise log of the sigmoid derivative, log sigmoid(z) + log sigmoid(-z).
static Value emitLogSigmoidDerivative(OpBuilder &builder, Location loc,
                                      Value z) {
  Value logSigZ = emitLogSigmoid(builder, loc, z);
  Value negZ = arith::NegFOp::create(builder, loc, z);
  Value logSigNegZ = emitLogSigmoid(builder, loc, negZ);
  return arith::AddFOp::create(builder, loc, logSigZ, logSigNegZ);
}

/// log|det J| of the forward map of the distribution transform `xform`,
/// evaluated at its input `forwardInput` and summed over the event.
static Value emitXformLogAbsDetJac(OpBuilder &builder, Location loc,
                                   XformAttr xform, Value forwardInput,
                                   ValueRange distArgs,
                                   RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return scalarConst(builder, loc, elemType, 0.0);
  case XformKind::EXP:
    return reduceSumToScalar(builder, loc, forwardInput);
  case XformKind::SIGMOID:
    return reduceSumToScalar(
        builder, loc, emitLogSigmoidDerivative(builder, loc, forwardInput));
  case XformKind::SOFTPLUS: {
    Value logSigZ = impulse::emitLogSigmoid(builder, loc, forwardInput);
    return reduceSumToScalar(builder, loc, logSigZ);
  }
  case XformKind::AFFINE: {
    // log(scale), not log|scale|: a negative scale gives NaN.
    Value scaleB = broadcastToSample(builder, loc, distArgs[1], sampleType);
    Value logScale = math::LogOp::create(builder, loc, scaleB);
    return reduceSumToScalar(builder, loc, logScale);
  }
  case XformKind::COMPOSE:
  case XformKind::INTERVAL:
  case XformKind::GREATER_THAN:
  case XformKind::LESS_THAN:
  case XformKind::STICK_BREAKING:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    // Not used in distribution chains, which are flat.
    return scalarConst(builder, loc, elemType, 0.0);
  }
  llvm_unreachable("unknown XformKind");
}

Value impulse::emitBijectorForward(OpBuilder &builder, Location loc,
                                   XformAttr xform, SupportAttr support,
                                   Value z) {
  return emitXformForward(builder, loc, xform, support, /*distArgs=*/{}, z);
}

Value impulse::emitBijectorInverse(OpBuilder &builder, Location loc,
                                   XformAttr xform, SupportAttr support,
                                   Value y) {
  return emitXformInverse(builder, loc, xform, support, /*distArgs=*/{}, y);
}

Value impulse::emitBijectorLogAbsDetJac(OpBuilder &builder, Location loc,
                                        XformAttr xform, SupportAttr support,
                                        Value unconstrained) {
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return emitSplatConstant(builder, loc, scalarType, 0.0);
  case XformKind::EXP:
  case XformKind::GREATER_THAN:
  case XformKind::LESS_THAN:
    return sumVector(builder, loc, unconstrained);
  case XformKind::SOFTPLUS: {
    Value logSigZ = impulse::emitLogSigmoid(builder, loc, unconstrained);
    return sumVector(builder, loc, logSigZ);
  }
  case XformKind::SIGMOID:
    return sumVector(builder, loc,
                     emitLogSigmoidDerivative(builder, loc, unconstrained));
  case XformKind::INTERVAL: {
    double scale = support.getUpperBound().getValueAsDouble() -
                   support.getLowerBound().getValueAsDouble();
    Value sumLogProduct = sumVector(
        builder, loc, emitLogSigmoidDerivative(builder, loc, unconstrained));

    int64_t n = zType.getNumElements();
    double logScaleTerm = n * std::log(scale);
    Value logScaleConst =
        emitSplatConstant(builder, loc, scalarType, logScaleTerm);
    return arith::AddFOp::create(builder, loc, sumLogProduct, logScaleConst);
  }
  case XformKind::STICK_BREAKING:
    return logDetSimplex(
        builder, loc, unconstrained,
        simplexDimsFromUnconstrained(support, zType.getNumElements()));
  case XformKind::COMPOSE: {
    Value total = emitSplatConstant(builder, loc, scalarType, 0.0);
    Value z = unconstrained;
    if (ArrayAttr comps = xform.getComponents())
      for (Attribute a : comps) {
        auto comp = cast<XformAttr>(a);
        Value ld = emitBijectorLogAbsDetJac(builder, loc, comp, support, z);
        total = arith::AddFOp::create(builder, loc, total, ld);
        z = emitBijectorForward(builder, loc, comp, support, z);
      }
    return total;
  }
  case XformKind::AFFINE:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    return emitSplatConstant(builder, loc, scalarType, 0.0);
  }
  llvm_unreachable("unknown XformKind");
}

int64_t impulse::bijectorForwardSize(int64_t unconstrainedSize,
                                     SupportAttr support) {
  if (!support)
    return unconstrainedSize;
  if (support.getKind() == SupportKind::SIMPLEX) {
    int64_t eventSize = support.getEventSize().getInt();
    int64_t m = unconstrainedSize / (eventSize - 1);
    return m * eventSize;
  }
  return unconstrainedSize;
}

int64_t impulse::bijectorInverseSize(int64_t constrainedSize,
                                     SupportAttr support) {
  if (!support)
    return constrainedSize;
  if (support.getKind() == SupportKind::SIMPLEX) {
    int64_t eventSize = support.getEventSize().getInt();
    int64_t m = constrainedSize / eventSize;
    return m * (eventSize - 1);
  }
  return constrainedSize;
}

bool impulse::isConstantJacobian(XformKind kind) {
  return kind == XformKind::IDENTITY || kind == XformKind::AFFINE;
}

impulse::ScaledUnitLowerCholeskyResult
impulse::emitScaledUnitLowerCholesky(OpBuilder &builder, Location loc,
                                     Value xFlat, int64_t d, Type elemType) {
  int64_t nOff = d * (d - 1) / 2;
  auto offDiagTy = RankedTensorType::get({nOff}, elemType);
  auto diagTy = RankedTensorType::get({d}, elemType);
  auto matTy = RankedTensorType::get({d, d}, elemType);
  auto scalarTy = RankedTensorType::get({}, elemType);

  Value offDiag, diagPre;
  if (nOff > 0) {
    offDiag = impulse::SliceOp::create(builder, loc, offDiagTy, xFlat,
                                       builder.getDenseI64ArrayAttr({0}),
                                       builder.getDenseI64ArrayAttr({nOff}),
                                       builder.getDenseI64ArrayAttr({1}));
  }
  diagPre = impulse::SliceOp::create(builder, loc, diagTy, xFlat,
                                     builder.getDenseI64ArrayAttr({nOff}),
                                     builder.getDenseI64ArrayAttr({nOff + d}),
                                     builder.getDenseI64ArrayAttr({1}));

  Value diag = impulse::emitSoftplus(builder, loc, diagPre);

  Value logDiag = math::LogOp::create(builder, loc, diag);
  Value sumLogDiag = sumVector(builder, loc, logDiag);

  Value scaleTril = emitSplatConstant(builder, loc, matTy, 0.0);

  auto sliceScalar = [&](Value vec, int64_t k) -> Value {
    auto oneTy = RankedTensorType::get({1}, elemType);
    Value s1 = impulse::SliceOp::create(builder, loc, oneTy, vec,
                                        builder.getDenseI64ArrayAttr({k}),
                                        builder.getDenseI64ArrayAttr({k + 1}),
                                        builder.getDenseI64ArrayAttr({1}));
    return impulse::ReshapeOp::create(builder, loc, scalarTy, s1);
  };
  auto reshapeTo11 = [&](Value scalar) -> Value {
    auto t11 = RankedTensorType::get({1, 1}, elemType);
    return impulse::ReshapeOp::create(builder, loc, t11, scalar);
  };

  int64_t k = 0;
  for (int64_t i = 0; i < d; ++i) {
    Value diagElem = sliceScalar(diag, i);
    for (int64_t j = 0; j < i; ++j) {
      Value offElem = sliceScalar(offDiag, k);
      Value entry = arith::MulFOp::create(builder, loc, offElem, diagElem);
      Value entry11 = reshapeTo11(entry);
      scaleTril = impulse::DynamicUpdateSliceOp::create(
          builder, loc, matTy, scaleTril, entry11,
          ValueRange{emitI64ScalarConstant(builder, loc, i),
                     emitI64ScalarConstant(builder, loc, j)});
      ++k;
    }
    Value diagElem11 = reshapeTo11(diagElem);
    scaleTril = impulse::DynamicUpdateSliceOp::create(
        builder, loc, matTy, scaleTril, diagElem11,
        ValueRange{emitI64ScalarConstant(builder, loc, i),
                   emitI64ScalarConstant(builder, loc, i)});
  }

  return {scaleTril, sumLogDiag};
}

/// Draws `sampleType` values from `distribution` with parameters 0 and 1,
/// that is uniform on [0, 1) or standard normal, and advances `rng`.
static Value emitRandomDraw(OpBuilder &builder, Location loc, Value &rng,
                            RankedTensorType sampleType,
                            RngDistribution distribution) {
  Type elemType = sampleType.getElementType();
  Value zero = scalarConst(builder, loc, elemType, 0.0);
  Value one = scalarConst(builder, loc, elemType, 1.0);
  auto op = impulse::RandomOp::create(
      builder, loc, TypeRange{rng.getType(), sampleType}, rng, zero, one,
      impulse::RngDistributionAttr::get(builder.getContext(), distribution));
  rng = op.getOutputRngState();
  return op.getResult();
}

Value impulse::emitUniformDraw(OpBuilder &builder, Location loc, Value &rng,
                               RankedTensorType sampleType) {
  return emitRandomDraw(builder, loc, rng, sampleType,
                        RngDistribution::UNIFORM);
}

/// Standard exponential sample -log1p(-u) from a uniform sample `u`.
static Value stdExpFromUniform(OpBuilder &builder, Location loc, Value u) {
  Value negU = arith::NegFOp::create(builder, loc, u);
  Value l1p = math::Log1pOp::create(builder, loc, negU);
  return arith::NegFOp::create(builder, loc, l1p);
}

/// Draws a `sampleType` sample of the untransformed `base` and advances `rng`.
static Value emitNativeSampler(OpBuilder &builder, Location loc,
                               DistBaseKind base, Value &rng,
                               ValueRange distArgs,
                               RankedTensorType sampleType) {
  auto bcast = [&](Value v) {
    return broadcastToSample(builder, loc, v, sampleType);
  };
  switch (base) {
  case DistBaseKind::STD_NORMAL:
    return emitRandomDraw(builder, loc, rng, sampleType,
                          RngDistribution::NORMAL);
  case DistBaseKind::STD_UNIFORM:
    return emitUniformDraw(builder, loc, rng, sampleType);
  case DistBaseKind::EXPONENTIAL: {
    Value u = emitUniformDraw(builder, loc, rng, sampleType);
    Value z = stdExpFromUniform(builder, loc, u);
    return arith::DivFOp::create(builder, loc, z, bcast(distArgs[0]));
  }
  case DistBaseKind::BERNOULLI: {
    Value u = emitUniformDraw(builder, loc, rng, sampleType);
    Value logits = bcast(distArgs[0]);
    Value negLogits = arith::NegFOp::create(builder, loc, logits);
    Value e = math::ExpOp::create(builder, loc, negLogits);
    Value one = emitSplatConstant(builder, loc, sampleType, 1.0);
    Value denom = arith::AddFOp::create(builder, loc, one, e);
    Value p = arith::DivFOp::create(builder, loc, one, denom);
    Value lt =
        arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT, u, p);
    Value zero = emitSplatConstant(builder, loc, sampleType, 0.0);
    return arith::SelectOp::create(builder, loc, lt, one, zero);
  }
  case DistBaseKind::DISCRETE_UNIFORM: {
    Value u = emitUniformDraw(builder, loc, rng, sampleType);
    Value low = bcast(distArgs[0]);
    Value high = bcast(distArgs[1]);
    Value one = emitSplatConstant(builder, loc, sampleType, 1.0);
    Value n = arith::AddFOp::create(
        builder, loc, arith::SubFOp::create(builder, loc, high, low), one);
    Value scaled = arith::MulFOp::create(builder, loc, u, n);
    Value fl = math::FloorOp::create(builder, loc, scaled);
    return arith::AddFOp::create(builder, loc, low, fl);
  }
  case DistBaseKind::HALF_NORMAL: {
    Value z =
        emitRandomDraw(builder, loc, rng, sampleType, RngDistribution::NORMAL);
    Value scaled = arith::MulFOp::create(builder, loc, bcast(distArgs[0]), z);
    return math::AbsFOp::create(builder, loc, scaled);
  }
  case DistBaseKind::HALF_CAUCHY: {
    Value u = emitUniformDraw(builder, loc, rng, sampleType);
    // Moves u off 0, the pole of tan(pi * (u - 0.5)).
    double epsF64 = std::numeric_limits<double>::epsilon();
    Value oneMinusEps =
        emitSplatConstant(builder, loc, sampleType, 1.0 - epsF64);
    Value epsV = emitSplatConstant(builder, loc, sampleType, epsF64);
    Value uc = arith::AddFOp::create(
        builder, loc, arith::MulFOp::create(builder, loc, u, oneMinusEps),
        epsV);
    Value half = emitSplatConstant(builder, loc, sampleType, 0.5);
    Value piV = emitSplatConstant(builder, loc, sampleType, M_PI);
    Value arg = arith::MulFOp::create(
        builder, loc, piV, arith::SubFOp::create(builder, loc, uc, half));
    Value t = math::TanOp::create(builder, loc, arg);
    Value at = math::AbsFOp::create(builder, loc, t);
    return arith::MulFOp::create(builder, loc, bcast(distArgs[0]), at);
  }
  case DistBaseKind::MVN: {
    // mu + U^T z with the upper Cholesky factor U of Sigma = U^T U.
    Value mu = distArgs[0];
    Value sigma = distArgs[1];
    int64_t n = sampleType.getNumElements();
    Type elemType = sampleType.getElementType();
    Value z =
        emitRandomDraw(builder, loc, rng, sampleType, RngDistribution::NORMAL);
    auto matType = cast<RankedTensorType>(sigma.getType());
    Value upperChol = impulse::CholeskyOp::create(
        builder, loc, matType, sigma, builder.getBoolAttr(/*lower=*/false));
    auto vecType = RankedTensorType::get({n}, elemType);
    // Contract U's dim 0: a lower factor need not be bitwise U^T.
    Value upperCholTz = impulse::DotOp::create(
        builder, loc, vecType, upperChol, z, builder.getDenseI64ArrayAttr({0}),
        builder.getDenseI64ArrayAttr({0}));
    return arith::AddFOp::create(builder, loc, mu, upperCholTz);
  }
  case DistBaseKind::GUMBEL: {
    // u = 0 maps to -inf.
    Value u = emitUniformDraw(builder, loc, rng, sampleType);
    Value g = arith::NegFOp::create(
        builder, loc,
        math::LogOp::create(
            builder, loc,
            arith::NegFOp::create(builder, loc,
                                  math::LogOp::create(builder, loc, u))));
    Value scaled = arith::MulFOp::create(builder, loc, bcast(distArgs[1]), g);
    return arith::AddFOp::create(builder, loc, bcast(distArgs[0]), scaled);
  }
  default:
    llvm::report_fatal_error(
        "emitNativeSampler: sampler not emittable for this base");
  }
}

SmallVector<Value> impulse::emitDistSampler(OpBuilder &builder, Location loc,
                                            DistAttr dist, Value rng,
                                            ValueRange distArgs,
                                            RankedTensorType sampleType) {
  Value val = emitNativeSampler(builder, loc, dist.getBase(), rng, distArgs,
                                sampleType);
  // The verifier allows chains only on parameterless bases, so the distArgs
  // belong to the chain.
  if (ArrayAttr transforms = dist.getTransforms())
    for (Attribute a : transforms)
      val = emitXformForward(builder, loc, cast<XformAttr>(a), SupportAttr(),
                             distArgs, val);
  return {rng, val};
}

/// Log density of `z` under the uniform distribution on [0, 1], summed over the
/// event.
static Value emitUnitUniformLogpdf(OpBuilder &builder, Location loc, Value z,
                                   RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  Value zero = broadcastToSample(
      builder, loc, scalarConst(builder, loc, elemType, 0.0), sampleType);
  Value one = broadcastToSample(
      builder, loc, scalarConst(builder, loc, elemType, 1.0), sampleType);
  Value negInf =
      broadcastToSample(builder, loc,
                        scalarConst(builder, loc, elemType,
                                    -std::numeric_limits<double>::infinity()),
                        sampleType);
  Value geLo =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGE, z, zero);
  Value leHi =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE, z, one);
  Value inside = arith::AndIOp::create(builder, loc, geLo, leHi);
  Value term = arith::SelectOp::create(builder, loc, inside, zero, negInf);
  return reduceSumToScalar(builder, loc, term);
}

/// Standard normal log density of `z`, summed over the event.
static Value emitStdNormalLogpdf(OpBuilder &builder, Location loc, Value z,
                                 RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  Value z2 = arith::MulFOp::create(builder, loc, z, z);
  Value log2pi = broadcastToSample(
      builder, loc, scalarConst(builder, loc, elemType, std::log(2.0 * M_PI)),
      sampleType);
  Value negHalf = broadcastToSample(
      builder, loc, scalarConst(builder, loc, elemType, -0.5), sampleType);
  Value summed = arith::AddFOp::create(builder, loc, z2, log2pi);
  Value kernel = arith::MulFOp::create(builder, loc, summed, negHalf);
  return reduceSumToScalar(builder, loc, kernel);
}

/// The parameter-only terms {z, coef} of the Student-t log density. Single
/// element df and scale are computed once and broadcast, which gives the same
/// bits as computing them per element.
static std::pair<Value, Value>
emitStudentTParamTerms(OpBuilder &builder, Location loc, Value dfArg,
                       Value scaleArg, RankedTensorType sampleType) {
  auto numel = [](Value v) {
    return cast<RankedTensorType>(v.getType()).getNumElements();
  };
  Type elemType = sampleType.getElementType();
  bool scalar = numel(dfArg) == 1 && numel(scaleArg) == 1;
  auto scalarType = RankedTensorType::get({}, elemType);
  auto toScalar = [&](Value v) -> Value {
    if (cast<RankedTensorType>(v.getType()).getRank() == 0)
      return v;
    return impulse::ReshapeOp::create(builder, loc, scalarType, v);
  };
  RankedTensorType ty = scalar ? scalarType : sampleType;
  Value df = scalar ? toScalar(dfArg)
                    : broadcastToSample(builder, loc, dfArg, sampleType);
  Value scale = scalar ? toScalar(scaleArg)
                       : broadcastToSample(builder, loc, scaleArg, sampleType);
  auto c = [&](double v) { return emitSplatConstant(builder, loc, ty, v); };
  Value half = c(0.5);
  Value logScale = math::LogOp::create(builder, loc, scale);
  Value halfLogDf = arith::MulFOp::create(
      builder, loc, half, math::LogOp::create(builder, loc, df));
  Value z = arith::AddFOp::create(builder, loc, logScale, halfLogDf);
  z = arith::AddFOp::create(builder, loc, z, c(0.5 * std::log(M_PI)));
  Value halfDf = arith::MulFOp::create(builder, loc, half, df);
  z = arith::AddFOp::create(
      builder, loc, z,
      impulse::LgammaOp::create(builder, loc, halfDf.getType(), halfDf));
  Value dfp1 = arith::AddFOp::create(builder, loc, df, c(1.0));
  Value halfDfp1 = arith::MulFOp::create(builder, loc, half, dfp1);
  z = arith::SubFOp::create(
      builder, loc, z,
      impulse::LgammaOp::create(builder, loc, halfDfp1.getType(), halfDfp1));
  Value coef = arith::MulFOp::create(builder, loc, c(-0.5), dfp1);
  if (scalar) {
    z = broadcastToSample(builder, loc, z, sampleType);
    coef = broadcastToSample(builder, loc, coef, sampleType);
  }
  return {z, coef};
}

/// Log density of `observed` under the untransformed `base`, summed over the
/// event.
static Value emitNativeLogpdf(OpBuilder &builder, Location loc,
                              DistBaseKind base, Value observed,
                              ValueRange distArgs) {
  auto sampleType = cast<RankedTensorType>(observed.getType());
  Type elemType = sampleType.getElementType();
  auto bcast = [&](Value v) {
    return broadcastToSample(builder, loc, v, sampleType);
  };
  auto splat = [&](double v) {
    return emitSplatConstant(builder, loc, sampleType, v);
  };
  auto reduce = [&](Value v) { return reduceSumToScalar(builder, loc, v); };
  auto zeroScalar = [&]() { return scalarConst(builder, loc, elemType, 0.0); };
  auto add = [&](Value lhs, Value rhs) -> Value {
    return arith::AddFOp::create(builder, loc, lhs, rhs);
  };

  switch (base) {
  case DistBaseKind::STD_UNIFORM: {
    Value logpdf = emitUnitUniformLogpdf(builder, loc, observed, sampleType);
    return add(logpdf, zeroScalar());
  }
  case DistBaseKind::EXPONENTIAL: {
    Value lam = bcast(distArgs[0]);
    Value unnormed = reduce(arith::NegFOp::create(
        builder, loc, arith::MulFOp::create(builder, loc, lam, observed)));
    Value lognorm = reduce(math::LogOp::create(builder, loc, lam));
    return add(unnormed, lognorm);
  }
  case DistBaseKind::BERNOULLI: {
    Value logits = bcast(distArgs[0]);
    Value unnormed =
        reduce(arith::MulFOp::create(builder, loc, observed, logits));
    Value maxL = arith::MaximumFOp::create(builder, loc, logits, splat(0.0));
    Value negAbs = arith::NegFOp::create(
        builder, loc, math::AbsFOp::create(builder, loc, logits));
    Value sp = impulse::emitSoftplus(builder, loc, negAbs);
    Value term = arith::NegFOp::create(
        builder, loc, arith::AddFOp::create(builder, loc, maxL, sp));
    return add(unnormed, reduce(term));
  }
  case DistBaseKind::DISCRETE_UNIFORM: {
    Value low = bcast(distArgs[0]);
    Value high = bcast(distArgs[1]);
    Value hp1 = arith::AddFOp::create(builder, loc, high, splat(1.0));
    Value range = arith::SubFOp::create(builder, loc, hp1, low);
    Value nlog = arith::NegFOp::create(
        builder, loc, math::LogOp::create(builder, loc, range));
    Value zero = zeroScalar();
    return add(zero, reduce(nlog));
  }
  case DistBaseKind::GAMMA: {
    Value alpha = bcast(distArgs[0]);
    Value beta = bcast(distArgs[1]);
    Value am1 = arith::SubFOp::create(builder, loc, alpha, splat(1.0));
    Value logx = math::LogOp::create(builder, loc, observed);
    Value am1LogX = arith::MulFOp::create(builder, loc, am1, logx);
    Value betaX = arith::MulFOp::create(builder, loc, beta, observed);
    Value kernel = arith::SubFOp::create(builder, loc, am1LogX, betaX);
    Value lgA = impulse::LgammaOp::create(builder, loc, alpha.getType(), alpha);
    Value aLogB = arith::MulFOp::create(
        builder, loc, alpha, math::LogOp::create(builder, loc, beta));
    Value normTerm = arith::SubFOp::create(builder, loc, aLogB, lgA);
    Value unnormed = reduce(kernel);
    return add(unnormed, reduce(normTerm));
  }
  case DistBaseKind::STUDENT_T: {
    auto [z, coef] = emitStudentTParamTerms(builder, loc, distArgs[0],
                                            distArgs[2], sampleType);
    Value df = bcast(distArgs[0]);
    Value locp = bcast(distArgs[1]);
    Value scale = bcast(distArgs[2]);
    Value y = arith::DivFOp::create(
        builder, loc, arith::SubFOp::create(builder, loc, observed, locp),
        scale);
    Value y2df = arith::DivFOp::create(
        builder, loc, arith::MulFOp::create(builder, loc, y, y), df);
    Value kernel = arith::MulFOp::create(
        builder, loc, coef, math::Log1pOp::create(builder, loc, y2df));
    // -(z - kernel), not kernel - z: the two contract to different FMAs.
    Value lp = arith::NegFOp::create(
        builder, loc, arith::SubFOp::create(builder, loc, z, kernel));
    return reduce(lp);
  }
  case DistBaseKind::POISSON: {
    Value rate = bcast(distArgs[0]);
    Value logRate = math::LogOp::create(builder, loc, rate);
    Value kp1 = arith::AddFOp::create(builder, loc, observed, splat(1.0));
    Value lgk = impulse::LgammaOp::create(builder, loc, kp1.getType(), kp1);
    Value kernel = arith::SubFOp::create(
        builder, loc, arith::MulFOp::create(builder, loc, observed, logRate),
        lgk);
    Value lognorm = reduce(arith::NegFOp::create(builder, loc, rate));
    return add(reduce(kernel), lognorm);
  }
  case DistBaseKind::HALF_NORMAL: {
    Value sigma = bcast(distArgs[0]);
    Value sig2 = arith::MulFOp::create(builder, loc, sigma, sigma);
    Value denom = arith::MulFOp::create(builder, loc, splat(2.0), sig2);
    Value x2 = arith::MulFOp::create(builder, loc, observed, observed);
    Value kernel = arith::NegFOp::create(
        builder, loc, arith::DivFOp::create(builder, loc, x2, denom));
    Value logSig = math::LogOp::create(builder, loc, sigma);
    Value norm = arith::SubFOp::create(
        builder, loc,
        arith::SubFOp::create(builder, loc, splat(std::log(2.0)), logSig),
        splat(0.5 * std::log(2.0 * M_PI)));
    Value unnormed = reduce(kernel);
    return add(unnormed, reduce(norm));
  }
  case DistBaseKind::HALF_CAUCHY: {
    Value scale = bcast(distArgs[0]);
    Value xOverS = arith::DivFOp::create(builder, loc, observed, scale);
    Value sq = arith::MulFOp::create(builder, loc, xOverS, xOverS);
    Value kernel = arith::NegFOp::create(
        builder, loc, math::Log1pOp::create(builder, loc, sq));
    Value logS = math::LogOp::create(builder, loc, scale);
    Value norm =
        arith::SubFOp::create(builder, loc, splat(std::log(2.0 / M_PI)), logS);
    Value unnormed = reduce(kernel);
    return add(unnormed, reduce(norm));
  }
  case DistBaseKind::HALF_STUDENT_T: {
    // Student-t(df, 0, scale) truncated at 0: cdf(0) is exactly 0.5, so the
    // normalizer is log(0.5).
    auto [z, coef] = emitStudentTParamTerms(builder, loc, distArgs[0],
                                            distArgs[1], sampleType);
    Value df = bcast(distArgs[0]);
    Value scale = bcast(distArgs[1]);
    Value y = arith::DivFOp::create(builder, loc, observed, scale);
    Value y2df = arith::DivFOp::create(
        builder, loc, arith::MulFOp::create(builder, loc, y, y), df);
    Value kernel = arith::MulFOp::create(
        builder, loc, coef, math::Log1pOp::create(builder, loc, y2df));
    Value lp = arith::SubFOp::create(builder, loc, kernel, z);
    lp = arith::SubFOp::create(builder, loc, lp, splat(std::log(0.5)));
    return reduce(lp);
  }
  case DistBaseKind::DIRICHLET: {
    // The event is the last axis; only lgamma(sum_K alpha) is summed per block.
    int64_t eventSize =
        sampleType.getShape().empty() ? 1 : sampleType.getShape().back();
    int64_t numBlocks = sampleType.getNumElements() / eventSize;
    auto twoDType = RankedTensorType::get({numBlocks, eventSize}, elemType);
    Value alpha = bcast(distArgs[0]);
    Value alpha2d = impulse::ReshapeOp::create(builder, loc, twoDType, alpha);
    Value x2d = impulse::ReshapeOp::create(builder, loc, twoDType, observed);
    Value one2d = emitSplatConstant(builder, loc, twoDType, 1.0);
    Value am1 = arith::SubFOp::create(builder, loc, alpha2d, one2d);
    Value logx = math::LogOp::create(builder, loc, x2d);
    Value kernel = arith::MulFOp::create(builder, loc, logx, am1);
    Value lgAlpha =
        impulse::LgammaOp::create(builder, loc, alpha2d.getType(), alpha2d);
    Value termA = reduce(lgAlpha);
    Value sumKAlpha =
        reduceLastAxis(builder, loc, alpha2d, numBlocks, eventSize);
    Value lgSumK =
        impulse::LgammaOp::create(builder, loc, sumKAlpha.getType(), sumKAlpha);
    Value termB = reduce(lgSumK);
    Value lognorm = arith::SubFOp::create(builder, loc, termB, termA);
    return add(reduce(kernel), lognorm);
  }
  case DistBaseKind::MVN: {
    // |U^-T diff|^2 with one triangular solve, where Sigma = U^T U.
    Value mu = distArgs[0];
    Value sigma = distArgs[1];
    int64_t n = sampleType.getNumElements();
    auto matType = cast<RankedTensorType>(sigma.getType());
    auto vecType = RankedTensorType::get({n}, elemType);
    auto colType = RankedTensorType::get({n, 1}, elemType);

    Value diff = arith::SubFOp::create(builder, loc, observed, mu);
    Value upperChol = impulse::CholeskyOp::create(
        builder, loc, matType, sigma, builder.getBoolAttr(/*lower=*/false));
    Value diffCol = impulse::ReshapeOp::create(builder, loc, colType, diff);
    Value wCol = impulse::TriangularSolveOp::create(
        builder, loc, colType, upperChol, diffCol,
        /*left_side=*/builder.getBoolAttr(true),
        /*lower=*/builder.getBoolAttr(false),
        /*unit_diagonal=*/builder.getBoolAttr(false),
        impulse::TransposeAttr::get(builder.getContext(),
                                    impulse::Transpose::TRANSPOSE));
    Value w = impulse::ReshapeOp::create(builder, loc, vecType, wCol);
    Value maha = reduce(arith::MulFOp::create(builder, loc, w, w));
    Value unnormed = arith::MulFOp::create(
        builder, loc, scalarConst(builder, loc, elemType, -0.5), maha);

    Value identity = emitIndicatorMatrix(
        builder, loc, matType, [](int64_t i, int64_t j) { return i == j; });
    Value upperCholMasked =
        arith::MulFOp::create(builder, loc, upperChol, identity);
    Value onesN = emitSplatConstant(builder, loc, vecType, 1.0);
    Value diagU = impulse::DotOp::create(
        builder, loc, vecType, upperCholMasked, onesN,
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
    Value sumLogDiag = reduce(math::LogOp::create(builder, loc, diagU));
    Value logDet = arith::MulFOp::create(
        builder, loc, scalarConst(builder, loc, elemType, 2.0), sumLogDiag);
    Value nLog2pi =
        scalarConst(builder, loc, elemType, n * std::log(2.0 * M_PI));
    Value constTerms = arith::AddFOp::create(builder, loc, logDet, nLog2pi);
    Value lognorm = arith::MulFOp::create(
        builder, loc, scalarConst(builder, loc, elemType, -0.5), constTerms);
    return add(unnormed, lognorm);
  }
  case DistBaseKind::GUMBEL: {
    Value locp = bcast(distArgs[0]);
    Value scale = bcast(distArgs[1]);
    Value z = arith::DivFOp::create(
        builder, loc, arith::SubFOp::create(builder, loc, observed, locp),
        scale);
    Value negZ = arith::NegFOp::create(builder, loc, z);
    Value expNegZ = math::ExpOp::create(builder, loc, negZ);
    Value kernel = arith::NegFOp::create(
        builder, loc, arith::AddFOp::create(builder, loc, z, expNegZ));
    Value lognorm = arith::NegFOp::create(
        builder, loc, math::LogOp::create(builder, loc, scale));
    Value unnormed = reduce(kernel);
    return add(unnormed, reduce(lognorm));
  }
  default:
    llvm::report_fatal_error("emitNativeLogpdf: unsupported base");
  }
}

Value impulse::emitDistLogpdf(OpBuilder &builder, Location loc, DistAttr dist,
                              Value observed, ValueRange distArgs) {
  auto sampleType = cast<RankedTensorType>(observed.getType());
  Type elemType = sampleType.getElementType();

  ArrayAttr chain = dist.getTransforms();
  bool hasChain = chain && !chain.empty();
  if (dist.getBase() != DistBaseKind::STD_NORMAL &&
      !(dist.getBase() == DistBaseKind::STD_UNIFORM && hasChain)) {
    return emitNativeLogpdf(builder, loc, dist.getBase(), observed, distArgs);
  }

  Value val = observed;
  Value logdetTotal = scalarConst(builder, loc, elemType, 0.0);

  if (ArrayAttr transforms = dist.getTransforms()) {
    for (int64_t i = transforms.size() - 1; i >= 0; --i) {
      auto xform = cast<XformAttr>(transforms[i]);
      Value inv =
          emitXformInverse(builder, loc, xform, SupportAttr(), distArgs, val);
      Value ld =
          emitXformLogAbsDetJac(builder, loc, xform, inv, distArgs, sampleType);
      logdetTotal = arith::AddFOp::create(builder, loc, logdetTotal, ld);
      val = inv;
    }
  }

  Value base = dist.getBase() == DistBaseKind::STD_NORMAL
                   ? emitStdNormalLogpdf(builder, loc, val, sampleType)
                   : emitUnitUniformLogpdf(builder, loc, val, sampleType);
  return arith::SubFOp::create(builder, loc, base, logdetTotal);
}
