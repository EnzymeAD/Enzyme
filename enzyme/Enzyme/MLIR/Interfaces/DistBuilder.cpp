//===- DistBuilder.cpp - Emit distribution sampler/logpdf IR --------------===//
// F2 spike: base-plus-transform distribution/bijector emitter. See
// DistBuilder.h.

#include "Interfaces/DistBuilder.h"
#include "Dialect/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <cmath>

using namespace mlir;
using namespace mlir::impulse;

static SupportKind applyXformToSupport(XformAttr xform, SupportKind in) {
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
  case XformKind::AFFINE:
    return in;
  case XformKind::EXP:
    return SupportKind::POSITIVE;
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

SupportAttr impulse::deriveSupportFromDist(MLIRContext *ctx, DistAttr dist) {
  SupportKind kind = SupportKind::REAL; // std_normal base support
  if (ArrayAttr transforms = dist.getTransforms())
    for (Attribute a : transforms)
      kind = applyXformToSupport(cast<XformAttr>(a), kind);
  return SupportAttr::get(ctx, kind, /*lower=*/FloatAttr(),
                          /*upper=*/FloatAttr(), /*event_size=*/IntegerAttr());
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
    default:
      kind = XformKind::IDENTITY; // spike skeleton: only REAL/POSITIVE
      break;
    }
  }
  return {XformAttr::get(ctx, kind, /*components=*/ArrayAttr())};
}

namespace {

Value scalarConst(OpBuilder &b, Location loc, Type elemType, double v) {
  auto scalarType = RankedTensorType::get({}, elemType);
  return arith::ConstantOp::create(
      b, loc, scalarType,
      DenseElementsAttr::get(scalarType, b.getFloatAttr(elemType, v)));
}

// `v` must hold one element unless it already has `sampleType`.
Value broadcastToSample(OpBuilder &b, Location loc, Value v,
                        RankedTensorType sampleType) {
  auto vt = cast<RankedTensorType>(v.getType());
  if (vt == sampleType)
    return v;
  Type elemType = sampleType.getElementType();
  Value scalar = v;
  if (vt.getRank() != 0)
    scalar = impulse::ReshapeOp::create(b, loc,
                                        RankedTensorType::get({}, elemType), v);
  return enzyme::BroadcastOp::create(
      b, loc, sampleType, scalar,
      b.getDenseI64ArrayAttr(sampleType.getShape()));
}

Value reduceSumToScalar(OpBuilder &b, Location loc, Value v) {
  auto vt = cast<RankedTensorType>(v.getType());
  Type elemType = vt.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  int64_t n = 1;
  for (int64_t d : vt.getShape())
    n *= d;
  if (n == 1)
    return impulse::ReshapeOp::create(b, loc, scalarType, v);

  auto flatType = RankedTensorType::get({n}, elemType);
  Value flat = v;
  if (vt.getRank() != 1 || vt.getShape()[0] != n)
    flat = impulse::ReshapeOp::create(b, loc, flatType, v);
  auto ones = arith::ConstantOp::create(
      b, loc, flatType,
      DenseElementsAttr::get(flatType, b.getFloatAttr(elemType, 1.0)));
  return impulse::DotOp::create(
      b, loc, scalarType, flat, ones, b.getDenseI64ArrayAttr({}),
      b.getDenseI64ArrayAttr({}), b.getDenseI64ArrayAttr({0}),
      b.getDenseI64ArrayAttr({0}));
}

Value emitXformForward(OpBuilder &b, Location loc, XformAttr xform, Value z,
                       ValueRange distArgs, RankedTensorType sampleType) {
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return z;
  case XformKind::EXP:
    return math::ExpOp::create(b, loc, z);
  case XformKind::AFFINE: {
    Value locB = broadcastToSample(b, loc, distArgs[0], sampleType);
    Value scaleB = broadcastToSample(b, loc, distArgs[1], sampleType);
    Value scaled = arith::MulFOp::create(b, loc, scaleB, z);
    return arith::AddFOp::create(b, loc, locB, scaled);
  }
  case XformKind::COMPOSE: {
    Value val = z;
    if (ArrayAttr components = xform.getComponents())
      for (Attribute a : components)
        val = emitXformForward(b, loc, cast<XformAttr>(a), val, distArgs,
                               sampleType);
    return val;
  }
  }
  return z;
}

Value emitXformInverse(OpBuilder &b, Location loc, XformAttr xform, Value y,
                       ValueRange distArgs, RankedTensorType sampleType) {
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return y;
  case XformKind::EXP:
    return math::LogOp::create(b, loc, y);
  case XformKind::AFFINE: {
    Value locB = broadcastToSample(b, loc, distArgs[0], sampleType);
    Value scaleB = broadcastToSample(b, loc, distArgs[1], sampleType);
    Value centered = arith::SubFOp::create(b, loc, y, locB);
    return arith::DivFOp::create(b, loc, centered, scaleB);
  }
  case XformKind::COMPOSE: {
    Value val = y;
    if (ArrayAttr components = xform.getComponents()) {
      for (int64_t i = components.size() - 1; i >= 0; --i)
        val = emitXformInverse(b, loc, cast<XformAttr>(components[i]), val,
                               distArgs, sampleType);
    }
    return val;
  }
  }
  return y;
}

// Evaluated at the forward map's input, summed over the event.
Value emitXformLogAbsDetJac(OpBuilder &b, Location loc, XformAttr xform,
                            Value forwardInput, ValueRange distArgs,
                            RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return scalarConst(b, loc, elemType, 0.0);
  case XformKind::EXP:
    return reduceSumToScalar(b, loc, forwardInput);
  case XformKind::AFFINE: {
    // log(scale) without abs, as in the traced Normal logpdf.
    Value scaleB = broadcastToSample(b, loc, distArgs[1], sampleType);
    Value logScale = math::LogOp::create(b, loc, scaleB);
    return reduceSumToScalar(b, loc, logScale);
  }
  case XformKind::COMPOSE: {
    // Not exercised in the spike (chains are flattened at the dist level).
    return scalarConst(b, loc, elemType, 0.0);
  }
  }
  return scalarConst(b, loc, elemType, 0.0);
}

// Standard normal logpdf, summed over the event.
Value emitBaseLogpdf(OpBuilder &b, Location loc, Value z,
                     RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  Value z2 = arith::MulFOp::create(b, loc, z, z);
  Value log2pi = broadcastToSample(
      b, loc, scalarConst(b, loc, elemType, std::log(2.0 * M_PI)), sampleType);
  Value negHalf = broadcastToSample(b, loc, scalarConst(b, loc, elemType, -0.5),
                                    sampleType);
  Value summed = arith::AddFOp::create(b, loc, z2, log2pi);
  Value kernel = arith::MulFOp::create(b, loc, summed, negHalf);
  return reduceSumToScalar(b, loc, kernel);
}

} // namespace

SmallVector<Value> impulse::emitDistSampler(OpBuilder &b, Location loc,
                                            DistAttr dist, Value rng,
                                            ValueRange distArgs,
                                            RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  MLIRContext *ctx = b.getContext();

  // Base draw z ~ StdNormal via impulse.random NORMAL(0, 1).
  Value zero = scalarConst(b, loc, elemType, 0.0);
  Value one = scalarConst(b, loc, elemType, 1.0);
  auto randomOp = impulse::RandomOp::create(
      b, loc, TypeRange{rng.getType(), sampleType}, rng, zero, one,
      impulse::RngDistributionAttr::get(ctx, impulse::RngDistribution::NORMAL));
  Value newRng = randomOp.getOutputRngState();
  Value val = randomOp.getResult();

  if (ArrayAttr transforms = dist.getTransforms())
    for (Attribute a : transforms)
      val = emitXformForward(b, loc, cast<XformAttr>(a), val, distArgs,
                             sampleType);

  return {newRng, val};
}

Value impulse::emitDistLogpdf(OpBuilder &b, Location loc, DistAttr dist,
                              Value observed, ValueRange distArgs) {
  auto sampleType = cast<RankedTensorType>(observed.getType());
  Type elemType = sampleType.getElementType();

  Value val = observed;
  Value logdetTotal = scalarConst(b, loc, elemType, 0.0);

  if (ArrayAttr transforms = dist.getTransforms()) {
    for (int64_t i = transforms.size() - 1; i >= 0; --i) {
      auto xform = cast<XformAttr>(transforms[i]);
      Value inv = emitXformInverse(b, loc, xform, val, distArgs, sampleType);
      Value ld =
          emitXformLogAbsDetJac(b, loc, xform, inv, distArgs, sampleType);
      logdetTotal = arith::AddFOp::create(b, loc, logdetTotal, ld);
      val = inv;
    }
  }

  Value base = emitBaseLogpdf(b, loc, val, sampleType);
  return arith::SubFOp::create(b, loc, base, logdetTotal);
}
