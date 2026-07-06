//===- DistBuilder.cpp - Emit distribution sampler/logpdf IR --------------===//
// F2 spike: base-plus-transform distribution/bijector emitter. See
// DistBuilder.h.

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
  case XformKind::AFFINE:
    return in;
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

// Must agree with the frontend `support(::Type)`; the verifier compares them.
static SupportKind baseSupport(DistBaseKind base) {
  switch (base) {
  case DistBaseKind::STD_UNIFORM:
    return SupportKind::UNIT_INTERVAL;
  case DistBaseKind::EXPONENTIAL:
  case DistBaseKind::GAMMA:
  case DistBaseKind::HALF_NORMAL:
  case DistBaseKind::HALF_CAUCHY:
    return SupportKind::POSITIVE;
  case DistBaseKind::DIRICHLET:
    return SupportKind::SIMPLEX;
  case DistBaseKind::STD_NORMAL:
  case DistBaseKind::STUDENT_T:
  case DistBaseKind::BERNOULLI:
  case DistBaseKind::POISSON:
  case DistBaseKind::DISCRETE_UNIFORM:
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

bool impulse::distSamplerEmittable(DistAttr dist) {
  switch (dist.getBase()) {
  case DistBaseKind::GAMMA:
  case DistBaseKind::STUDENT_T:
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

Value reduceLastAxis(OpBuilder &b, Location loc, Value v, int64_t M,
                     int64_t K) {
  Type elemType = cast<RankedTensorType>(v.getType()).getElementType();
  auto vecType = RankedTensorType::get({M}, elemType);
  auto onesType = RankedTensorType::get({K}, elemType);
  auto ones = arith::ConstantOp::create(
      b, loc, onesType,
      DenseElementsAttr::get(onesType, b.getFloatAttr(elemType, 1.0)));
  return impulse::DotOp::create(
      b, loc, vecType, v, ones, b.getDenseI64ArrayAttr({}),
      b.getDenseI64ArrayAttr({}), b.getDenseI64ArrayAttr({1}),
      b.getDenseI64ArrayAttr({0}));
}

Value emitXformForward(OpBuilder &b, Location loc, XformAttr xform, Value z,
                       ValueRange distArgs, RankedTensorType sampleType) {
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return z;
  case XformKind::EXP:
    return math::ExpOp::create(b, loc, z);
  case XformKind::SIGMOID:
    return impulse::LogisticOp::create(b, loc, z.getType(), z);
  case XformKind::SOFTPLUS:
    return impulse::emitSoftplus(b, loc, z);
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
  case XformKind::INTERVAL:
  case XformKind::GREATER_THAN:
  case XformKind::LESS_THAN:
  case XformKind::STICK_BREAKING:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    // Not used in distribution chains; see emitBijectorForward.
    return z;
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
  case XformKind::SIGMOID:
    return impulse::emitLogit(b, loc, y);
  case XformKind::SOFTPLUS:
    return impulse::emitSoftplusInv(b, loc, y);
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
  case XformKind::INTERVAL:
  case XformKind::GREATER_THAN:
  case XformKind::LESS_THAN:
  case XformKind::STICK_BREAKING:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    return y;
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
  case XformKind::SIGMOID: {
    Value logSigZ = impulse::emitLogSigmoid(b, loc, forwardInput);
    Value negZ = arith::NegFOp::create(b, loc, forwardInput);
    Value logSigNegZ = impulse::emitLogSigmoid(b, loc, negZ);
    Value logProduct = arith::AddFOp::create(b, loc, logSigZ, logSigNegZ);
    return reduceSumToScalar(b, loc, logProduct);
  }
  case XformKind::SOFTPLUS: {
    Value logSigZ = impulse::emitLogSigmoid(b, loc, forwardInput);
    return reduceSumToScalar(b, loc, logSigZ);
  }
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
  case XformKind::INTERVAL:
  case XformKind::GREATER_THAN:
  case XformKind::LESS_THAN:
  case XformKind::STICK_BREAKING:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    return scalarConst(b, loc, elemType, 0.0);
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

// `lognorm` depends only on the parameters, so it can be hoisted.
struct LogpdfSplit {
  Value unnormed;
  Value lognorm; // value-independent normalizer (scalar tensor)
};

Value splatSample(OpBuilder &b, Location loc, RankedTensorType sampleType,
                  double v) {
  return arith::ConstantOp::create(
      b, loc, sampleType,
      DenseElementsAttr::get(sampleType,
                             b.getFloatAttr(sampleType.getElementType(), v)));
}

// Lowers to the same bits as Reactant's `rand`, matching the traced path.
Value emitUniformDraw(OpBuilder &b, Location loc, Value &rng,
                      RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  Value zero = scalarConst(b, loc, elemType, 0.0);
  Value one = scalarConst(b, loc, elemType, 1.0);
  auto op = impulse::RandomOp::create(
      b, loc, TypeRange{rng.getType(), sampleType}, rng, zero, one,
      impulse::RngDistributionAttr::get(b.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  rng = op.getOutputRngState();
  return op.getResult();
}

// Op for op as Reactant's `randexp`.
Value stdExpFromUniform(OpBuilder &b, Location loc, Value u) {
  Value negU = arith::NegFOp::create(b, loc, u);
  Value l1p = math::Log1pOp::create(b, loc, negU);
  return arith::NegFOp::create(b, loc, l1p);
}

Value emitNativeSampler(OpBuilder &b, Location loc, DistBaseKind base,
                        Value &rng, ValueRange distArgs,
                        RankedTensorType sampleType) {
  auto bcast = [&](Value v) {
    return broadcastToSample(b, loc, v, sampleType);
  };
  switch (base) {
  case DistBaseKind::STD_UNIFORM:
    return emitUniformDraw(b, loc, rng, sampleType);
  case DistBaseKind::EXPONENTIAL: {
    Value u = emitUniformDraw(b, loc, rng, sampleType);
    Value z = stdExpFromUniform(b, loc, u);
    return arith::DivFOp::create(b, loc, z, bcast(distArgs[0]));
  }
  case DistBaseKind::BERNOULLI: {
    Value u = emitUniformDraw(b, loc, rng, sampleType);
    Value logits = bcast(distArgs[0]);
    // 1 / (1 + exp(-logits)), not LogisticOp: bitwise as the traced sampler.
    Value negLogits = arith::NegFOp::create(b, loc, logits);
    Value e = math::ExpOp::create(b, loc, negLogits);
    Value one = splatSample(b, loc, sampleType, 1.0);
    Value denom = arith::AddFOp::create(b, loc, one, e);
    Value p = arith::DivFOp::create(b, loc, one, denom);
    Value lt = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, u, p);
    Value zero = splatSample(b, loc, sampleType, 0.0);
    return arith::SelectOp::create(b, loc, lt, one, zero);
  }
  case DistBaseKind::DISCRETE_UNIFORM: {
    Value u = emitUniformDraw(b, loc, rng, sampleType);
    Value low = bcast(distArgs[0]);
    Value high = bcast(distArgs[1]);
    Value one = splatSample(b, loc, sampleType, 1.0);
    // (high - low) + 1 in the traced sampler's order.
    Value n = arith::AddFOp::create(
        b, loc, arith::SubFOp::create(b, loc, high, low), one);
    Value scaled = arith::MulFOp::create(b, loc, u, n);
    Value fl = math::FloorOp::create(b, loc, scaled);
    return arith::AddFOp::create(b, loc, low, fl);
  }
  case DistBaseKind::HALF_NORMAL: {
    // Matches the traced abs.(sigma .* randn) only up to sub-ULP differences.
    Type elemType = sampleType.getElementType();
    Value zero = scalarConst(b, loc, elemType, 0.0);
    Value one = scalarConst(b, loc, elemType, 1.0);
    auto op = impulse::RandomOp::create(
        b, loc, TypeRange{rng.getType(), sampleType}, rng, zero, one,
        impulse::RngDistributionAttr::get(b.getContext(),
                                          impulse::RngDistribution::NORMAL));
    rng = op.getOutputRngState();
    Value scaled =
        arith::MulFOp::create(b, loc, bcast(distArgs[0]), op.getResult());
    return math::AbsFOp::create(b, loc, scaled);
  }
  case DistBaseKind::HALF_CAUCHY: {
    // u * (1 - eps) + eps, as in the traced JAX construction.
    Value u = emitUniformDraw(b, loc, rng, sampleType);
    double epsF64 = std::numeric_limits<double>::epsilon();
    Value oneMinusEps = splatSample(b, loc, sampleType, 1.0 - epsF64);
    Value epsV = splatSample(b, loc, sampleType, epsF64);
    Value uc = arith::AddFOp::create(
        b, loc, arith::MulFOp::create(b, loc, u, oneMinusEps), epsV);
    Value half = splatSample(b, loc, sampleType, 0.5);
    Value piV = splatSample(b, loc, sampleType, M_PI);
    Value arg = arith::MulFOp::create(b, loc, piV,
                                      arith::SubFOp::create(b, loc, uc, half));
    Value t = math::TanOp::create(b, loc, arg);
    Value at = math::AbsFOp::create(b, loc, t);
    return arith::MulFOp::create(b, loc, bcast(distArgs[0]), at);
  }
  default:
    llvm::report_fatal_error(
        "emitNativeSampler: sampler not emittable for this base");
  }
}

LogpdfSplit emitNativeLogpdfSplit(OpBuilder &b, Location loc, DistBaseKind base,
                                  Value observed, ValueRange distArgs) {
  auto sampleType = cast<RankedTensorType>(observed.getType());
  Type elemType = sampleType.getElementType();
  auto bcast = [&](Value v) {
    return broadcastToSample(b, loc, v, sampleType);
  };
  auto splat = [&](double v) { return splatSample(b, loc, sampleType, v); };
  auto reduce = [&](Value v) { return reduceSumToScalar(b, loc, v); };
  auto zeroScalar = [&]() { return scalarConst(b, loc, elemType, 0.0); };

  switch (base) {
  case DistBaseKind::STD_UNIFORM:
    // Density 1 on [0, 1): both terms vanish.
    return {zeroScalar(), zeroScalar()};
  case DistBaseKind::EXPONENTIAL: {
    Value lam = bcast(distArgs[0]);
    Value unnormed = reduce(arith::NegFOp::create(
        b, loc, arith::MulFOp::create(b, loc, lam, observed)));
    Value lognorm = reduce(math::LogOp::create(b, loc, lam));
    return {unnormed, lognorm};
  }
  case DistBaseKind::BERNOULLI: {
    Value logits = bcast(distArgs[0]);
    Value unnormed = reduce(arith::MulFOp::create(b, loc, observed, logits));
    Value maxL = arith::MaximumFOp::create(b, loc, logits, splat(0.0));
    Value negAbs =
        arith::NegFOp::create(b, loc, math::AbsFOp::create(b, loc, logits));
    Value sp = impulse::emitSoftplus(b, loc, negAbs);
    Value term =
        arith::NegFOp::create(b, loc, arith::AddFOp::create(b, loc, maxL, sp));
    return {unnormed, reduce(term)};
  }
  case DistBaseKind::DISCRETE_UNIFORM: {
    Value low = bcast(distArgs[0]);
    Value high = bcast(distArgs[1]);
    Value hp1 = arith::AddFOp::create(b, loc, high, splat(1.0));
    Value range = arith::SubFOp::create(b, loc, hp1, low);
    Value nlog =
        arith::NegFOp::create(b, loc, math::LogOp::create(b, loc, range));
    return {zeroScalar(), reduce(nlog)};
  }
  case DistBaseKind::GAMMA: {
    Value alpha = bcast(distArgs[0]);
    Value beta = bcast(distArgs[1]);
    Value am1 = arith::SubFOp::create(b, loc, alpha, splat(1.0));
    Value logx = math::LogOp::create(b, loc, observed);
    Value am1LogX = arith::MulFOp::create(b, loc, am1, logx);
    Value betaX = arith::MulFOp::create(b, loc, beta, observed);
    Value kernel = arith::SubFOp::create(b, loc, am1LogX, betaX);
    Value lgA = impulse::LgammaOp::create(b, loc, alpha.getType(), alpha);
    Value aLogB =
        arith::MulFOp::create(b, loc, alpha, math::LogOp::create(b, loc, beta));
    Value normTerm = arith::SubFOp::create(b, loc, aLogB, lgA);
    return {reduce(kernel), reduce(normTerm)};
  }
  case DistBaseKind::STUDENT_T: {
    // log p = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi) - log(scale)
    //         - (df+1)/2 * log1p(((x-loc)/scale)^2 / df)
    Value df = bcast(distArgs[0]);
    Value locp = bcast(distArgs[1]);
    Value scale = bcast(distArgs[2]);
    Value half = splat(0.5);
    Value y = arith::DivFOp::create(
        b, loc, arith::SubFOp::create(b, loc, observed, locp), scale);
    Value y2df =
        arith::DivFOp::create(b, loc, arith::MulFOp::create(b, loc, y, y), df);
    Value l1p = math::Log1pOp::create(b, loc, y2df);
    Value dfp1 = arith::AddFOp::create(b, loc, df, splat(1.0));
    Value dfp1half = arith::MulFOp::create(b, loc, dfp1, half);
    Value kernel = arith::NegFOp::create(
        b, loc, arith::MulFOp::create(b, loc, dfp1half, l1p));
    Value lgTop =
        impulse::LgammaOp::create(b, loc, dfp1half.getType(), dfp1half);
    Value dfhalf = arith::MulFOp::create(b, loc, df, half);
    Value lgBot = impulse::LgammaOp::create(b, loc, dfhalf.getType(), dfhalf);
    Value dfpi = arith::MulFOp::create(b, loc, df, splat(M_PI));
    Value halfLogdfpi =
        arith::MulFOp::create(b, loc, half, math::LogOp::create(b, loc, dfpi));
    Value logscale = math::LogOp::create(b, loc, scale);
    Value normTerm = arith::SubFOp::create(
        b, loc,
        arith::SubFOp::create(
            b, loc, arith::SubFOp::create(b, loc, lgTop, lgBot), halfLogdfpi),
        logscale);
    return {reduce(kernel), reduce(normTerm)};
  }
  case DistBaseKind::POISSON: {
    Value rate = bcast(distArgs[0]);
    Value logRate = math::LogOp::create(b, loc, rate);
    Value kp1 = arith::AddFOp::create(b, loc, observed, splat(1.0));
    Value lgk = impulse::LgammaOp::create(b, loc, kp1.getType(), kp1);
    Value kernel = arith::SubFOp::create(
        b, loc, arith::MulFOp::create(b, loc, observed, logRate), lgk);
    Value lognorm = reduce(arith::NegFOp::create(b, loc, rate));
    return {reduce(kernel), lognorm};
  }
  case DistBaseKind::HALF_NORMAL: {
    Value sigma = bcast(distArgs[0]);
    Value sig2 = arith::MulFOp::create(b, loc, sigma, sigma);
    Value denom = arith::MulFOp::create(b, loc, splat(2.0), sig2);
    Value x2 = arith::MulFOp::create(b, loc, observed, observed);
    Value kernel =
        arith::NegFOp::create(b, loc, arith::DivFOp::create(b, loc, x2, denom));
    Value logSig = math::LogOp::create(b, loc, sigma);
    Value norm = arith::SubFOp::create(
        b, loc, arith::SubFOp::create(b, loc, splat(std::log(2.0)), logSig),
        splat(0.5 * std::log(2.0 * M_PI)));
    return {reduce(kernel), reduce(norm)};
  }
  case DistBaseKind::HALF_CAUCHY: {
    Value scale = bcast(distArgs[0]);
    Value xOverS = arith::DivFOp::create(b, loc, observed, scale);
    Value sq = arith::MulFOp::create(b, loc, xOverS, xOverS);
    Value kernel =
        arith::NegFOp::create(b, loc, math::Log1pOp::create(b, loc, sq));
    Value logS = math::LogOp::create(b, loc, scale);
    Value norm =
        arith::SubFOp::create(b, loc, splat(std::log(2.0 / M_PI)), logS);
    return {reduce(kernel), reduce(norm)};
  }
  case DistBaseKind::DIRICHLET: {
    // The event is the last axis; only lgamma(sum_K alpha) is summed per block.
    int64_t K =
        sampleType.getShape().empty() ? 1 : sampleType.getShape().back();
    int64_t total = 1;
    for (int64_t dsz : sampleType.getShape())
      total *= dsz;
    int64_t Mb = (K == 0) ? 0 : total / K;
    auto twoDType = RankedTensorType::get({Mb, K}, elemType);
    Value alpha = bcast(distArgs[0]);
    Value alpha2d = impulse::ReshapeOp::create(b, loc, twoDType, alpha);
    Value x2d = impulse::ReshapeOp::create(b, loc, twoDType, observed);
    Value one2d = arith::ConstantOp::create(
        b, loc, twoDType,
        DenseElementsAttr::get(twoDType, b.getFloatAttr(elemType, 1.0)));
    Value am1 = arith::SubFOp::create(b, loc, alpha2d, one2d);
    Value logx = math::LogOp::create(b, loc, x2d);
    Value kernel = arith::MulFOp::create(b, loc, logx, am1);
    Value lgAlpha =
        impulse::LgammaOp::create(b, loc, alpha2d.getType(), alpha2d);
    Value termA = reduce(lgAlpha);
    Value sumKAlpha = reduceLastAxis(b, loc, alpha2d, Mb, K);
    Value lgSumK =
        impulse::LgammaOp::create(b, loc, sumKAlpha.getType(), sumKAlpha);
    Value termB = reduce(lgSumK);
    Value lognorm = arith::SubFOp::create(b, loc, termB, termA);
    return {reduce(kernel), lognorm};
  }
  default:
    llvm::report_fatal_error("emitNativeLogpdfSplit: unsupported base");
  }
}

} // namespace

SmallVector<Value> impulse::emitDistSampler(OpBuilder &b, Location loc,
                                            DistAttr dist, Value rng,
                                            ValueRange distArgs,
                                            RankedTensorType sampleType) {
  Type elemType = sampleType.getElementType();
  MLIRContext *ctx = b.getContext();

  if (dist.getBase() != DistBaseKind::STD_NORMAL) {
    Value val =
        emitNativeSampler(b, loc, dist.getBase(), rng, distArgs, sampleType);
    return {rng, val};
  }

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

  if (dist.getBase() != DistBaseKind::STD_NORMAL) {
    LogpdfSplit split =
        emitNativeLogpdfSplit(b, loc, dist.getBase(), observed, distArgs);
    return arith::AddFOp::create(b, loc, split.unnormed, split.lognorm);
  }

  // StdNormal + transform chain: base_logpdf(inverse(y)) - Sum logabsdetjac.
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

Value impulse::emitLogit(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  auto oneConst = arith::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 1.0)));
  auto oneMinusX = arith::SubFOp::create(builder, loc, oneConst, x);
  auto logX = math::LogOp::create(builder, loc, x);
  auto logOneMinusX = math::LogOp::create(builder, loc, oneMinusX);
  return arith::SubFOp::create(builder, loc, logX, logOneMinusX);
}

Value impulse::emitLogSigmoid(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  // -log_add_exp(-x, 0), stable for large |x|.
  auto negX = arith::NegFOp::create(builder, loc, x);
  auto zeroConst = arith::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 0.0)));
  auto softplusNegX =
      impulse::LogAddExpOp::create(builder, loc, xType, negX, zeroConst);
  return arith::NegFOp::create(builder, loc, softplusNegX);
}

namespace {

// m contiguous simplex blocks of K constrained (n = K - 1 free) entries.
struct SimplexDims {
  int64_t K;
  int64_t n;
  int64_t m;
};

Value splatTensorConst(OpBuilder &builder, Location loc, RankedTensorType ty,
                       double v) {
  return arith::ConstantOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getFloatAttr(ty.getElementType(), v)));
}

Value matConst(OpBuilder &builder, Location loc, RankedTensorType ty,
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

Value triOnes(OpBuilder &builder, Location loc, int64_t n, bool strict,
              Type elemType) {
  auto ty = RankedTensorType::get({n, n}, elemType);
  SmallVector<APFloat> vals;
  vals.reserve(n * n);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j) {
      bool one = strict ? (i < j) : (i <= j);
      vals.push_back(
          cast<FloatAttr>(builder.getFloatAttr(elemType, one ? 1.0 : 0.0))
              .getValue());
    }
  return arith::ConstantOp::create(builder, loc, ty,
                                   DenseElementsAttr::get(ty, vals));
}

Value prefixSum(OpBuilder &builder, Location loc, Value mat, Value tri) {
  return impulse::DotOp::create(
      builder, loc, mat.getType(), mat, tri, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({1}),
      builder.getDenseI64ArrayAttr({0}));
}

Value shiftMat(OpBuilder &builder, Location loc, const SimplexDims &d,
               Type elemType) {
  SmallVector<double> rowVals(d.n);
  for (int64_t c = 0; c < d.n; ++c)
    rowVals[c] = std::log((double)(d.n - c));
  return matConst(builder, loc, RankedTensorType::get({d.m, d.n}, elemType),
                  rowVals);
}

Value constrainSimplex(OpBuilder &builder, Location loc, Value unconstrained,
                       const SimplexDims &d) {
  auto elemType =
      cast<RankedTensorType>(unconstrained.getType()).getElementType();
  auto xMatType = RankedTensorType::get({d.m, d.n}, elemType);
  auto yBufType = RankedTensorType::get({d.m, d.K}, elemType);
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
  Value P = math::ExpOp::create(builder, loc, logPexcl);
  Value ybody = arith::MulFOp::create(builder, loc, z, P);

  auto onesCol = splatTensorConst(
      builder, loc, RankedTensorType::get({d.n, 1}, elemType), 1.0);
  Value total = impulse::DotOp::create(
      builder, loc, colType, log1mZ, onesCol, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({1}),
      builder.getDenseI64ArrayAttr({0}));
  Value remainder = math::ExpOp::create(builder, loc, total);

  auto i64Scalar = RankedTensorType::get({}, builder.getI64Type());
  auto idx = [&](int64_t v) {
    return arith::ConstantOp::create(
        builder, loc, i64Scalar,
        DenseElementsAttr::get(i64Scalar, builder.getI64IntegerAttr(v)));
  };
  Value ybuf = splatTensorConst(builder, loc, yBufType, 0.0);
  ybuf = impulse::DynamicUpdateSliceOp::create(
      builder, loc, yBufType, ybuf, ybody, ValueRange{idx(0), idx(0)});
  ybuf = impulse::DynamicUpdateSliceOp::create(
      builder, loc, yBufType, ybuf, remainder,
      ValueRange{idx(0), idx(d.K - 1)});

  return impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.K}, elemType), ybuf);
}

Value unconstrainSimplex(OpBuilder &builder, Location loc, Value constrained,
                         const SimplexDims &d) {
  auto elemType =
      cast<RankedTensorType>(constrained.getType()).getElementType();
  auto yMatType = RankedTensorType::get({d.m, d.K}, elemType);
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
      builder, loc, splatTensorConst(builder, loc, xMatType, 1.0), cumIncl);
  Value ratio = arith::DivFOp::create(builder, loc, ycrop, z1m);
  Value logRatio = math::LogOp::create(builder, loc, ratio);
  Value x = arith::AddFOp::create(builder, loc, logRatio,
                                  shiftMat(builder, loc, d, elemType));

  return impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.n}, elemType), x);
}

Value logDetSimplex(OpBuilder &builder, Location loc, Value unconstrained,
                    const SimplexDims &d) {
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

  auto flatType = RankedTensorType::get({d.m * d.n}, elemType);
  Value flat = impulse::ReshapeOp::create(builder, loc, flatType, term);
  Value ones = splatTensorConst(builder, loc, flatType, 1.0);
  auto scalarType = RankedTensorType::get({}, elemType);
  return impulse::DotOp::create(
      builder, loc, scalarType, flat, ones, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({0}));
}

SimplexDims simplexDimsFromConstrained(SupportAttr support,
                                       int64_t constrainedSize) {
  int64_t K = support.getEventSize().getInt();
  return {K, K - 1, constrainedSize / K};
}

SimplexDims simplexDimsFromUnconstrained(SupportAttr support,
                                         int64_t unconstrainedSize) {
  int64_t K = support.getEventSize().getInt();
  return {K, K - 1, unconstrainedSize / (K - 1)};
}

} // namespace

int64_t impulse::bijectorForwardSize(int64_t unconstrainedSize,
                                     SupportAttr support) {
  if (!support)
    return unconstrainedSize;
  if (support.getKind() == SupportKind::SIMPLEX) {
    int64_t K = support.getEventSize().getInt();
    int64_t m = unconstrainedSize / (K - 1);
    return m * K;
  }
  return unconstrainedSize;
}

int64_t impulse::bijectorInverseSize(int64_t constrainedSize,
                                     SupportAttr support) {
  if (!support)
    return constrainedSize;
  if (support.getKind() == SupportKind::SIMPLEX) {
    int64_t K = support.getEventSize().getInt();
    int64_t m = constrainedSize / K;
    return m * (K - 1);
  }
  return constrainedSize;
}

bool impulse::isConstantJacobian(XformKind kind) {
  return kind == XformKind::IDENTITY || kind == XformKind::AFFINE;
}

Value impulse::emitBijectorForward(OpBuilder &builder, Location loc,
                                   XformAttr xform, SupportAttr support,
                                   Value unconstrained) {
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();

  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return unconstrained;
  case XformKind::EXP:
    return math::ExpOp::create(builder, loc, unconstrained);
  case XformKind::SOFTPLUS:
    return impulse::emitSoftplus(builder, loc, unconstrained);
  case XformKind::SIGMOID:
    return impulse::LogisticOp::create(builder, loc, unconstrained.getType(),
                                       unconstrained);
  case XformKind::INTERVAL: {
    double lower = support.getLowerBound().getValueAsDouble();
    double upper = support.getUpperBound().getValueAsDouble();
    auto lowerConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, lower)));
    auto scaleConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType,
                               builder.getFloatAttr(elemType, upper - lower)));
    auto sigmoid = impulse::LogisticOp::create(
        builder, loc, unconstrained.getType(), unconstrained);
    auto scaled = arith::MulFOp::create(builder, loc, scaleConst, sigmoid);
    return arith::AddFOp::create(builder, loc, lowerConst, scaled);
  }
  case XformKind::GREATER_THAN: {
    double lower = support.getLowerBound().getValueAsDouble();
    auto lowerConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, lower)));
    auto expZ = math::ExpOp::create(builder, loc, unconstrained);
    return arith::AddFOp::create(builder, loc, lowerConst, expZ);
  }
  case XformKind::LESS_THAN: {
    double upper = support.getUpperBound().getValueAsDouble();
    auto upperConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, upper)));
    auto expZ = math::ExpOp::create(builder, loc, unconstrained);
    return arith::SubFOp::create(builder, loc, upperConst, expZ);
  }
  case XformKind::STICK_BREAKING:
    return constrainSimplex(
        builder, loc, unconstrained,
        simplexDimsFromUnconstrained(support, zType.getNumElements()));
  case XformKind::COMPOSE: {
    Value v = unconstrained;
    if (ArrayAttr comps = xform.getComponents())
      for (Attribute a : comps)
        v = emitBijectorForward(builder, loc, cast<XformAttr>(a), support, v);
    return v;
  }
  case XformKind::AFFINE:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    // Not produced by bijectorFor.
    return unconstrained;
  }
  return unconstrained;
}

Value impulse::emitBijectorInverse(OpBuilder &builder, Location loc,
                                   XformAttr xform, SupportAttr support,
                                   Value constrained) {
  auto xType = cast<RankedTensorType>(constrained.getType());
  auto elemType = xType.getElementType();

  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return constrained;
  case XformKind::EXP:
    return math::LogOp::create(builder, loc, constrained);
  case XformKind::SOFTPLUS:
    return impulse::emitSoftplusInv(builder, loc, constrained);
  case XformKind::SIGMOID:
    return impulse::emitLogit(builder, loc, constrained);
  case XformKind::INTERVAL: {
    double lower = support.getLowerBound().getValueAsDouble();
    double upper = support.getUpperBound().getValueAsDouble();
    auto lowerConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, lower)));
    auto scaleConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType,
                               builder.getFloatAttr(elemType, upper - lower)));
    auto shifted = arith::SubFOp::create(builder, loc, constrained, lowerConst);
    auto normalized = arith::DivFOp::create(builder, loc, shifted, scaleConst);
    return impulse::emitLogit(builder, loc, normalized);
  }
  case XformKind::GREATER_THAN: {
    double lower = support.getLowerBound().getValueAsDouble();
    auto lowerConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, lower)));
    auto shifted = arith::SubFOp::create(builder, loc, constrained, lowerConst);
    return math::LogOp::create(builder, loc, shifted);
  }
  case XformKind::LESS_THAN: {
    double upper = support.getUpperBound().getValueAsDouble();
    auto upperConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, upper)));
    auto shifted = arith::SubFOp::create(builder, loc, upperConst, constrained);
    return math::LogOp::create(builder, loc, shifted);
  }
  case XformKind::STICK_BREAKING:
    return unconstrainSimplex(
        builder, loc, constrained,
        simplexDimsFromConstrained(support, xType.getNumElements()));
  case XformKind::COMPOSE: {
    Value v = constrained;
    if (ArrayAttr comps = xform.getComponents())
      for (int64_t i = comps.size() - 1; i >= 0; --i)
        v = emitBijectorInverse(builder, loc, cast<XformAttr>(comps[i]),
                                support, v);
    return v;
  }
  case XformKind::AFFINE:
  case XformKind::SCALED_UNIT_LOWER_CHOLESKY:
    return constrained;
  }
  return constrained;
}

Value impulse::emitBijectorLogAbsDetJac(OpBuilder &builder, Location loc,
                                        XformAttr xform, SupportAttr support,
                                        Value unconstrained) {
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  auto sumToScalar = [&](Value v) -> Value {
    auto vt = cast<RankedTensorType>(v.getType());
    auto ones = arith::ConstantOp::create(
        builder, loc, vt,
        DenseElementsAttr::get(vt, builder.getFloatAttr(elemType, 1.0)));
    return impulse::DotOp::create(
        builder, loc, scalarType, v, ones, builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0}),
        builder.getDenseI64ArrayAttr({0}));
  };

  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 0.0)));
  case XformKind::EXP:
  case XformKind::GREATER_THAN:
  case XformKind::LESS_THAN:
    return sumToScalar(unconstrained);
  case XformKind::SOFTPLUS: {
    Value logSigZ = impulse::emitLogSigmoid(builder, loc, unconstrained);
    return sumToScalar(logSigZ);
  }
  case XformKind::SIGMOID: {
    auto logSigZ = impulse::emitLogSigmoid(builder, loc, unconstrained);
    auto negZ = arith::NegFOp::create(builder, loc, unconstrained);
    auto logSigNegZ = impulse::emitLogSigmoid(builder, loc, negZ);
    auto logProduct = arith::AddFOp::create(builder, loc, logSigZ, logSigNegZ);
    return sumToScalar(logProduct);
  }
  case XformKind::INTERVAL: {
    double scale = support.getUpperBound().getValueAsDouble() -
                   support.getLowerBound().getValueAsDouble();
    auto logSigZ = impulse::emitLogSigmoid(builder, loc, unconstrained);
    auto negZ = arith::NegFOp::create(builder, loc, unconstrained);
    auto logSigNegZ = impulse::emitLogSigmoid(builder, loc, negZ);
    auto logProduct = arith::AddFOp::create(builder, loc, logSigZ, logSigNegZ);
    auto sumLogProduct = sumToScalar(logProduct);

    int64_t n = zType.getNumElements();
    double logScaleTerm = n * std::log(scale);
    auto logScaleConst = arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, logScaleTerm)));
    return arith::AddFOp::create(builder, loc, sumLogProduct, logScaleConst);
  }
  case XformKind::STICK_BREAKING:
    return logDetSimplex(
        builder, loc, unconstrained,
        simplexDimsFromUnconstrained(support, zType.getNumElements()));
  case XformKind::COMPOSE: {
    Value total = arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 0.0)));
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
    return arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 0.0)));
  }
  return arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
}

Value impulse::emitSoftplus(OpBuilder &builder, Location loc, Value x) {
  Value expX = math::ExpOp::create(builder, loc, x);
  return math::Log1pOp::create(builder, loc, expX);
}

Value impulse::emitSoftplusInv(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();
  Value expX = math::ExpOp::create(builder, loc, x);
  Value one = arith::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 1.0)));
  Value expXMinus1 = arith::SubFOp::create(builder, loc, expX, one);
  return math::LogOp::create(builder, loc, expXMinus1);
}

impulse::ScaledUnitLowerCholeskyResult
impulse::emitScaledUnitLowerCholesky(OpBuilder &builder, Location loc,
                                     Value xFlat, int64_t d, Type elemType) {
  int64_t nOff = d * (d - 1) / 2;
  auto offDiagTy = RankedTensorType::get({nOff}, elemType);
  auto diagTy = RankedTensorType::get({d}, elemType);
  auto matTy = RankedTensorType::get({d, d}, elemType);
  auto scalarTy = RankedTensorType::get({}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

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
  Value onesD = arith::ConstantOp::create(
      builder, loc, diagTy,
      DenseElementsAttr::get(diagTy, builder.getFloatAttr(elemType, 1.0)));
  Value sumLogDiag = impulse::DotOp::create(
      builder, loc, scalarTy, logDiag, onesD, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({0}));

  Value L = arith::ConstantOp::create(
      builder, loc, matTy,
      DenseElementsAttr::get(matTy, builder.getFloatAttr(elemType, 0.0)));

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
  auto i64Const = [&](int64_t v) -> Value {
    return arith::ConstantOp::create(
        builder, loc, i64T,
        DenseElementsAttr::get(i64T, builder.getI64IntegerAttr(v)));
  };

  int64_t k = 0;
  for (int64_t i = 0; i < d; ++i) {
    Value diag_i = sliceScalar(diag, i);
    for (int64_t j = 0; j < i; ++j) {
      Value off_k = sliceScalar(offDiag, k);
      Value entry = arith::MulFOp::create(builder, loc, off_k, diag_i);
      Value entry11 = reshapeTo11(entry);
      L = impulse::DynamicUpdateSliceOp::create(
          builder, loc, matTy, L, entry11,
          ValueRange{i64Const(i), i64Const(j)});
      ++k;
    }
    Value diag_i_11 = reshapeTo11(diag_i);
    L = impulse::DynamicUpdateSliceOp::create(
        builder, loc, matTy, L, diag_i_11,
        ValueRange{i64Const(i), i64Const(i)});
  }

  return {L, sumLogDiag};
}
