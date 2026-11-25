//===- VIUtils.cpp - Variational inference lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The key splits, draws and update formulas fix the bits of the result; do not
// reorder them.
//
//===----------------------------------------------------------------------===//

#include "VIUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Impulse/InferStateLayout.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Interfaces/HMCUtils.h"
#include "Interfaces/ImpulseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <cmath>
#include <optional>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;
using namespace mlir::impulse::VI;

namespace {
/// The model and configuration of one VI inference op.
struct VIContext : public ModelContext {
  Value learningRate;
  /// Number of optimizer steps.
  int64_t numSteps = 0;
  GuideKind guideKind;
  OptimizerKind optimizerKind;
  int64_t numParticles = 1;
  double initScale = 0.1;
  double adamB1 = 0.9;
  double adamB2 = 0.999;
  double adamEps = 1.0e-8;
  /// Per-element gradient clip of CLIPPED_ADAM.
  double clipNorm = 10.0;
  /// Half-width of the uniform initial location used without `init_position`.
  double initRadius = 2.0;
  double momentumMass = 0.9;
  double rmspropGamma = 0.9;
  double rmspropEps = 1.0e-8;
  /// Skip steps whose ELBO or gradient is not finite; the key still advances.
  bool stableUpdate = false;
};
} // namespace

/// Sums all elements of `tensor` into a rank-0 tensor with a dot against ones.
static Value sumAllElements(OpBuilder &builder, Location loc, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto scalarType = RankedTensorType::get({}, tensorType.getElementType());
  Value ones = emitSplatConstant(builder, loc, tensorType, 1.0);
  DenseI64ArrayAttr allDims = builder.getDenseI64ArrayAttr(
      llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank())));
  return impulse::DotOp::create(builder, loc, scalarType, tensor, ones, allDims,
                                allDims);
}

/// Whether `elbo` and every gradient in `grads` are finite. A gradient is
/// tested through its sum, which is not finite when any element is not.
static Value computeIsFiniteForStableUpdate(OpBuilder &builder, Location loc,
                                            Value elbo, ValueRange grads) {
  Value isFinite = math::IsFiniteOp::create(builder, loc, elbo);
  for (Value grad : grads) {
    Value gradSum = sumAllElements(builder, loc, grad);
    Value gradFinite = math::IsFiniteOp::create(builder, loc, gradSum);
    isFinite = arith::AndIOp::create(builder, loc, isFinite, gradFinite);
  }
  return isFinite;
}

/// Log-density of `loc + L @ eps` under N(loc, L L^T), where `sumLogScale` is
/// the sum of the logs of L's diagonal and `eps` the standard normal draw.
static Value emitGaussianLogQ(OpBuilder &builder, Location loc,
                              Value sumLogScale, Value eps,
                              const VIContext &ctx) {
  auto scalarType = ctx.getScalarType();

  Value epsSq = arith::MulFOp::create(builder, loc, eps, eps);
  Value sumEpsSq = sumAllElements(builder, loc, epsSq);
  Value halfSumEpsSq = arith::MulFOp::create(
      builder, loc, sumEpsSq, emitSplatConstant(builder, loc, scalarType, 0.5));
  Value log2piTerm = emitSplatConstant(builder, loc, scalarType,
                                       static_cast<double>(ctx.positionSize) *
                                           (0.5 * std::log(2.0 * M_PI)));

  Value negSumLogScale = arith::NegFOp::create(builder, loc, sumLogScale);
  Value logQPart =
      arith::SubFOp::create(builder, loc, negSumLogScale, halfSumEpsSq);
  return arith::SubFOp::create(builder, loc, logQPart, log2piTerm);
}

namespace {
/// The ELBO of one particle and the model's rng state after scoring it.
struct ParticleELBO {
  Value elbo;
  Value rng;
};

/// An ELBO estimate and its gradient with respect to each guide parameter.
struct ELBOGradient {
  Value elbo;
  SmallVector<Value> grads;
};

/// A guide parameter and its two optimizer state slots. An update rule that
/// keeps fewer slots passes the unused ones through.
struct ParamState {
  Value param;
  Value m;
  Value v;
};

/// The values an SVI loop leaves behind.
struct SVILoopResult {
  Value traceBuf;
  Value elboBuf;
  Value rng;
  SmallVector<ParamState> params;
  Value stepCount;
};
} // namespace

/// Emits an autodiff region that differentiates, with respect to `params`, the
/// ELBO that `emitELBO` builds inside the region from the region arguments
/// standing for `params`.
static ELBOGradient
emitELBOGradient(OpBuilder &builder, Location loc, ValueRange params,
                 Type rngType, const VIContext &ctx,
                 function_ref<ParticleELBO(ValueRange paramArgs)> emitELBO) {
  RankedTensorType scalarType = ctx.getScalarType();
  Value gradSeed = emitSplatConstant(builder, loc, scalarType, 1.0);
  SmallVector<Value> adInputs = llvm::to_vector(params);
  adInputs.push_back(gradSeed);
  SmallVector<Type> resultTypes{scalarType, rngType};
  llvm::append_range(resultTypes, params.getTypes());
  auto adOp = enzyme::AutoDiffRegionOp::create(
      builder, loc, resultTypes, adInputs,
      buildGradientRegionAttrs(builder, params.size(), ctx.autodiffAttrs));

  Block *adBlock = builder.createBlock(&adOp.getBody());
  for (Value param : params)
    adBlock->addArgument(param.getType(), loc);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(adBlock);
    ParticleELBO particle = emitELBO(adBlock->getArguments());
    enzyme::YieldOp::create(builder, loc,
                            ValueRange{particle.elbo, particle.rng});
  }
  builder.setInsertionPointAfter(adOp);

  return {adOp.getResult(0),
          llvm::to_vector_of<Value>(adOp.getResults().drop_front(2))};
}

/// One ELBO particle for a mean-field normal guide. `params` are the location
/// and the unconstrained scale, whose softplus is the standard deviation; the
/// sample is constrained and its log-Jacobian enters log q.
static ELBOGradient emitAutoNormalParticle(OpBuilder &builder, Location loc,
                                           ValueRange params, Value pkey,
                                           const VIContext &ctx) {
  Type rngType = pkey.getType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  Type elemType = ctx.elementType;

  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);
  Value guideSeed = seedSplit.getResult(1);

  // eps is drawn outside the autodiff region so that the primal and the shadow
  // see the same sample; each site draws from its own key.
  auto zeroScalar = emitSplatConstant(builder, loc, scalarType, 0.0);
  auto oneScalar = emitSplatConstant(builder, loc, scalarType, 1.0);
  Value c0i64 = emitI64ScalarConstant(builder, loc, 0);

  Value eps = emitSplatConstant(builder, loc, positionType, 0.0);
  Value chain = guideSeed;
  for (const auto &info : ctx.supports) {
    auto siteSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rngType, rngType}, chain);
    chain = siteSplit.getResult(0);
    Value siteKey = siteSplit.getResult(1);
    auto siteEpsType =
        RankedTensorType::get({1, info.constrainedSize}, elemType);
    auto randOp = impulse::RandomOp::create(
        builder, loc, TypeRange{rngType, siteEpsType}, siteKey, zeroScalar,
        oneScalar,
        impulse::RngDistributionAttr::get(builder.getContext(),
                                          impulse::RngDistribution::NORMAL));
    Value siteEps = randOp.getResult();
    Value offsetI64 =
        emitI64ScalarConstant(builder, loc, info.constrainedOffset);
    eps = impulse::DynamicUpdateSliceOp::create(
        builder, loc, positionType, eps, siteEps, ValueRange{c0i64, offsetI64});
  }

  eps = conditionalDump(builder, loc, eps, "VI: eps (combined per-site)",
                        ctx.debugDump);

  return emitELBOGradient(
      builder, loc, params, rngType, ctx,
      [&](ValueRange paramArgs) -> ParticleELBO {
        Value locArg = paramArgs[0];
        Value scaleArg = paramArgs[1];

        Value scale = impulse::emitSoftplus(builder, loc, scaleArg);

        Value scaleDumped = conditionalDump(
            builder, loc, scale, "VI: scale=softplus(scaleArg)", ctx.debugDump);
        Value locDumped =
            conditionalDump(builder, loc, locArg, "VI: locArg", ctx.debugDump);

        Value scaleEps = arith::MulFOp::create(builder, loc, scaleDumped, eps);
        Value zUnc = arith::AddFOp::create(builder, loc, locDumped, scaleEps);
        zUnc = conditionalDump(builder, loc, zUnc, "VI: zUnc=loc+scale*eps",
                               ctx.debugDump);

        Value zConstrained =
            constrainPosition(builder, loc, zUnc, ctx.supports);
        zConstrained = conditionalDump(builder, loc, zConstrained,
                                       "VI: zConstrained", ctx.debugDump);

        Value logScale = math::LogOp::create(builder, loc, scaleDumped);
        Value sumLogScale = sumAllElements(builder, loc, logScale);
        Value logQUnc = emitGaussianLogQ(builder, loc, sumLogScale, eps, ctx);
        logQUnc = conditionalDump(builder, loc, logQUnc, "VI: logQUnc",
                                  ctx.debugDump);
        Value jacCorrection =
            computeTotalJacobianCorrection(builder, loc, zUnc, ctx.supports);
        jacCorrection = conditionalDump(builder, loc, jacCorrection,
                                        "VI: jacCorrection", ctx.debugDump);
        Value logQ =
            arith::SubFOp::create(builder, loc, logQUnc, jacCorrection);
        logQ = conditionalDump(builder, loc, logQ,
                               "VI: logQ=logQUnc-jacCorrection", ctx.debugDump);

        Value modelRng = modelSeed;
        Value logP =
            emitModelLogDensity(builder, loc, zConstrained, modelRng, ctx);
        logP = conditionalDump(builder, loc, logP, "VI: logP", ctx.debugDump);

        Value elbo = arith::SubFOp::create(builder, loc, logP, logQ);
        elbo = conditionalDump(builder, loc, elbo, "VI: elbo=logP-logQ",
                               ctx.debugDump);
        return {elbo, modelRng};
      });
}

/// One ELBO particle for a point-mass guide at `params[0]`: the model
/// log-density at the constrained location, with no log q or Jacobian term.
/// `pkey` is still split into a model and an unused guide seed.
static ELBOGradient emitAutoDeltaParticle(OpBuilder &builder, Location loc,
                                          ValueRange params, Value pkey,
                                          const VIContext &ctx) {
  Type rngType = pkey.getType();

  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);

  return emitELBOGradient(builder, loc, params, rngType, ctx,
                          [&](ValueRange paramArgs) -> ParticleELBO {
                            Value zConstrained = constrainPosition(
                                builder, loc, paramArgs[0], ctx.supports);
                            Value modelRng = modelSeed;
                            Value logP = emitModelLogDensity(
                                builder, loc, zConstrained, modelRng, ctx);
                            return {logP, modelRng};
                          });
}

/// One ELBO particle for a full-rank normal guide. `params` are the location
/// and the flat unconstrained scale_tril, which emitScaledUnitLowerCholesky
/// maps to the lower Cholesky factor.
static ELBOGradient emitAutoMVNParticle(OpBuilder &builder, Location loc,
                                        ValueRange params, Value pkey,
                                        const VIContext &ctx) {
  Type rngType = pkey.getType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  Type elemType = ctx.elementType;
  int64_t d = ctx.positionSize;

  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);
  Value guideSeed = seedSplit.getResult(1);

  // The guide draws all latents as one site, so it takes a single site key.
  auto siteSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, guideSeed);
  Value siteKey = siteSplit.getResult(1);

  auto zeroScalar = emitSplatConstant(builder, loc, scalarType, 0.0);
  auto oneScalar = emitSplatConstant(builder, loc, scalarType, 1.0);
  auto randOp = impulse::RandomOp::create(
      builder, loc, TypeRange{rngType, positionType}, siteKey, zeroScalar,
      oneScalar,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::NORMAL));
  Value eps = randOp.getResult();

  return emitELBOGradient(
      builder, loc, params, rngType, ctx,
      [&](ValueRange paramArgs) -> ParticleELBO {
        impulse::ScaledUnitLowerCholeskyResult chol =
            impulse::emitScaledUnitLowerCholesky(builder, loc, paramArgs[1], d,
                                                 elemType);

        // L @ eps in row-vector form: eps @ L^T.
        Value scaledEps = impulse::DotOp::create(
            builder, loc, positionType, eps, chol.scaleTril,
            builder.getDenseI64ArrayAttr({1}),
            builder.getDenseI64ArrayAttr({1}));
        Value z = arith::AddFOp::create(builder, loc, paramArgs[0], scaledEps);

        Value logQ = emitGaussianLogQ(builder, loc, chol.sumLogDiag, eps, ctx);

        Value modelRng = modelSeed;
        Value logP = emitModelLogDensity(builder, loc, z, modelRng, ctx);

        Value elbo = arith::SubFOp::create(builder, loc, logP, logQ);
        return {elbo, modelRng};
      });
}

/// Signature shared by the per-guide particle emitters: one ELBO estimate and
/// its gradient for the guide parameters `params` and particle key `pkey`.
using ParticleEmitter = function_ref<ELBOGradient(
    OpBuilder &builder, Location loc, ValueRange params, Value pkey,
    const VIContext &ctx)>;

/// Averages the ELBO and gradients of `ctx.numParticles` particles. A single
/// particle uses `rngStep` as its key; more particles split it into one key
/// each.
static ELBOGradient averageParticles(OpBuilder &builder, Location loc,
                                     ValueRange params, Value rngStep,
                                     const VIContext &ctx,
                                     ParticleEmitter emitParticle) {
  int64_t numParticles = ctx.numParticles;
  if (numParticles == 1)
    return emitParticle(builder, loc, params, rngStep, ctx);

  SmallVector<Type> keyTypes(numParticles, rngStep.getType());
  auto keys = impulse::RandomSplitOp::create(builder, loc, keyTypes, rngStep);

  RankedTensorType scalarType = ctx.getScalarType();
  Value elboSum = emitSplatConstant(builder, loc, scalarType, 0.0);
  SmallVector<Value> gradSums;
  for (Value param : params)
    gradSums.push_back(emitSplatConstant(
        builder, loc, cast<RankedTensorType>(param.getType()), 0.0));

  for (Value pkey : keys.getResults()) {
    ELBOGradient particle = emitParticle(builder, loc, params, pkey, ctx);
    elboSum = arith::AddFOp::create(builder, loc, elboSum, particle.elbo);
    for (auto [sum, grad] : llvm::zip_equal(gradSums, particle.grads))
      sum = arith::AddFOp::create(builder, loc, sum, grad);
  }

  Value count = emitSplatConstant(builder, loc, scalarType,
                                  static_cast<double>(numParticles));
  DenseMap<Type, Value> countSplats;
  for (Value param : params) {
    auto type = cast<RankedTensorType>(param.getType());
    if (!countSplats.contains(type))
      countSplats[type] = emitBroadcastScalar(builder, loc, count, type);
  }
  ELBOGradient mean;
  mean.elbo = arith::DivFOp::create(builder, loc, elboSum, count);
  for (auto [param, sum] : llvm::zip_equal(params, gradSums))
    mean.grads.push_back(arith::DivFOp::create(
        builder, loc, sum, countSplats.lookup(param.getType())));
  return mean;
}

/// SGD step on the loss -elbo, where `gradElbo` is d(elbo)/d(param). Both
/// slots pass through.
static ParamState applySGDUpdate(OpBuilder &builder, Location loc, Value param,
                                 Value gradElbo, Value m, Value v, Value lr) {
  auto ty = cast<RankedTensorType>(param.getType());
  Value lrBcast = emitBroadcastScalar(builder, loc, lr, ty);
  Value scaledGrad = arith::MulFOp::create(builder, loc, lrBcast, gradElbo);
  Value paramNew = arith::AddFOp::create(builder, loc, param, scaledGrad);
  return {paramNew, m, v};
}

/// Bias-corrected Adam step; `m` and `v` are the moment estimates and `stepIdx`
/// counts from 0. A positive `clipNorm` first clamps each gradient element to
/// [-clipNorm, clipNorm].
static ParamState applyAdamUpdate(OpBuilder &builder, Location loc, Value param,
                                  Value gradElbo, Value m, Value v,
                                  Value stepIdx, Value lr, double b1, double b2,
                                  double eps, double clipNorm) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  if (clipNorm > 0.0) {
    auto negClip = emitSplatConstant(builder, loc, scalarType, -clipNorm);
    auto posClip = emitSplatConstant(builder, loc, scalarType, clipNorm);
    Value negClipBcast = emitBroadcastScalar(builder, loc, negClip, paramType);
    Value posClipBcast = emitBroadcastScalar(builder, loc, posClip, paramType);
    g = arith::MaximumFOp::create(builder, loc, g, negClipBcast);
    g = arith::MinimumFOp::create(builder, loc, g, posClipBcast);
  }

  auto b1Const = emitSplatConstant(builder, loc, scalarType, b1);
  auto oneMinusB1 = emitSplatConstant(builder, loc, scalarType, 1.0 - b1);
  auto b2Const = emitSplatConstant(builder, loc, scalarType, b2);
  auto oneMinusB2 = emitSplatConstant(builder, loc, scalarType, 1.0 - b2);
  auto epsConst = emitSplatConstant(builder, loc, scalarType, eps);
  auto oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);

  Value b1Bcast = emitBroadcastScalar(builder, loc, b1Const, paramType);
  Value oneMinusB1Bcast =
      emitBroadcastScalar(builder, loc, oneMinusB1, paramType);
  Value b2Bcast = emitBroadcastScalar(builder, loc, b2Const, paramType);
  Value oneMinusB2Bcast =
      emitBroadcastScalar(builder, loc, oneMinusB2, paramType);

  Value oneMinusB1G = arith::MulFOp::create(builder, loc, oneMinusB1Bcast, g);
  Value b1M = arith::MulFOp::create(builder, loc, b1Bcast, m);
  Value mNew = arith::AddFOp::create(builder, loc, oneMinusB1G, b1M);

  Value gSq = arith::MulFOp::create(builder, loc, g, g);
  Value oneMinusB2GSq =
      arith::MulFOp::create(builder, loc, oneMinusB2Bcast, gSq);
  Value b2V = arith::MulFOp::create(builder, loc, b2Bcast, v);
  Value vNew = arith::AddFOp::create(builder, loc, oneMinusB2GSq, b2V);

  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value stepPlus1 = arith::AddIOp::create(builder, loc, stepIdx, oneI64);
  Value stepFloat =
      arith::SIToFPOp::create(builder, loc, scalarType, stepPlus1);

  Value b1PowS = math::PowFOp::create(builder, loc, b1Const, stepFloat);
  Value b2PowS = math::PowFOp::create(builder, loc, b2Const, stepFloat);
  Value oneMinusB1Pow = arith::SubFOp::create(builder, loc, oneConst, b1PowS);
  Value oneMinusB2Pow = arith::SubFOp::create(builder, loc, oneConst, b2PowS);
  Value oneMinusB1PowBcast =
      emitBroadcastScalar(builder, loc, oneMinusB1Pow, paramType);
  Value oneMinusB2PowBcast =
      emitBroadcastScalar(builder, loc, oneMinusB2Pow, paramType);

  Value mhat = arith::DivFOp::create(builder, loc, mNew, oneMinusB1PowBcast);
  Value vhat = arith::DivFOp::create(builder, loc, vNew, oneMinusB2PowBcast);

  Value sqrtVhat = math::SqrtOp::create(builder, loc, vhat);
  Value epsBcast = emitBroadcastScalar(builder, loc, epsConst, paramType);
  Value denom = arith::AddFOp::create(builder, loc, sqrtVhat, epsBcast);
  Value mhatOverDenom = arith::DivFOp::create(builder, loc, mhat, denom);
  Value lrBcast = emitBroadcastScalar(builder, loc, lr, paramType);
  Value update = arith::MulFOp::create(builder, loc, lrBcast, mhatOverDenom);
  Value paramNew = arith::SubFOp::create(builder, loc, param, update);

  return {paramNew, mNew, vNew};
}

/// Heavy-ball momentum step; `m` is the velocity and `v` passes through.
static ParamState applyMomentumUpdate(OpBuilder &builder, Location loc,
                                      Value param, Value gradElbo, Value m,
                                      Value v, Value lr, double mass) {
  auto paramType = cast<RankedTensorType>(param.getType());

  Value g = arith::NegFOp::create(builder, loc, gradElbo);
  Value massBcast = emitBroadcastConstant(builder, loc, paramType, mass);
  Value mScaled = arith::MulFOp::create(builder, loc, massBcast, m);
  Value mNew = arith::AddFOp::create(builder, loc, mScaled, g);
  Value lrBcast = emitBroadcastScalar(builder, loc, lr, paramType);
  Value scaledM = arith::MulFOp::create(builder, loc, lrBcast, mNew);
  Value paramNew = arith::SubFOp::create(builder, loc, param, scaledM);
  return {paramNew, mNew, v};
}

/// RMSProp step; `m` is the running squared-gradient average. With
/// `momentumMass`, `v` is the momentum, else it passes through.
static ParamState applyRMSPropUpdate(OpBuilder &builder, Location loc,
                                     Value param, Value gradElbo, Value m,
                                     Value v, Value lr, double gamma,
                                     double eps,
                                     std::optional<double> momentumMass) {
  auto paramType = cast<RankedTensorType>(param.getType());

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  Value gammaBcast = emitBroadcastConstant(builder, loc, paramType, gamma);
  Value oneMinusGammaBcast =
      emitBroadcastConstant(builder, loc, paramType, 1.0 - gamma);
  Value epsBcast = emitBroadcastConstant(builder, loc, paramType, eps);
  Value momBcast;
  if (momentumMass)
    momBcast = emitBroadcastConstant(builder, loc, paramType, *momentumMass);

  Value gSq = arith::MulFOp::create(builder, loc, g, g);
  Value oneMinusGammaGSq =
      arith::MulFOp::create(builder, loc, oneMinusGammaBcast, gSq);
  Value gammaAvg = arith::MulFOp::create(builder, loc, gammaBcast, m);
  Value avgNew =
      arith::AddFOp::create(builder, loc, gammaAvg, oneMinusGammaGSq);

  Value avgPlusEps = arith::AddFOp::create(builder, loc, avgNew, epsBcast);
  Value sqrtAvgPlusEps = math::SqrtOp::create(builder, loc, avgPlusEps);
  Value gOverSqrt = arith::DivFOp::create(builder, loc, g, sqrtAvgPlusEps);
  Value lrBcast = emitBroadcastScalar(builder, loc, lr, paramType);
  Value update = arith::MulFOp::create(builder, loc, lrBcast, gOverSqrt);
  if (!momentumMass) {
    Value paramNew = arith::SubFOp::create(builder, loc, param, update);
    return {paramNew, avgNew, v};
  }
  Value momTimesMom = arith::MulFOp::create(builder, loc, momBcast, v);
  Value momNew = arith::AddFOp::create(builder, loc, momTimesMom, update);
  Value paramNew = arith::SubFOp::create(builder, loc, param, momNew);
  return {paramNew, avgNew, momNew};
}

/// Adagrad step with momentum; `m` is the squared-gradient sum and `v` the
/// momentum.
static ParamState applyAdagradUpdate(OpBuilder &builder, Location loc,
                                     Value param, Value gradElbo, Value m,
                                     Value v, Value lr, double momentum) {
  auto paramType = cast<RankedTensorType>(param.getType());

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  Value gSquared = arith::MulFOp::create(builder, loc, g, g);
  Value gSqNew = arith::AddFOp::create(builder, loc, m, gSquared);

  // g is 0 wherever g_sq is 0; select 0 there, as g * 1/sqrt(0) is NaN.
  Value sqrtGSq = math::SqrtOp::create(builder, loc, gSqNew);
  Value zeroBcast = emitSplatConstant(builder, loc, paramType, 0.0);
  Value isPositive = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OGT, gSqNew, zeroBcast);
  Value oneBcast = emitSplatConstant(builder, loc, paramType, 1.0);
  Value invSqrt = arith::DivFOp::create(builder, loc, oneBcast, sqrtGSq);
  Value gSqInvSqrt =
      arith::SelectOp::create(builder, loc, isPositive, invSqrt, zeroBcast);

  Value lrBcast = emitBroadcastScalar(builder, loc, lr, paramType);
  Value gTimesInv = arith::MulFOp::create(builder, loc, g, gSqInvSqrt);
  Value lrG = arith::MulFOp::create(builder, loc, lrBcast, gTimesInv);
  Value dx = arith::NegFOp::create(builder, loc, lrG);

  Value momBcast = emitBroadcastConstant(builder, loc, paramType, momentum);
  Value oneMinusMomBcast =
      emitBroadcastConstant(builder, loc, paramType, 1.0 - momentum);
  Value oneMinusMomDx =
      arith::MulFOp::create(builder, loc, oneMinusMomBcast, dx);
  Value momTimesV = arith::MulFOp::create(builder, loc, momBcast, v);
  Value vNew = arith::AddFOp::create(builder, loc, oneMinusMomDx, momTimesV);

  Value paramNew = arith::AddFOp::create(builder, loc, param, vNew);
  return {paramNew, gSqNew, vNew};
}

/// Applies `ctx.optimizerKind` to `param` with gradient `gradElbo` and state
/// slots `m` and `v`; `step` is the zero-based global step index.
static ParamState applyOptimizer(OpBuilder &builder, Location loc,
                                 const VIContext &ctx, Value step, Value param,
                                 Value gradElbo, Value m, Value v) {
  switch (ctx.optimizerKind) {
  case OptimizerKind::ADAM:
    return applyAdamUpdate(builder, loc, param, gradElbo, m, v, step,
                           ctx.learningRate, ctx.adamB1, ctx.adamB2,
                           ctx.adamEps,
                           /*clipNorm=*/0.0);
  case OptimizerKind::CLIPPED_ADAM:
    return applyAdamUpdate(builder, loc, param, gradElbo, m, v, step,
                           ctx.learningRate, ctx.adamB1, ctx.adamB2,
                           ctx.adamEps, ctx.clipNorm);
  case OptimizerKind::MOMENTUM:
    return applyMomentumUpdate(builder, loc, param, gradElbo, m, v,
                               ctx.learningRate, ctx.momentumMass);
  case OptimizerKind::RMSPROP:
    return applyRMSPropUpdate(builder, loc, param, gradElbo, m, v,
                              ctx.learningRate, ctx.rmspropGamma,
                              ctx.rmspropEps,
                              /*momentumMass=*/std::nullopt);
  case OptimizerKind::ADAGRAD:
    return applyAdagradUpdate(builder, loc, param, gradElbo, m, v,
                              ctx.learningRate, ctx.momentumMass);
  case OptimizerKind::RMSPROP_MOMENTUM:
    return applyRMSPropUpdate(builder, loc, param, gradElbo, m, v,
                              ctx.learningRate, ctx.rmspropGamma,
                              ctx.rmspropEps, ctx.momentumMass);
  case OptimizerKind::SGD:
    return applySGDUpdate(builder, loc, param, gradElbo, m, v,
                          ctx.learningRate);
  }
  llvm_unreachable("unknown VI optimizer kind");
}

/// Emits `ctx.numSteps` SVI steps over the guide parameters `params0`. Null
/// optimizer slots start at zero and a null `initStepCount` at step 0. Each
/// step records its ELBO and the first parameter, constrained when
/// `recordConstrained` is set.
static SVILoopResult emitSVILoop(OpBuilder &builder, Location loc,
                                 ArrayRef<ParamState> params0, Value rng0,
                                 Value initStepCount, const VIContext &ctx,
                                 ParticleEmitter emitParticle,
                                 bool recordConstrained) {
  Type elemType = ctx.elementType;
  auto scalarType = ctx.getScalarType();
  auto traceBufType =
      RankedTensorType::get({ctx.numSteps, ctx.positionSize}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSteps}, elemType);
  auto i64Type = RankedTensorType::get({}, builder.getI64Type());
  size_t numParams = params0.size();

  Value c0i = emitI64ScalarConstant(builder, loc, 0);
  Value c1i = emitI64ScalarConstant(builder, loc, 1);
  Value cNi = emitI64ScalarConstant(builder, loc, ctx.numSteps);
  Value traceInit = emitSplatConstant(builder, loc, traceBufType, 0.0);
  Value elboInit = emitSplatConstant(builder, loc, elboBufType, 0.0);
  Value elboFinalSeed = emitSplatConstant(builder, loc, scalarType, 0.0);
  DenseMap<Type, Value> zeroSlots;
  for (const ParamState &p : params0) {
    auto type = cast<RankedTensorType>(p.param.getType());
    if (!zeroSlots.contains(type))
      zeroSlots[type] = emitSplatConstant(builder, loc, type, 0.0);
  }
  if (!initStepCount)
    initStepCount = emitI64ScalarConstant(builder, loc, 0);

  // Iteration arguments: params, rng, trace, ELBO trace, last ELBO, then the
  // (m, v) slots of each param.
  SmallVector<Value> iterArgs;
  for (const ParamState &p : params0)
    iterArgs.push_back(p.param);
  iterArgs.append({rng0, traceInit, elboInit, elboFinalSeed});
  for (const ParamState &p : params0) {
    Value zero = zeroSlots.lookup(p.param.getType());
    iterArgs.push_back(p.m ? p.m : zero);
    iterArgs.push_back(p.v ? p.v : zero);
  }
  SmallVector<Type> iterTypes(ValueRange(iterArgs).getTypes());

  auto forOp =
      impulse::ForOp::create(builder, loc, iterTypes, c0i, cNi, c1i, iterArgs);
  Block *body = builder.createBlock(&forOp.getRegion());
  body->addArgument(i64Type, loc);
  for (Type type : iterTypes)
    body->addArgument(type, loc);

  Value iter = body->getArgument(0);
  ValueRange iterValues = body->getArguments().drop_front();
  ValueRange params = iterValues.take_front(numParams);
  Value rngIter = iterValues[numParams];
  Value traceBufIter = iterValues[numParams + 1];
  Value elboBufIter = iterValues[numParams + 2];
  ValueRange slots = iterValues.drop_front(numParams + 4);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);

    auto updateSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rng0.getType(), rng0.getType()}, rngIter);
    Value rngNext = updateSplit.getResult(0);
    Value rngStep = updateSplit.getResult(1);

    ELBOGradient estimate =
        averageParticles(builder, loc, params, rngStep, ctx, emitParticle);

    Value step = arith::AddIOp::create(builder, loc, iter, initStepCount);
    SmallVector<Value> newParams, newSlots;
    for (size_t i = 0; i < numParams; ++i) {
      ParamState next =
          applyOptimizer(builder, loc, ctx, step, params[i], estimate.grads[i],
                         slots[2 * i], slots[2 * i + 1]);
      newParams.push_back(next.param);
      newSlots.append({next.m, next.v});
    }

    // A step that is not finite keeps the previous parameters and optimizer
    // state; its ELBO is still recorded and the key still advances.
    if (ctx.stableUpdate) {
      Value isFinite = computeIsFiniteForStableUpdate(
          builder, loc, estimate.elbo, estimate.grads);
      auto keepIfFinite = [&](Value next, Value prev) -> Value {
        return impulse::SelectOp::create(builder, loc, prev.getType(), isFinite,
                                         next, prev);
      };
      for (auto [next, prev] : llvm::zip_equal(newParams, params))
        next = keepIfFinite(next, prev);
      for (auto [next, prev] : llvm::zip_equal(newSlots, slots))
        next = keepIfFinite(next, prev);
    }

    Value recorded = newParams.front();
    if (recordConstrained)
      recorded = constrainPosition(builder, loc, recorded, ctx.supports);
    Value recordedRow = impulse::ReshapeOp::create(
        builder, loc, ctx.getPositionType(), recorded);
    Value traceBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, traceBufType, traceBufIter, recordedRow,
        ValueRange{iter, c0i});

    Value elbo1d = impulse::ReshapeOp::create(
        builder, loc, RankedTensorType::get({1}, elemType), estimate.elbo);
    Value elboBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, elboBufType, elboBufIter, elbo1d, ValueRange{iter});

    SmallVector<Value> yielded(newParams);
    yielded.append({rngNext, traceBufNew, elboBufNew, estimate.elbo});
    llvm::append_range(yielded, newSlots);
    impulse::YieldOp::create(builder, loc, yielded);
  }

  builder.setInsertionPointAfter(forOp);

  Value numStepsI64 = emitI64ScalarConstant(builder, loc, ctx.numSteps);
  SVILoopResult result;
  result.traceBuf = forOp.getResult(numParams + 1);
  result.elboBuf = forOp.getResult(numParams + 2);
  result.rng = forOp.getResult(numParams);
  for (size_t i = 0; i < numParams; ++i)
    result.params.push_back({forOp.getResult(i),
                             forOp.getResult(numParams + 4 + 2 * i),
                             forOp.getResult(numParams + 5 + 2 * i)});
  result.stepCount =
      arith::AddIOp::create(builder, loc, initStepCount, numStepsI64);
  return result;
}

/// Configures VI over `model` from the `vi_config`, `num_samples` and
/// `init_step_size` of `op`.
template <typename OpT>
static VIContext makeVIContext(OpT op, ModelContext model) {
  VIConfigAttr config = *op.getViConfig();
  VIContext ctx{std::move(model)};
  ctx.learningRate = op.getInitStepSize();
  ctx.elementType =
      cast<RankedTensorType>(ctx.learningRate.getType()).getElementType();
  ctx.numSteps = op.getNumSamples();
  ctx.guideKind = config.getGuideKind();
  ctx.optimizerKind = config.getOptimizerKind();
  ctx.numParticles = config.getNumParticles();
  auto readIfSet = [](double &field, FloatAttr attr) {
    if (attr)
      field = attr.getValueAsDouble();
  };
  readIfSet(ctx.initScale, config.getInitScale());
  readIfSet(ctx.adamB1, config.getAdamB1());
  readIfSet(ctx.adamB2, config.getAdamB2());
  readIfSet(ctx.adamEps, config.getAdamEps());
  readIfSet(ctx.clipNorm, config.getClipNorm());
  readIfSet(ctx.initRadius, config.getInitRadius());
  readIfSet(ctx.momentumMass, config.getMomentumMass());
  readIfSet(ctx.rmspropGamma, config.getRmspropGamma());
  readIfSet(ctx.rmspropEps, config.getRmspropEps());
  ctx.stableUpdate = config.getStableUpdate();
  return ctx;
}

template <typename OpT>
LogicalResult mlir::impulse::VI::lowerVI(PatternRewriter &rewriter, OpT op,
                                         ModelContext model) {
  Location loc = op.getLoc();
  VIContext ctx = makeVIContext(op, std::move(model));

  if (ctx.numParticles <= 0) {
    op.emitError("Impulse VI: num_particles must be > 0");
    return failure();
  }
  if (ctx.hasCustomLogpdf()) {
    op.emitError("Impulse VI: custom logpdf mode is not yet supported");
    return failure();
  }

  rewriter.setInsertionPoint(op);

  Type elemType = ctx.elementType;
  auto positionType = ctx.getPositionType();

  ValueRange stateIn = op.getStateIn();
  bool isResume = !stateIn.empty();
  ArrayAttr schema = op.getStateSchemaAttr();
  auto slotIn = [&](StringRef name) {
    return stateInSlot(schema, stateIn, name);
  };

  // A cold start splits the key into the loop key, an unused model seed and
  // the guide's init seed; a resumed run's key is already the loop key.
  Value rngInput = op.getInputs()[0];
  Type rngType = rngInput.getType();
  Value rng0;
  Value guideInitSeed;
  if (isResume) {
    rng0 = rngInput;
  } else {
    auto rngInitSplit = impulse::RandomSplitOp::create(
        rewriter, loc, TypeRange{rngType, rngType, rngType}, rngInput);
    rng0 = rngInitSplit.getResult(0);
    guideInitSeed = rngInitSplit.getResult(2);
  }

  Value loc0;
  if (isResume) {
    loc0 = slotIn(state_slot::kLoc);
  } else if (Value initPosition = op.getInitPosition()) {
    if (initPosition.getType() == positionType) {
      loc0 = initPosition;
    } else {
      loc0 =
          impulse::ReshapeOp::create(rewriter, loc, positionType, initPosition);
    }
  } else {
    // Each site starts uniform in [-initRadius, initRadius], from its own key.
    RankedTensorType scalarType = ctx.getScalarType();
    Value c0i64 = emitI64ScalarConstant(rewriter, loc, 0);
    Value negRadius =
        emitSplatConstant(rewriter, loc, scalarType, -ctx.initRadius);
    Value posRadius =
        emitSplatConstant(rewriter, loc, scalarType, ctx.initRadius);
    auto uniformAttr = impulse::RngDistributionAttr::get(
        rewriter.getContext(), impulse::RngDistribution::UNIFORM);

    loc0 = emitSplatConstant(rewriter, loc, positionType, 0.0);
    auto preSplit = impulse::RandomSplitOp::create(
        rewriter, loc, TypeRange{rngType, rngType}, guideInitSeed);
    Value chain = preSplit.getResult(1);
    for (const auto &info : ctx.supports) {
      auto siteSplit = impulse::RandomSplitOp::create(
          rewriter, loc, TypeRange{rngType, rngType}, chain);
      chain = siteSplit.getResult(0);
      Value siteKey = siteSplit.getResult(1);

      auto siteFlatType =
          RankedTensorType::get({info.constrainedSize}, elemType);
      auto draw = impulse::RandomOp::create(
          rewriter, loc, TypeRange{rngType, siteFlatType}, siteKey, negRadius,
          posRadius, uniformAttr);
      Value initDraw = draw.getResult();

      auto sliceType =
          RankedTensorType::get({1, info.constrainedSize}, elemType);
      Value initDraw2D =
          impulse::ReshapeOp::create(rewriter, loc, sliceType, initDraw);
      Value offsetIdx =
          emitI64ScalarConstant(rewriter, loc, info.constrainedOffset);
      loc0 = impulse::DynamicUpdateSliceOp::create(
          rewriter, loc, positionType, loc0, initDraw2D,
          ValueRange{c0i64, offsetIdx});
    }
  }

  ParamState locState{loc0, slotIn(state_slot::kOptMLoc),
                      slotIn(state_slot::kOptVLoc)};
  Value initStepCount = slotIn(state_slot::kOptStepCount);
  SVILoopResult loopRes;
  if (ctx.guideKind == impulse::GuideKind::AUTO_DELTA) {
    loopRes = emitSVILoop(rewriter, loc, {locState}, rng0, initStepCount, ctx,
                          emitAutoDeltaParticle, /*recordConstrained=*/true);
  } else if (ctx.guideKind == impulse::GuideKind::AUTO_NORMAL) {
    Value scale0;
    if (isResume) {
      scale0 = slotIn(state_slot::kScaleUnconstrained);
    } else if (Value initScale = op.getInitScale()) {
      if (initScale.getType() != positionType)
        initScale =
            impulse::ReshapeOp::create(rewriter, loc, positionType, initScale);
      scale0 = impulse::emitSoftplusInv(rewriter, loc, initScale);
    } else {
      double initScaleU = std::log(std::expm1(ctx.initScale));
      scale0 = emitSplatConstant(rewriter, loc, positionType, initScaleU);
    }
    ParamState scaleState{scale0, slotIn(state_slot::kOptMScale),
                          slotIn(state_slot::kOptVScale)};
    loopRes = emitSVILoop(rewriter, loc, {locState, scaleState}, rng0,
                          initStepCount, ctx, emitAutoNormalParticle,
                          /*recordConstrained=*/false);
  } else {
    // scale_tril = init_scale * I: zero off-diagonals, softplus_inv diagonal.
    int64_t d = ctx.positionSize;
    int64_t flatSize = d * (d + 1) / 2;
    int64_t nOff = d * (d - 1) / 2;
    auto flatType = RankedTensorType::get({flatSize}, elemType);
    Value xScaleTril0;
    if (isResume) {
      xScaleTril0 = slotIn(state_slot::kScaleTrilUnconstrained);
    } else {
      double initU = std::log(std::expm1(ctx.initScale));
      SmallVector<Attribute> vals;
      vals.reserve(flatSize);
      for (int64_t i = 0; i < nOff; ++i)
        vals.push_back(rewriter.getFloatAttr(elemType, 0.0));
      for (int64_t i = 0; i < d; ++i)
        vals.push_back(rewriter.getFloatAttr(elemType, initU));
      xScaleTril0 = arith::ConstantOp::create(
          rewriter, loc, flatType, DenseElementsAttr::get(flatType, vals));
    }
    ParamState scaleTrilState{xScaleTril0, slotIn(state_slot::kOptMScaleTril),
                              slotIn(state_slot::kOptVScaleTril)};
    loopRes = emitSVILoop(rewriter, loc, {locState, scaleTrilState}, rng0,
                          initStepCount, ctx, emitAutoMVNParticle,
                          /*recordConstrained=*/false);
  }

  auto slotValue = [&](StringRef name) -> Value {
    if (name == state_slot::kLoc)
      return loopRes.params[0].param;
    if (name == state_slot::kScaleUnconstrained ||
        name == state_slot::kScaleTrilUnconstrained)
      return loopRes.params[1].param;
    if (name == state_slot::kOptMLoc)
      return loopRes.params[0].m;
    if (name == state_slot::kOptVLoc)
      return loopRes.params[0].v;
    if (name == state_slot::kOptMScale || name == state_slot::kOptMScaleTril)
      return loopRes.params[1].m;
    if (name == state_slot::kOptVScale || name == state_slot::kOptVScaleTril)
      return loopRes.params[1].v;
    if (name == state_slot::kOptStepCount)
      return loopRes.stepCount;
    llvm_unreachable("unknown VI state slot");
  };

  SmallVector<Value> results;
  results.push_back(loopRes.traceBuf);
  results.push_back(loopRes.elboBuf); // diagnostics (per-step ELBO)
  results.push_back(arith::ConstantOp::create(
      rewriter, loc, op.getLogDensities().getType(),
      DenseElementsAttr::get(
          cast<ShapedType>(op.getLogDensities().getType()),
          rewriter.getFloatAttr(elemType, 0.0)))); // log_densities (unused)
  results.push_back(loopRes.rng);
  for (const impulse::StateSlot &slot :
       impulse::buildVIStateSchema(ctx.guideKind, ctx.positionSize, elemType))
    results.push_back(slotValue(slot.name));

  rewriter.replaceOp(op, results);
  return success();
}

template LogicalResult
mlir::impulse::VI::lowerVI<impulse::InferOp>(PatternRewriter &,
                                             impulse::InferOp, ModelContext);
template LogicalResult mlir::impulse::VI::lowerVI<impulse::InferRegionOp>(
    PatternRewriter &, impulse::InferRegionOp, ModelContext);
