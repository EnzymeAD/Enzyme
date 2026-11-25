//===- VIUtils.cpp - Variational Inference lowering utilities -------------===//

#include "VIUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Passes/InferStateLayout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"

#include <cmath>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;
using namespace mlir::impulse::VI;

static Value scalarConst(OpBuilder &builder, Location loc, RankedTensorType ty,
                         double v) {
  auto elem = ty.getElementType();
  return arith::ConstantOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getFloatAttr(elem, v)));
}

static Value i64Const(OpBuilder &builder, Location loc, int64_t v) {
  auto ty = RankedTensorType::get({}, builder.getI64Type());
  return arith::ConstantOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getI64IntegerAttr(v)));
}

static Value broadcastScalar(OpBuilder &builder, Location loc, Value scalar,
                             RankedTensorType targetType) {
  return enzyme::BroadcastOp::create(
      builder, loc, targetType, scalar,
      builder.getDenseI64ArrayAttr(targetType.getShape()));
}

static Value scatterPositionToTrace(OpBuilder &builder, Location loc,
                                    Value position2d, Value fullTrace,
                                    const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto traceType = RankedTensorType::get({1, ctx.getFullTraceSize()}, elemType);
  auto c0 = i64Const(builder, loc, 0);

  Value result = fullTrace;
  for (const auto &info : ctx.supports) {
    auto sliceType = RankedTensorType::get({1, info.size}, elemType);
    auto posOffset = i64Const(builder, loc, info.offset);
    SmallVector<Value> extractIdx{c0, posOffset};
    auto slice = impulse::DynamicSliceOp::create(
        builder, loc, sliceType, position2d, extractIdx,
        builder.getDenseI64ArrayAttr({1, info.size}));

    auto traceOffset = i64Const(builder, loc, info.traceOffset);
    SmallVector<Value> updateIdx{c0, traceOffset};
    result = impulse::DynamicUpdateSliceOp::create(builder, loc, traceType,
                                                   result, slice, updateIdx);
  }
  return result;
}

/// Sums all elements of `tensor` into a rank-0 tensor with a dot against ones.
static Value sumAllElements(OpBuilder &builder, Location loc, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto scalarType = RankedTensorType::get({}, tensorType.getElementType());
  Value ones = scalarConst(builder, loc, tensorType, 1.0);
  DenseI64ArrayAttr allDims = builder.getDenseI64ArrayAttr(
      llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank())));
  return impulse::DotOp::create(builder, loc, scalarType, tensor, ones, allDims,
                                allDims);
}

/// Sum as canary: a NaN/Inf gradient element makes its sum non-finite.
static Value computeIsFiniteForStableUpdate(OpBuilder &builder, Location loc,
                                            Value elboScalar, Value gradLoc,
                                            Value gradScale) {
  Value isFinite = math::IsFiniteOp::create(builder, loc, elboScalar);
  if (gradLoc) {
    Value gradLocSum = sumAllElements(builder, loc, gradLoc);
    Value gradLocFinite = math::IsFiniteOp::create(builder, loc, gradLocSum);
    isFinite = arith::AndIOp::create(builder, loc, isFinite, gradLocFinite);
  }
  if (gradScale) {
    Value gradScaleSum = sumAllElements(builder, loc, gradScale);
    Value gradScaleFinite =
        math::IsFiniteOp::create(builder, loc, gradScaleSum);
    isFinite = arith::AndIOp::create(builder, loc, isFinite, gradScaleFinite);
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
      builder, loc, sumEpsSq, scalarConst(builder, loc, scalarType, 0.5));
  Value log2piTerm = scalarConst(builder, loc, scalarType,
                                 static_cast<double>(ctx.positionSize) *
                                     (0.5 * std::log(2.0 * M_PI)));

  Value negSumLogScale = arith::NegFOp::create(builder, loc, sumLogScale);
  Value logQPart =
      arith::SubFOp::create(builder, loc, negSumLogScale, halfSumEpsSq);
  return arith::SubFOp::create(builder, loc, logQPart, log2piTerm);
}

/// Attributes for an autodiff region with `numActiveArgs` active arguments
/// that differentiates its first result and passes its second through as a
/// constant, followed by the entries of `extra` when it is non-null.
static SmallVector<NamedAttribute>
buildGradientRegionAttrs(Builder &builder, unsigned numActiveArgs,
                         DictionaryAttr extra) {
  MLIRContext *context = builder.getContext();
  enzyme::ActivityAttr active =
      enzyme::ActivityAttr::get(context, enzyme::Activity::enzyme_active);
  enzyme::ActivityAttr constant =
      enzyme::ActivityAttr::get(context, enzyme::Activity::enzyme_const);
  SmallVector<NamedAttribute> attrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr(SmallVector<Attribute>(numActiveArgs, active))),
      builder.getNamedAttr("ret_activity",
                           builder.getArrayAttr({active, constant})),
  };
  if (extra)
    llvm::append_range(attrs, extra);
  return attrs;
}

namespace {
struct LogPResult {
  Value logP;
  Value rng;
};

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

/// A parameter and its two optimizer state slots after one step. A rule that
/// keeps fewer slots returns the unused ones unchanged.
struct OptimizerUpdate {
  Value param;
  Value m;
  Value v;
};

struct SVILoopResult {
  Value traceBuf;
  Value elboBuf;
  Value rngFinal;
  Value locFinal; // unconstrained
  // Unconstrained; flat scale_tril for AutoMVN, null for AutoDelta.
  Value scaleFinal;
  Value elboFinal;
  // Final optimizer state; the scale moments are null for AutoDelta.
  Value mLocFinal;
  Value vLocFinal;
  Value mScaleFinal;
  Value vScaleFinal;
  Value stepCountFinal;
};
} // namespace

/// Values captured from above the region are referenced as-is.
static Value cloneVILogpdfRegion(OpBuilder &builder, Location loc, Value z,
                                 const VIContext &ctx) {
  Block &lpBlock = ctx.logpdfRegion->front();
  auto elemType = ctx.getElementType();

  IRMapping mapping;
  for (auto [arg, slice] :
       llvm::zip(lpBlock.getArguments(), ctx.logpdfArgSlices)) {
    int64_t offset = slice.first;
    int64_t size = slice.second;
    auto argType = cast<RankedTensorType>(arg.getType());
    auto sliceType = RankedTensorType::get({1, size}, elemType);
    Value s = impulse::SliceOp::create(
        builder, loc, sliceType, z, builder.getDenseI64ArrayAttr({0, offset}),
        builder.getDenseI64ArrayAttr({1, offset + size}),
        builder.getDenseI64ArrayAttr({1, 1}));
    Value component = impulse::ReshapeOp::create(builder, loc, argType, s);
    mapping.map(arg, component);
  }

  for (Operation &op : lpBlock.without_terminator())
    builder.clone(op, mapping);
  return mapping.lookupOrDefault(lpBlock.getTerminator()->getOperand(0));
}

static LogPResult callModelGenerate(OpBuilder &builder, Location loc, Value z,
                                    Value modelRng, const VIContext &ctx) {
  // The region consumes no RNG, so the model rng is returned unadvanced.
  if (ctx.hasLogpdfRegion())
    return {cloneVILogpdfRegion(builder, loc, z, ctx), modelRng};

  auto elemType = ctx.getElementType();
  auto scalarType = ctx.getScalarType();
  auto traceType = RankedTensorType::get({1, ctx.getFullTraceSize()}, elemType);

  Value fullTrace =
      scatterPositionToTrace(builder, loc, z, ctx.originalTrace, ctx);

  SmallVector<Value> genInputs;
  genInputs.push_back(modelRng);
  genInputs.append(ctx.fnInputs.begin(), ctx.fnInputs.end());

  SmallVector<Type> genResultTypes;
  genResultTypes.push_back(traceType);
  genResultTypes.push_back(scalarType);
  genResultTypes.append(ctx.fnResultTypes.begin(), ctx.fnResultTypes.end());

  auto genOp = impulse::GenerateOp::create(
      builder, loc, genResultTypes, ctx.fn, genInputs, fullTrace,
      ctx.allAddresses, ctx.allAddresses, builder.getStringAttr(""));

  // Result 2 is the model's first output, the rng by convention.
  Value rngAfter = genOp.getNumResults() >= 3 ? genOp.getResult(2) : modelRng;
  return {genOp.getWeight(), rngAfter};
}

/// Emits an autodiff region that differentiates, with respect to `params`, the
/// ELBO that `emitELBO` builds inside the region from the region arguments
/// standing for `params`.
static ELBOGradient
emitELBOGradient(OpBuilder &builder, Location loc, ValueRange params,
                 Type rngType, const VIContext &ctx,
                 function_ref<ParticleELBO(ValueRange paramArgs)> emitELBO) {
  RankedTensorType scalarType = ctx.getScalarType();
  Value gradSeed = scalarConst(builder, loc, scalarType, 1.0);
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

/// One particle of NumPyro's Trace_ELBO with an AutoNormal guide.
static ELBOGradient emitAutoNormalParticle(OpBuilder &builder, Location loc,
                                           ValueRange params, Value pkey,
                                           const VIContext &ctx) {
  Type rngType = pkey.getType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();

  // NumPyro order: model_seed, guide_seed = split(pkey).
  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);
  Value guideSeed = seedSplit.getResult(1);

  // eps is drawn outside autodiff so the primal and shadow share it, with
  // NumPyro's (chain, site_key) = split(chain) per site in declaration order.
  auto zeroScalar = scalarConst(builder, loc, scalarType, 0.0);
  auto oneScalar = scalarConst(builder, loc, scalarType, 1.0);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto c0i64 = arith::ConstantOp::create(
      builder, loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  Value eps = scalarConst(builder, loc, positionType, 0.0);
  Value chain = guideSeed;
  for (const auto &info : ctx.supports) {
    auto siteSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rngType, rngType}, chain);
    chain = siteSplit.getResult(0);
    Value siteKey = siteSplit.getResult(1);
    auto siteEpsType = RankedTensorType::get({1, info.size}, elemType);
    auto randOp = impulse::RandomOp::create(
        builder, loc, TypeRange{rngType, siteEpsType}, siteKey, zeroScalar,
        oneScalar,
        impulse::RngDistributionAttr::get(builder.getContext(),
                                          impulse::RngDistribution::NORMAL));
    Value siteEps = randOp.getResult();
    auto offsetI64 = arith::ConstantOp::create(
        builder, loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(info.offset)));
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

        // NumPyro's AutoNormal: scale = softplus(unconstrained_scale).
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

        LogPResult lp =
            callModelGenerate(builder, loc, zConstrained, modelSeed, ctx);
        Value logP =
            conditionalDump(builder, loc, lp.logP, "VI: logP", ctx.debugDump);

        Value elbo = arith::SubFOp::create(builder, loc, logP, logQ);
        elbo = conditionalDump(builder, loc, elbo, "VI: elbo=logP-logQ",
                               ctx.debugDump);
        return {elbo, lp.rng};
      });
}

/// log q = 0, but guide_seed is still split per site as NumPyro does.
static ELBOGradient emitAutoDeltaParticle(OpBuilder &builder, Location loc,
                                          ValueRange params, Value pkey,
                                          const VIContext &ctx) {
  Type rngType = pkey.getType();

  // NumPyro order: model_seed, guide_seed = split(pkey).
  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);

  return emitELBOGradient(builder, loc, params, rngType, ctx,
                          [&](ValueRange paramArgs) -> ParticleELBO {
                            // No Jacobian term: NumPyro's AutoDelta has
                            // log_density = 0.
                            Value zConstrained = constrainPosition(
                                builder, loc, paramArgs[0], ctx.supports);
                            LogPResult lp = callModelGenerate(
                                builder, loc, zConstrained, modelSeed, ctx);
                            return {lp.logP, lp.rng};
                          });
}

/// AutoMultivariateNormal variant; the scale parameter is the unconstrained
/// flat scale_tril.
static ELBOGradient emitAutoMVNParticle(OpBuilder &builder, Location loc,
                                        ValueRange params, Value pkey,
                                        const VIContext &ctx) {
  Type rngType = pkey.getType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();
  int64_t d = ctx.positionSize;

  // NumPyro order: model_seed, guide_seed = split(pkey).
  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);
  Value guideSeed = seedSplit.getResult(1);

  // One split for NumPyro's single `_auto_latent` site.
  auto siteSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, guideSeed);
  Value siteKey = siteSplit.getResult(1);

  auto zeroScalar = scalarConst(builder, loc, scalarType, 0.0);
  auto oneScalar = scalarConst(builder, loc, scalarType, 1.0);
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

        LogPResult lp = callModelGenerate(builder, loc, z, modelSeed, ctx);

        Value elbo = arith::SubFOp::create(builder, loc, lp.logP, logQ);
        return {elbo, lp.rng};
      });
}

/// Signature shared by the per-guide particle emitters: one ELBO estimate and
/// its gradient for the guide parameters `params` and particle key `pkey`.
using ParticleEmitter = function_ref<ELBOGradient(
    OpBuilder &builder, Location loc, ValueRange params, Value pkey,
    const VIContext &ctx)>;

/// NumPyro Trace_ELBO: K == 1 uses `rngStep` directly; K > 1 splits it K ways
/// and averages the particles.
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
  Value elboSum = scalarConst(builder, loc, scalarType, 0.0);
  SmallVector<Value> gradSums;
  for (Value param : params)
    gradSums.push_back(scalarConst(
        builder, loc, cast<RankedTensorType>(param.getType()), 0.0));

  for (Value pkey : keys.getResults()) {
    ELBOGradient particle = emitParticle(builder, loc, params, pkey, ctx);
    elboSum = arith::AddFOp::create(builder, loc, elboSum, particle.elbo);
    for (auto [sum, grad] : llvm::zip_equal(gradSums, particle.grads))
      sum = arith::AddFOp::create(builder, loc, sum, grad);
  }

  Value count =
      scalarConst(builder, loc, scalarType, static_cast<double>(numParticles));
  DenseMap<Type, Value> countSplats;
  for (Value param : params) {
    auto type = cast<RankedTensorType>(param.getType());
    if (!countSplats.contains(type))
      countSplats[type] = broadcastScalar(builder, loc, count, type);
  }
  ELBOGradient mean;
  mean.elbo = arith::DivFOp::create(builder, loc, elboSum, count);
  for (auto [param, sum] : llvm::zip_equal(params, gradSums))
    mean.grads.push_back(arith::DivFOp::create(
        builder, loc, sum, countSplats.lookup(param.getType())));
  return mean;
}

/// `gradElbo` is d(elbo)/dp; the minimized loss is -elbo.
static OptimizerUpdate applySGDUpdate(OpBuilder &builder, Location loc,
                                      Value param, Value gradElbo, Value m,
                                      Value v, Value lr) {
  auto ty = cast<RankedTensorType>(param.getType());
  Value lrBcast = broadcastScalar(builder, loc, lr, ty);
  Value scaledGrad = arith::MulFOp::create(builder, loc, lrBcast, gradElbo);
  Value paramNew = arith::AddFOp::create(builder, loc, param, scaledGrad);
  return {paramNew, m, v};
}

/// jax.example_libraries.optimizers.adam; `stepIdx` counts from 0. A positive
/// `clipNorm` clips each gradient element, as ClippedAdam does.
static OptimizerUpdate applyAdamUpdate(OpBuilder &builder, Location loc,
                                       Value param, Value gradElbo, Value m,
                                       Value v, Value stepIdx, Value lr,
                                       double b1, double b2, double eps,
                                       double clipNorm) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  if (clipNorm > 0.0) {
    auto negClip = scalarConst(builder, loc, scalarType, -clipNorm);
    auto posClip = scalarConst(builder, loc, scalarType, clipNorm);
    Value negClipBcast = broadcastScalar(builder, loc, negClip, paramType);
    Value posClipBcast = broadcastScalar(builder, loc, posClip, paramType);
    g = arith::MaximumFOp::create(builder, loc, g, negClipBcast);
    g = arith::MinimumFOp::create(builder, loc, g, posClipBcast);
  }

  auto b1Const = scalarConst(builder, loc, scalarType, b1);
  auto oneMinusB1 = scalarConst(builder, loc, scalarType, 1.0 - b1);
  auto b2Const = scalarConst(builder, loc, scalarType, b2);
  auto oneMinusB2 = scalarConst(builder, loc, scalarType, 1.0 - b2);
  auto epsConst = scalarConst(builder, loc, scalarType, eps);
  auto oneConst = scalarConst(builder, loc, scalarType, 1.0);

  Value b1Bcast = broadcastScalar(builder, loc, b1Const, paramType);
  Value oneMinusB1Bcast = broadcastScalar(builder, loc, oneMinusB1, paramType);
  Value b2Bcast = broadcastScalar(builder, loc, b2Const, paramType);
  Value oneMinusB2Bcast = broadcastScalar(builder, loc, oneMinusB2, paramType);

  Value oneMinusB1G = arith::MulFOp::create(builder, loc, oneMinusB1Bcast, g);
  Value b1M = arith::MulFOp::create(builder, loc, b1Bcast, m);
  Value mNew = arith::AddFOp::create(builder, loc, oneMinusB1G, b1M);

  Value gSq = arith::MulFOp::create(builder, loc, g, g);
  Value oneMinusB2GSq =
      arith::MulFOp::create(builder, loc, oneMinusB2Bcast, gSq);
  Value b2V = arith::MulFOp::create(builder, loc, b2Bcast, v);
  Value vNew = arith::AddFOp::create(builder, loc, oneMinusB2GSq, b2V);

  auto i64Ty = RankedTensorType::get({}, builder.getI64Type());
  Value oneI64 = arith::ConstantOp::create(
      builder, loc, i64Ty,
      DenseElementsAttr::get(i64Ty, builder.getI64IntegerAttr(1)));
  Value stepPlus1 = arith::AddIOp::create(builder, loc, stepIdx, oneI64);
  Value stepFloat =
      arith::SIToFPOp::create(builder, loc, scalarType, stepPlus1);

  Value b1PowS = math::PowFOp::create(builder, loc, b1Const, stepFloat);
  Value b2PowS = math::PowFOp::create(builder, loc, b2Const, stepFloat);
  Value oneMinusB1Pow = arith::SubFOp::create(builder, loc, oneConst, b1PowS);
  Value oneMinusB2Pow = arith::SubFOp::create(builder, loc, oneConst, b2PowS);
  Value oneMinusB1PowBcast =
      broadcastScalar(builder, loc, oneMinusB1Pow, paramType);
  Value oneMinusB2PowBcast =
      broadcastScalar(builder, loc, oneMinusB2Pow, paramType);

  Value mhat = arith::DivFOp::create(builder, loc, mNew, oneMinusB1PowBcast);
  Value vhat = arith::DivFOp::create(builder, loc, vNew, oneMinusB2PowBcast);

  Value sqrtVhat = math::SqrtOp::create(builder, loc, vhat);
  Value epsBcast = broadcastScalar(builder, loc, epsConst, paramType);
  Value denom = arith::AddFOp::create(builder, loc, sqrtVhat, epsBcast);
  Value mhatOverDenom = arith::DivFOp::create(builder, loc, mhat, denom);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value update = arith::MulFOp::create(builder, loc, lrBcast, mhatOverDenom);
  Value paramNew = arith::SubFOp::create(builder, loc, param, update);

  return {paramNew, mNew, vNew};
}

/// jax.example_libraries.optimizers.momentum; `m` is the velocity.
static OptimizerUpdate applyMomentumUpdate(OpBuilder &builder, Location loc,
                                           Value param, Value gradElbo, Value m,
                                           Value v, Value lr, double mass) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);
  Value massBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, mass), paramType);
  Value mScaled = arith::MulFOp::create(builder, loc, massBcast, m);
  Value mNew = arith::AddFOp::create(builder, loc, mScaled, g);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value scaledM = arith::MulFOp::create(builder, loc, lrBcast, mNew);
  Value paramNew = arith::SubFOp::create(builder, loc, param, scaledM);
  return {paramNew, mNew, v};
}

/// jax.example_libraries.optimizers.rmsprop; `m` is the squared-gradient
/// average.
static OptimizerUpdate applyRMSPropUpdate(OpBuilder &builder, Location loc,
                                          Value param, Value gradElbo, Value m,
                                          Value v, Value lr, double gamma,
                                          double eps) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  Value gammaBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, gamma), paramType);
  Value oneMinusGammaBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, 1.0 - gamma),
      paramType);
  Value epsBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, eps), paramType);

  Value gSq = arith::MulFOp::create(builder, loc, g, g);
  Value oneMinusGammaGSq =
      arith::MulFOp::create(builder, loc, oneMinusGammaBcast, gSq);
  Value gammaAvg = arith::MulFOp::create(builder, loc, gammaBcast, m);
  Value avgNew =
      arith::AddFOp::create(builder, loc, gammaAvg, oneMinusGammaGSq);

  Value avgPlusEps = arith::AddFOp::create(builder, loc, avgNew, epsBcast);
  Value sqrtAvgPlusEps = math::SqrtOp::create(builder, loc, avgPlusEps);
  Value gOverSqrt = arith::DivFOp::create(builder, loc, g, sqrtAvgPlusEps);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value update = arith::MulFOp::create(builder, loc, lrBcast, gOverSqrt);
  Value paramNew = arith::SubFOp::create(builder, loc, param, update);
  return {paramNew, avgNew, v};
}

/// jax.example_libraries.optimizers.adagrad; `m` is g_sq, `v` the momentum.
static OptimizerUpdate applyAdagradUpdate(OpBuilder &builder, Location loc,
                                          Value param, Value gradElbo, Value m,
                                          Value v, Value lr, double momentum) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  Value gSquared = arith::MulFOp::create(builder, loc, g, g);
  Value gSqNew = arith::AddFOp::create(builder, loc, m, gSquared);

  // where(g_sq > 0, 1/sqrt(g_sq), 0), as JAX's adagrad.
  Value sqrtGSq = math::SqrtOp::create(builder, loc, gSqNew);
  Value zeroBcast = scalarConst(builder, loc, paramType, 0.0);
  Value isPositive = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OGT, gSqNew, zeroBcast);
  Value oneBcast = scalarConst(builder, loc, paramType, 1.0);
  Value invSqrt = arith::DivFOp::create(builder, loc, oneBcast, sqrtGSq);
  Value gSqInvSqrt =
      arith::SelectOp::create(builder, loc, isPositive, invSqrt, zeroBcast);

  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value gTimesInv = arith::MulFOp::create(builder, loc, g, gSqInvSqrt);
  Value lrG = arith::MulFOp::create(builder, loc, lrBcast, gTimesInv);
  Value dx = arith::NegFOp::create(builder, loc, lrG);

  Value momBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, momentum), paramType);
  Value oneMinusMomBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, 1.0 - momentum),
      paramType);
  Value oneMinusMomDx =
      arith::MulFOp::create(builder, loc, oneMinusMomBcast, dx);
  Value momTimesV = arith::MulFOp::create(builder, loc, momBcast, v);
  Value vNew = arith::AddFOp::create(builder, loc, oneMinusMomDx, momTimesV);

  Value paramNew = arith::AddFOp::create(builder, loc, param, vNew);
  return {paramNew, gSqNew, vNew};
}

/// jax.example_libraries.optimizers.rmsprop_momentum; `m` is the
/// squared-gradient average, `v` the momentum.
static OptimizerUpdate
applyRMSPropMomentumUpdate(OpBuilder &builder, Location loc, Value param,
                           Value gradElbo, Value m, Value v, Value lr,
                           double gamma, double eps, double momentumMass) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  Value gammaBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, gamma), paramType);
  Value oneMinusGammaBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, 1.0 - gamma),
      paramType);
  Value epsBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, eps), paramType);
  Value momBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, momentumMass),
      paramType);

  Value gSq = arith::MulFOp::create(builder, loc, g, g);
  Value oneMinusGammaGSq =
      arith::MulFOp::create(builder, loc, oneMinusGammaBcast, gSq);
  Value gammaAvg = arith::MulFOp::create(builder, loc, gammaBcast, m);
  Value avgNew =
      arith::AddFOp::create(builder, loc, gammaAvg, oneMinusGammaGSq);

  Value avgPlusEps = arith::AddFOp::create(builder, loc, avgNew, epsBcast);
  Value sqrtAvgPlusEps = math::SqrtOp::create(builder, loc, avgPlusEps);
  Value gOverSqrt = arith::DivFOp::create(builder, loc, g, sqrtAvgPlusEps);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value lrGOverSqrt = arith::MulFOp::create(builder, loc, lrBcast, gOverSqrt);
  Value momTimesMom = arith::MulFOp::create(builder, loc, momBcast, v);
  Value momNew = arith::AddFOp::create(builder, loc, momTimesMom, lrGOverSqrt);

  Value paramNew = arith::SubFOp::create(builder, loc, param, momNew);
  return {paramNew, avgNew, momNew};
}

/// Applies `ctx.optimizerKind` to `param` with gradient `gradElbo` and state
/// slots `m` and `v`; `step` is the zero-based global step index.
static OptimizerUpdate applyOptimizer(OpBuilder &builder, Location loc,
                                      const VIContext &ctx, Value step,
                                      Value param, Value gradElbo, Value m,
                                      Value v) {
  switch (ctx.optimizerKind) {
  case OptimizerKind::ADAM:
    return applyAdamUpdate(builder, loc, param, gradElbo, m, v, step,
                           ctx.stepSize, ctx.adamB1, ctx.adamB2, ctx.adamEps,
                           /*clipNorm=*/0.0);
  case OptimizerKind::CLIPPED_ADAM:
    return applyAdamUpdate(builder, loc, param, gradElbo, m, v, step,
                           ctx.stepSize, ctx.adamB1, ctx.adamB2, ctx.adamEps,
                           ctx.clipNorm);
  case OptimizerKind::MOMENTUM:
    return applyMomentumUpdate(builder, loc, param, gradElbo, m, v,
                               ctx.stepSize, ctx.momentumMass);
  case OptimizerKind::RMSPROP:
    return applyRMSPropUpdate(builder, loc, param, gradElbo, m, v, ctx.stepSize,
                              ctx.rmspropGamma, ctx.rmspropEps);
  case OptimizerKind::ADAGRAD:
    return applyAdagradUpdate(builder, loc, param, gradElbo, m, v, ctx.stepSize,
                              ctx.momentumMass);
  case OptimizerKind::RMSPROP_MOMENTUM:
    return applyRMSPropMomentumUpdate(builder, loc, param, gradElbo, m, v,
                                      ctx.stepSize, ctx.rmspropGamma,
                                      ctx.rmspropEps, ctx.momentumMass);
  case OptimizerKind::SGD:
    return applySGDUpdate(builder, loc, param, gradElbo, m, v, ctx.stepSize);
  }
  llvm_unreachable("unknown VI optimizer kind");
}

/// `ctx.numSamples` iterations of NumPyro's SVI.update.
static SVILoopResult emitSVILoop(OpBuilder &builder, Location loc, Value loc0,
                                 Value scale0, Value rng0,
                                 const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  Type rngType = rng0.getType();
  auto traceBufType =
      RankedTensorType::get({ctx.numSamples, ctx.positionSize}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSamples}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  Value c0i = i64Const(builder, loc, 0);
  Value c1i = i64Const(builder, loc, 1);
  Value cNi = i64Const(builder, loc, ctx.numSamples);
  Value traceInit = scalarConst(builder, loc, traceBufType, 0.0);
  Value elboInit = scalarConst(builder, loc, elboBufType, 0.0);
  Value elboFinalSeed = scalarConst(builder, loc, scalarType, 0.0);
  Value zeroPos = scalarConst(builder, loc, positionType, 0.0);
  Value mLoc0 = ctx.initMLoc ? ctx.initMLoc : zeroPos;
  Value vLoc0 = ctx.initVLoc ? ctx.initVLoc : zeroPos;
  Value mScale0 = ctx.initMScale ? ctx.initMScale : zeroPos;
  Value vScale0 = ctx.initVScale ? ctx.initVScale : zeroPos;
  Value initStepCountI64 =
      ctx.initStepCount ? ctx.initStepCount : i64Const(builder, loc, 0);

  SmallVector<Value> iterArgs{loc0,          scale0, rng0,  traceInit, elboInit,
                              elboFinalSeed, mLoc0,  vLoc0, mScale0,   vScale0};
  SmallVector<Type> iterTypes{
      positionType, positionType, rngType,      traceBufType, elboBufType,
      scalarType,   positionType, positionType, positionType, positionType};

  auto forOp =
      impulse::ForOp::create(builder, loc, iterTypes, c0i, cNi, c1i, iterArgs);
  Block *body = builder.createBlock(&forOp.getRegion());
  body->addArgument(i64T, loc);
  for (auto t : iterTypes)
    body->addArgument(t, loc);

  Value iter = body->getArgument(0);
  Value locIter = body->getArgument(1);
  Value scaleIter = body->getArgument(2);
  Value rngIter = body->getArgument(3);
  Value traceBufIter = body->getArgument(4);
  Value elboBufIter = body->getArgument(5);
  Value mLocIter = body->getArgument(7);
  Value vLocIter = body->getArgument(8);
  Value mScaleIter = body->getArgument(9);
  Value vScaleIter = body->getArgument(10);

  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(body);

    // NumPyro SVI.update: rng_next, rng_step = split(rng).
    auto updateSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rngType, rngType}, rngIter);
    Value rngNext = updateSplit.getResult(0);
    Value rngStep = updateSplit.getResult(1);

    ELBOGradient elboRes =
        averageParticles(builder, loc, ValueRange{locIter, scaleIter}, rngStep,
                         ctx, emitAutoNormalParticle);
    Value elboGradLoc = elboRes.grads[0];
    Value elboGradScale = elboRes.grads[1];

    Value effectiveStep =
        arith::AddIOp::create(builder, loc, iter, initStepCountI64);

    OptimizerUpdate locUpd =
        applyOptimizer(builder, loc, ctx, effectiveStep, locIter, elboGradLoc,
                       mLocIter, vLocIter);
    OptimizerUpdate scaleUpd =
        applyOptimizer(builder, loc, ctx, effectiveStep, scaleIter,
                       elboGradScale, mScaleIter, vScaleIter);
    Value locNew = locUpd.param, mLocNew = locUpd.m, vLocNew = locUpd.v;
    Value scaleNew = scaleUpd.param, mScaleNew = scaleUpd.m,
          vScaleNew = scaleUpd.v;

    // A skipped step still records its ELBO, as numpyro's stable_update does.
    if (ctx.stableUpdate) {
      Value isFinite = computeIsFiniteForStableUpdate(
          builder, loc, elboRes.elbo, elboGradLoc, elboGradScale);
      locNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                         locNew, locIter);
      scaleNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                           scaleNew, scaleIter);
      mLocNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                          mLocNew, mLocIter);
      vLocNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                          vLocNew, vLocIter);
      mScaleNew = impulse::SelectOp::create(builder, loc, positionType,
                                            isFinite, mScaleNew, mScaleIter);
      vScaleNew = impulse::SelectOp::create(builder, loc, positionType,
                                            isFinite, vScaleNew, vScaleIter);
    }

    Value locFlat = impulse::ReshapeOp::create(
        builder, loc, RankedTensorType::get({1, ctx.positionSize}, elemType),
        locNew);
    Value traceBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, traceBufType, traceBufIter, locFlat,
        ValueRange{iter, c0i});

    auto scalar1d = RankedTensorType::get({1}, elemType);
    Value elbo1d =
        impulse::ReshapeOp::create(builder, loc, scalar1d, elboRes.elbo);
    Value elboBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, elboBufType, elboBufIter, elbo1d, ValueRange{iter});

    impulse::YieldOp::create(builder, loc,
                             ValueRange{locNew, scaleNew, rngNext, traceBufNew,
                                        elboBufNew, elboRes.elbo, mLocNew,
                                        vLocNew, mScaleNew, vScaleNew});
  }

  builder.setInsertionPointAfter(forOp);

  Value numSamplesI64 = i64Const(builder, loc, ctx.numSamples);
  Value finalStepCountI64 =
      arith::AddIOp::create(builder, loc, initStepCountI64, numSamplesI64);

  SVILoopResult r;
  r.traceBuf = forOp.getResult(3);
  r.elboBuf = forOp.getResult(4);
  r.rngFinal = forOp.getResult(2);
  r.locFinal = forOp.getResult(0);
  r.scaleFinal = forOp.getResult(1);
  r.elboFinal = forOp.getResult(5);
  r.mLocFinal = forOp.getResult(6);
  r.vLocFinal = forOp.getResult(7);
  r.mScaleFinal = forOp.getResult(8);
  r.vScaleFinal = forOp.getResult(9);
  r.stepCountFinal = finalStepCountI64;
  return r;
}

/// AutoDelta variant of emitSVILoop; the trace records the constrained loc.
static SVILoopResult emitSVILoopAutoDelta(OpBuilder &builder, Location loc,
                                          Value loc0, Value rng0,
                                          const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  Type rngType = rng0.getType();
  auto traceBufType =
      RankedTensorType::get({ctx.numSamples, ctx.positionSize}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSamples}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  Value c0i = i64Const(builder, loc, 0);
  Value c1i = i64Const(builder, loc, 1);
  Value cNi = i64Const(builder, loc, ctx.numSamples);
  Value traceInit = scalarConst(builder, loc, traceBufType, 0.0);
  Value elboInit = scalarConst(builder, loc, elboBufType, 0.0);
  Value elboFinalSeed = scalarConst(builder, loc, scalarType, 0.0);
  Value zeroPos = scalarConst(builder, loc, positionType, 0.0);
  Value mLoc0 = ctx.initMLoc ? ctx.initMLoc : zeroPos;
  Value vLoc0 = ctx.initVLoc ? ctx.initVLoc : zeroPos;
  Value initStepCountI64 =
      ctx.initStepCount ? ctx.initStepCount : i64Const(builder, loc, 0);

  SmallVector<Value> iterArgs{loc0,          rng0,  traceInit, elboInit,
                              elboFinalSeed, mLoc0, vLoc0};
  SmallVector<Type> iterTypes{positionType, rngType,    traceBufType,
                              elboBufType,  scalarType, positionType,
                              positionType};

  auto forOp =
      impulse::ForOp::create(builder, loc, iterTypes, c0i, cNi, c1i, iterArgs);
  Block *body = builder.createBlock(&forOp.getRegion());
  body->addArgument(i64T, loc);
  for (auto t : iterTypes)
    body->addArgument(t, loc);

  Value iter = body->getArgument(0);
  Value locIter = body->getArgument(1);
  Value rngIter = body->getArgument(2);
  Value traceBufIter = body->getArgument(3);
  Value elboBufIter = body->getArgument(4);
  Value mLocIter = body->getArgument(6);
  Value vLocIter = body->getArgument(7);

  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(body);

    // NumPyro SVI.update: rng_next, rng_step = split(rng).
    auto updateSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rngType, rngType}, rngIter);
    Value rngNext = updateSplit.getResult(0);
    Value rngStep = updateSplit.getResult(1);

    ELBOGradient elboRes = averageParticles(
        builder, loc, ValueRange{locIter}, rngStep, ctx, emitAutoDeltaParticle);
    Value elboMean = elboRes.elbo;
    Value gradLocMean = elboRes.grads[0];

    Value effectiveStep =
        arith::AddIOp::create(builder, loc, iter, initStepCountI64);

    OptimizerUpdate locUpd =
        applyOptimizer(builder, loc, ctx, effectiveStep, locIter, gradLocMean,
                       mLocIter, vLocIter);
    Value locNew = locUpd.param, mLocNew = locUpd.m, vLocNew = locUpd.v;

    if (ctx.stableUpdate) {
      Value isFinite = computeIsFiniteForStableUpdate(
          builder, loc, elboMean, gradLocMean, /*gradScale=*/Value());
      locNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                         locNew, locIter);
      mLocNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                          mLocNew, mLocIter);
      vLocNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                          vLocNew, vLocIter);
    }

    // Constrained, as NumPyro reports the `<site>_auto_loc` params.
    Value locConstrained =
        constrainPosition(builder, loc, locNew, ctx.supports);
    Value locFlat = impulse::ReshapeOp::create(
        builder, loc, RankedTensorType::get({1, ctx.positionSize}, elemType),
        locConstrained);
    Value traceBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, traceBufType, traceBufIter, locFlat,
        ValueRange{iter, c0i});

    auto scalar1d = RankedTensorType::get({1}, elemType);
    Value elbo1d = impulse::ReshapeOp::create(builder, loc, scalar1d, elboMean);
    Value elboBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, elboBufType, elboBufIter, elbo1d, ValueRange{iter});

    impulse::YieldOp::create(builder, loc,
                             ValueRange{locNew, rngNext, traceBufNew,
                                        elboBufNew, elboMean, mLocNew,
                                        vLocNew});
  }

  builder.setInsertionPointAfter(forOp);

  Value numSamplesI64 = i64Const(builder, loc, ctx.numSamples);
  Value finalStepCountI64 =
      arith::AddIOp::create(builder, loc, initStepCountI64, numSamplesI64);

  SVILoopResult r;
  r.traceBuf = forOp.getResult(2);
  r.elboBuf = forOp.getResult(3);
  r.rngFinal = forOp.getResult(1);
  r.locFinal = forOp.getResult(0);
  r.scaleFinal = Value();
  r.elboFinal = forOp.getResult(4);
  r.mLocFinal = forOp.getResult(5);
  r.vLocFinal = forOp.getResult(6);
  r.mScaleFinal = Value();
  r.vScaleFinal = Value();
  r.stepCountFinal = finalStepCountI64;
  return r;
}

/// `xScaleTril0` is the unconstrained flat vector of length d*(d+1)/2.
static SVILoopResult emitSVILoopMVN(OpBuilder &builder, Location loc,
                                    Value loc0, Value xScaleTril0, Value rng0,
                                    const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  Type rngType = rng0.getType();
  int64_t d = ctx.positionSize;
  int64_t flatSize = d * (d + 1) / 2;
  auto flatXTy = RankedTensorType::get({flatSize}, elemType);
  auto traceBufType = RankedTensorType::get({ctx.numSamples, d}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSamples}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  Value c0i = i64Const(builder, loc, 0);
  Value c1i = i64Const(builder, loc, 1);
  Value cNi = i64Const(builder, loc, ctx.numSamples);
  Value traceInit = scalarConst(builder, loc, traceBufType, 0.0);
  Value elboInit = scalarConst(builder, loc, elboBufType, 0.0);
  Value elboFinalSeed = scalarConst(builder, loc, scalarType, 0.0);
  Value zeroPos = scalarConst(builder, loc, positionType, 0.0);
  Value zeroFlat = scalarConst(builder, loc, flatXTy, 0.0);
  Value mLoc0 = ctx.initMLoc ? ctx.initMLoc : zeroPos;
  Value vLoc0 = ctx.initVLoc ? ctx.initVLoc : zeroPos;
  Value mX0 = ctx.initMScale ? ctx.initMScale : zeroFlat;
  Value vX0 = ctx.initVScale ? ctx.initVScale : zeroFlat;
  Value initStepCountI64 =
      ctx.initStepCount ? ctx.initStepCount : i64Const(builder, loc, 0);

  SmallVector<Value> iterArgs{
      loc0,          xScaleTril0, rng0,  traceInit, elboInit,
      elboFinalSeed, mLoc0,       vLoc0, mX0,       vX0};
  SmallVector<Type> iterTypes{
      positionType, flatXTy,      rngType,      traceBufType, elboBufType,
      scalarType,   positionType, positionType, flatXTy,      flatXTy};

  auto forOp =
      impulse::ForOp::create(builder, loc, iterTypes, c0i, cNi, c1i, iterArgs);
  Block *body = builder.createBlock(&forOp.getRegion());
  body->addArgument(i64T, loc);
  for (auto t : iterTypes)
    body->addArgument(t, loc);

  Value iter = body->getArgument(0);
  Value locIter = body->getArgument(1);
  Value xIter = body->getArgument(2);
  Value rngIter = body->getArgument(3);
  Value traceBufIter = body->getArgument(4);
  Value elboBufIter = body->getArgument(5);
  Value mLocIter = body->getArgument(7);
  Value vLocIter = body->getArgument(8);
  Value mXIter = body->getArgument(9);
  Value vXIter = body->getArgument(10);

  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(body);

    // NumPyro SVI.update: rng_next, rng_step = split(rng).
    auto updateSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rngType, rngType}, rngIter);
    Value rngNext = updateSplit.getResult(0);
    Value rngStep = updateSplit.getResult(1);

    ELBOGradient elboRes =
        averageParticles(builder, loc, ValueRange{locIter, xIter}, rngStep, ctx,
                         emitAutoMVNParticle);
    Value elbo = elboRes.elbo;
    Value gradLoc = elboRes.grads[0];
    Value gradX = elboRes.grads[1];

    Value effectiveStep =
        arith::AddIOp::create(builder, loc, iter, initStepCountI64);

    OptimizerUpdate locUpd = applyOptimizer(
        builder, loc, ctx, effectiveStep, locIter, gradLoc, mLocIter, vLocIter);
    OptimizerUpdate xUpd = applyOptimizer(builder, loc, ctx, effectiveStep,
                                          xIter, gradX, mXIter, vXIter);
    Value locNew = locUpd.param, mLocNew = locUpd.m, vLocNew = locUpd.v;
    Value xNew = xUpd.param, mXNew = xUpd.m, vXNew = xUpd.v;

    if (ctx.stableUpdate) {
      Value isFinite =
          computeIsFiniteForStableUpdate(builder, loc, elbo, gradLoc, gradX);
      auto xType = cast<RankedTensorType>(xIter.getType());
      locNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                         locNew, locIter);
      xNew =
          impulse::SelectOp::create(builder, loc, xType, isFinite, xNew, xIter);
      mLocNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                          mLocNew, mLocIter);
      vLocNew = impulse::SelectOp::create(builder, loc, positionType, isFinite,
                                          vLocNew, vLocIter);
      mXNew = impulse::SelectOp::create(builder, loc, xType, isFinite, mXNew,
                                        mXIter);
      vXNew = impulse::SelectOp::create(builder, loc, xType, isFinite, vXNew,
                                        vXIter);
    }

    Value locFlat =
        impulse::ReshapeOp::create(builder, loc, positionType, locNew);
    Value traceBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, traceBufType, traceBufIter, locFlat,
        ValueRange{iter, c0i});

    auto scalar1d = RankedTensorType::get({1}, elemType);
    Value elbo1d = impulse::ReshapeOp::create(builder, loc, scalar1d, elbo);
    Value elboBufNew = impulse::DynamicUpdateSliceOp::create(
        builder, loc, elboBufType, elboBufIter, elbo1d, ValueRange{iter});

    impulse::YieldOp::create(builder, loc,
                             ValueRange{locNew, xNew, rngNext, traceBufNew,
                                        elboBufNew, elbo, mLocNew, vLocNew,
                                        mXNew, vXNew});
  }

  builder.setInsertionPointAfter(forOp);

  Value numSamplesI64 = i64Const(builder, loc, ctx.numSamples);
  Value finalStepCountI64 =
      arith::AddIOp::create(builder, loc, initStepCountI64, numSamplesI64);

  SVILoopResult r;
  r.traceBuf = forOp.getResult(3);
  r.elboBuf = forOp.getResult(4);
  r.rngFinal = forOp.getResult(2);
  r.locFinal = forOp.getResult(0);
  r.scaleFinal = forOp.getResult(1);
  r.elboFinal = forOp.getResult(5);
  r.mLocFinal = forOp.getResult(6);
  r.vLocFinal = forOp.getResult(7);
  r.mScaleFinal = forOp.getResult(8);
  r.vScaleFinal = forOp.getResult(9);
  r.stepCountFinal = finalStepCountI64;
  return r;
}

template <typename OpT>
LogicalResult mlir::impulse::VI::lowerVI(PatternRewriter &rewriter, OpT op,
                                         VIContext ctx, bool debugDump) {
  Location loc = op.getLoc();

  if (ctx.guideKind != impulse::GuideKind::AUTO_NORMAL &&
      ctx.guideKind != impulse::GuideKind::AUTO_MULTIVARIATE_NORMAL &&
      ctx.guideKind != impulse::GuideKind::AUTO_DELTA) {
    op.emitError("Impulse VI: guide_kind = ")
        << stringifyGuideKind(ctx.guideKind) << " is not yet supported";
    return failure();
  }
  if (ctx.numParticles <= 0) {
    op.emitError("Impulse VI: num_particles must be > 0");
    return failure();
  }
  if (ctx.hasCustomLogpdf()) {
    op.emitError("Impulse VI: custom logpdf mode is not yet supported");
    return failure();
  }

  rewriter.setInsertionPoint(op);

  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();

  bool isResume = !op.getStateIn().empty();
  ctx.isResume = isResume;
  ArrayAttr schema = op.getStateSchemaAttr();
  auto slotIn = [&](StringRef name) -> Value {
    auto idx = impulse::slotIndex(schema, name);
    if (!idx)
      return Value();
    auto stateIn = op.getStateIn();
    return *idx < stateIn.size() ? stateIn[*idx] : Value();
  };

  ctx.initMLoc = slotIn("opt_m_loc");
  ctx.initVLoc = slotIn("opt_v_loc");
  if (ctx.guideKind == impulse::GuideKind::AUTO_MULTIVARIATE_NORMAL) {
    ctx.initMScale = slotIn("opt_m_scale_tril");
    ctx.initVScale = slotIn("opt_v_scale_tril");
  } else {
    ctx.initMScale = slotIn("opt_m_scale");
    ctx.initVScale = slotIn("opt_v_scale");
  }
  ctx.initStepCount = slotIn("opt_step_count");

  // NumPyro SVI.init: rng_key, model_seed, guide_seed = split(key, 3).
  // A resumed run already holds the per-update rng_key, so it skips the split.
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
    loc0 = slotIn("loc");
  } else if (auto ip = op.getInitPosition()) {
    auto ipType = cast<RankedTensorType>(ip.getType());
    if (ipType == positionType) {
      loc0 = ip;
    } else {
      loc0 = impulse::ReshapeOp::create(rewriter, loc, positionType, ip);
    }
  } else {
    // NumPyro's init_to_uniform: one split per site in declaration order.
    auto scalarTy = ctx.getScalarType();
    auto i64T = RankedTensorType::get({}, rewriter.getI64Type());
    auto c0i64 = arith::ConstantOp::create(
        rewriter, loc, i64T,
        DenseElementsAttr::get(i64T, rewriter.getI64IntegerAttr(0)));
    auto negRadius = arith::ConstantOp::create(
        rewriter, loc, scalarTy,
        DenseElementsAttr::get(
            scalarTy, rewriter.getFloatAttr(elemType, -ctx.initRadius)));
    auto posRadius = arith::ConstantOp::create(
        rewriter, loc, scalarTy,
        DenseElementsAttr::get(
            scalarTy, rewriter.getFloatAttr(elemType, ctx.initRadius)));
    auto uniformAttr = impulse::RngDistributionAttr::get(
        rewriter.getContext(), impulse::RngDistribution::UNIFORM);

    loc0 = arith::ConstantOp::create(
        rewriter, loc, positionType,
        DenseElementsAttr::get(positionType,
                               rewriter.getFloatAttr(elemType, 0.0)));
    // AutoGuide._setup_prototype's numpyro.prng_key() consumes one split.
    auto preSplit = impulse::RandomSplitOp::create(
        rewriter, loc, TypeRange{rngType, rngType}, guideInitSeed);
    Value chain = preSplit.getResult(1);
    for (const auto &info : ctx.supports) {
      auto sp = impulse::RandomSplitOp::create(
          rewriter, loc, TypeRange{rngType, rngType}, chain);
      chain = sp.getResult(0);
      Value siteKey = sp.getResult(1);

      auto siteFlatTy = RankedTensorType::get({info.size}, elemType);
      auto rop = impulse::RandomOp::create(
          rewriter, loc, TypeRange{rngType, siteFlatTy}, siteKey, negRadius,
          posRadius, uniformAttr);
      Value initDraw = rop.getResult();

      auto sliceTy = RankedTensorType::get({1, info.size}, elemType);
      Value initDraw2D =
          impulse::ReshapeOp::create(rewriter, loc, sliceTy, initDraw);
      auto offsetIdx = arith::ConstantOp::create(
          rewriter, loc, i64T,
          DenseElementsAttr::get(i64T,
                                 rewriter.getI64IntegerAttr(info.offset)));
      loc0 = impulse::DynamicUpdateSliceOp::create(
          rewriter, loc, positionType, loc0, initDraw2D,
          ValueRange{c0i64, offsetIdx});
    }
  }

  SVILoopResult loopRes;
  impulse::VIGuide guide;
  if (ctx.guideKind == impulse::GuideKind::AUTO_DELTA) {
    guide = impulse::VIGuide::AutoDelta;
    loopRes = emitSVILoopAutoDelta(rewriter, loc, loc0, rng0, ctx);
  } else if (ctx.guideKind == impulse::GuideKind::AUTO_NORMAL) {
    guide = impulse::VIGuide::AutoNormal;
    Value scale0;
    if (isResume) {
      scale0 = slotIn("scale_unconstrained");
    } else if (auto initScaleOp = op.getInitScale()) {
      auto isType = cast<RankedTensorType>(initScaleOp.getType());
      Value initScaleConstrained =
          (isType == positionType)
              ? Value(initScaleOp)
              : Value(impulse::ReshapeOp::create(rewriter, loc, positionType,
                                                 initScaleOp));
      scale0 = impulse::emitSoftplusInv(rewriter, loc, initScaleConstrained);
    } else {
      double initScaleU = std::log(std::expm1(ctx.initScale));
      scale0 = scalarConst(rewriter, loc, positionType, initScaleU);
    }
    loopRes = emitSVILoop(rewriter, loc, loc0, scale0, rng0, ctx);
  } else {
    // scale_tril = init_scale * I: zero off-diagonals, softplus_inv diagonal.
    guide = impulse::VIGuide::AutoMVN;
    int64_t d = ctx.positionSize;
    int64_t flatSize = d * (d + 1) / 2;
    int64_t nOff = d * (d - 1) / 2;
    auto flatTy = RankedTensorType::get({flatSize}, elemType);
    Value xScaleTril0;
    if (isResume) {
      xScaleTril0 = slotIn("scale_tril_unconstrained");
    } else {
      double initU = std::log(std::expm1(ctx.initScale));
      SmallVector<Attribute> vals;
      vals.reserve(flatSize);
      for (int64_t i = 0; i < nOff; ++i)
        vals.push_back(rewriter.getFloatAttr(elemType, 0.0));
      for (int64_t i = 0; i < d; ++i)
        vals.push_back(rewriter.getFloatAttr(elemType, initU));
      xScaleTril0 = arith::ConstantOp::create(
          rewriter, loc, flatTy, DenseElementsAttr::get(flatTy, vals));
    }
    loopRes = emitSVILoopMVN(rewriter, loc, loc0, xScaleTril0, rng0, ctx);
  }

  auto slotValue = [&](StringRef name) -> Value {
    if (name == "loc")
      return loopRes.locFinal;
    if (name == "scale_unconstrained" || name == "scale_tril_unconstrained")
      return loopRes.scaleFinal;
    if (name == "opt_m_loc")
      return loopRes.mLocFinal;
    if (name == "opt_v_loc")
      return loopRes.vLocFinal;
    if (name == "opt_m_scale" || name == "opt_m_scale_tril")
      return loopRes.mScaleFinal;
    if (name == "opt_v_scale" || name == "opt_v_scale_tril")
      return loopRes.vScaleFinal;
    if (name == "opt_step_count")
      return loopRes.stepCountFinal;
    return Value();
  };

  SmallVector<Value> results;
  results.push_back(loopRes.traceBuf);
  results.push_back(loopRes.elboBuf); // diagnostics (per-step ELBO)
  results.push_back(arith::ConstantOp::create(
      rewriter, loc, op.getLogDensities().getType(),
      DenseElementsAttr::get(
          cast<ShapedType>(op.getLogDensities().getType()),
          rewriter.getFloatAttr(elemType, 0.0)))); // log_densities (unused)
  results.push_back(loopRes.rngFinal);
  for (const impulse::StateSlot &slot :
       impulse::buildVIStateSchema(guide, ctx.positionSize, elemType))
    results.push_back(slotValue(slot.name));

  rewriter.replaceOp(op, results);
  return success();
}

template LogicalResult
mlir::impulse::VI::lowerVI<impulse::InferOp>(PatternRewriter &,
                                             impulse::InferOp, VIContext, bool);
template LogicalResult mlir::impulse::VI::lowerVI<impulse::InferRegionOp>(
    PatternRewriter &, impulse::InferRegionOp, VIContext, bool);
