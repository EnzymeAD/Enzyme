//===- VIUtils.cpp - Variational Inference lowering utilities -------------===//

#include "VIUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Passes/InferStateLayout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <cmath>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;
using namespace mlir::impulse::VI;

namespace {

Value scalarConst(OpBuilder &builder, Location loc, RankedTensorType ty,
                  double v) {
  auto elem = ty.getElementType();
  return arith::ConstantOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getFloatAttr(elem, v)));
}

Value i64Const(OpBuilder &builder, Location loc, int64_t v) {
  auto ty = RankedTensorType::get({}, builder.getI64Type());
  return arith::ConstantOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getI64IntegerAttr(v)));
}

Value broadcastScalar(OpBuilder &builder, Location loc, Value scalar,
                      RankedTensorType targetType) {
  return enzyme::BroadcastOp::create(
      builder, loc, targetType, scalar,
      builder.getDenseI64ArrayAttr(targetType.getShape()));
}

Value scatterPositionToTrace(OpBuilder &builder, Location loc, Value position2d,
                             Value fullTrace, const VIContext &ctx) {
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

Value sumVector1xD(OpBuilder &builder, Location loc, Value v,
                   RankedTensorType positionType) {
  auto scalarType = RankedTensorType::get({}, positionType.getElementType());
  auto ones = scalarConst(builder, loc, positionType, 1.0);
  return impulse::DotOp::create(
      builder, loc, scalarType, v, ones,
      /*lhs_batch=*/builder.getDenseI64ArrayAttr({}),
      /*rhs_batch=*/builder.getDenseI64ArrayAttr({}),
      /*lhs_contract=*/builder.getDenseI64ArrayAttr({0, 1}),
      /*rhs_contract=*/builder.getDenseI64ArrayAttr({0, 1}));
}

Value sumAllElements(OpBuilder &builder, Location loc, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto elemType = tensorType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto ones = arith::ConstantOp::create(
      builder, loc, tensorType,
      DenseElementsAttr::get(tensorType, builder.getFloatAttr(elemType, 1.0)));
  SmallVector<int64_t> allDims;
  for (int64_t i = 0; i < tensorType.getRank(); ++i)
    allDims.push_back(i);
  return impulse::DotOp::create(
      builder, loc, scalarType, tensor, ones, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr(allDims),
      builder.getDenseI64ArrayAttr(allDims));
}

/// Sum as canary: a NaN/Inf gradient element makes its sum non-finite.
Value computeIsFiniteForStableUpdate(OpBuilder &builder, Location loc,
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

/// log N(loc + scale * eps; loc, scale), constant included as in NumPyro.
Value computeAutoNormalLogQ(OpBuilder &builder, Location loc, Value scale,
                            Value eps, const VIContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();

  Value logScale = math::LogOp::create(builder, loc, scale);
  Value sumLogScale = sumVector1xD(builder, loc, logScale, positionType);

  Value epsSq = arith::MulFOp::create(builder, loc, eps, eps);
  Value sumEpsSq = sumVector1xD(builder, loc, epsSq, positionType);
  Value halfSumEpsSq = arith::MulFOp::create(
      builder, loc, sumEpsSq, scalarConst(builder, loc, scalarType, 0.5));

  double log2piHalf = 0.5 * 1.8378770664093453; // = 0.5 * log(2π)
  Value log2piTerm =
      scalarConst(builder, loc, scalarType,
                  static_cast<double>(ctx.positionSize) * log2piHalf);

  Value neg1 = arith::NegFOp::create(builder, loc, sumLogScale);
  Value neg2 = arith::SubFOp::create(builder, loc, neg1, halfSumEpsSq);
  Value logQ = arith::SubFOp::create(builder, loc, neg2, log2piTerm);
  return logQ;
}

struct LogPResult {
  Value logP;
  Value rng;
};

LogPResult callModelGenerate(OpBuilder &builder, Location loc, Value z,
                             Value modelRng, const VIContext &ctx) {
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

struct CholeskyTransformResult {
  Value scaleTril;
  Value sumLogDiag;
};

CholeskyTransformResult applyScaledUnitLowerCholesky(OpBuilder &builder,
                                                     Location loc, Value xFlat,
                                                     int64_t d, Type elemType) {
  int64_t nOff = d * (d - 1) / 2;
  auto offDiagTy = RankedTensorType::get({nOff}, elemType);
  auto diagTy = RankedTensorType::get({d}, elemType);
  auto matTy = RankedTensorType::get({d, d}, elemType);
  auto scalarTy = RankedTensorType::get({}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  // Slice off_diag = x[0:nOff], diag_pre = x[nOff:nOff+d].
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

  // diag = softplus(diag_pre) = log1p(exp(diag_pre)). Shape (d,).
  Value expDP = math::ExpOp::create(builder, loc, diagPre);
  Value diag = math::Log1pOp::create(builder, loc, expDP);

  // log_diag = log(diag); sum_log_diag = sum(log_diag) — scalar.
  Value logDiag = math::LogOp::create(builder, loc, diag);
  Value onesD = arith::ConstantOp::create(
      builder, loc, diagTy,
      DenseElementsAttr::get(diagTy, builder.getFloatAttr(elemType, 1.0)));
  Value sumLogDiag = impulse::DotOp::create(
      builder, loc, scalarTy, logDiag, onesD, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({0}));

  // Build L. Start from zeros, then scatter:
  //   L[i, i] = diag[i]                              (d entries)
  //   L[i, j] = off_diag[k] * diag[i] for j < i      (nOff entries)
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
      // L[i, j] = off_diag[k] * diag[i]
      Value off_k = sliceScalar(offDiag, k);
      Value entry = arith::MulFOp::create(builder, loc, off_k, diag_i);
      Value entry11 = reshapeTo11(entry);
      L = impulse::DynamicUpdateSliceOp::create(
          builder, loc, matTy, L, entry11,
          ValueRange{i64Const(i), i64Const(j)});
      ++k;
    }
    // L[i, i] = diag[i]
    Value diag_i_11 = reshapeTo11(diag_i);
    L = impulse::DynamicUpdateSliceOp::create(
        builder, loc, matTy, L, diag_i_11,
        ValueRange{i64Const(i), i64Const(i)});
  }

  return {L, sumLogDiag};
}

} // namespace

mlir::impulse::VI::SingleParticleELBOResult
mlir::impulse::VI::emitSingleParticleELBO(OpBuilder &builder, Location loc,
                                          Value locParam, Value scaleParam,
                                          Value pkey, const VIContext &ctx) {
  auto rngType = pkey.getType();
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

  Value gradSeed = scalarConst(builder, loc, scalarType, 1.0);

  SmallVector<Value> adInputs{locParam, scaleParam, gradSeed};
  SmallVector<NamedAttribute> adAttrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active)})),
      builder.getNamedAttr(
          "ret_activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_const)})),
  };
  if (ctx.autodiffAttrs) {
    for (auto a : ctx.autodiffAttrs)
      adAttrs.push_back(a);
  }

  auto adOp = enzyme::AutoDiffRegionOp::create(
      builder, loc, TypeRange{scalarType, rngType, positionType, positionType},
      adInputs, adAttrs);

  Block *adBlock = builder.createBlock(&adOp.getBody());
  adBlock->addArgument(positionType, loc);
  adBlock->addArgument(positionType, loc);
  Value locArg = adBlock->getArgument(0);
  Value scaleArg = adBlock->getArgument(1);

  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(adBlock);

    // `scale = softplus(unconstrained_scale)`. Replicate this bijection so
    // gradients flow through softplus automatically.
    Value expU = math::ExpOp::create(builder, loc, scaleArg);
    Value scale = math::Log1pOp::create(builder, loc, expU);

    Value scaleDumped = conditionalDump(
        builder, loc, scale, "VI: scale=softplus(scaleArg)", ctx.debugDump);
    Value locDumped =
        conditionalDump(builder, loc, locArg, "VI: locArg", ctx.debugDump);

    Value scaleEps = arith::MulFOp::create(builder, loc, scaleDumped, eps);
    Value zUnc = arith::AddFOp::create(builder, loc, locDumped, scaleEps);
    zUnc = conditionalDump(builder, loc, zUnc, "VI: zUnc=loc+scale*eps",
                           ctx.debugDump);

    Value zConstrained = constrainPosition(builder, loc, zUnc, ctx.supports);
    zConstrained = conditionalDump(builder, loc, zConstrained,
                                   "VI: zConstrained", ctx.debugDump);

    Value logQUnc = computeAutoNormalLogQ(builder, loc, scaleDumped, eps, ctx);
    logQUnc =
        conditionalDump(builder, loc, logQUnc, "VI: logQUnc", ctx.debugDump);
    Value jacCorrection =
        computeTotalJacobianCorrection(builder, loc, zUnc, ctx.supports);
    jacCorrection = conditionalDump(builder, loc, jacCorrection,
                                    "VI: jacCorrection", ctx.debugDump);
    Value logQ = arith::SubFOp::create(builder, loc, logQUnc, jacCorrection);
    logQ = conditionalDump(builder, loc, logQ, "VI: logQ=logQUnc-jacCorrection",
                           ctx.debugDump);

    LogPResult lp =
        callModelGenerate(builder, loc, zConstrained, modelSeed, ctx);
    Value logP =
        conditionalDump(builder, loc, lp.logP, "VI: logP", ctx.debugDump);

    Value elbo = arith::SubFOp::create(builder, loc, logP, logQ);
    elbo = conditionalDump(builder, loc, elbo, "VI: elbo=logP-logQ",
                           ctx.debugDump);

    enzyme::YieldOp::create(builder, loc, ValueRange{elbo, lp.rng});
  }

  builder.setInsertionPointAfter(adOp);

  Value elbo = adOp.getResult(0);
  Value modelRngAfter = adOp.getResult(1);
  Value gradLoc = adOp.getResult(2);
  Value gradScale = adOp.getResult(3);

  return {elbo, gradLoc, gradScale, modelRngAfter};
}

mlir::impulse::VI::SingleParticleAutoDeltaResult
mlir::impulse::VI::emitSingleParticleAutoDelta(OpBuilder &builder, Location loc,
                                               Value locParam, Value pkey,
                                               const VIContext &ctx) {
  auto rngType = pkey.getType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();

  // NumPyro order: model_seed, guide_seed = split(pkey).
  auto seedSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rngType, rngType}, pkey);
  Value modelSeed = seedSplit.getResult(0);
  Value guideSeed = seedSplit.getResult(1);

  Value gradSeed = scalarConst(builder, loc, scalarType, 1.0);

  SmallVector<Value> adInputs{locParam, gradSeed};
  SmallVector<NamedAttribute> adAttrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr({enzyme::ActivityAttr::get(
              builder.getContext(), enzyme::Activity::enzyme_active)})),
      builder.getNamedAttr(
          "ret_activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_const)})),
  };
  if (ctx.autodiffAttrs) {
    for (auto a : ctx.autodiffAttrs)
      adAttrs.push_back(a);
  }

  auto adOp = enzyme::AutoDiffRegionOp::create(
      builder, loc, TypeRange{scalarType, rngType, positionType}, adInputs,
      adAttrs);

  Block *adBlock = builder.createBlock(&adOp.getBody());
  adBlock->addArgument(positionType, loc);
  Value locArg = adBlock->getArgument(0);

  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(adBlock);

    // No Jacobian term: NumPyro's AutoDelta has log_density = 0.
    Value zConstrained = constrainPosition(builder, loc, locArg, ctx.supports);

    LogPResult lp =
        callModelGenerate(builder, loc, zConstrained, modelSeed, ctx);

    enzyme::YieldOp::create(builder, loc, ValueRange{lp.logP, lp.rng});
  }

  builder.setInsertionPointAfter(adOp);

  return {adOp.getResult(0), adOp.getResult(2), adOp.getResult(1)};
}

mlir::impulse::VI::SingleParticleMVNResult
mlir::impulse::VI::emitSingleParticleMVNELBO(OpBuilder &builder, Location loc,
                                             Value locParam, Value xScaleTril,
                                             Value pkey, const VIContext &ctx) {
  auto rngType = pkey.getType();
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
  (void)randOp.getOutputRngState();

  Value gradSeed = scalarConst(builder, loc, scalarType, 1.0);
  SmallVector<Value> adInputs{locParam, xScaleTril, gradSeed};
  SmallVector<NamedAttribute> adAttrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active)})),
      builder.getNamedAttr(
          "ret_activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_const)})),
  };
  if (ctx.autodiffAttrs) {
    for (auto a : ctx.autodiffAttrs)
      adAttrs.push_back(a);
  }

  auto adOp = enzyme::AutoDiffRegionOp::create(
      builder, loc,
      TypeRange{scalarType, rngType, positionType, xScaleTril.getType()},
      adInputs, adAttrs);

  Block *adBlock = builder.createBlock(&adOp.getBody());
  adBlock->addArgument(positionType, loc);
  adBlock->addArgument(xScaleTril.getType(), loc);
  Value locArg = adBlock->getArgument(0);
  Value xArg = adBlock->getArgument(1);

  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(adBlock);

    auto chol = applyScaledUnitLowerCholesky(builder, loc, xArg, d, elemType);
    Value scaleTril = chol.scaleTril;
    Value sumLogDiag = chol.sumLogDiag;

    // L @ eps in row-vector form: eps @ L^T.
    Value Leps = impulse::DotOp::create(
        builder, loc, positionType, eps, scaleTril,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({1}));
    Value z = arith::AddFOp::create(builder, loc, locArg, Leps);

    Value epsSq = arith::MulFOp::create(builder, loc, eps, eps);
    Value sumEpsSq = sumVector1xD(builder, loc, epsSq, positionType);
    Value halfSumEpsSq = arith::MulFOp::create(
        builder, loc, sumEpsSq, scalarConst(builder, loc, scalarType, 0.5));
    double log2piHalf = 0.5 * 1.8378770664093453;
    Value log2piTerm = scalarConst(builder, loc, scalarType,
                                   static_cast<double>(d) * log2piHalf);
    Value negSumLogDiag = arith::NegFOp::create(builder, loc, sumLogDiag);
    Value logQPart =
        arith::SubFOp::create(builder, loc, negSumLogDiag, halfSumEpsSq);
    Value logQ = arith::SubFOp::create(builder, loc, logQPart, log2piTerm);

    LogPResult lp = callModelGenerate(builder, loc, z, modelSeed, ctx);

    Value elbo = arith::SubFOp::create(builder, loc, lp.logP, logQ);
    enzyme::YieldOp::create(builder, loc, ValueRange{elbo, lp.rng});
  }

  builder.setInsertionPointAfter(adOp);

  return {adOp.getResult(0), adOp.getResult(2), adOp.getResult(3),
          adOp.getResult(1)};
}

mlir::impulse::VI::MultiParticleELBOResult
mlir::impulse::VI::emitELBOStep(OpBuilder &builder, Location loc,
                                Value locParam, Value scaleParam, Value rngStep,
                                const VIContext &ctx) {
  int64_t K = ctx.numParticles;
  if (K == 1) {
    auto r = emitSingleParticleELBO(builder, loc, locParam, scaleParam, rngStep,
                                    ctx);
    return {r.elbo, r.gradLoc, r.gradScale};
  }

  auto rngType = rngStep.getType();
  SmallVector<Type> splitResultTypes(K, rngType);
  auto splitK =
      impulse::RandomSplitOp::create(builder, loc, splitResultTypes, rngStep);

  auto scalarType = ctx.getScalarType();
  auto positionType = ctx.getPositionType();

  Value elboSum = scalarConst(builder, loc, scalarType, 0.0);
  Value gradLocSum = scalarConst(builder, loc, positionType, 0.0);
  Value gradScaleSum = scalarConst(builder, loc, positionType, 0.0);

  for (int64_t k = 0; k < K; ++k) {
    Value pkey = splitK.getResult(k);
    auto r =
        emitSingleParticleELBO(builder, loc, locParam, scaleParam, pkey, ctx);
    elboSum = arith::AddFOp::create(builder, loc, elboSum, r.elbo);
    gradLocSum = arith::AddFOp::create(builder, loc, gradLocSum, r.gradLoc);
    gradScaleSum =
        arith::AddFOp::create(builder, loc, gradScaleSum, r.gradScale);
  }

  auto Kf = scalarConst(builder, loc, scalarType, static_cast<double>(K));
  Value KfBcast = broadcastScalar(builder, loc, Kf, positionType);
  Value elboMean = arith::DivFOp::create(builder, loc, elboSum, Kf);
  Value gradLocMean = arith::DivFOp::create(builder, loc, gradLocSum, KfBcast);
  Value gradScaleMean =
      arith::DivFOp::create(builder, loc, gradScaleSum, KfBcast);

  return {elboMean, gradLocMean, gradScaleMean};
}

Value mlir::impulse::VI::applySGDUpdate(OpBuilder &builder, Location loc,
                                        Value param, Value gradElbo, Value lr) {
  auto ty = cast<RankedTensorType>(param.getType());
  Value lrBcast = broadcastScalar(builder, loc, lr, ty);
  Value scaledGrad = arith::MulFOp::create(builder, loc, lrBcast, gradElbo);
  return arith::AddFOp::create(builder, loc, param, scaledGrad);
}

mlir::impulse::VI::AdamStepResult
mlir::impulse::VI::applyAdamUpdate(OpBuilder &builder, Location loc,
                                   Value param, Value gradElbo, Value m,
                                   Value v, Value stepIdx, Value lr, double b1,
                                   double b2, double eps, double clipNorm) {
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
  Value one_i = arith::ConstantOp::create(
      builder, loc, i64Ty,
      DenseElementsAttr::get(i64Ty, builder.getI64IntegerAttr(1)));
  Value stepPlus1 = arith::AddIOp::create(builder, loc, stepIdx, one_i);
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

mlir::impulse::VI::GenericOptimStepResult
mlir::impulse::VI::applyMomentumUpdate(OpBuilder &builder, Location loc,
                                       Value param, Value gradElbo,
                                       Value velocity, Value lr, double mass) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);
  Value massBcast = broadcastScalar(
      builder, loc, scalarConst(builder, loc, scalarType, mass), paramType);
  Value vScaled = arith::MulFOp::create(builder, loc, massBcast, velocity);
  Value vNew = arith::AddFOp::create(builder, loc, vScaled, g);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value scaledV = arith::MulFOp::create(builder, loc, lrBcast, vNew);
  Value paramNew = arith::SubFOp::create(builder, loc, param, scaledV);
  Value zeroSlot = scalarConst(builder, loc, paramType, 0.0);
  return {paramNew, vNew, zeroSlot};
}

mlir::impulse::VI::GenericOptimStepResult mlir::impulse::VI::applyRMSPropUpdate(
    OpBuilder &builder, Location loc, Value param, Value gradElbo,
    Value avgSqGrad, Value lr, double gamma, double eps) {
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
  Value gammaAvg = arith::MulFOp::create(builder, loc, gammaBcast, avgSqGrad);
  Value avgNew =
      arith::AddFOp::create(builder, loc, gammaAvg, oneMinusGammaGSq);

  Value avgPlusEps = arith::AddFOp::create(builder, loc, avgNew, epsBcast);
  Value sqrtAvgPlusEps = math::SqrtOp::create(builder, loc, avgPlusEps);
  Value gOverSqrt = arith::DivFOp::create(builder, loc, g, sqrtAvgPlusEps);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value update = arith::MulFOp::create(builder, loc, lrBcast, gOverSqrt);
  Value paramNew = arith::SubFOp::create(builder, loc, param, update);

  Value zeroSlot = scalarConst(builder, loc, paramType, 0.0);
  return {paramNew, avgNew, zeroSlot};
}

mlir::impulse::VI::GenericOptimStepResult
mlir::impulse::VI::applyAdagradUpdate(OpBuilder &builder, Location loc,
                                      Value param, Value gradElbo, Value gSq,
                                      Value m, Value lr, double momentum) {
  auto paramType = cast<RankedTensorType>(param.getType());
  auto elemType = paramType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value g = arith::NegFOp::create(builder, loc, gradElbo);

  Value gSquared = arith::MulFOp::create(builder, loc, g, g);
  Value gSqNew = arith::AddFOp::create(builder, loc, gSq, gSquared);

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
  Value momTimesM = arith::MulFOp::create(builder, loc, momBcast, m);
  Value mNew = arith::AddFOp::create(builder, loc, oneMinusMomDx, momTimesM);

  Value paramNew = arith::AddFOp::create(builder, loc, param, mNew);

  return {paramNew, gSqNew, mNew};
}

mlir::impulse::VI::GenericOptimStepResult
mlir::impulse::VI::applyRMSPropMomentumUpdate(OpBuilder &builder, Location loc,
                                              Value param, Value gradElbo,
                                              Value avgSqGrad, Value mom,
                                              Value lr, double gamma,
                                              double eps, double momentumMass) {
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
  Value gammaAvg = arith::MulFOp::create(builder, loc, gammaBcast, avgSqGrad);
  Value avgNew =
      arith::AddFOp::create(builder, loc, gammaAvg, oneMinusGammaGSq);

  Value avgPlusEps = arith::AddFOp::create(builder, loc, avgNew, epsBcast);
  Value sqrtAvgPlusEps = math::SqrtOp::create(builder, loc, avgPlusEps);
  Value gOverSqrt = arith::DivFOp::create(builder, loc, g, sqrtAvgPlusEps);
  Value lrBcast = broadcastScalar(builder, loc, lr, paramType);
  Value lrGOverSqrt = arith::MulFOp::create(builder, loc, lrBcast, gOverSqrt);
  Value momTimesMom = arith::MulFOp::create(builder, loc, momBcast, mom);
  Value momNew = arith::AddFOp::create(builder, loc, momTimesMom, lrGOverSqrt);

  Value paramNew = arith::SubFOp::create(builder, loc, param, momNew);
  return {paramNew, avgNew, momNew};
}

mlir::impulse::VI::SVILoopResult
mlir::impulse::VI::emitSVILoop(OpBuilder &builder, Location loc, Value loc0,
                               Value scale0, Value rng0, const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto rngType = rng0.getType();
  auto traceBufType =
      RankedTensorType::get({ctx.numSamples, ctx.positionSize}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSamples}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  double clipNorm =
      (ctx.optimizerKind == OptimizerKind::CLIPPED_ADAM) ? ctx.clipNorm : 0.0;

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

    auto elboRes = emitELBOStep(builder, loc, locIter, scaleIter, rngStep, ctx);

    Value effectiveStep =
        arith::AddIOp::create(builder, loc, iter, initStepCountI64);

    auto runOpt = [&](Value param, Value gradElbo, Value m,
                      Value v) -> std::tuple<Value, Value, Value> {
      switch (ctx.optimizerKind) {
      case OptimizerKind::ADAM:
      case OptimizerKind::CLIPPED_ADAM: {
        auto r = applyAdamUpdate(builder, loc, param, gradElbo, m, v,
                                 effectiveStep, ctx.stepSize, ctx.adamB1,
                                 ctx.adamB2, ctx.adamEps, clipNorm);
        return {r.paramNew, r.mNew, r.vNew};
      }
      case OptimizerKind::MOMENTUM: {
        auto r = applyMomentumUpdate(builder, loc, param, gradElbo, m,
                                     ctx.stepSize, ctx.momentumMass);
        return {r.paramNew, r.slot1New, v};
      }
      case OptimizerKind::RMSPROP: {
        auto r =
            applyRMSPropUpdate(builder, loc, param, gradElbo, m, ctx.stepSize,
                               ctx.rmspropGamma, ctx.rmspropEps);
        return {r.paramNew, r.slot1New, v};
      }
      case OptimizerKind::ADAGRAD: {
        auto r = applyAdagradUpdate(builder, loc, param, gradElbo, m, v,
                                    ctx.stepSize, ctx.momentumMass);
        return {r.paramNew, r.slot1New, r.slot2New};
      }
      case OptimizerKind::RMSPROP_MOMENTUM: {
        auto r = applyRMSPropMomentumUpdate(builder, loc, param, gradElbo, m, v,
                                            ctx.stepSize, ctx.rmspropGamma,
                                            ctx.rmspropEps, ctx.momentumMass);
        return {r.paramNew, r.slot1New, r.slot2New};
      }
      case OptimizerKind::SGD:
      default: {
        Value pNew =
            applySGDUpdate(builder, loc, param, gradElbo, ctx.stepSize);
        return {pNew, m, v};
      }
      }
    };

    Value locNew, scaleNew, mLocNew, vLocNew, mScaleNew, vScaleNew;
    std::tie(locNew, mLocNew, vLocNew) =
        runOpt(locIter, elboRes.gradLoc, mLocIter, vLocIter);
    std::tie(scaleNew, mScaleNew, vScaleNew) =
        runOpt(scaleIter, elboRes.gradScale, mScaleIter, vScaleIter);

    // A skipped step still records its ELBO, as numpyro's stable_update does.
    if (ctx.stableUpdate) {
      Value isFinite = computeIsFiniteForStableUpdate(
          builder, loc, elboRes.elbo, elboRes.gradLoc, elboRes.gradScale);
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

mlir::impulse::VI::SVILoopResult
mlir::impulse::VI::emitSVILoopAutoDelta(OpBuilder &builder, Location loc,
                                        Value loc0, Value rng0,
                                        const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto rngType = rng0.getType();
  auto traceBufType =
      RankedTensorType::get({ctx.numSamples, ctx.positionSize}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSamples}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  double clipNorm =
      (ctx.optimizerKind == OptimizerKind::CLIPPED_ADAM) ? ctx.clipNorm : 0.0;

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

    // Particles are identical; the K-way split keeps NumPyro's RNG stream.
    Value elboMean, gradLocMean;
    int64_t K = ctx.numParticles;
    if (K == 1) {
      auto r = emitSingleParticleAutoDelta(builder, loc, locIter, rngStep, ctx);
      elboMean = r.elbo;
      gradLocMean = r.gradLoc;
    } else {
      SmallVector<Type> splitResultTypes(K, rngType);
      auto splitK = impulse::RandomSplitOp::create(builder, loc,
                                                   splitResultTypes, rngStep);
      Value elboSum = scalarConst(builder, loc, scalarType, 0.0);
      Value gradLocSum = scalarConst(builder, loc, positionType, 0.0);
      for (int64_t k = 0; k < K; ++k) {
        Value pkey = splitK.getResult(k);
        auto r = emitSingleParticleAutoDelta(builder, loc, locIter, pkey, ctx);
        elboSum = arith::AddFOp::create(builder, loc, elboSum, r.elbo);
        gradLocSum = arith::AddFOp::create(builder, loc, gradLocSum, r.gradLoc);
      }
      auto Kf = scalarConst(builder, loc, scalarType, static_cast<double>(K));
      Value KfBcast = broadcastScalar(builder, loc, Kf, positionType);
      elboMean = arith::DivFOp::create(builder, loc, elboSum, Kf);
      gradLocMean = arith::DivFOp::create(builder, loc, gradLocSum, KfBcast);
    }

    Value effectiveStep =
        arith::AddIOp::create(builder, loc, iter, initStepCountI64);

    auto runOpt = [&](Value param, Value gradElbo, Value m,
                      Value v) -> std::tuple<Value, Value, Value> {
      switch (ctx.optimizerKind) {
      case OptimizerKind::ADAM:
      case OptimizerKind::CLIPPED_ADAM: {
        auto r = applyAdamUpdate(builder, loc, param, gradElbo, m, v,
                                 effectiveStep, ctx.stepSize, ctx.adamB1,
                                 ctx.adamB2, ctx.adamEps, clipNorm);
        return {r.paramNew, r.mNew, r.vNew};
      }
      case OptimizerKind::MOMENTUM: {
        auto r = applyMomentumUpdate(builder, loc, param, gradElbo, m,
                                     ctx.stepSize, ctx.momentumMass);
        return {r.paramNew, r.slot1New, v};
      }
      case OptimizerKind::RMSPROP: {
        auto r =
            applyRMSPropUpdate(builder, loc, param, gradElbo, m, ctx.stepSize,
                               ctx.rmspropGamma, ctx.rmspropEps);
        return {r.paramNew, r.slot1New, v};
      }
      case OptimizerKind::ADAGRAD: {
        auto r = applyAdagradUpdate(builder, loc, param, gradElbo, m, v,
                                    ctx.stepSize, ctx.momentumMass);
        return {r.paramNew, r.slot1New, r.slot2New};
      }
      case OptimizerKind::RMSPROP_MOMENTUM: {
        auto r = applyRMSPropMomentumUpdate(builder, loc, param, gradElbo, m, v,
                                            ctx.stepSize, ctx.rmspropGamma,
                                            ctx.rmspropEps, ctx.momentumMass);
        return {r.paramNew, r.slot1New, r.slot2New};
      }
      case OptimizerKind::SGD:
      default: {
        Value pNew =
            applySGDUpdate(builder, loc, param, gradElbo, ctx.stepSize);
        return {pNew, m, v};
      }
      }
    };

    Value locNew, mLocNew, vLocNew;
    std::tie(locNew, mLocNew, vLocNew) =
        runOpt(locIter, gradLocMean, mLocIter, vLocIter);

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

mlir::impulse::VI::SVILoopResult
mlir::impulse::VI::emitSVILoopMVN(OpBuilder &builder, Location loc, Value loc0,
                                  Value xScaleTril0, Value rng0,
                                  const VIContext &ctx) {
  auto elemType = ctx.getElementType();
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto rngType = rng0.getType();
  int64_t d = ctx.positionSize;
  int64_t flatSize = d * (d + 1) / 2;
  auto flatXTy = RankedTensorType::get({flatSize}, elemType);
  auto traceBufType = RankedTensorType::get({ctx.numSamples, d}, elemType);
  auto elboBufType = RankedTensorType::get({ctx.numSamples}, elemType);
  auto i64T = RankedTensorType::get({}, builder.getI64Type());

  double clipNorm =
      (ctx.optimizerKind == OptimizerKind::CLIPPED_ADAM) ? ctx.clipNorm : 0.0;

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

    int64_t K = ctx.numParticles;
    Value elbo, gradLoc, gradX;
    if (K == 1) {
      auto r =
          emitSingleParticleMVNELBO(builder, loc, locIter, xIter, rngStep, ctx);
      elbo = r.elbo;
      gradLoc = r.gradLoc;
      gradX = r.gradScaleTril;
    } else {
      SmallVector<Type> splitTypes(K, rngType);
      auto splitK =
          impulse::RandomSplitOp::create(builder, loc, splitTypes, rngStep);
      Value elboSum = scalarConst(builder, loc, scalarType, 0.0);
      Value gradLocSum = scalarConst(builder, loc, positionType, 0.0);
      Value gradXSum = scalarConst(builder, loc, flatXTy, 0.0);
      for (int64_t k = 0; k < K; ++k) {
        Value pkey = splitK.getResult(k);
        auto r =
            emitSingleParticleMVNELBO(builder, loc, locIter, xIter, pkey, ctx);
        elboSum = arith::AddFOp::create(builder, loc, elboSum, r.elbo);
        gradLocSum = arith::AddFOp::create(builder, loc, gradLocSum, r.gradLoc);
        gradXSum =
            arith::AddFOp::create(builder, loc, gradXSum, r.gradScaleTril);
      }
      auto Kf = scalarConst(builder, loc, scalarType, static_cast<double>(K));
      Value KfBcastP = broadcastScalar(builder, loc, Kf, positionType);
      Value KfBcastX = broadcastScalar(builder, loc, Kf, flatXTy);
      elbo = arith::DivFOp::create(builder, loc, elboSum, Kf);
      gradLoc = arith::DivFOp::create(builder, loc, gradLocSum, KfBcastP);
      gradX = arith::DivFOp::create(builder, loc, gradXSum, KfBcastX);
    }

    Value effectiveStep =
        arith::AddIOp::create(builder, loc, iter, initStepCountI64);

    auto runOpt = [&](Value param, Value gradElbo, Value m,
                      Value v) -> std::tuple<Value, Value, Value> {
      switch (ctx.optimizerKind) {
      case OptimizerKind::ADAM:
      case OptimizerKind::CLIPPED_ADAM: {
        auto r = applyAdamUpdate(builder, loc, param, gradElbo, m, v,
                                 effectiveStep, ctx.stepSize, ctx.adamB1,
                                 ctx.adamB2, ctx.adamEps, clipNorm);
        return {r.paramNew, r.mNew, r.vNew};
      }
      case OptimizerKind::MOMENTUM: {
        auto r = applyMomentumUpdate(builder, loc, param, gradElbo, m,
                                     ctx.stepSize, ctx.momentumMass);
        return {r.paramNew, r.slot1New, v};
      }
      case OptimizerKind::RMSPROP: {
        auto r =
            applyRMSPropUpdate(builder, loc, param, gradElbo, m, ctx.stepSize,
                               ctx.rmspropGamma, ctx.rmspropEps);
        return {r.paramNew, r.slot1New, v};
      }
      case OptimizerKind::ADAGRAD: {
        auto r = applyAdagradUpdate(builder, loc, param, gradElbo, m, v,
                                    ctx.stepSize, ctx.momentumMass);
        return {r.paramNew, r.slot1New, r.slot2New};
      }
      case OptimizerKind::RMSPROP_MOMENTUM: {
        auto r = applyRMSPropMomentumUpdate(builder, loc, param, gradElbo, m, v,
                                            ctx.stepSize, ctx.rmspropGamma,
                                            ctx.rmspropEps, ctx.momentumMass);
        return {r.paramNew, r.slot1New, r.slot2New};
      }
      case OptimizerKind::SGD:
      default: {
        Value pNew =
            applySGDUpdate(builder, loc, param, gradElbo, ctx.stepSize);
        return {pNew, m, v};
      }
      }
    };

    Value locNew, xNew, mLocNew, vLocNew, mXNew, vXNew;
    std::tie(locNew, mLocNew, vLocNew) =
        runOpt(locIter, gradLoc, mLocIter, vLocIter);
    std::tie(xNew, mXNew, vXNew) = runOpt(xIter, gradX, mXIter, vXIter);

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

LogicalResult mlir::impulse::VI::lowerVI(PatternRewriter &rewriter, InferOp op,
                                         VIContext ctx, bool debugDump) {
  auto loc = op.getLoc();

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
  auto rngType = rngInput.getType();
  Value rng0;
  Value guideInitSeed;
  if (isResume) {
    rng0 = rngInput;
    guideInitSeed = Value();
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
      Value expS = math::ExpOp::create(rewriter, loc, initScaleConstrained);
      Value oneBcast = scalarConst(rewriter, loc, positionType, 1.0);
      Value expSMinus1 = arith::SubFOp::create(rewriter, loc, expS, oneBcast);
      scale0 = math::LogOp::create(rewriter, loc, expSMinus1);
    } else {
      double initScaleU = std::log(std::expm1(ctx.initScale));
      scale0 = scalarConst(rewriter, loc, positionType, initScaleU);
    }
    loopRes = emitSVILoop(rewriter, loc, loc0, scale0, rng0, ctx);
  } else {
    // scale_tril = init_scale * I: zero off-diagonals, softplus_inv diagonal.
    guide = impulse::VIGuide::AutoMVN;
    auto elemTy = ctx.getElementType();
    int64_t d = ctx.positionSize;
    int64_t flatSize = d * (d + 1) / 2;
    int64_t nOff = d * (d - 1) / 2;
    auto flatTy = RankedTensorType::get({flatSize}, elemTy);
    Value xScaleTril0;
    if (isResume) {
      xScaleTril0 = slotIn("scale_tril_unconstrained");
    } else {
      double initU = std::log(std::expm1(ctx.initScale));
      SmallVector<Attribute> vals;
      vals.reserve(flatSize);
      for (int64_t i = 0; i < nOff; ++i)
        vals.push_back(rewriter.getFloatAttr(elemTy, 0.0));
      for (int64_t i = 0; i < d; ++i)
        vals.push_back(rewriter.getFloatAttr(elemTy, initU));
      xScaleTril0 = arith::ConstantOp::create(
          rewriter, loc, flatTy, DenseElementsAttr::get(flatTy, vals));
    }
    loopRes = emitSVILoopMVN(rewriter, loc, loc0, xScaleTril0, rng0, ctx);
  }

  //   trace, diagnostics, output_rng_state, then the VI state schema slots.
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
  results.push_back(loopRes.rngFinal);
  for (const impulse::StateSlot &slot :
       impulse::buildVIStateSchema(guide, ctx.positionSize, elemType))
    results.push_back(slotValue(slot.name));

  rewriter.replaceOp(op, results);
  return success();
}
