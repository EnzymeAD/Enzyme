//===- HMCUtils.cpp - Utilities for HMC/NUTS inference -------* C++ -*-===//
//
// This file implements utility functions for Hamiltonian Monte Carlo (HMC) and
// No-U-Turn Sampler (NUTS) implementations.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/hmc_util.py
//
//===----------------------------------------------------------------------===//

#include "HMCUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Interfaces/ImpulseUtils.h"
#include "Interfaces/TransformUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

SmallVector<Value> NUTSTreeState::toValues() const {
  return {q_left,        p_left,        grad_left,
          q_right,       p_right,       grad_right,
          q_proposal,    grad_proposal, U_proposal,
          H_proposal,    depth,         weight,
          turning,       diverging,     sum_accept_probs,
          num_proposals, p_sum,         rng};
}

NUTSTreeState NUTSTreeState::fromValues(ArrayRef<Value> values) {
  assert(values.size() == 18 && "Expected 18 NUTSTreeState fields");
  return {.q_left = values[0],
          .p_left = values[1],
          .grad_left = values[2],
          .q_right = values[3],
          .p_right = values[4],
          .grad_right = values[5],
          .q_proposal = values[6],
          .grad_proposal = values[7],
          .U_proposal = values[8],
          .H_proposal = values[9],
          .depth = values[10],
          .weight = values[11],
          .turning = values[12],
          .diverging = values[13],
          .sum_accept_probs = values[14],
          .num_proposals = values[15],
          .p_sum = values[16],
          .rng = values[17]};
}

SmallVector<Type> NUTSTreeState::getTypes() const {
  SmallVector<Type> types;
  for (auto val : toValues())
    types.push_back(val.getType());
  return types;
}

Value impulse::conditionalDump(OpBuilder &builder, Location loc, Value value,
                               StringRef label, bool debugDump) {
  if (debugDump) {
    return enzyme::DumpOp::create(builder, loc, value.getType(), value,
                                  builder.getStringAttr(label))
        .getOutput();
  }
  return value;
}

/// Whether the HMC kernels treat a mass tensor of type `t` as a diagonal:
/// `tensor<N>` for every N and `tensor<1xN>` for N > 1. `tensor<1x1>` is a
/// dense 1x1 matrix.
static bool isDiagonalMass(RankedTensorType t) {
  return t.getRank() == 1 ||
         (t.getRank() == 2 && t.getShape()[0] == 1 && t.getShape()[1] != 1);
}

/// Creates a 2D identity matrix of the specified type.
static Value createIdentityMatrix(OpBuilder &builder, Location loc,
                                  RankedTensorType matrixType) {
  return emitIndicatorMatrix(builder, loc, matrixType,
                             [](int64_t i, int64_t j) { return i == j; });
}

/// Computes A[::-1, ::-1] using permutation matrix through P @ A @ P.
static Value reverseRowsAndColumns(OpBuilder &builder, Location loc,
                                   Value matrix) {
  auto matrixType = cast<RankedTensorType>(matrix.getType());
  int64_t size = matrixType.getShape()[0];
  Value permutation = emitIndicatorMatrix(
      builder, loc, matrixType,
      [size](int64_t i, int64_t j) { return j == size - 1 - i; });

  // PA = P @ A
  auto rowsReversed = impulse::DotOp::create(
      builder, loc, matrixType, permutation, matrix,
      builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({0}));

  // PAP = PA @ P
  return impulse::DotOp::create(builder, loc, matrixType, rowsReversed,
                                permutation, builder.getDenseI64ArrayAttr({1}),
                                builder.getDenseI64ArrayAttr({0}));
}

Value impulse::applyInverseMassMatrix(OpBuilder &builder, Location loc,
                                      Value invMass, Value momentum,
                                      RankedTensorType positionType) {
  if (!invMass) {
    return momentum;
  }

  auto invMassType = cast<RankedTensorType>(invMass.getType());

  if (isDiagonalMass(invMassType)) {
    // Diagonal: element-wise
    auto diagMass =
        impulse::ReshapeOp::create(builder, loc, positionType, invMass);
    return arith::MulFOp::create(builder, loc, diagMass, momentum);
  } else if (invMassType.getRank() == 2) {
    // Dense: v = invMass @ p
    return impulse::DotOp::create(builder, loc, positionType, momentum, invMass,
                                  builder.getDenseI64ArrayAttr({1}),
                                  builder.getDenseI64ArrayAttr({0}));
  }

  emitError(loc, "ProbProg: Provided invMass must have rank 1 or 2, got rank " +
                     std::to_string(invMassType.getRank()));
  return nullptr;
}

Value impulse::computeKineticEnergy(OpBuilder &builder, Location loc,
                                    Value momentum, Value invMass,
                                    RankedTensorType positionType) {
  auto elemType = positionType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value halfConst = emitSplatConstant(builder, loc, scalarType, 0.5);

  // v = M^-1 @ p
  auto v =
      applyInverseMassMatrix(builder, loc, invMass, momentum, positionType);

  // K = 0.5 * p^T @ v
  // For 2D tensors [1, N], contract over both dimensions to get scalar
  auto pDotV = impulse::DotOp::create(builder, loc, scalarType, momentum, v,
                                      builder.getDenseI64ArrayAttr({0, 1}),
                                      builder.getDenseI64ArrayAttr({0, 1}));

  return arith::MulFOp::create(builder, loc, halfConst, pDotV);
}

Value impulse::computeMassMatrixSqrt(OpBuilder &builder, Location loc,
                                     Value invMass,
                                     RankedTensorType positionType) {
  if (!invMass) {
    return Value();
  }

  auto invMassType = cast<RankedTensorType>(invMass.getType());

  if (isDiagonalMass(invMassType)) {
    // Diagonal: mass_matrix_sqrt = 1/sqrt(invMass)
    auto sqrtInvMass = math::SqrtOp::create(builder, loc, invMass);
    Value onesVector = emitSplatConstant(builder, loc, invMassType, 1.0);
    return arith::DivFOp::create(builder, loc, onesVector, sqrtInvMass);
  } else {
    // Dense: mass_matrix_sqrt = M^{1/2}
    // TODO: improve
    // Reference:
    // https://github.com/pyro-ppl/numpyro/blob/6a9cb9a530fe53897edb6c472368e58965b034e4/numpyro/infer/hmc_util.py#L499
    auto reversedInvMass = reverseRowsAndColumns(builder, loc, invMass);
    auto L_reversed =
        impulse::CholeskyOp::create(builder, loc, invMassType, reversedInvMass,
                                    /*lower=*/builder.getBoolAttr(true));
    auto massMatrixSqrtInvT = reverseRowsAndColumns(builder, loc, L_reversed);
    auto identityMatrix = createIdentityMatrix(builder, loc, invMassType);
    auto massMatrixSqrt = impulse::TriangularSolveOp::create(
        builder, loc, invMassType, massMatrixSqrtInvT, identityMatrix,
        /*left_side=*/builder.getBoolAttr(true),
        /*lower=*/builder.getBoolAttr(false),
        /*unit_diagonal=*/builder.getBoolAttr(false),
        /*transpose_a=*/
        impulse::TransposeAttr::get(builder.getContext(),
                                    impulse::Transpose::TRANSPOSE));

    return massMatrixSqrt;
  }
}

Value impulse::sampleMomentum(OpBuilder &builder, Location loc, Value rng,
                              Value invMass, Value massMatrixSqrt,
                              RankedTensorType positionType, bool debugDump) {
  auto elemType = positionType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);

  rng = conditionalDump(builder, loc, rng, "sampleMomentum: input rng state",
                        debugDump);

  // Sample eps ~ N(0, I)
  auto randomOp = impulse::RandomOp::create(
      builder, loc, TypeRange{rng.getType(), positionType}, rng, zeroConst,
      oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::NORMAL));

  auto eps = randomOp.getResult();

  if (!massMatrixSqrt) {
    return eps;
  }

  auto massMatrixSqrtType = cast<RankedTensorType>(massMatrixSqrt.getType());

  if (isDiagonalMass(massMatrixSqrtType)) {
    // Diagonal: p = massMatrixSqrt * eps (element-wise)
    auto diagSqrt =
        impulse::ReshapeOp::create(builder, loc, positionType, massMatrixSqrt);
    return arith::MulFOp::create(builder, loc, diagSqrt, eps);
  } else {
    // Dense: p = massMatrixSqrt @ eps
    return impulse::DotOp::create(
        builder, loc, positionType, eps, massMatrixSqrt,
        builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({1}));
  }
}

/// Copies the columns of every site in `supports` from row 0 of `src` into
/// row 0 of `dst`, from position to trace columns if `toTrace`, else back.
/// `zeroIndex` is a rank-0 `i64` zero.
static Value copySiteColumns(OpBuilder &builder, Location loc, Value src,
                             Value dst, Value zeroIndex,
                             ArrayRef<SupportInfo> supports, bool toTrace) {
  Type elemType = cast<RankedTensorType>(src.getType()).getElementType();
  Value result = dst;
  for (const SupportInfo &info : supports) {
    int64_t srcColumn = toTrace ? info.constrainedOffset : info.traceOffset;
    int64_t dstColumn = toTrace ? info.traceOffset : info.constrainedOffset;
    auto sliceType = RankedTensorType::get({1, info.constrainedSize}, elemType);
    Value srcIndex = emitI64ScalarConstant(builder, loc, srcColumn);
    Value slice = impulse::DynamicSliceOp::create(
        builder, loc, sliceType, src, ValueRange{zeroIndex, srcIndex},
        builder.getDenseI64ArrayAttr({1, info.constrainedSize}));
    Value dstIndex = emitI64ScalarConstant(builder, loc, dstColumn);
    result = impulse::DynamicUpdateSliceOp::create(
        builder, loc, dst.getType(), result, slice,
        ValueRange{zeroIndex, dstIndex});
  }
  return result;
}

static Value gatherPositionFromTrace(OpBuilder &builder, Location loc,
                                     Value fullTrace, const HMCContext &ctx) {
  auto positionType2d = RankedTensorType::get(
      {1, ctx.getConstrainedPositionSize()}, ctx.elementType);
  Value zeroIndex = emitI64ScalarConstant(builder, loc, 0);
  Value position = emitSplatConstant(builder, loc, positionType2d, 0.0);
  return copySiteColumns(builder, loc, fullTrace, position, zeroIndex,
                         ctx.supports, /*toTrace=*/false);
}

/// Writes `qConstrained` into the original trace and emits a generate op of
/// the model on that trace.
static GenerateOp emitGenerateAt(OpBuilder &builder, Location loc,
                                 Value qConstrained, Value rng,
                                 const ModelContext &ctx) {
  Value zeroIndex = emitI64ScalarConstant(builder, loc, 0);
  Value fullTrace =
      copySiteColumns(builder, loc, qConstrained, ctx.originalTrace, zeroIndex,
                      ctx.supports, /*toTrace=*/true);
  SmallVector<Value> inputs{rng};
  llvm::append_range(inputs, ctx.fnInputs);
  SmallVector<Type> resultTypes{
      RankedTensorType::get({1, ctx.getFullTraceSize()}, ctx.elementType),
      ctx.getScalarType()};
  llvm::append_range(resultTypes, ctx.fnResultTypes);
  return impulse::GenerateOp::create(
      builder, loc, resultTypes, ctx.fn, inputs, fullTrace, ctx.allAddresses,
      ctx.allAddresses, builder.getStringAttr(""));
}

Value impulse::emitModelLogDensity(OpBuilder &builder, Location loc,
                                   Value qConstrained, Value &rng,
                                   const ModelContext &ctx) {
  if (ctx.hasLogpdfRegion()) {
    Block &lpBlock = ctx.logpdfRegion->front();
    IRMapping mapping;
    for (auto [arg, column] :
         llvm::zip(lpBlock.getArguments(), ctx.logpdfArgOffsets))
      mapping.map(arg, emitRowSlice(builder, loc, qConstrained, column,
                                    cast<RankedTensorType>(arg.getType())));
    return cloneBlockBody(builder, lpBlock, mapping).front();
  }
  GenerateOp generateOp = emitGenerateAt(builder, loc, qConstrained, rng, ctx);
  if (!generateOp.getOutputs().empty())
    rng = generateOp.getOutputs().front();
  return generateOp.getWeight();
}

/// Emits the logpdf symbol at `position` and returns the log density. A
/// single-block body is inlined because the adjoint of a call re-runs the
/// callee's forward pass; anything else stays a call.
static Value emitLogpdfFnInAutodiff(OpBuilder &builder, Location loc,
                                    Value position, const HMCContext &ctx) {
  SmallVector<Value> args{position};
  args.append(ctx.fnInputs.begin(), ctx.fnInputs.end());
  auto fn = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      builder.getInsertionBlock()->getParentOp(), ctx.logpdfFn);
  if (!fn || !llvm::hasSingleElement(fn.getFunctionBody())) {
    auto callOp = func::CallOp::create(builder, loc, ctx.logpdfFn,
                                       TypeRange{ctx.getScalarType()}, args);
    return callOp.getResult(0);
  }

  Block &entry = fn.getFunctionBody().front();
  IRMapping mapping;
  mapping.map(entry.getArguments(), args);
  return cloneBlockBody(builder, entry, mapping).front();
}

/// Emits the potential U(q) = -log p(qConstrained) - log|det J(q)| and
/// advances `rng` as emitModelLogDensity does.
static Value emitPotential(OpBuilder &builder, Location loc, Value q,
                           Value qConstrained, Value &rng,
                           const HMCContext &ctx) {
  Value logDensity = emitModelLogDensity(builder, loc, qConstrained, rng, ctx);
  Value negLogDensity = arith::NegFOp::create(builder, loc, logDensity);
  Value jacobianCorrection =
      computeTotalJacobianCorrection(builder, loc, q, ctx.supports);
  return arith::SubFOp::create(builder, loc, negLogDensity, jacobianCorrection);
}

/// Whether the potential calls the logpdf symbol, which takes the position
/// flattened to `tensor<N>`.
static bool usesLogpdfSymbol(const HMCContext &ctx) {
  return ctx.hasCustomLogpdf() && !ctx.hasLogpdfRegion();
}

/// Reshapes a `1xN` position to the `tensor<N>` argument of the logpdf symbol.
static Value flattenPosition(OpBuilder &builder, Location loc, Value position,
                             const HMCContext &ctx) {
  auto flatType = RankedTensorType::get({ctx.positionSize}, ctx.elementType);
  return impulse::ReshapeOp::create(builder, loc, flatType, position);
}

/// Emits the gradient region of the potential at `autodiffPosition`, which is
/// flat when the potential calls the logpdf symbol, seeded with `gradSeed`.
/// The returned gradient has the position type.
static GradientResult emitPotentialGradient(OpBuilder &builder, Location loc,
                                            Value autodiffPosition,
                                            Value gradSeed, Value rng,
                                            const HMCContext &ctx) {
  RankedTensorType positionType = ctx.getPositionType();
  bool useSymbol = usesLogpdfSymbol(ctx);
  RankedTensorType autodiffType =
      useSymbol ? RankedTensorType::get({ctx.positionSize}, ctx.elementType)
                : positionType;

  auto autodiffOp = enzyme::AutoDiffRegionOp::create(
      builder, loc, TypeRange{ctx.getScalarType(), rng.getType(), autodiffType},
      ValueRange{autodiffPosition, gradSeed},
      buildGradientRegionAttrs(builder, /*numActiveArgs=*/1,
                               ctx.autodiffAttrs));

  Block *autodiffBlock = builder.createBlock(&autodiffOp.getBody());
  Value qArg = autodiffBlock->addArgument(autodiffType, loc);
  builder.setInsertionPointToStart(autodiffBlock);

  if (useSymbol) {
    Value logDensity = emitLogpdfFnInAutodiff(builder, loc, qArg, ctx);
    Value potentialEnergy = arith::NegFOp::create(builder, loc, logDensity);
    // TODO(#2695): handle hybrid case
    enzyme::YieldOp::create(builder, loc, {potentialEnergy, rng});
  } else {
    // `supports` is empty in logpdf mode; that logpdf has its own Jacobians.
    Value qConstrained = constrainPosition(builder, loc, qArg, ctx.supports);
    Value rngOut = rng;
    Value potentialEnergy =
        emitPotential(builder, loc, qArg, qConstrained, rngOut, ctx);
    enzyme::YieldOp::create(builder, loc, {potentialEnergy, rngOut});
  }

  builder.setInsertionPointAfter(autodiffOp);

  Value grad = autodiffOp.getResult(2);
  if (useSymbol)
    grad = impulse::ReshapeOp::create(builder, loc, positionType, grad);
  return {autodiffOp.getResult(0), grad, autodiffOp.getResult(1)};
}

GradientResult impulse::computePotentialAndGradient(OpBuilder &builder,
                                                    Location loc,
                                                    Value position, Value rng,
                                                    const HMCContext &ctx) {
  Value gradSeed = emitSplatConstant(builder, loc, ctx.getScalarType(), 1.0);
  Value autodiffPosition = usesLogpdfSymbol(ctx)
                               ? flattenPosition(builder, loc, position, ctx)
                               : position;
  return emitPotentialGradient(builder, loc, autodiffPosition, gradSeed, rng,
                               ctx);
}

IntegrationResult impulse::computeIntegrationStep(OpBuilder &builder,
                                                  Location loc,
                                                  const IntegratorState &leaf,
                                                  Value rng, Value direction,
                                                  const HMCContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();

  auto negStepSize = arith::NegFOp::create(builder, loc, ctx.stepSize);
  Value signedStepSize = impulse::SelectOp::create(
      builder, loc, scalarType, direction, ctx.stepSize, negStepSize);

  Value halfConst = emitSplatConstant(builder, loc, scalarType, 0.5);

  Value stepSizeBroadcast =
      emitBroadcastScalar(builder, loc, signedStepSize, positionType);
  auto halfStep =
      arith::MulFOp::create(builder, loc, halfConst, signedStepSize);
  Value halfStepBroadcast =
      emitBroadcastScalar(builder, loc, halfStep, positionType);

  // 1. Half step momentum: p_half = p - 0.5 * eps * grad
  auto deltaP1 =
      arith::MulFOp::create(builder, loc, halfStepBroadcast, leaf.grad);
  Value pHalf = arith::SubFOp::create(builder, loc, leaf.p, deltaP1);

  // 2. Full step position: q_new = q + eps * M^-1 * p_half
  Value v =
      applyInverseMassMatrix(builder, loc, ctx.invMass, pHalf, positionType);
  auto deltaQ = arith::MulFOp::create(builder, loc, stepSizeBroadcast, v);
  Value qNew = arith::AddFOp::create(builder, loc, leaf.q, deltaQ);

  // 3. Compute gradient at new position
  auto gradResult = computePotentialAndGradient(builder, loc, qNew, rng, ctx);

  // 4. Final half step momentum: p_new = p_half - 0.5 * eps * grad_new
  auto deltaP2 =
      arith::MulFOp::create(builder, loc, halfStepBroadcast, gradResult.grad);
  Value pNew = arith::SubFOp::create(builder, loc, pHalf, deltaP2);

  return {qNew, pNew, gradResult.grad, gradResult.U, gradResult.rng};
}

Value impulse::checkTurning(OpBuilder &builder, Location loc, Value pLeft,
                            Value pRight, Value pSum, const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();

  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value halfConst = emitSplatConstant(builder, loc, scalarType, 0.5);

  Value vLeft =
      applyInverseMassMatrix(builder, loc, ctx.invMass, pLeft, positionType);
  Value vRight =
      applyInverseMassMatrix(builder, loc, ctx.invMass, pRight, positionType);

  // p_sum_centered = p_sum - (p_left + p_right) / 2
  Value halfBroadcast =
      emitBroadcastScalar(builder, loc, halfConst, positionType);

  auto pLeftPlusPRight = arith::AddFOp::create(builder, loc, pLeft, pRight);
  auto halfSum =
      arith::MulFOp::create(builder, loc, halfBroadcast, pLeftPlusPRight);
  Value pSumCentered = arith::SubFOp::create(builder, loc, pSum, halfSum);

  auto leftAngle =
      impulse::DotOp::create(builder, loc, scalarType, vLeft, pSumCentered,
                             builder.getDenseI64ArrayAttr({0, 1}),
                             builder.getDenseI64ArrayAttr({0, 1}));
  auto rightAngle =
      impulse::DotOp::create(builder, loc, scalarType, vRight, pSumCentered,
                             builder.getDenseI64ArrayAttr({0, 1}),
                             builder.getDenseI64ArrayAttr({0, 1}));

  // turning = (left_angle <= 0) OR (right_angle <= 0)
  auto leftNeg = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                       leftAngle, zeroConst);
  auto rightNeg = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                        rightAngle, zeroConst);

  return arith::OrIOp::create(builder, loc, leftNeg, rightNeg);
}

Value impulse::computeUniformTransitionProb(OpBuilder &builder, Location loc,
                                            Value currentWeight,
                                            Value newWeight) {
  Value weightDiff =
      arith::SubFOp::create(builder, loc, newWeight, currentWeight);
  return impulse::LogisticOp::create(builder, loc, weightDiff.getType(),
                                     weightDiff);
}

Value impulse::computeBiasedTransitionProb(OpBuilder &builder, Location loc,
                                           Value currentWeight, Value newWeight,
                                           Value turning, Value diverging) {
  auto resultType = cast<RankedTensorType>(currentWeight.getType());

  Value zeroConst = emitSplatConstant(builder, loc, resultType, 0.0);
  Value oneConst = emitSplatConstant(builder, loc, resultType, 1.0);

  Value weightDiff =
      arith::SubFOp::create(builder, loc, newWeight, currentWeight);
  Value expDiff = math::ExpOp::create(builder, loc, weightDiff);
  Value clippedProb =
      arith::MinimumFOp::create(builder, loc, oneConst, expDiff);
  Value turningOrDiverging =
      arith::OrIOp::create(builder, loc, turning, diverging);
  return arith::SelectOp::create(builder, loc, resultType, turningOrDiverging,
                                 zeroConst, clippedProb);
}

NUTSTreeState impulse::combineTrees(OpBuilder &builder, Location loc,
                                    const NUTSTreeState &tree,
                                    const NUTSTreeState &subTree,
                                    Value direction, Value rng, bool biased,
                                    const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();

  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);

  auto qLeft = impulse::SelectOp::create(builder, loc, positionType, direction,
                                         tree.q_left, subTree.q_left);
  auto pLeft = impulse::SelectOp::create(builder, loc, positionType, direction,
                                         tree.p_left, subTree.p_left);
  auto gradLeft = impulse::SelectOp::create(
      builder, loc, positionType, direction, tree.grad_left, subTree.grad_left);
  auto qRight = impulse::SelectOp::create(builder, loc, positionType, direction,
                                          subTree.q_right, tree.q_right);
  auto pRight = impulse::SelectOp::create(builder, loc, positionType, direction,
                                          subTree.p_right, tree.p_right);
  auto gradRight =
      impulse::SelectOp::create(builder, loc, positionType, direction,
                                subTree.grad_right, tree.grad_right);

  auto combinedWeight =
      impulse::emitLogAddExp(builder, loc, tree.weight, subTree.weight);

  // Compute transition probability
  Value transitionProb;
  if (biased) {
    transitionProb =
        computeBiasedTransitionProb(builder, loc, tree.weight, subTree.weight,
                                    subTree.turning, subTree.diverging);
  } else {
    transitionProb =
        computeUniformTransitionProb(builder, loc, tree.weight, subTree.weight);
  }

  auto randomOp = impulse::RandomOp::create(
      builder, loc, TypeRange{rng.getType(), scalarType}, rng, zeroConst,
      oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  auto rngOut = randomOp.getOutputRngState();
  auto uniformSample = randomOp.getResult();

  auto acceptNew = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLT, uniformSample, transitionProb);

  auto qProposal =
      impulse::SelectOp::create(builder, loc, positionType, acceptNew,
                                subTree.q_proposal, tree.q_proposal);
  auto gradProposal =
      impulse::SelectOp::create(builder, loc, positionType, acceptNew,
                                subTree.grad_proposal, tree.grad_proposal);
  auto UProposal = impulse::SelectOp::create(
      builder, loc, scalarType, acceptNew, subTree.U_proposal, tree.U_proposal);
  auto HProposal = impulse::SelectOp::create(
      builder, loc, scalarType, acceptNew, subTree.H_proposal, tree.H_proposal);

  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  auto combinedDepth = arith::AddIOp::create(builder, loc, tree.depth, oneI64);

  Value combinedTurning;
  if (biased) {
    auto turningCheck = checkTurning(
        builder, loc, pLeft, pRight,
        arith::AddFOp::create(builder, loc, tree.p_sum, subTree.p_sum), ctx);
    combinedTurning =
        arith::OrIOp::create(builder, loc, subTree.turning, turningCheck);
  } else {
    combinedTurning = tree.turning;
  }

  auto combinedDiverging =
      arith::OrIOp::create(builder, loc, tree.diverging, subTree.diverging);
  auto sumAcceptProbs = arith::AddFOp::create(
      builder, loc, tree.sum_accept_probs, subTree.sum_accept_probs);
  auto numProposals = arith::AddIOp::create(builder, loc, tree.num_proposals,
                                            subTree.num_proposals);
  auto pSum = arith::AddFOp::create(builder, loc, tree.p_sum, subTree.p_sum);

  return {.q_left = qLeft,
          .p_left = pLeft,
          .grad_left = gradLeft,
          .q_right = qRight,
          .p_right = pRight,
          .grad_right = gradRight,
          .q_proposal = qProposal,
          .grad_proposal = gradProposal,
          .U_proposal = UProposal,
          .H_proposal = HProposal,
          .depth = combinedDepth,
          .weight = combinedWeight,
          .turning = combinedTurning,
          .diverging = combinedDiverging,
          .sum_accept_probs = sumAcceptProbs,
          .num_proposals = numProposals,
          .p_sum = pSum,
          .rng = rngOut};
}

InitialHMCState impulse::InitHMC(OpBuilder &builder, Location loc, Value rng,
                                 const HMCContext &ctx, Value initialPosition) {
  RankedTensorType scalarType = ctx.getScalarType();

  auto initSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rng.getType(), rng.getType()}, rng);
  auto kernelSplit = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rng.getType(), rng.getType(), rng.getType()},
      initSplit.getResult(0));
  auto rngForSampleKernel = kernelSplit.getResult(0);
  auto rngForAutodiff = kernelSplit.getResult(1);

  Value q0;
  Value U0;
  bool useSymbol = usesLogpdfSymbol(ctx);
  if (useSymbol) {
    q0 = initialPosition;
    SmallVector<Value> callArgs{flattenPosition(builder, loc, q0, ctx)};
    llvm::append_range(callArgs, ctx.fnInputs);
    auto callOp = func::CallOp::create(builder, loc, ctx.logpdfFn,
                                       TypeRange{scalarType}, callArgs);
    U0 = arith::NegFOp::create(builder, loc, callOp.getResult(0));
  } else {
    Value q0Constrained;
    if (ctx.hasCustomLogpdf()) {
      q0 = initialPosition;
      q0Constrained = constrainPosition(builder, loc, q0, ctx.supports);
    } else {
      q0Constrained =
          gatherPositionFromTrace(builder, loc, ctx.originalTrace, ctx);
      q0 = unconstrainPosition(builder, loc, q0Constrained, ctx.supports);
    }
    Value rngForPotential = rngForAutodiff;
    U0 = emitPotential(builder, loc, q0, q0Constrained, rngForPotential, ctx);
  }

  Value autodiffQ0 = useSymbol ? flattenPosition(builder, loc, q0, ctx) : q0;
  Value gradSeed = emitSplatConstant(builder, loc, scalarType, 1.0);
  GradientResult gradient = emitPotentialGradient(
      builder, loc, autodiffQ0, gradSeed, rngForAutodiff, ctx);

  return {q0, U0, gradient.grad, rngForSampleKernel};
}

namespace {
/// RNG states, momentum and Hamiltonian at the start of one MCMC transition.
struct TransitionStart {
  Value rngNext;
  Value rngKernel;
  Value momentum;
  Value hamiltonian;
};
} // namespace

/// Splits `rng`, samples a fresh momentum and adds its kinetic energy to
/// `potentialEnergy`. `rngNext` seeds the next transition and `rngKernel` the
/// trajectory of this one.
static TransitionStart startTransition(OpBuilder &builder, Location loc,
                                       Value potentialEnergy, Value rng,
                                       const HMCContext &ctx) {
  RankedTensorType positionType = ctx.getPositionType();
  auto split = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{rng.getType(), rng.getType(), rng.getType()},
      rng);
  Value rngMomentum = split.getResult(1);
  // The logpdf mode draws the momentum from the unsplit key.
  if (!ctx.hasCustomLogpdf()) {
    auto momentumSplit = impulse::RandomSplitOp::create(
        builder, loc, TypeRange{rng.getType(), rng.getType()}, rngMomentum);
    rngMomentum = momentumSplit.getResult(0);
  }
  Value momentum =
      sampleMomentum(builder, loc, rngMomentum, ctx.invMass, ctx.massMatrixSqrt,
                     positionType, ctx.debugDump);
  Value kineticEnergy =
      computeKineticEnergy(builder, loc, momentum, ctx.invMass, positionType);
  Value hamiltonian =
      arith::AddFOp::create(builder, loc, potentialEnergy, kineticEnergy);
  return {split.getResult(0), split.getResult(2), momentum, hamiltonian};
}

MCMCKernelResult impulse::SampleHMC(OpBuilder &builder, Location loc, Value q,
                                    Value grad, Value U, Value rng,
                                    const HMCContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  Value numSteps, adjustedStepSize;
  if (ctx.numSteps > 0) {
    numSteps = emitI64ScalarConstant(builder, loc, ctx.numSteps);
    adjustedStepSize = ctx.stepSize;
  } else {
    auto trajDivStep =
        arith::DivFOp::create(builder, loc, ctx.trajectoryLength, ctx.stepSize);
    auto numStepsF64 = math::CeilOp::create(builder, loc, trajDivStep);
    numSteps =
        arith::FPToSIOp::create(builder, loc, i64TensorType, numStepsF64);
    adjustedStepSize =
        arith::DivFOp::create(builder, loc, ctx.trajectoryLength, numStepsF64);
  }

  auto adjustedCtx = ctx.withStepSize(adjustedStepSize);

  auto [rngNext, rngTransition, p0, initialHamiltonian] =
      startTransition(builder, loc, U, rng, ctx);

  Value direction = emitI1ScalarConstant(builder, loc, true);

  Value c0 = emitI64ScalarConstant(builder, loc, 0);
  Value c1 = emitI64ScalarConstant(builder, loc, 1);

  SmallVector<Type> loopResultTypes = {positionType, positionType, positionType,
                                       scalarType, rngTransition.getType()};
  auto forLoopOp =
      impulse::ForOp::create(builder, loc, loopResultTypes, c0, numSteps, c1,
                             ValueRange{q, p0, grad, U, rngTransition});

  Block *loopBody = builder.createBlock(&forLoopOp.getRegion());
  loopBody->addArgument(i64TensorType, loc);
  for (Type type : loopResultTypes)
    loopBody->addArgument(type, loc);

  builder.setInsertionPointToStart(loopBody);
  auto qLoop = loopBody->getArgument(1);
  auto pLoop = loopBody->getArgument(2);
  auto gradLoop = loopBody->getArgument(3);
  auto rngLoop = loopBody->getArgument(5);

  IntegratorState leaf = {qLoop, pLoop, gradLoop};
  auto step = computeIntegrationStep(builder, loc, leaf, rngLoop, direction,
                                     adjustedCtx);

  impulse::YieldOp::create(
      builder, loc, ValueRange{step.q, step.p, step.grad, step.U, step.rng});

  builder.setInsertionPointAfter(forLoopOp);
  auto qProposal = forLoopOp.getResult(0);
  auto pProposal = forLoopOp.getResult(1);
  auto gradProposal = forLoopOp.getResult(2);
  auto UProposal = forLoopOp.getResult(3);
  auto rngAfterLeapfrog = forLoopOp.getResult(4);

  auto K1 =
      computeKineticEnergy(builder, loc, pProposal, ctx.invMass, positionType);
  auto H1 = arith::AddFOp::create(builder, loc, UProposal, K1);

  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);

  auto dH = arith::SubFOp::create(builder, loc, initialHamiltonian, H1);
  auto expDH = math::ExpOp::create(builder, loc, dH);
  auto accProb = arith::MinimumFOp::create(builder, loc, oneConst, expDH);

  auto randomOp = impulse::RandomOp::create(
      builder, loc, TypeRange{rngAfterLeapfrog.getType(), scalarType},
      rngAfterLeapfrog, zeroConst, oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  auto randUniform = randomOp.getResult();

  auto acceptedTensor = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLT, randUniform, accProb);

  // HMC does not detect divergent trajectories.
  Value falseConst = emitI1ScalarConstant(builder, loc, false);

  auto qFinal = impulse::SelectOp::create(builder, loc, positionType,
                                          acceptedTensor, qProposal, q);
  auto gradFinal = impulse::SelectOp::create(
      builder, loc, positionType, acceptedTensor, gradProposal, grad);
  auto UFinal = impulse::SelectOp::create(builder, loc, scalarType,
                                          acceptedTensor, UProposal, U);

  return {qFinal,     gradFinal, UFinal, acceptedTensor,
          falseConst, accProb,   rngNext};
}

MCMCKernelResult impulse::SampleNUTS(OpBuilder &builder, Location loc, Value q,
                                     Value grad, Value U, Value rng,
                                     const NUTSContext &ctx) {
  auto scalarType = ctx.getScalarType();

  auto [rngNext, rngTree, p0, initialHamiltonian] =
      startTransition(builder, loc, U, rng, ctx);

  auto iterCtx = ctx.withH0(initialHamiltonian);

  Value zeroI64 = emitI64ScalarConstant(builder, loc, 0);
  Value falseConst = emitI1ScalarConstant(builder, loc, false);
  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);

  NUTSTreeState initialTree = {.q_left = q,
                               .p_left = p0,
                               .grad_left = grad,
                               .q_right = q,
                               .p_right = p0,
                               .grad_right = grad,
                               .q_proposal = q,
                               .grad_proposal = grad,
                               .U_proposal = U,
                               .H_proposal = initialHamiltonian,
                               .depth = zeroI64,
                               .weight = zeroConst,
                               .turning = falseConst,
                               .diverging = falseConst,
                               .sum_accept_probs = zeroConst,
                               .num_proposals = zeroI64,
                               .p_sum = p0,
                               .rng = rngTree};

  auto finalTree = buildTree(builder, loc, initialTree, iterCtx);

  // The tree itself selects the proposal, so the transition always accepts.
  Value trueConst = emitI1ScalarConstant(builder, loc, true);

  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  auto numProposalsClamped =
      arith::MaxSIOp::create(builder, loc, finalTree.num_proposals, oneI64);
  auto numProposalsFloat =
      arith::SIToFPOp::create(builder, loc, scalarType, numProposalsClamped);
  auto meanAcceptProb = arith::DivFOp::create(
      builder, loc, finalTree.sum_accept_probs, numProposalsFloat);

  return {finalTree.q_proposal,
          finalTree.grad_proposal,
          finalTree.U_proposal,
          trueConst,
          finalTree.diverging,
          meanAcceptProb,
          rngNext};
}

NUTSTreeState impulse::buildBaseTree(OpBuilder &builder, Location loc,
                                     const IntegratorState &leaf, Value rng,
                                     Value direction, const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();

  Value oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);
  Value zeroI64 = emitI64ScalarConstant(builder, loc, 0);
  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value falseConst = emitI1ScalarConstant(builder, loc, false);

  IntegrationResult leap =
      computeIntegrationStep(builder, loc, leaf, rng, direction, ctx);

  auto qNew = leap.q;
  auto pNew = leap.p;
  auto gradNew = leap.grad;
  auto UNew = leap.U;
  auto rngOut = leap.rng;

  auto KNew =
      computeKineticEnergy(builder, loc, pNew, ctx.invMass, positionType);
  auto HNew = arith::AddFOp::create(builder, loc, UNew, KNew);
  Value deltaH = arith::SubFOp::create(builder, loc, HNew, ctx.H0);

  // NaN check
  auto isNan = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::UNE,
                                     deltaH, deltaH);
  Value infConst = emitSplatConstant(builder, loc, scalarType,
                                     std::numeric_limits<double>::infinity());
  deltaH = arith::SelectOp::create(builder, loc, scalarType, isNan, infConst,
                                   deltaH);

  auto treeWeight = arith::NegFOp::create(builder, loc, deltaH);

  // Check for divergence
  auto diverging = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OGT, deltaH, ctx.maxDeltaEnergy);

  auto negDeltaH = arith::NegFOp::create(builder, loc, deltaH);
  auto expNegDelta = math::ExpOp::create(builder, loc, negDeltaH);
  auto acceptProb =
      arith::MinimumFOp::create(builder, loc, oneConst, expNegDelta);

  return {.q_left = qNew,
          .p_left = pNew,
          .grad_left = gradNew,
          .q_right = qNew,
          .p_right = pNew,
          .grad_right = gradNew,
          .q_proposal = qNew,
          .grad_proposal = gradNew,
          .U_proposal = UNew,
          .H_proposal = HNew,
          .depth = zeroI64,
          .weight = treeWeight,
          .turning = falseConst,
          .diverging = diverging,
          .sum_accept_probs = acceptProb,
          .num_proposals = oneI64,
          .p_sum = pNew,
          .rng = rngOut};
}

IntegratorState impulse::getLeafFromTree(OpBuilder &builder, Location loc,
                                         const NUTSTreeState &tree,
                                         Value direction,
                                         const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();

  auto leafQ = impulse::SelectOp::create(builder, loc, positionType, direction,
                                         tree.q_right, tree.q_left);
  auto leafP = impulse::SelectOp::create(builder, loc, positionType, direction,
                                         tree.p_right, tree.p_left);
  auto leafGrad = impulse::SelectOp::create(
      builder, loc, positionType, direction, tree.grad_right, tree.grad_left);
  return {leafQ, leafP, leafGrad};
}

SubtreeBuildResult impulse::buildIterativeSubtree(
    OpBuilder &builder, Location loc, const NUTSTreeState &initialTree,
    Value direction, Value pCkpts, Value pSumCkpts, const NUTSContext &ctx) {
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto pCkptsType = cast<RankedTensorType>(pCkpts.getType());
  Value trueConst = emitI1ScalarConstant(builder, loc, true);
  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value zeroI64 = emitI64ScalarConstant(builder, loc, 0);

  // 2 ^ (initialTree.depth)
  auto maxNumProposals =
      arith::ShLIOp::create(builder, loc, oneI64, initialTree.depth);

  SmallVector<Type> whileTypes = initialTree.getTypes();
  whileTypes.push_back(pCkptsType);
  whileTypes.push_back(pCkptsType);
  whileTypes.push_back(i64TensorType);

  SmallVector<Value> whileInitVals = initialTree.toValues();
  whileInitVals[15] = zeroI64; // zero `num_proposals`
  whileInitVals.push_back(pCkpts);
  whileInitVals.push_back(pSumCkpts);
  whileInitVals.push_back(zeroI64);

  auto whileOp =
      impulse::WhileOp::create(builder, loc, whileTypes, whileInitVals);

  // Check: num_proposals < max_num_proposals && !turning && !diverging
  Block *condBlock = builder.createBlock(&whileOp.getConditionRegion());
  for (auto type : whileTypes)
    condBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(condBlock);
  SmallVector<Value> condTreeArgs(condBlock->getArguments().begin(),
                                  condBlock->getArguments().begin() + 18);
  NUTSTreeState condTree = NUTSTreeState::fromValues(condTreeArgs);

  auto numProposalsCheck =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                            condTree.num_proposals, maxNumProposals);
  auto notTurning =
      arith::XOrIOp::create(builder, loc, condTree.turning, trueConst);
  auto notDiverging =
      arith::XOrIOp::create(builder, loc, condTree.diverging, trueConst);
  auto continueCond = arith::AndIOp::create(
      builder, loc,
      arith::AndIOp::create(builder, loc, numProposalsCheck, notTurning),
      notDiverging);

  // Yield continue condition
  impulse::YieldOp::create(builder, loc, ValueRange{continueCond});

  Block *bodyBlock = builder.createBlock(&whileOp.getBodyRegion());
  for (auto type : whileTypes)
    bodyBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(bodyBlock);

  SmallVector<Value> bodyTreeArgs(bodyBlock->getArguments().begin(),
                                  bodyBlock->getArguments().begin() + 18);
  NUTSTreeState bodyTree = NUTSTreeState::fromValues(bodyTreeArgs);
  auto bodyPCkpts = bodyBlock->getArgument(18);
  auto bodyPSumCkpts = bodyBlock->getArgument(19);
  auto bodyLeafIdx = bodyBlock->getArgument(20);

  // Extract leaf based on direction
  IntegratorState leaf =
      getLeafFromTree(builder, loc, bodyTree, direction, ctx);

  auto rngSplit2 = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{bodyTree.rng.getType(), bodyTree.rng.getType()},
      bodyTree.rng);
  auto rngNext = rngSplit2.getResult(0);
  auto rngCombine = rngSplit2.getResult(1);

  // Build base tree
  NUTSTreeState newLeaf =
      buildBaseTree(builder, loc, leaf, rngNext, direction, ctx);

  // First leaf check
  auto isFirstLeaf = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, bodyTree.num_proposals, zeroI64);

  SmallVector<Type> treeTypes = newLeaf.getTypes();
  auto ifOp = impulse::IfOp::create(builder, loc, treeTypes, isFirstLeaf);
  {
    Block *trueBranch = builder.createBlock(&ifOp.getTrueBranch());
    builder.setInsertionPointToStart(trueBranch);
    impulse::YieldOp::create(builder, loc, newLeaf.toValues());
  }
  {
    Block *falseBranch = builder.createBlock(&ifOp.getFalseBranch());
    builder.setInsertionPointToStart(falseBranch);
    NUTSTreeState combinedTree =
        combineTrees(builder, loc, bodyTree, newLeaf, direction, rngCombine,
                     /*biased=*/false, ctx);
    impulse::YieldOp::create(builder, loc, combinedTree.toValues());
  }

  builder.setInsertionPointAfter(ifOp);
  NUTSTreeState updatedTree = NUTSTreeState::fromValues(
      SmallVector<Value>(ifOp.getResults().begin(), ifOp.getResults().end()));
  updatedTree.rng = rngNext;

  // Update and check iterative turning
  auto [ckptIdxMin, ckptIdxMax] =
      leafIdxToCheckpointIdxs(builder, loc, bodyLeafIdx);
  auto [updatedPCkpts, updatedPSumCkpts] =
      updateCheckpoints(builder, loc, bodyLeafIdx, ckptIdxMax, newLeaf.p_right,
                        updatedTree.p_sum, bodyPCkpts, bodyPSumCkpts);
  auto iterativeTurning = checkIterativeTurning(
      builder, loc, newLeaf.p_right, updatedTree.p_sum, updatedPCkpts,
      updatedPSumCkpts, ckptIdxMin, ckptIdxMax, ctx);

  updatedTree.turning =
      impulse::SelectOp::create(builder, loc, i1TensorType, isFirstLeaf,
                                newLeaf.turning, iterativeTurning);

  auto nextLeafIdx = arith::AddIOp::create(builder, loc, bodyLeafIdx, oneI64);

  SmallVector<Value> yieldVals = updatedTree.toValues();
  yieldVals.push_back(updatedPCkpts);
  yieldVals.push_back(updatedPSumCkpts);
  yieldVals.push_back(nextLeafIdx);
  impulse::YieldOp::create(builder, loc, yieldVals);

  builder.setInsertionPointAfter(whileOp);

  SmallVector<Value> resultTreeArgs(whileOp.getResults().begin(),
                                    whileOp.getResults().begin() + 18);
  NUTSTreeState resultTree = NUTSTreeState::fromValues(resultTreeArgs);
  auto resultPCkpts = whileOp.getResult(18);
  auto resultPSumCkpts = whileOp.getResult(19);

  // `combineTrees` increments depth at each leaf building step; we need to
  // restore to target depth here
  resultTree.depth = initialTree.depth;

  return {resultTree, resultPCkpts, resultPSumCkpts};
}

SubtreeBuildResult impulse::doubleTree(OpBuilder &builder, Location loc,
                                       const NUTSTreeState &tree,
                                       Value direction, Value pCkpts,
                                       Value pSumCkpts,
                                       const NUTSContext &ctx) {
  auto rngSplit2 = impulse::RandomSplitOp::create(
      builder, loc, TypeRange{tree.rng.getType(), tree.rng.getType()},
      tree.rng);
  auto rngSubtree = rngSplit2.getResult(0);
  auto rngTransition = rngSplit2.getResult(1);

  NUTSTreeState subTreeInit = tree;
  subTreeInit.rng = rngSubtree;
  auto subtreeResult = buildIterativeSubtree(builder, loc, subTreeInit,
                                             direction, pCkpts, pSumCkpts, ctx);

  // Tree combine using *biased* transition kernel
  NUTSTreeState combinedTree =
      combineTrees(builder, loc, tree, subtreeResult.tree, direction,
                   rngTransition, /*biased=*/true, ctx);

  return {combinedTree, subtreeResult.pCkpts, subtreeResult.pSumCkpts};
}

NUTSTreeState impulse::buildTree(OpBuilder &builder, Location loc,
                                 const NUTSTreeState &initialTree,
                                 const NUTSContext &ctx) {
  Type elemType = ctx.elementType;
  RankedTensorType scalarType = ctx.getScalarType();

  Value trueConst = emitI1ScalarConstant(builder, loc, true);
  Value halfConst = emitSplatConstant(builder, loc, scalarType, 0.5);
  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);

  Value maxTreeDepth = emitI64ScalarConstant(builder, loc, ctx.maxTreeDepth);

  auto checkpointType =
      RankedTensorType::get({ctx.maxTreeDepth, ctx.positionSize}, elemType);

  SmallVector<Type> whileTypes = initialTree.getTypes();
  SmallVector<Value> whileInitVals = initialTree.toValues();

  auto whileOp =
      impulse::WhileOp::create(builder, loc, whileTypes, whileInitVals);

  // Check: (depth < maxTreeDepth) && !turning && !diverging
  Block *condBlock = builder.createBlock(&whileOp.getConditionRegion());
  for (auto type : whileTypes)
    condBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(condBlock);

  SmallVector<Value> condArgs(condBlock->getArguments().begin(),
                              condBlock->getArguments().end());
  NUTSTreeState condTree = NUTSTreeState::fromValues(condArgs);

  auto depthCheck = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, condTree.depth, maxTreeDepth);
  auto notTurning =
      arith::XOrIOp::create(builder, loc, condTree.turning, trueConst);
  auto notDiverging =
      arith::XOrIOp::create(builder, loc, condTree.diverging, trueConst);

  // Yield continue condition
  auto continueCond = arith::AndIOp::create(
      builder, loc, arith::AndIOp::create(builder, loc, depthCheck, notTurning),
      notDiverging);

  impulse::YieldOp::create(builder, loc, ValueRange{continueCond});

  Block *bodyBlock = builder.createBlock(&whileOp.getBodyRegion());
  for (auto type : whileTypes)
    bodyBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(bodyBlock);

  SmallVector<Value> bodyArgs(bodyBlock->getArguments().begin(),
                              bodyBlock->getArguments().end());
  NUTSTreeState bodyTree = NUTSTreeState::fromValues(bodyArgs);

  // Create fresh checkpoint tensors
  Value zeroCkpts = emitSplatConstant(builder, loc, checkpointType, 0.0);
  Value bodyPCkpts = zeroCkpts;
  Value bodyPSumCkpts = zeroCkpts;

  auto rngSplit3 = impulse::RandomSplitOp::create(
      builder, loc,
      TypeRange{bodyTree.rng.getType(), bodyTree.rng.getType(),
                bodyTree.rng.getType()},
      bodyTree.rng);
  auto rngNext = rngSplit3.getResult(0);
  auto rngDir = rngSplit3.getResult(1);
  auto rngDbl = rngSplit3.getResult(2);

  auto directionRandom = impulse::RandomOp::create(
      builder, loc, TypeRange{rngDir.getType(), scalarType}, rngDir, zeroConst,
      oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  auto direction =
      arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT,
                            directionRandom.getResult(), halfConst);

  // Double the tree
  NUTSTreeState treeToDouble = bodyTree;
  treeToDouble.rng = rngDbl;
  auto doubleResult = doubleTree(builder, loc, treeToDouble, direction,
                                 bodyPCkpts, bodyPSumCkpts, ctx);

  NUTSTreeState treeToYield = doubleResult.tree;
  treeToYield.rng = rngNext;
  impulse::YieldOp::create(builder, loc, treeToYield.toValues());

  builder.setInsertionPointAfter(whileOp);

  SmallVector<Value> results(whileOp.getResults().begin(),
                             whileOp.getResults().end());
  return NUTSTreeState::fromValues(results);
}

std::pair<Value, Value> impulse::leafIdxToCheckpointIdxs(OpBuilder &builder,
                                                         Location loc,
                                                         Value leafIdx) {

  Value oneConst = emitI64ScalarConstant(builder, loc, 1);

  // idx_max = popcount(leafIdx >> 1)
  auto shiftedIdx = arith::ShRUIOp::create(builder, loc, leafIdx, oneConst);
  auto idxMax = math::CtPopOp::create(builder, loc, shiftedIdx);

  // num_subtrees = popcount((~leafIdx & (leafIdx + 1)) - 1)
  auto leafIdxPlusOne = arith::AddIOp::create(builder, loc, leafIdx, oneConst);

  Value minusOneConst = emitI64ScalarConstant(builder, loc, -1);
  auto notLeafIdx = arith::XOrIOp::create(builder, loc, leafIdx, minusOneConst);

  Value andResult =
      arith::AndIOp::create(builder, loc, notLeafIdx, leafIdxPlusOne);
  Value andMinusOne = arith::SubIOp::create(builder, loc, andResult, oneConst);
  Value numSubtrees = math::CtPopOp::create(builder, loc, andMinusOne);

  // idx_min = idx_max - num_subtrees + 1
  Value idxMaxMinusNumSubtrees =
      arith::SubIOp::create(builder, loc, idxMax, numSubtrees);
  Value idxMin =
      arith::AddIOp::create(builder, loc, idxMaxMinusNumSubtrees, oneConst);

  return {idxMin, idxMax};
}

Value impulse::checkIterativeTurning(OpBuilder &builder, Location loc, Value p,
                                     Value pSum, Value pCkpts, Value pSumCkpts,
                                     Value idxMin, Value idxMax,
                                     const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());

  Value falseConst = emitI1ScalarConstant(builder, loc, false);
  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);

  // Iterate from `idx_max` down to `idx_min`, check turning at each checkpoint
  SmallVector<Type> whileTypes = {i64TensorType, i1TensorType};
  SmallVector<Value> whileInitVals = {idxMax, falseConst};

  auto whileOp =
      impulse::WhileOp::create(builder, loc, whileTypes, whileInitVals);
  Block *condBlock = builder.createBlock(&whileOp.getConditionRegion());
  condBlock->addArgument(i64TensorType, loc);
  condBlock->addArgument(i1TensorType, loc);
  builder.setInsertionPointToStart(condBlock);

  Value iCond = condBlock->getArgument(0);
  Value turningCond = condBlock->getArgument(1);

  auto iGeMin = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                      iCond, idxMin);
  Value trueConst = emitI1ScalarConstant(builder, loc, true);
  auto notTurning = arith::XOrIOp::create(builder, loc, turningCond, trueConst);
  auto continueLoop = arith::AndIOp::create(builder, loc, iGeMin, notTurning);

  impulse::YieldOp::create(builder, loc, ValueRange{continueLoop.getResult()});

  Block *bodyBlock = builder.createBlock(&whileOp.getBodyRegion());
  bodyBlock->addArgument(i64TensorType, loc);
  bodyBlock->addArgument(i1TensorType, loc);
  builder.setInsertionPointToStart(bodyBlock);
  Value iBody = bodyBlock->getArgument(0);

  Value zeroI64 = emitI64ScalarConstant(builder, loc, 0);
  Value pLeft = impulse::DynamicSliceOp::create(
      builder, loc, positionType, pCkpts, ValueRange{iBody, zeroI64},
      builder.getDenseI64ArrayAttr({1, ctx.positionSize}));
  Value pSumCkptI = impulse::DynamicSliceOp::create(
      builder, loc, positionType, pSumCkpts, ValueRange{iBody, zeroI64},
      builder.getDenseI64ArrayAttr({1, ctx.positionSize}));

  // Compute subtree momentum sum: pSum - pSumCkpts[i] + pCkpts[i]
  auto pSumMinusCkpt = arith::SubFOp::create(builder, loc, pSum, pSumCkptI);
  Value subtreePSum = arith::AddFOp::create(builder, loc, pSumMinusCkpt, pLeft);

  // Check turning
  Value turningAtCkpt = checkTurning(builder, loc, pLeft, p, subtreePSum, ctx);

  Value iNext = arith::SubIOp::create(builder, loc, iBody, oneI64);
  impulse::YieldOp::create(builder, loc, ValueRange{iNext, turningAtCkpt});

  builder.setInsertionPointAfter(whileOp);
  return whileOp.getResult(1);
}

std::pair<Value, Value> impulse::updateCheckpoints(OpBuilder &builder,
                                                   Location loc, Value leafIdx,
                                                   Value ckptIdxMax, Value p,
                                                   Value pSum, Value pCkpts,
                                                   Value pSumCkpts) {
  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value zeroI64 = emitI64ScalarConstant(builder, loc, 0);

  Value leafIdxBit0 = arith::AndIOp::create(builder, loc, leafIdx, oneI64);
  Value isEven = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       leafIdxBit0, zeroI64);

  auto pCkptsType = cast<RankedTensorType>(pCkpts.getType());

  // Compute updates only on even leafIdx
  SmallVector<Type> ifResultTypes = {pCkptsType, pCkptsType};
  auto ifOp = impulse::IfOp::create(builder, loc, ifResultTypes, isEven);
  {
    Block *trueBranch = builder.createBlock(&ifOp.getTrueBranch());
    builder.setInsertionPointToStart(trueBranch);

    auto updatedPCkpts = impulse::DynamicUpdateSliceOp::create(
        builder, loc, pCkptsType, pCkpts, p, ValueRange{ckptIdxMax, zeroI64});
    auto updatedPSumCkpts = impulse::DynamicUpdateSliceOp::create(
        builder, loc, pCkptsType, pSumCkpts, pSum,
        ValueRange{ckptIdxMax, zeroI64});

    impulse::YieldOp::create(builder, loc,
                             ValueRange{updatedPCkpts, updatedPSumCkpts});
  }
  {
    Block *falseBranch = builder.createBlock(&ifOp.getFalseBranch());
    builder.setInsertionPointToStart(falseBranch);

    impulse::YieldOp::create(builder, loc, ValueRange{pCkpts, pSumCkpts});
  }

  builder.setInsertionPointAfter(ifOp);

  Value finalPCkpts = ifOp.getResult(0);
  Value finalPSumCkpts = ifOp.getResult(1);

  return {finalPCkpts, finalPSumCkpts};
}

DualAveragingState impulse::initDualAveraging(OpBuilder &builder, Location loc,
                                              Value stepSize,
                                              bool logOfProduct) {
  auto stepSizeType = cast<RankedTensorType>(stepSize.getType());
  auto elemType = stepSizeType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value proxCenter;
  if (logOfProduct) {
    Value tenConst = emitSplatConstant(builder, loc, scalarType, 10.0);
    Value scaled = arith::MulFOp::create(builder, loc, tenConst, stepSize);
    proxCenter = math::LogOp::create(builder, loc, scaled);
  } else {
    Value log10Const =
        emitSplatConstant(builder, loc, scalarType, std::log(10.0));
    Value logStepSize = math::LogOp::create(builder, loc, stepSize);
    proxCenter = arith::AddFOp::create(builder, loc, log10Const, logStepSize);
  }

  Value zeroConst = emitSplatConstant(builder, loc, scalarType, 0.0);
  Value zeroI64 = emitI64ScalarConstant(builder, loc, 0);

  return {
      .log_step_size = zeroConst,
      .log_step_size_avg = zeroConst,
      .gradient_avg = zeroConst,
      .step_count = zeroI64,
      .prox_center = proxCenter,
  };
}

DualAveragingState
impulse::updateDualAveraging(OpBuilder &builder, Location loc,
                             const DualAveragingState &state, Value acceptProb,
                             const DualAveragingConfig &config) {
  // Dual Averaging update:
  //   g = target_accept_prob - accept_prob
  //   g_avg = (1 - 1/(t+t0)) * g_avg + g/(t+t0)
  //   log_step_size = prox_center - sqrt(t)/gamma * g_avg
  //   log_step_size_avg = (1 - t^(-kappa)) * log_step_size_avg +
  //                        t^(-kappa) * log_step_size
  auto acceptProbType = cast<RankedTensorType>(acceptProb.getType());
  auto elemType = acceptProbType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  // t = t + 1
  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value tNew = arith::AddIOp::create(builder, loc, state.step_count, oneI64);
  Value tFloat = arith::SIToFPOp::create(builder, loc, scalarType, tNew);

  // t0
  Value t0Const = emitSplatConstant(builder, loc, scalarType, config.t0);

  // g = target_accept_prob - accept_prob
  Value targetConst =
      emitSplatConstant(builder, loc, scalarType, config.target_accept_prob);
  Value g = arith::SubFOp::create(builder, loc, targetConst, acceptProb);

  // t_plus_t0 = t + t0
  Value tPlusT0 = arith::AddFOp::create(builder, loc, tFloat, t0Const);

  // weight = 1 / (t + t0)
  Value oneConst = emitSplatConstant(builder, loc, scalarType, 1.0);
  Value weight = arith::DivFOp::create(builder, loc, oneConst, tPlusT0);

  // decay = 1 - weight = 1 - 1/(t + t0)
  Value decay = arith::SubFOp::create(builder, loc, oneConst, weight);

  // g_avg = decay * g_avg + weight * g
  // g_avg = (1 - 1/(t+t0)) * g_avg + g/(t+t0)
  Value gAvgDecayed =
      arith::MulFOp::create(builder, loc, decay, state.gradient_avg);
  Value gWeighted = arith::MulFOp::create(builder, loc, weight, g);
  Value gAvgNew = arith::AddFOp::create(builder, loc, gAvgDecayed, gWeighted);

  // x_t = prox_center - sqrt(t) / gamma * g_avg
  Value sqrtT = math::SqrtOp::create(builder, loc, tFloat);
  Value gammaConst = emitSplatConstant(builder, loc, scalarType, config.gamma);
  Value sqrtTOverGamma = arith::DivFOp::create(builder, loc, sqrtT, gammaConst);
  Value adjustment =
      arith::MulFOp::create(builder, loc, sqrtTOverGamma, gAvgNew);
  Value logStepSizeNew =
      arith::SubFOp::create(builder, loc, state.prox_center, adjustment);

  // weight_t = t^(-kappa)
  Value negKappaConst =
      emitSplatConstant(builder, loc, scalarType, -config.kappa);
  Value weightT = math::PowFOp::create(builder, loc, tFloat, negKappaConst);

  // x_avg = (1 - weight_t) * x_avg + weight_t * x_t
  Value oneMinusWeightT =
      arith::SubFOp::create(builder, loc, oneConst, weightT);
  Value avgDecayed = arith::MulFOp::create(builder, loc, oneMinusWeightT,
                                           state.log_step_size_avg);
  Value newContribution =
      arith::MulFOp::create(builder, loc, weightT, logStepSizeNew);
  Value logStepSizeAvgNew =
      arith::AddFOp::create(builder, loc, avgDecayed, newContribution);

  return {
      .log_step_size = logStepSizeNew,
      .log_step_size_avg = logStepSizeAvgNew,
      .gradient_avg = gAvgNew,
      .step_count = tNew,
      .prox_center = state.prox_center,
  };
}

Value impulse::getStepSizeFromDualAveraging(OpBuilder &builder, Location loc,
                                            const DualAveragingState &state,
                                            bool final) {
  Value logStepSize = final ? state.log_step_size_avg : state.log_step_size;
  return math::ExpOp::create(builder, loc, logStepSize);
}

WelfordState impulse::initWelford(OpBuilder &builder, Location loc,
                                  int64_t positionSize, bool diagonal) {
  auto elemType = builder.getF64Type();
  auto meanType = RankedTensorType::get({positionSize}, elemType);

  Value mean = emitSplatConstant(builder, loc, meanType, 0.0);

  // Diagonal -> tensor<positionSize>
  // Dense -> tensor<positionSize x positionSize>
  Value m2;
  if (diagonal) {
    m2 = emitSplatConstant(builder, loc, meanType, 0.0);
  } else {
    auto m2Type = RankedTensorType::get({positionSize, positionSize}, elemType);
    m2 = emitSplatConstant(builder, loc, m2Type, 0.0);
  }

  Value n = emitI64ScalarConstant(builder, loc, 0);

  return {mean, m2, n};
}

WelfordState impulse::updateWelford(OpBuilder &builder, Location loc,
                                    const WelfordState &state, Value sample,
                                    const WelfordConfig &config) {
  // Algorithm:
  //   n = n + 1
  //   delta_pre = sample - mean
  //   mean = mean + delta_pre / n
  //   delta_post = sample - mean
  //   (if diagonal) m2 = m2 + delta_pre * delta_post
  //   (if dense)    m2 = m2 + outer(delta_post, delta_pre)

  auto sampleType = cast<RankedTensorType>(sample.getType());
  auto elemType = sampleType.getElementType();

  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value nNew = arith::AddIOp::create(builder, loc, state.n, oneI64);

  auto scalarType = RankedTensorType::get({}, elemType);
  Value nFloat = arith::SIToFPOp::create(builder, loc, scalarType, nNew);

  Value nBroadcast = emitBroadcastScalar(builder, loc, nFloat, sampleType);

  Value deltaPre = arith::SubFOp::create(builder, loc, sample, state.mean);

  Value deltaPreOverN =
      arith::DivFOp::create(builder, loc, deltaPre, nBroadcast);
  Value meanNew =
      arith::AddFOp::create(builder, loc, state.mean, deltaPreOverN);

  Value deltaPost = arith::SubFOp::create(builder, loc, sample, meanNew);

  Value m2New;
  if (config.diagonal) {
    Value product = arith::MulFOp::create(builder, loc, deltaPre, deltaPost);
    m2New = arith::AddFOp::create(builder, loc, state.m2, product);
  } else { // Dense
    auto m2Type = cast<RankedTensorType>(state.m2.getType());
    Value outerProduct = impulse::DotOp::create(
        builder, loc, m2Type, deltaPost, deltaPre,
        /*lhs_contracting_dimensions=*/builder.getDenseI64ArrayAttr({}),
        /*rhs_contracting_dimensions=*/builder.getDenseI64ArrayAttr({}));
    m2New = arith::AddFOp::create(builder, loc, state.m2, outerProduct);
  }

  return {meanNew, m2New, nNew};
}

Value impulse::finalizeWelford(OpBuilder &builder, Location loc,
                               const WelfordState &state,
                               const WelfordConfig &config) {
  // Compute sample covariance: cov = m2 / (n - 1)
  auto m2Type = cast<RankedTensorType>(state.m2.getType());
  auto elemType = m2Type.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value oneI64 = emitI64ScalarConstant(builder, loc, 1);
  Value nMinus1 = arith::SubIOp::create(builder, loc, state.n, oneI64);
  Value nMinus1Float =
      arith::SIToFPOp::create(builder, loc, scalarType, nMinus1);

  Value nMinus1Bcast = emitBroadcastScalar(builder, loc, nMinus1Float, m2Type);

  Value cov = arith::DivFOp::create(builder, loc, state.m2, nMinus1Bcast);

  // (Optional) Regularization (Stan's shrinkage):
  //   scaled_cov = (n / (n + 5)) * cov
  //   shrinkage = 1e-3 * (5 / (n + 5))
  //   (if diagonal) cov = scaled_cov + shrinkage
  //   (if dense)    cov = scaled_cov + shrinkage * I
  if (config.regularize) {
    Value nFloat = arith::SIToFPOp::create(builder, loc, scalarType, state.n);

    Value fiveConst = emitSplatConstant(builder, loc, scalarType, 5.0);
    Value nPlusFive = arith::AddFOp::create(builder, loc, nFloat, fiveConst);

    Value scale = arith::DivFOp::create(builder, loc, nFloat, nPlusFive);
    Value scaleBcast = emitBroadcastScalar(builder, loc, scale, m2Type);
    Value scaledCov = arith::MulFOp::create(builder, loc, scaleBcast, cov);

    Value shrinkageBaseConst =
        emitSplatConstant(builder, loc, scalarType, 1e-3 * 5.0);
    Value shrinkage =
        arith::DivFOp::create(builder, loc, shrinkageBaseConst, nPlusFive);

    if (config.diagonal) {
      Value shrinkageBcast =
          emitBroadcastScalar(builder, loc, shrinkage, m2Type);
      cov = arith::AddFOp::create(builder, loc, scaledCov, shrinkageBcast);
    } else {
      Value identity = createIdentityMatrix(builder, loc, m2Type);
      Value shrinkageBcast =
          emitBroadcastScalar(builder, loc, shrinkage, m2Type);
      Value shrinkageI =
          arith::MulFOp::create(builder, loc, shrinkageBcast, identity);
      cov = arith::AddFOp::create(builder, loc, scaledCov, shrinkageI);
    }
  }

  return cov;
}

SmallVector<AdaptWindow> impulse::buildAdaptationSchedule(int64_t numSteps) {
  // |<-- start buffer -->|<-- middle windows (doubling) -->|<-- end buffer -->|
  // |      (no mass)     |     (collect + adapt mass)      |    (no mass)     |
  // |   step size only   |   step size + mass matrix       |  step size only  |

  SmallVector<AdaptWindow> schedule;

  if (numSteps < 20) {
    schedule.push_back({0, numSteps - 1});
    return schedule;
  }

  // Stan-style window schedule
  int64_t startBufferSize = 75;
  int64_t endBufferSize = 50;
  int64_t initWindowSize = 25;

  if ((startBufferSize + endBufferSize + initWindowSize) > numSteps) {
    startBufferSize = static_cast<int64_t>(0.15 * numSteps);
    endBufferSize = static_cast<int64_t>(0.1 * numSteps);
    initWindowSize = numSteps - startBufferSize - endBufferSize;
  }

  // Start buffer window
  schedule.push_back({0, startBufferSize - 1});

  int64_t endWindowStart = numSteps - endBufferSize;
  int64_t nextWindowSize = initWindowSize;
  int64_t nextWindowStart = startBufferSize;

  // Middle windows
  while (nextWindowStart < endWindowStart) {
    int64_t curWindowStart = nextWindowStart;
    int64_t curWindowSize = nextWindowSize;

    if (3 * curWindowSize <= endWindowStart - curWindowStart) {
      nextWindowSize = 2 * curWindowSize;
    } else {
      curWindowSize = endWindowStart - curWindowStart;
    }

    nextWindowStart = curWindowStart + curWindowSize;
    schedule.push_back({curWindowStart, nextWindowStart - 1});
  }

  // End buffer window
  schedule.push_back({endWindowStart, numSteps - 1});

  return schedule;
}

/// Whether the site's support needs a bijector, i.e. is not all of R.
static bool isConstrainedSite(const SupportInfo &info) {
  return info.support && info.support.getKind() != impulse::SupportKind::REAL;
}

/// Maps a `1xN` position site by site from constrained to unconstrained
/// coordinates or back. Returns `position` itself when no site is constrained.
static Value transformPosition(OpBuilder &builder, Location loc, Value position,
                               ArrayRef<SupportInfo> supports,
                               bool toUnconstrained) {
  if (llvm::none_of(supports, isConstrainedSite))
    return position;

  auto inputType = cast<RankedTensorType>(position.getType());
  Type elemType = inputType.getElementType();
  int64_t inSize = inputType.getShape()[1];
  int64_t outSize = 0;
  for (const SupportInfo &info : supports)
    outSize += toUnconstrained ? info.unconstrainedSize : info.constrainedSize;

  auto positionType1D = RankedTensorType::get({inSize}, elemType);
  Value position1D =
      impulse::ReshapeOp::create(builder, loc, positionType1D, position);

  SmallVector<Value> slices;
  for (const SupportInfo &info : supports) {
    int64_t offset =
        toUnconstrained ? info.constrainedOffset : info.unconstrainedOffset;
    int64_t size =
        toUnconstrained ? info.constrainedSize : info.unconstrainedSize;
    auto sliceType = RankedTensorType::get({size}, elemType);
    Value slice =
        impulse::SliceOp::create(builder, loc, sliceType, position1D,
                                 builder.getDenseI64ArrayAttr({offset}),
                                 builder.getDenseI64ArrayAttr({offset + size}),
                                 builder.getDenseI64ArrayAttr({1}));
    if (isConstrainedSite(info))
      slice = toUnconstrained
                  ? transforms::unconstrain(builder, loc, slice, info.support)
                  : transforms::constrain(builder, loc, slice, info.support);
    slices.push_back(slice);
  }

  Value result1D = slices.front();
  if (slices.size() > 1) {
    auto resultType1D = RankedTensorType::get({outSize}, elemType);
    result1D = emitSplatConstant(builder, loc, resultType1D, 0.0);

    // The slices tile [0, outSize) exactly, so the zero init is never read.
    int64_t offset = 0;
    for (Value slice : slices) {
      Value resultIdx = emitI64ScalarConstant(builder, loc, offset);
      result1D = impulse::DynamicUpdateSliceOp::create(
          builder, loc, resultType1D, result1D, slice, ValueRange{resultIdx});
      offset += cast<RankedTensorType>(slice.getType()).getShape()[0];
    }
  }

  auto resultType2D = RankedTensorType::get({1, outSize}, elemType);
  return impulse::ReshapeOp::create(builder, loc, resultType2D, result1D);
}

Value impulse::unconstrainPosition(OpBuilder &builder, Location loc,
                                   Value constrained,
                                   ArrayRef<SupportInfo> supports) {
  return transformPosition(builder, loc, constrained, supports,
                           /*toUnconstrained=*/true);
}

Value impulse::constrainPosition(OpBuilder &builder, Location loc,
                                 Value unconstrained,
                                 ArrayRef<SupportInfo> supports) {
  return transformPosition(builder, loc, unconstrained, supports,
                           /*toUnconstrained=*/false);
}

Value impulse::computeTotalJacobianCorrection(OpBuilder &builder, Location loc,
                                              Value unconstrained,
                                              ArrayRef<SupportInfo> supports) {
  auto inputType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = inputType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value total = emitSplatConstant(builder, loc, scalarType, 0.0);

  if (supports.empty())
    return total;

  int64_t positionSize = inputType.getShape()[1];
  auto positionType1D = RankedTensorType::get({positionSize}, elemType);
  Value unconstrained1D =
      impulse::ReshapeOp::create(builder, loc, positionType1D, unconstrained);

  for (const auto &info : supports) {
    if (!isConstrainedSite(info))
      continue;

    auto sliceType = RankedTensorType::get({info.unconstrainedSize}, elemType);
    auto slice = impulse::SliceOp::create(
        builder, loc, sliceType, unconstrained1D,
        builder.getDenseI64ArrayAttr({info.unconstrainedOffset}),
        builder.getDenseI64ArrayAttr(
            {info.unconstrainedOffset + info.unconstrainedSize}),
        builder.getDenseI64ArrayAttr({1}));
    auto jacobian =
        transforms::logAbsDetJacobian(builder, loc, slice, info.support);

    total = arith::AddFOp::create(builder, loc, total, jacobian);
  }

  return total;
}
