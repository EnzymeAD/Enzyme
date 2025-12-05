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

#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;
using namespace mlir::enzyme;

SmallVector<Value> NUTSTreeState::toValues() const {
  return {
      q_left,        p_left,     grad_left,     q_right,    p_right,
      grad_right,    q_proposal, grad_proposal, U_proposal, H_proposal,
      depth,         weight,     turning,       diverging,  sum_accept_probs,
      num_proposals, p_sum};
}

NUTSTreeState NUTSTreeState::fromValues(ArrayRef<Value> values) {
  assert(values.size() == NUM_FIELDS);
  NUTSTreeState tree;
  tree.q_left = values[0];
  tree.p_left = values[1];
  tree.grad_left = values[2];
  tree.q_right = values[3];
  tree.p_right = values[4];
  tree.grad_right = values[5];
  tree.q_proposal = values[6];
  tree.grad_proposal = values[7];
  tree.U_proposal = values[8];
  tree.H_proposal = values[9];
  tree.depth = values[10];
  tree.weight = values[11];
  tree.turning = values[12];
  tree.diverging = values[13];
  tree.sum_accept_probs = values[14];
  tree.num_proposals = values[15];
  tree.p_sum = values[16];
  return tree;
}

SmallVector<Type> NUTSTreeState::getTypes() const {
  SmallVector<Type> types;
  for (auto val : toValues())
    types.push_back(val.getType());
  return types;
}

Value enzyme::createIdentityMatrix(OpBuilder &builder, Location loc,
                                   RankedTensorType matrixType) {
  auto shape = matrixType.getShape();
  assert(shape.size() == 2 && shape[0] == shape[1] &&
         "Identity matrix must be square");
  int64_t n = shape[0];

  SmallVector<double> identityData(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    identityData[i * n + i] = 1.0;
  }

  return arith::ConstantOp::create(
      builder, loc, matrixType,
      DenseElementsAttr::get(matrixType, ArrayRef<double>(identityData)));
}

Value enzyme::createSigmoid(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  auto oneConst = arith::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 1.0)));
  auto negX = arith::NegFOp::create(builder, loc, x);
  auto expNegX = math::ExpOp::create(builder, loc, negX);
  auto onePlusExp = arith::AddFOp::create(builder, loc, oneConst, expNegX);
  auto result = arith::DivFOp::create(builder, loc, oneConst, onePlusExp);
  return result;
}

Value enzyme::conditionalDump(OpBuilder &builder, Location loc, Value value,
                              StringRef label, bool debugDump) {
  if (debugDump) {
    return enzyme::DumpOp::create(builder, loc, value.getType(), value,
                                  builder.getStringAttr(label))
        .getOutput();
  }
  return value;
}

Value enzyme::applyInverseMassMatrix(OpBuilder &builder, Location loc,
                                     Value invMass, Value momentum,
                                     RankedTensorType positionType) {
  if (!invMass) {
    return momentum;
  }

  auto invMassType = cast<RankedTensorType>(invMass.getType());

  if (invMassType.getRank() == 1) {
    // Diagonal: element-wise
    return arith::MulFOp::create(builder, loc, invMass, momentum);
  } else if (invMassType.getRank() == 2) {
    // Dense: v = invMass @ p
    return enzyme::DotOp::create(
        builder, loc, positionType, invMass, momentum,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({0}));
  }

  emitError(loc, "ProbProg: Provided invMass must have rank 1 or 2, got rank " +
                     std::to_string(invMassType.getRank()));
  return nullptr;
}

Value enzyme::computeKineticEnergy(OpBuilder &builder, Location loc,
                                   Value momentum, Value invMass,
                                   Value halfConst, RankedTensorType scalarType,
                                   RankedTensorType positionType) {
  // v = M^-1 @ p
  Value v =
      applyInverseMassMatrix(builder, loc, invMass, momentum, positionType);

  // K = 0.5 * p^T @ v
  auto pDotV = enzyme::DotOp::create(
      builder, loc, scalarType, momentum, v, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({0}));

  return arith::MulFOp::create(builder, loc, halfConst, pDotV);
}

std::pair<Value, Value> enzyme::sampleMomentum(OpBuilder &builder, Location loc,
                                               Value rngState, Value invMass,
                                               Value zeroConst, Value oneConst,
                                               RankedTensorType positionType) {
  // Sample eps ~ N(0, I)
  auto randomOp = enzyme::RandomOp::create(
      builder, loc, TypeRange{rngState.getType(), positionType}, rngState,
      zeroConst, oneConst,
      enzyme::RngDistributionAttr::get(builder.getContext(),
                                       enzyme::RngDistribution::NORMAL));

  Value rngOut = randomOp.getOutputRngState();
  Value eps = randomOp.getResult();

  if (!invMass) {
    return {eps, rngOut};
  }

  auto invMassType = cast<RankedTensorType>(invMass.getType());

  if (invMassType.getRank() == 1) {
    // Diagonal: p = (1/sqrt(invMass)) * eps = sqrt(M) * eps
    auto sqrtInvMass = math::SqrtOp::create(builder, loc, invMass);
    auto onesVector = arith::ConstantOp::create(
        builder, loc, invMassType,
        DenseElementsAttr::get(invMassType, builder.getF64FloatAttr(1.0)));
    auto massMatrixSqrt =
        arith::DivFOp::create(builder, loc, onesVector, sqrtInvMass);
    Value p = arith::MulFOp::create(builder, loc, massMatrixSqrt, eps);
    return {p, rngOut};
  } else {
    // Dense: p = chol(M) @ eps where M = inv(invMass)
    auto identityMatrix = createIdentityMatrix(builder, loc, invMassType);
    auto massMatrixSqrt = enzyme::CholeskySolveOp::create(
        builder, loc, invMassType, invMass, identityMatrix);
    Value p = enzyme::DotOp::create(
        builder, loc, positionType, massMatrixSqrt, eps,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({0}));
    return {p, rngOut};
  }
}

GradientResult enzyme::computePotentialAndGradient(
    OpBuilder &builder, Location loc, Value position, Value rng,
    FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs, Value originalTrace,
    ArrayAttr selection, enzyme::TraceType traceType,
    RankedTensorType scalarType, RankedTensorType positionType) {

  auto gradSeed = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getF64FloatAttr(1.0)));

  auto autodiffOp = enzyme::AutoDiffRegionOp::create(
      builder, loc, TypeRange{scalarType, rng.getType(), positionType},
      ValueRange{position, gradSeed},
      builder.getArrayAttr({enzyme::ActivityAttr::get(
          builder.getContext(), enzyme::Activity::enzyme_active)}),
      builder.getArrayAttr(
          {enzyme::ActivityAttr::get(builder.getContext(),
                                     enzyme::Activity::enzyme_active),
           enzyme::ActivityAttr::get(builder.getContext(),
                                     enzyme::Activity::enzyme_const)}),
      builder.getI64IntegerAttr(1), builder.getBoolAttr(false), nullptr);

  Block *autodiffBlock = builder.createBlock(&autodiffOp.getBody());
  autodiffBlock->addArgument(positionType, loc);

  builder.setInsertionPointToStart(autodiffBlock);
  Value qArg = autodiffBlock->getArgument(0);

  SmallVector<Value> updateInputs;
  updateInputs.push_back(rng);
  updateInputs.append(fnInputs.begin(), fnInputs.end());

  auto updateOp = enzyme::UpdateOp::create(
      builder, loc, TypeRange{traceType, scalarType, rng.getType()}, fn,
      updateInputs, originalTrace, qArg, selection, builder.getStringAttr(""));

  Value U = arith::NegFOp::create(builder, loc, updateOp.getWeight());

  enzyme::YieldOp::create(builder, loc,
                          ValueRange{U, updateOp.getOutputRngState()});

  builder.setInsertionPointAfter(autodiffOp);

  return {
      autodiffOp.getResult(0), // potential
      autodiffOp.getResult(2), // gradient
      autodiffOp.getResult(1)  // rngOut
  };
}

LeapfrogResult enzyme::emitLeapfrogStep(
    OpBuilder &builder, Location loc, Value q, Value p, Value grad, Value rng,
    Value stepSize, Value invMass, FlatSymbolRefAttr fn,
    ArrayRef<Value> fnInputs, Value originalTrace, ArrayAttr selection,
    enzyme::TraceType traceType, RankedTensorType scalarType,
    RankedTensorType positionType) {

  auto halfConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getF64FloatAttr(0.5)));

  ArrayRef<int64_t> shape = positionType.getShape();
  auto stepSizeBroadcast =
      enzyme::BroadcastOp::create(builder, loc, positionType, stepSize,
                                  builder.getDenseI64ArrayAttr(shape));
  auto halfStep = arith::MulFOp::create(builder, loc, halfConst, stepSize);
  auto halfStepBroadcast =
      enzyme::BroadcastOp::create(builder, loc, positionType, halfStep,
                                  builder.getDenseI64ArrayAttr(shape));

  // 1. Half step momentum: p_half = p - 0.5 * eps * grad
  auto deltaP1 = arith::MulFOp::create(builder, loc, halfStepBroadcast, grad);
  Value pHalf = arith::SubFOp::create(builder, loc, p, deltaP1);

  // 2. Full step position: q_new = q + eps * M^-1 * p_half
  Value v = applyInverseMassMatrix(builder, loc, invMass, pHalf, positionType);
  auto deltaQ = arith::MulFOp::create(builder, loc, stepSizeBroadcast, v);
  Value qNew = arith::AddFOp::create(builder, loc, q, deltaQ);

  // 3. Compute gradient at new position
  auto gradResult = computePotentialAndGradient(
      builder, loc, qNew, rng, fn, fnInputs, originalTrace, selection,
      traceType, scalarType, positionType);

  // 4. Final half step momentum: p_new = p_half - 0.5 * eps * grad_new
  auto deltaP2 = arith::MulFOp::create(builder, loc, halfStepBroadcast,
                                       gradResult.gradient);
  Value pNew = arith::SubFOp::create(builder, loc, pHalf, deltaP2);

  return {qNew, pNew, gradResult.gradient, gradResult.potential,
          gradResult.rngOut};
}

Value enzyme::checkTurning(OpBuilder &builder, Location loc, Value invMass,
                           Value pLeft, Value pRight, Value pSum,
                           Value zeroConst, RankedTensorType scalarType,
                           RankedTensorType positionType) {
  Value vLeft =
      applyInverseMassMatrix(builder, loc, invMass, pLeft, positionType);
  Value vRight =
      applyInverseMassMatrix(builder, loc, invMass, pRight, positionType);

  // p_sum_centered = p_sum - (p_left + p_right) / 2
  auto halfConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getF64FloatAttr(0.5)));
  auto halfBroadcast = enzyme::BroadcastOp::create(
      builder, loc, positionType, halfConst,
      builder.getDenseI64ArrayAttr(positionType.getShape()));

  auto pLeftPlusPRight = arith::AddFOp::create(builder, loc, pLeft, pRight);
  auto halfSum =
      arith::MulFOp::create(builder, loc, halfBroadcast, pLeftPlusPRight);
  Value pSumCentered = arith::SubFOp::create(builder, loc, pSum, halfSum);

  auto leftAngle = enzyme::DotOp::create(
      builder, loc, scalarType, vLeft, pSumCentered,
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  auto rightAngle = enzyme::DotOp::create(
      builder, loc, scalarType, vRight, pSumCentered,
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));

  // turning = (left_angle <= 0) OR (right_angle <= 0)
  auto leftNeg = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                       leftAngle, zeroConst);
  auto rightNeg = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLE,
                                        rightAngle, zeroConst);

  return arith::OrIOp::create(builder, loc, leftNeg, rightNeg);
}

Value enzyme::computeUniformTransitionProb(OpBuilder &builder, Location loc,
                                           Value currentWeight,
                                           Value newWeight) {
  Value weightDiff =
      arith::SubFOp::create(builder, loc, newWeight, currentWeight);
  return createSigmoid(builder, loc, weightDiff);
}

Value enzyme::computeBiasedTransitionProb(OpBuilder &builder, Location loc,
                                          Value currentWeight, Value newWeight,
                                          Value turning, Value diverging,
                                          Value zeroConst, Value oneConst) {
  auto resultType = cast<RankedTensorType>(currentWeight.getType());

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

NUTSTreeState enzyme::combineTrees(
    OpBuilder &builder, Location loc, const NUTSTreeState &currentTree,
    const NUTSTreeState &newTree, Value invMass, Value goingRight, Value rngKey,
    bool biasedTransition, Value zeroConst, Value oneConst,
    RankedTensorType scalarType, RankedTensorType positionType,
    RankedTensorType i64TensorType, RankedTensorType i1TensorType) {

  auto goingRightBroadcast = enzyme::BroadcastOp::create(
      builder, loc,
      RankedTensorType::get(positionType.getShape(), builder.getI1Type()),
      goingRight, builder.getDenseI64ArrayAttr(positionType.getShape()));

  Value qLeft =
      arith::SelectOp::create(builder, loc, positionType, goingRightBroadcast,
                              currentTree.q_left, newTree.q_left);
  Value pLeft =
      arith::SelectOp::create(builder, loc, positionType, goingRightBroadcast,
                              currentTree.p_left, newTree.p_left);
  Value gradLeft =
      arith::SelectOp::create(builder, loc, positionType, goingRightBroadcast,
                              currentTree.grad_left, newTree.grad_left);
  Value qRight =
      arith::SelectOp::create(builder, loc, positionType, goingRightBroadcast,
                              newTree.q_right, currentTree.q_right);
  Value pRight =
      arith::SelectOp::create(builder, loc, positionType, goingRightBroadcast,
                              newTree.p_right, currentTree.p_right);
  Value gradRight =
      arith::SelectOp::create(builder, loc, positionType, goingRightBroadcast,
                              newTree.grad_right, currentTree.grad_right);

  Value combinedWeight = enzyme::LogAddExpOp::create(
      builder, loc, scalarType, currentTree.weight, newTree.weight);

  // Compute transition probability
  Value transitionProb;
  if (biasedTransition) {
    transitionProb = computeBiasedTransitionProb(
        builder, loc, currentTree.weight, newTree.weight, newTree.turning,
        newTree.diverging, zeroConst, oneConst);
  } else {
    transitionProb = computeUniformTransitionProb(
        builder, loc, currentTree.weight, newTree.weight);
  }

  auto randomOp = enzyme::RandomOp::create(
      builder, loc, TypeRange{rngKey.getType(), scalarType}, rngKey, zeroConst,
      oneConst,
      enzyme::RngDistributionAttr::get(builder.getContext(),
                                       enzyme::RngDistribution::UNIFORM));
  Value uniformSample = randomOp.getResult();

  auto acceptNew = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLT, uniformSample, transitionProb);
  auto acceptNewBroadcast = enzyme::BroadcastOp::create(
      builder, loc,
      RankedTensorType::get(positionType.getShape(), builder.getI1Type()),
      acceptNew, builder.getDenseI64ArrayAttr(positionType.getShape()));

  Value qProposal =
      arith::SelectOp::create(builder, loc, positionType, acceptNewBroadcast,
                              newTree.q_proposal, currentTree.q_proposal);
  Value gradProposal =
      arith::SelectOp::create(builder, loc, positionType, acceptNewBroadcast,
                              newTree.grad_proposal, currentTree.grad_proposal);
  Value UProposal =
      arith::SelectOp::create(builder, loc, scalarType, acceptNew,
                              newTree.U_proposal, currentTree.U_proposal);
  Value HProposal =
      arith::SelectOp::create(builder, loc, scalarType, acceptNew,
                              newTree.H_proposal, currentTree.H_proposal);

  auto oneI64 = arith::ConstantOp::create(
      builder, loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));

  Value combinedDepth =
      arith::AddIOp::create(builder, loc, currentTree.depth, oneI64);

  Value combinedTurning;
  if (biasedTransition) {
    Value turningCheck = checkTurning(
        builder, loc, invMass, pLeft, pRight,
        arith::AddFOp::create(builder, loc, currentTree.p_sum, newTree.p_sum),
        zeroConst, scalarType, positionType);
    combinedTurning =
        arith::OrIOp::create(builder, loc, newTree.turning, turningCheck);
  } else {
    combinedTurning = currentTree.turning;
  }

  Value combinedDiverging = arith::OrIOp::create(
      builder, loc, currentTree.diverging, newTree.diverging);
  Value sumAcceptProbs = arith::AddFOp::create(
      builder, loc, currentTree.sum_accept_probs, newTree.sum_accept_probs);
  Value numProposals = arith::AddIOp::create(
      builder, loc, currentTree.num_proposals, newTree.num_proposals);
  Value pSum =
      arith::AddFOp::create(builder, loc, currentTree.p_sum, newTree.p_sum);

  return NUTSTreeState{.q_left = qLeft,
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
                       .p_sum = pSum};
}

InitialHMCState enzyme::initializeHMCState(
    OpBuilder &builder, Location loc, Value rngState, Value originalTrace,
    ArrayAttr selection, Value invMass,
    Value initialMomentum, // Debug
    FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs, RankedTensorType scalarType,
    RankedTensorType positionType, bool debugDump) {

  auto traceType = enzyme::TraceType::get(builder.getContext());
  auto elemType = scalarType.getElementType();

  // 1. Extract initial position vector q0
  auto q0 = enzyme::GetFlattenedSamplesFromTraceOp::create(
      builder, loc, positionType, originalTrace, selection);

  // 2. Compute initial potential energy U0 = -weight
  auto weight0 = enzyme::GetWeightFromTraceOp::create(builder, loc, scalarType,
                                                      originalTrace);
  Value U0 = conditionalDump(builder, loc,
                             arith::NegFOp::create(builder, loc, weight0),
                             "HMC: initial potential energy U0", debugDump);

  auto zeroConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto oneConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));
  auto halfConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.5)));

  Value rng1;
  Value p0;

  // 3. Sample initial momentum p0 ~ N(0, M) if not provided
  if (initialMomentum) {
    p0 = initialMomentum;
    rng1 = rngState;
  } else {
    std::tie(p0, rng1) = sampleMomentum(builder, loc, rngState, invMass,
                                        zeroConst, oneConst, positionType);
  }

  // 4. Compute initial kinetic energy K0 = 0.5 * p^T * M^-1 * p
  Value K0 =
      conditionalDump(builder, loc,
                      computeKineticEnergy(builder, loc, p0, invMass, halfConst,
                                           scalarType, positionType),
                      "HMC: initial kinetic energy K0", debugDump);

  Value H0 =
      conditionalDump(builder, loc, arith::AddFOp::create(builder, loc, U0, K0),
                      "HMC: initial Hamiltonian H0", debugDump);

  // 5. Compute initial gradient at q0
  auto gradSeedInit = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));
  auto autodiffInit = enzyme::AutoDiffRegionOp::create(
      builder, loc, TypeRange{rng1.getType(), positionType},
      ValueRange{q0, gradSeedInit},
      builder.getArrayAttr({enzyme::ActivityAttr::get(
          builder.getContext(), enzyme::Activity::enzyme_active)}),
      builder.getArrayAttr(
          {enzyme::ActivityAttr::get(builder.getContext(),
                                     enzyme::Activity::enzyme_activenoneed),
           enzyme::ActivityAttr::get(builder.getContext(),
                                     enzyme::Activity::enzyme_const)}),
      builder.getI64IntegerAttr(1), builder.getBoolAttr(false), nullptr);

  Block *autodiffInitBlock = builder.createBlock(&autodiffInit.getBody());
  autodiffInitBlock->addArgument(positionType, loc);

  builder.setInsertionPointToStart(autodiffInitBlock);
  Value q0Arg = autodiffInitBlock->getArgument(0);

  SmallVector<Value> updateInputsInit;
  updateInputsInit.push_back(rng1);
  updateInputsInit.append(fnInputs.begin(), fnInputs.end());

  auto updateOpInit = enzyme::UpdateOp::create(
      builder, loc, TypeRange{traceType, scalarType, rng1.getType()}, fn,
      updateInputsInit, originalTrace, q0Arg, selection,
      builder.getStringAttr(""));
  Value w0 = updateOpInit.getWeight();
  Value rng0_out = updateOpInit.getOutputRngState();
  Value U0_init = arith::NegFOp::create(builder, loc, w0);

  enzyme::YieldOp::create(builder, loc, ValueRange{U0_init, rng0_out});

  builder.setInsertionPointAfter(autodiffInit);
  Value rng0_final = autodiffInit.getResult(0);
  Value grad0 = autodiffInit.getResult(1);

  return InitialHMCState{
      .position = q0,
      .momentum = p0,
      .potential = U0,
      .kinetic = K0,
      .hamiltonian = H0,
      .gradient = grad0,
      .rng = rng0_final,
  };
}

std::tuple<Value, Value, Value> enzyme::finalizeHMCStep(
    OpBuilder &builder, Location loc, Value proposedPosition,
    Value proposedMomentum, Value H0, Value invMass, Value rngState,
    Value originalTrace, ArrayAttr selection, FlatSymbolRefAttr fn,
    ArrayRef<Value> fnInputs, RankedTensorType scalarType,
    RankedTensorType positionType, bool debugDump) {

  auto traceType = enzyme::TraceType::get(builder.getContext());
  auto elemType = scalarType.getElementType();

  auto zeroConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto oneConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));
  auto halfConst = arith::ConstantOp::create(
      builder, loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.5)));

  SmallVector<Value> finalUpdateInputs;
  finalUpdateInputs.push_back(rngState);
  finalUpdateInputs.append(fnInputs.begin(), fnInputs.end());

  auto finalUpdateOp = enzyme::UpdateOp::create(
      builder, loc, TypeRange{traceType, scalarType, rngState.getType()}, fn,
      finalUpdateInputs, originalTrace, proposedPosition, selection,
      builder.getStringAttr(""));
  Value finalTrace = finalUpdateOp.getUpdatedTrace();
  Value weight1 = finalUpdateOp.getWeight();
  Value rngAfterUpdate = finalUpdateOp.getOutputRngState();

  Value U1_final = conditionalDump(builder, loc,
                                   arith::NegFOp::create(builder, loc, weight1),
                                   "HMC: final potential energy U1", debugDump);

  // K1 = 0.5 * pL^T * M^-1 * pL
  Value K1 = conditionalDump(
      builder, loc,
      computeKineticEnergy(builder, loc, proposedMomentum, invMass, halfConst,
                           scalarType, positionType),
      "HMC: final kinetic energy K1", debugDump);

  Value H1 = conditionalDump(builder, loc,
                             arith::AddFOp::create(builder, loc, U1_final, K1),
                             "HMC: final Hamiltonian H1", debugDump);

  // Metropolis-Hastings accept/reject step
  // with acceptance probability: α = min(1, exp(H0 - H1))
  auto dH = arith::SubFOp::create(builder, loc, H0, H1);
  auto expDH = math::ExpOp::create(builder, loc, dH);
  Value accProb = conditionalDump(
      builder, loc, arith::MinimumFOp::create(builder, loc, oneConst, expDH),
      "HMC: acceptance probability α", debugDump);

  auto randomOp = enzyme::RandomOp::create(
      builder, loc, TypeRange{rngAfterUpdate.getType(), scalarType},
      rngAfterUpdate, zeroConst, oneConst,
      enzyme::RngDistributionAttr::get(builder.getContext(),
                                       enzyme::RngDistribution::UNIFORM));
  Value rngFinal = randomOp.getOutputRngState();
  Value randUniform = randomOp.getResult();

  auto acceptedTensor = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OLT, randUniform, accProb);
  auto selectedTrace = enzyme::SelectTraceOp::create(
      builder, loc, traceType, acceptedTensor, finalTrace, originalTrace);

  return {selectedTrace, acceptedTensor, rngFinal};
}

std::tuple<Value, Value, Value> enzyme::finalizeNUTSStep(
    OpBuilder &builder, Location loc, Value proposedPosition, Value rngState,
    Value originalTrace, ArrayAttr selection, FlatSymbolRefAttr fn,
    ArrayRef<Value> fnInputs, RankedTensorType scalarType,
    RankedTensorType positionType, RankedTensorType i1TensorType) {

  auto traceType = enzyme::TraceType::get(builder.getContext());

  SmallVector<Value> finalUpdateInputs;
  finalUpdateInputs.push_back(rngState);
  finalUpdateInputs.append(fnInputs.begin(), fnInputs.end());

  auto finalUpdateOp = enzyme::UpdateOp::create(
      builder, loc, TypeRange{traceType, scalarType, rngState.getType()}, fn,
      finalUpdateInputs, originalTrace, proposedPosition, selection,
      builder.getStringAttr(""));
  Value finalTrace = finalUpdateOp.getUpdatedTrace();
  Value rngAfterUpdate = finalUpdateOp.getOutputRngState();

  // Always accept the proposal
  auto acceptedTensor = arith::ConstantOp::create(
      builder, loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(true)));

  return {finalTrace, acceptedTensor, rngAfterUpdate};
}
