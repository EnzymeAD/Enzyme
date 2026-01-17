//===- TransformUtils.cpp - Constraint transforms for HMC ------* C++ -*-===//
//
// This file implements constraint transforms for HMC inference.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
//
//===----------------------------------------------------------------------===//

#include "TransformUtils.h"

#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <cmath>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::transforms;

Value transforms::createLogit(OpBuilder &builder, Location loc, Value x) {
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

Value transforms::createLogSigmoid(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  // log_sigmoid(x) = -softplus(-x) = -log_add_exp(-x, 0)
  auto negX = arith::NegFOp::create(builder, loc, x);
  auto zeroConst = arith::ConstantOp::create(
      builder, loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 0.0)));
  auto softplusNegX =
      enzyme::LogAddExpOp::create(builder, loc, xType, negX, zeroConst);
  return arith::NegFOp::create(builder, loc, softplusNegX);
}

int64_t transforms::getUnconstrainedSize(int64_t constrainedSize,
                                         SupportKind kind) {
  switch (kind) {
  case SupportKind::REAL:
  case SupportKind::POSITIVE:
  case SupportKind::UNIT_INTERVAL:
  case SupportKind::INTERVAL:
  case SupportKind::GREATER_THAN:
  case SupportKind::LESS_THAN:
    return constrainedSize;
  }
  llvm_unreachable("Unknown SupportKind");
}

int64_t transforms::getConstrainedSize(int64_t unconstrainedSize,
                                       SupportKind kind) {
  switch (kind) {
  case SupportKind::REAL:
  case SupportKind::POSITIVE:
  case SupportKind::UNIT_INTERVAL:
  case SupportKind::INTERVAL:
  case SupportKind::GREATER_THAN:
  case SupportKind::LESS_THAN:
    return unconstrainedSize;
  }
  llvm_unreachable("Unknown SupportKind");
}

Value transforms::unconstrain(OpBuilder &builder, Location loc,
                              Value constrained, SupportAttr support) {
  auto kind = support.getKind();
  auto xType = cast<RankedTensorType>(constrained.getType());
  auto elemType = xType.getElementType();

  switch (kind) {
  case SupportKind::REAL:
    // Identity
    return constrained;
  case SupportKind::POSITIVE:
    return math::LogOp::create(builder, loc, constrained);
  case SupportKind::UNIT_INTERVAL:
    return createLogit(builder, loc, constrained);
  case SupportKind::INTERVAL: {
    // z = logit((x - a) / (b - a))
    auto lowerAttr = support.getLowerBound();
    auto upperAttr = support.getUpperBound();
    if (!lowerAttr || !upperAttr) {
      llvm_unreachable("INTERVAL support requires lower and upper bounds");
    }
    double lower = lowerAttr.getValueAsDouble();
    double upper = upperAttr.getValueAsDouble();

    auto lowerConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, lower)));
    auto scaleConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType,
                               builder.getFloatAttr(elemType, upper - lower)));

    auto shifted = arith::SubFOp::create(builder, loc, constrained, lowerConst);
    auto normalized = arith::DivFOp::create(builder, loc, shifted, scaleConst);
    return createLogit(builder, loc, normalized);
  }
  case SupportKind::GREATER_THAN: {
    // z = log(x - lower)
    auto lowerAttr = support.getLowerBound();
    if (!lowerAttr) {
      llvm_unreachable("GREATER_THAN support requires lower bound");
    }
    double lower = lowerAttr.getValueAsDouble();

    auto lowerConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, lower)));
    auto shifted = arith::SubFOp::create(builder, loc, constrained, lowerConst);
    return math::LogOp::create(builder, loc, shifted);
  }
  case SupportKind::LESS_THAN: {
    // z = log(upper - x)
    auto upperAttr = support.getUpperBound();
    if (!upperAttr) {
      llvm_unreachable("LESS_THAN support requires upper bound");
    }
    double upper = upperAttr.getValueAsDouble();

    auto upperConst = arith::ConstantOp::create(
        builder, loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, upper)));
    auto shifted = arith::SubFOp::create(builder, loc, upperConst, constrained);
    return math::LogOp::create(builder, loc, shifted);
  }
  }
  llvm_unreachable("Unknown SupportKind");
}

Value transforms::constrain(OpBuilder &builder, Location loc,
                            Value unconstrained, SupportAttr support) {
  auto kind = support.getKind();
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();

  switch (kind) {
  case SupportKind::REAL:
    // Identity
    return unconstrained;
  case SupportKind::POSITIVE:
    return math::ExpOp::create(builder, loc, unconstrained);
  case SupportKind::UNIT_INTERVAL:
    // x = sigmoid(z)
    return enzyme::LogisticOp::create(builder, loc, unconstrained.getType(),
                                      unconstrained);
  case SupportKind::INTERVAL: {
    // x = a + (b - a) * sigmoid(z)
    auto lowerAttr = support.getLowerBound();
    auto upperAttr = support.getUpperBound();
    if (!lowerAttr || !upperAttr) {
      llvm_unreachable("INTERVAL support requires lower and upper bounds");
    }
    double lower = lowerAttr.getValueAsDouble();
    double upper = upperAttr.getValueAsDouble();

    auto lowerConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, lower)));
    auto scaleConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType,
                               builder.getFloatAttr(elemType, upper - lower)));

    auto sigmoid = enzyme::LogisticOp::create(
        builder, loc, unconstrained.getType(), unconstrained);
    auto scaled = arith::MulFOp::create(builder, loc, scaleConst, sigmoid);
    return arith::AddFOp::create(builder, loc, lowerConst, scaled);
  }
  case SupportKind::GREATER_THAN: {
    // x = lower + exp(z)
    auto lowerAttr = support.getLowerBound();
    if (!lowerAttr) {
      llvm_unreachable("GREATER_THAN support requires lower bound");
    }
    double lower = lowerAttr.getValueAsDouble();

    auto lowerConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, lower)));
    auto expZ = math::ExpOp::create(builder, loc, unconstrained);
    return arith::AddFOp::create(builder, loc, lowerConst, expZ);
  }
  case SupportKind::LESS_THAN: {
    // x = upper - exp(z)
    auto upperAttr = support.getUpperBound();
    if (!upperAttr) {
      llvm_unreachable("LESS_THAN support requires upper bound");
    }
    double upper = upperAttr.getValueAsDouble();

    auto upperConst = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, upper)));
    auto expZ = math::ExpOp::create(builder, loc, unconstrained);
    return arith::SubFOp::create(builder, loc, upperConst, expZ);
  }
  }

  llvm_unreachable("Unknown SupportKind");
}

Value transforms::logAbsDetJacobian(OpBuilder &builder, Location loc,
                                    Value unconstrained, SupportAttr support) {
  auto kind = support.getKind();
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  switch (kind) {
  case SupportKind::REAL: {
    // Identity: log|det(I)| = 0
    return arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 0.0)));
  }
  case SupportKind::POSITIVE: {
    // x = exp(z), dx/dz = exp(z)
    // log|det(J)| = sum(log|dx_i/dz_i|) = sum(z)
    auto ones = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return enzyme::DotOp::create(
        builder, loc, scalarType, unconstrained, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case SupportKind::UNIT_INTERVAL: {
    // x = sigmoid(z), dx/dz = sigmoid(z) * (1 - sigmoid(z))
    // log|det(J)| = sum(log(sigmoid(z)) + log(1 - sigmoid(z)))
    //             = sum(log_sigmoid(z) + log_sigmoid(-z))
    auto logSigZ = createLogSigmoid(builder, loc, unconstrained);
    auto negZ = arith::NegFOp::create(builder, loc, unconstrained);
    auto logSigNegZ = createLogSigmoid(builder, loc, negZ);
    auto logProduct = arith::AddFOp::create(builder, loc, logSigZ, logSigNegZ);
    auto ones = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return enzyme::DotOp::create(
        builder, loc, scalarType, logProduct, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case SupportKind::INTERVAL: {
    // log|det(J)| = sum(log_sigmoid(z) + log(1 - sigmoid(z))) + n*log(scale)
    auto lowerAttr = support.getLowerBound();
    auto upperAttr = support.getUpperBound();
    if (!lowerAttr || !upperAttr) {
      llvm_unreachable("INTERVAL support requires lower and upper bounds");
    }
    double scale = upperAttr.getValueAsDouble() - lowerAttr.getValueAsDouble();

    auto logSigZ = createLogSigmoid(builder, loc, unconstrained);
    auto negZ = arith::NegFOp::create(builder, loc, unconstrained);
    auto logSigNegZ = createLogSigmoid(builder, loc, negZ);
    auto logProduct = arith::AddFOp::create(builder, loc, logSigZ, logSigNegZ);
    auto ones = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    auto sumLogProduct = enzyme::DotOp::create(
        builder, loc, scalarType, logProduct, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));

    int64_t n = zType.getNumElements();
    double logScaleTerm = n * std::log(scale);
    auto logScaleConst = arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, logScaleTerm)));
    return arith::AddFOp::create(builder, loc, sumLogProduct, logScaleConst);
  }
  case SupportKind::GREATER_THAN:
  case SupportKind::LESS_THAN: {
    // log|det(J)| = sum(z)
    auto ones = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return enzyme::DotOp::create(
        builder, loc, scalarType, unconstrained, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  }

  llvm_unreachable("Unknown SupportKind");
}
