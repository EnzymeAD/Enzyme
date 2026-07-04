//===- TransformUtils.cpp - Constraint transforms for HMC ------* C++ -*-===//
//
// This file implements constraint transforms for HMC inference.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
//
//===----------------------------------------------------------------------===//

#include "TransformUtils.h"

#include "Dialect/Impulse/Impulse.h"
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
      impulse::LogAddExpOp::create(builder, loc, xType, negX, zeroConst);
  return arith::NegFOp::create(builder, loc, softplusNegX);
}

static int64_t simplexEventSize(impulse::SupportAttr support) {
  auto attr = support.getEventSize();
  assert(attr && "SIMPLEX support requires an event_size");
  return attr.getInt();
}

int64_t transforms::getUnconstrainedSize(int64_t constrainedSize,
                                         impulse::SupportAttr support) {
  if (!support)
    return constrainedSize;
  switch (support.getKind()) {
  case impulse::SupportKind::REAL:
  case impulse::SupportKind::POSITIVE:
  case impulse::SupportKind::UNIT_INTERVAL:
  case impulse::SupportKind::INTERVAL:
  case impulse::SupportKind::GREATER_THAN:
  case impulse::SupportKind::LESS_THAN:
    return constrainedSize;
  case impulse::SupportKind::SIMPLEX: {
    int64_t K = simplexEventSize(support);
    int64_t m = constrainedSize / K;
    return m * (K - 1);
  }
  }
  llvm_unreachable("Unknown SupportKind");
}

int64_t transforms::getConstrainedSize(int64_t unconstrainedSize,
                                       impulse::SupportAttr support) {
  if (!support)
    return unconstrainedSize;
  switch (support.getKind()) {
  case impulse::SupportKind::REAL:
  case impulse::SupportKind::POSITIVE:
  case impulse::SupportKind::UNIT_INTERVAL:
  case impulse::SupportKind::INTERVAL:
  case impulse::SupportKind::GREATER_THAN:
  case impulse::SupportKind::LESS_THAN:
    return unconstrainedSize;
  case impulse::SupportKind::SIMPLEX: {
    int64_t K = simplexEventSize(support);
    int64_t m = unconstrainedSize / (K - 1);
    return m * K;
  }
  }
  llvm_unreachable("Unknown SupportKind");
}

namespace {

// Stick-breaking helpers matching NumPyro's StickBreakingTransform. A slice is
// laid out as m contiguous simplex blocks; we reshape it to [m, .] and operate
// per-block along the last (event) axis. K = constrained block length, n = K-1.

struct SimplexDims {
  int64_t K;
  int64_t n; // K - 1
  int64_t m; // number of independent blocks
};

Value splatConst(OpBuilder &builder, Location loc, RankedTensorType ty,
                 double v) {
  return arith::ConstantOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getFloatAttr(ty.getElementType(), v)));
}

// A [m, ncol] constant whose every row equals `rowVals`.
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

// A [n, n] lower-shift matrix of ones: T[i][j] = 1 iff (strict ? i < j : i <=
// j). Right-multiplying a row vector by T gives the (exclusive/inclusive)
// prefix sum along the row.
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

// prefix sum of `mat` [m,n] along the row axis: `mat @ tri`.
Value prefixSum(OpBuilder &builder, Location loc, Value mat, Value tri) {
  return impulse::DotOp::create(
      builder, loc, mat.getType(), mat, tri, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({1}),
      builder.getDenseI64ArrayAttr({0}));
}

// Balanced shift vector as a [m,n] constant (row = [log n, log(n-1), ..., 0]).
Value shiftMat(OpBuilder &builder, Location loc, const SimplexDims &d,
               Type elemType) {
  SmallVector<double> rowVals(d.n);
  for (int64_t c = 0; c < d.n; ++c)
    rowVals[c] = std::log((double)(d.n - c));
  return matConst(builder, loc, RankedTensorType::get({d.m, d.n}, elemType),
                  rowVals);
}

// unconstrained x [m*n] -> constrained y [m*K].
// Operates on whole [m,n] tensors (no per-column [1,1] ops) so the
// differentiated potential mirrors the vectorised scalar-transform jacobians.
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
  Value log1mZ = transforms::createLogSigmoid(builder, loc, negXsh);

  Value logPexcl =
      prefixSum(builder, loc, log1mZ,
                triOnes(builder, loc, d.n, /*strict=*/true, elemType));
  Value P = math::ExpOp::create(builder, loc, logPexcl);
  Value ybody = arith::MulFOp::create(builder, loc, z, P);

  // remainder = prod(1 - z) = exp(sum_j log(1 - z_j))
  auto onesCol =
      splatConst(builder, loc, RankedTensorType::get({d.n, 1}, elemType), 1.0);
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
  Value ybuf = splatConst(builder, loc, yBufType, 0.0);
  ybuf = impulse::DynamicUpdateSliceOp::create(
      builder, loc, yBufType, ybuf, ybody, ValueRange{idx(0), idx(0)});
  ybuf = impulse::DynamicUpdateSliceOp::create(
      builder, loc, yBufType, ybuf, remainder,
      ValueRange{idx(0), idx(d.K - 1)});

  return impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.K}, elemType), ybuf);
}

// constrained y [m*K] -> unconstrained x [m*n]. Not differentiated, so plain
// log/div are fine here.
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

  // z1m = 1 - cumsum(ycrop)  (numpyro's clipped z1m_cumprod)
  Value cumIncl =
      prefixSum(builder, loc, ycrop,
                triOnes(builder, loc, d.n, /*strict=*/false, elemType));
  Value z1m = arith::SubFOp::create(
      builder, loc, splatConst(builder, loc, xMatType, 1.0), cumIncl);
  Value ratio = arith::DivFOp::create(builder, loc, ycrop, z1m);
  Value logRatio = math::LogOp::create(builder, loc, ratio);
  Value x = arith::AddFOp::create(builder, loc, logRatio,
                                  shiftMat(builder, loc, d, elemType));

  return impulse::ReshapeOp::create(
      builder, loc, RankedTensorType::get({d.m * d.n}, elemType), x);
}

// log|det J| of the constrain (unconstrained->constrained) transform.
// Stan's identity per block: sum_k [ log z_k + log(1-z_k) + log P_k ] with
// P_k = prod_{j<k}(1 - z_j). Uses only log_sigmoid (no math.log / math.exp) on
// whole [m,n] tensors so the differentiated potential avoids the log-derivative
// NaN guards.
Value logDetSimplex(OpBuilder &builder, Location loc, Value unconstrained,
                    const SimplexDims &d) {
  auto elemType =
      cast<RankedTensorType>(unconstrained.getType()).getElementType();
  auto xMatType = RankedTensorType::get({d.m, d.n}, elemType);

  Value xmat =
      impulse::ReshapeOp::create(builder, loc, xMatType, unconstrained);
  Value xsh = arith::SubFOp::create(builder, loc, xmat,
                                    shiftMat(builder, loc, d, elemType));
  Value logZ = transforms::createLogSigmoid(builder, loc, xsh);
  Value negXsh = arith::NegFOp::create(builder, loc, xsh);
  Value log1mZ = transforms::createLogSigmoid(builder, loc, negXsh);
  Value logPexcl =
      prefixSum(builder, loc, log1mZ,
                triOnes(builder, loc, d.n, /*strict=*/true, elemType));
  Value term = arith::AddFOp::create(builder, loc, logZ, log1mZ);
  term = arith::AddFOp::create(builder, loc, term, logPexcl);

  auto flatType = RankedTensorType::get({d.m * d.n}, elemType);
  Value flat = impulse::ReshapeOp::create(builder, loc, flatType, term);
  Value ones = splatConst(builder, loc, flatType, 1.0);
  auto scalarType = RankedTensorType::get({}, elemType);
  return impulse::DotOp::create(
      builder, loc, scalarType, flat, ones, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0}),
      builder.getDenseI64ArrayAttr({0}));
}

SimplexDims simplexDimsFromConstrained(impulse::SupportAttr support,
                                       int64_t constrainedSize) {
  int64_t K = support.getEventSize().getInt();
  return {K, K - 1, constrainedSize / K};
}

SimplexDims simplexDimsFromUnconstrained(impulse::SupportAttr support,
                                         int64_t unconstrainedSize) {
  int64_t K = support.getEventSize().getInt();
  return {K, K - 1, unconstrainedSize / (K - 1)};
}

} // namespace

Value transforms::unconstrain(OpBuilder &builder, Location loc,
                              Value constrained, impulse::SupportAttr support) {
  auto kind = support.getKind();
  auto xType = cast<RankedTensorType>(constrained.getType());
  auto elemType = xType.getElementType();

  switch (kind) {
  case impulse::SupportKind::REAL:
    // Identity
    return constrained;
  case impulse::SupportKind::POSITIVE:
    return math::LogOp::create(builder, loc, constrained);
  case impulse::SupportKind::UNIT_INTERVAL:
    return createLogit(builder, loc, constrained);
  case impulse::SupportKind::INTERVAL: {
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
  case impulse::SupportKind::GREATER_THAN: {
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
  case impulse::SupportKind::LESS_THAN: {
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
  case impulse::SupportKind::SIMPLEX:
    return unconstrainSimplex(
        builder, loc, constrained,
        simplexDimsFromConstrained(support, xType.getNumElements()));
  }
  llvm_unreachable("Unknown SupportKind");
}

Value transforms::constrain(OpBuilder &builder, Location loc,
                            Value unconstrained, impulse::SupportAttr support) {
  auto kind = support.getKind();
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();

  switch (kind) {
  case impulse::SupportKind::REAL:
    // Identity
    return unconstrained;
  case impulse::SupportKind::POSITIVE:
    return math::ExpOp::create(builder, loc, unconstrained);
  case impulse::SupportKind::UNIT_INTERVAL:
    // x = sigmoid(z)
    return impulse::LogisticOp::create(builder, loc, unconstrained.getType(),
                                       unconstrained);
  case impulse::SupportKind::INTERVAL: {
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

    auto sigmoid = impulse::LogisticOp::create(
        builder, loc, unconstrained.getType(), unconstrained);
    auto scaled = arith::MulFOp::create(builder, loc, scaleConst, sigmoid);
    return arith::AddFOp::create(builder, loc, lowerConst, scaled);
  }
  case impulse::SupportKind::GREATER_THAN: {
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
  case impulse::SupportKind::LESS_THAN: {
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
  case impulse::SupportKind::SIMPLEX:
    return constrainSimplex(
        builder, loc, unconstrained,
        simplexDimsFromUnconstrained(support, zType.getNumElements()));
  }

  llvm_unreachable("Unknown SupportKind");
}

Value transforms::logAbsDetJacobian(OpBuilder &builder, Location loc,
                                    Value unconstrained,
                                    impulse::SupportAttr support) {
  auto kind = support.getKind();
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  switch (kind) {
  case impulse::SupportKind::REAL: {
    // Identity: log|det(I)| = 0
    return arith::ConstantOp::create(
        builder, loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 0.0)));
  }
  case impulse::SupportKind::POSITIVE: {
    // x = exp(z), dx/dz = exp(z)
    // log|det(J)| = sum(log|dx_i/dz_i|) = sum(z)
    auto ones = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return impulse::DotOp::create(
        builder, loc, scalarType, unconstrained, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case impulse::SupportKind::UNIT_INTERVAL: {
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
    return impulse::DotOp::create(
        builder, loc, scalarType, logProduct, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case impulse::SupportKind::INTERVAL: {
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
    auto sumLogProduct = impulse::DotOp::create(
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
  case impulse::SupportKind::GREATER_THAN:
  case impulse::SupportKind::LESS_THAN: {
    // log|det(J)| = sum(z)
    auto ones = arith::ConstantOp::create(
        builder, loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return impulse::DotOp::create(
        builder, loc, scalarType, unconstrained, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case impulse::SupportKind::SIMPLEX:
    return logDetSimplex(
        builder, loc, unconstrained,
        simplexDimsFromUnconstrained(support, zType.getNumElements()));
  }

  llvm_unreachable("Unknown SupportKind");
}
