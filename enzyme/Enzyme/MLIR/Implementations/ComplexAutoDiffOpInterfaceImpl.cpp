//===- ComplexAutoDiffOpInterfaceImpl.cpp - Interface external model --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR complex dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/ComplexDerivatives.inc"

bool isZero(mlir::Value v) {
  ArrayAttr lhs;
  matchPattern(v, m_Constant(&lhs));
  if (lhs) {
    for (auto e : lhs) {
      if (!cast<FloatAttr>(e).getValue().isZero())
        return false;
    }
    return true;
  }
  return false;
}

struct ComplexAddSimplifyMathInterface
    : public MathSimplifyInterface::ExternalModel<
          ComplexAddSimplifyMathInterface, complex::AddOp> {
  mlir::LogicalResult simplifyMath(Operation *src,
                                   PatternRewriter &rewriter) const {
    auto op = cast<complex::AddOp>(src);

    if (isZero(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    if (isZero(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    return failure();
  }
};

struct ComplexSubSimplifyMathInterface
    : public MathSimplifyInterface::ExternalModel<
          ComplexSubSimplifyMathInterface, complex::SubOp> {
  mlir::LogicalResult simplifyMath(Operation *src,
                                   PatternRewriter &rewriter) const {
    auto op = cast<complex::SubOp>(src);

    if (isZero(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (isZero(op.getLhs())) {
      rewriter.replaceOpWithNewOp<complex::NegOp>(op, op.getRhs());
      return success();
    }

    return failure();
  }
};

} // namespace

void mlir::enzyme::registerComplexDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, complex::ComplexDialect *) {
    complex::AddOp::attachInterface<ComplexAddSimplifyMathInterface>(*context);
    complex::SubOp::attachInterface<ComplexSubSimplifyMathInterface>(*context);
    registerInterfaces(context);
  });
}
