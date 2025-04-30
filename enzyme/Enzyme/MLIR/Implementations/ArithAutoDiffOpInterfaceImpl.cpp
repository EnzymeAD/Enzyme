//===- ArithAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR arithmetic dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

struct ArithConstantOpBatchInterface
    : public BatchOpInterface::ExternalModel<ArithConstantOpBatchInterface,
                                             arith::ConstantOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {

    SmallVector<Type> resultTypes(src->getResultTypes().begin(),
                                  src->getResultTypes().end());
    for (auto &Ty : resultTypes) {
      auto T = cast<TensorType>(Ty);
      SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
      shape.append(T.getShape().begin(), T.getShape().end());
      Ty = T.clone(shape);
    }
    mlir::NamedAttrList attrs;
    for (auto attr : src->getAttrs()) {
      auto eattr = cast<DenseElementsAttr>(attr.getValue());
      attr.setValue(eattr.resizeSplat(cast<ShapedType>(resultTypes[0])));
      attrs.append(attr);
    }
    auto cop = mlir::Operation::create(
        src->getLoc(), src->getName(), resultTypes, {}, std::move(attrs),
        OpaqueProperties(nullptr), mlir::BlockRange(), 0);
    builder.insert(cop);
    mapper.map(src->getResult(0), cop->getResult(0));
    return success();
  }
};

struct ArithAddFSimplifyMathInterface
    : public MathSimplifyInterface::ExternalModel<
          ArithAddFSimplifyMathInterface, arith::AddFOp> {
  mlir::LogicalResult simplifyMath(Operation *src,
                                   PatternRewriter &rewriter) const {
    auto op = cast<arith::AddFOp>(src);

    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    return failure();
  }
};

struct ArithSubFSimplifyMathInterface
    : public MathSimplifyInterface::ExternalModel<
          ArithAddFSimplifyMathInterface, arith::SubFOp> {
  mlir::LogicalResult simplifyMath(Operation *src,
                                   PatternRewriter &rewriter) const {
    auto op = cast<arith::SubFOp>(src);

    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, op.getRhs());
      return success();
    }

    return failure();
  }
};

#include "Implementations/ArithDerivatives.inc"
} // namespace

void mlir::enzyme::registerArithDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, arith::ArithDialect *) {
    registerInterfaces(context);
    arith::ConstantOp::attachInterface<ArithConstantOpBatchInterface>(*context);
    arith::AddFOp::attachInterface<ArithAddFSimplifyMathInterface>(*context);
    arith::SubFOp::attachInterface<ArithSubFSimplifyMathInterface>(*context);
  });
}

void mlir::enzyme::registerTensorDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, tensor::TensorDialect *) {
    registerInterfaces(context);
  });
}
