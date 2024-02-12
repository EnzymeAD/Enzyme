//===- AffineAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR Affine dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
affine::AffineForOp
createAffineForWithShadows(Operation *op, OpBuilder &builder,
                           MGradientUtils *gutils, Operation *original,
                           ValueRange remappedOperands, TypeRange rettys) {
  affine::AffineForOpAdaptor adaptor(remappedOperands,
                                     cast<affine::AffineForOp>(original));
  auto repFor = builder.create<affine::AffineForOp>(
      original->getLoc(), adaptor.getLowerBoundOperands(),
      adaptor.getLowerBoundMap(), adaptor.getUpperBoundOperands(),
      adaptor.getUpperBoundMap(), adaptor.getStep().getZExtValue(),
      // This dance is necessary because the adaptor accessors are based on the
      // internal attribute containing the number of operands associated with
      // each named operand group. This attribute is carried over from the
      // original operation and does not account for the shadow-related iter
      // args. Instead, assume lower/upper bound operands must not have shadows
      // since they are integer-typed and take the result of operands as iter
      // args.
      remappedOperands.drop_front(adaptor.getLowerBoundOperands().size() +
                                  adaptor.getUpperBoundOperands().size()));
  return repFor;
}

affine::AffineIfOp createAffineIfWithShadows(Operation *op, OpBuilder &builder,
                                             MGradientUtils *gutils,
                                             affine::AffineIfOp original,
                                             ValueRange remappedOperands,
                                             TypeRange rettys) {
  affine::AffineIfOpAdaptor adaptor(remappedOperands, original);
  return builder.create<affine::AffineIfOp>(
      original->getLoc(), rettys, original.getIntegerSet(),
      adaptor.getOperands(), !original.getElseRegion().empty());
}

#include "Implementations/AffineDerivatives.inc"
} // namespace

void mlir::enzyme::registerAffineDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, affine::AffineDialect *) {
    registerInterfaces(context);
  });
}
