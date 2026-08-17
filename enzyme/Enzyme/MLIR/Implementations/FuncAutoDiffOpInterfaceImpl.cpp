//===- FuncAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/FuncDerivatives.inc"
} // namespace

class AutoDiffFuncFuncFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffFuncFuncFunctionInterface, func::FuncOp> {
public:
  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &types) const {}

  // A func.func carries no linkage of its own, but one raised from an
  // llvm.func still holds the primal's comdat as a plain attribute, and the
  // clone inherits it. The derivative is not part of the primal's
  // deduplication group (see the llvm.func model above for why that group is
  // wrong for it), and a func.func cannot express a comdat anyway -- what
  // remains is a dangling nested symbol reference that breaks symbol-use
  // walks such as gpu-kernel-outlining's. Drop it.
  void detachFromPrimalDefinition(Operation *self) const {
    self->removeAttr("comdat");
  }

  Operation *createCall(Operation *self, OpBuilder &builder, Location loc,
                        ValueRange args) const {
    return func::CallOp::create(builder, loc, cast<func::FuncOp>(self), args);
  }

  Operation *createReturn(Operation *self, OpBuilder &builder, Location loc,
                          ValueRange args) const {
    return func::ReturnOp::create(builder, loc, args);
  }
};

void mlir::enzyme::registerFuncDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, func::FuncDialect *) {
    registerInterfaces(context);
    func::FuncOp::attachInterface<AutoDiffFuncFuncFunctionInterface>(*context);
  });
}
