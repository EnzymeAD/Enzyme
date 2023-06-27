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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct MulFOpInterface
    : public AutoDiffOpInterface::ExternalModel<MulFOpInterface,
                                                arith::MulFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a * b -> dr = a * db + da * b
    auto mulOp = cast<arith::MulFOp>(op);
    if (!gutils->isConstantValue(mulOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(mulOp.getOperand(i))) {
          mlir::Value tmp = builder.create<arith::MulFOp>(
              mulOp.getLoc(),
              gutils->invertPointerM(mulOp.getOperand(i), builder),
              gutils->getNewFromOriginal(mulOp.getOperand(1 - i)));
          if (res == nullptr)
            res = tmp;
          else
            res = builder.create<arith::AddFOp>(mulOp.getLoc(), res, tmp);
        }
      }
      gutils->setDiffe(mulOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct AddFOpInterface
    : public AutoDiffOpInterface::ExternalModel<AddFOpInterface,
                                                arith::AddFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a + b -> dr = da + db
    auto addOp = cast<arith::AddFOp>(op);
    if (!gutils->isConstantValue(addOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(addOp.getOperand(i))) {
          mlir::Value tmp =
              gutils->invertPointerM(addOp.getOperand(i), builder);
          if (res == nullptr)
            res = tmp;
          else
            res = builder.create<arith::AddFOp>(addOp.getLoc(), res, tmp);
        }
      }
      gutils->setDiffe(addOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

void addToGradient(Value oldGradient, Value addedGradient, OpBuilder &builder,
                   MGradientUtilsReverse *gutils) {
  Value gradient = addedGradient;
  if (gutils->hasInvertPointer(oldGradient)) {
    Value operandGradient = gutils->invertPointerM(oldGradient, builder);
    gradient = builder.create<arith::AddFOp>(oldGradient.getLoc(),
                                             operandGradient, addedGradient);
  }
  gutils->mapInvertPointer(oldGradient, gradient, builder);
}

struct AddFOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<AddFOpInterfaceReverse,
                                                       arith::AddFOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    // Derivative of r = a + b -> dr = da + db
    auto addOp = cast<arith::AddFOp>(op);

    if (gutils->hasInvertPointer(addOp)) {
      Value addedGradient = gutils->invertPointerM(addOp, builder);
      addToGradient(addOp.getLhs(), addedGradient, builder, gutils);
      addToGradient(addOp.getRhs(), addedGradient, builder, gutils);
    }
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct MulFOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<MulFOpInterfaceReverse,
                                                       arith::MulFOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto mulOp = cast<arith::MulFOp>(op);

    if (gutils->hasInvertPointer(mulOp)) {
      Value own_gradient = gutils->invertPointerM(mulOp, builder);
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(mulOp.getOperand(i))) {
          Value cache = caches[i];
          Value retrievedValue = gutils->popCache(cache, builder);
          Value addedGradient = builder.create<arith::MulFOp>(
              mulOp.getLoc(), own_gradient, retrievedValue);

          addToGradient(mulOp.getOperand(i), addedGradient, builder, gutils);
        }
      }
    }
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto mulOp = cast<arith::MulFOp>(op);
    if (gutils->hasInvertPointer(mulOp)) {
      OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
      SmallVector<Value> caches;
      for (int i = 0; i < 2; i++) {
        Value otherOperand = mulOp.getOperand((i + 1) % 2);
        Value cache = gutils->initAndPushCache(
            gutils->getNewFromOriginal(otherOperand), cacheBuilder);
        caches.push_back(cache);
      }
      return caches;
    }
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

} // namespace

void mlir::enzyme::registerArithDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, arith::ArithDialect *) {
    arith::AddFOp::attachInterface<AddFOpInterfaceReverse>(*context);
    arith::MulFOp::attachInterface<MulFOpInterfaceReverse>(*context);

    arith::AddFOp::attachInterface<AddFOpInterface>(*context);
    arith::MulFOp::attachInterface<MulFOpInterface>(*context);
  });
}
