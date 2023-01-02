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
          Value tmp = builder.create<arith::MulFOp>(
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

struct DivFOpInterface
    : public AutoDiffOpInterface::ExternalModel<DivFOpInterface,
                                                arith::DivFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a / b -> dr = (a * db - da * b) / (b * b)
    auto divOp = cast<arith::DivFOp>(op);
    if (!gutils->isConstantValue(divOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(divOp.getOperand(i))) {
          Value tmp = builder.create<arith::MulFOp>(
              divOp.getLoc(),
              gutils->invertPointerM(divOp.getOperand(i), builder),
              gutils->getNewFromOriginal(divOp.getOperand(1 - i)));
          if (res == nullptr)
            res = tmp;
          else
            res = builder.create<arith::SubFOp>(divOp.getLoc(), res, tmp);
        }
      }
      Value tmp = builder.create<arith::MulFOp>(divOp.getLoc(), gutils->getNewFromOriginal(divOp.getOperand(0)), gutils->getNewFromOriginal(divOp.getOperand(1)));
      res = builder.create<arith::DivFOp>(divOp.getLoc(), res, tmp);
      gutils->setDiffe(divOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct SubFOpInterface
    : public AutoDiffOpInterface::ExternalModel<SubFOpInterface,
                                                arith::SubFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a - b -> dr = da - db
    auto addOp = cast<arith::SubFOp>(op);
    if (!gutils->isConstantValue(addOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(addOp.getOperand(i))) {
          Value tmp = gutils->invertPointerM(addOp.getOperand(i), builder);
          if (res == nullptr)
            res = tmp;
          else
            res = builder.create<arith::SubFOp>(addOp.getLoc(), res, tmp);
        }
      }
      gutils->setDiffe(addOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct AddFOpInterface : public AutoDiffOpInterface::ExternalModel<AddFOpInterface, arith::AddFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    // Derivative of r = a + b -> dr = da + db
    auto addOp = cast<arith::AddFOp>(op);
    if (!gutils->isConstantValue(addOp)) {
      mlir::Value res = nullptr;
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(addOp.getOperand(i))) {
          Value tmp = gutils->invertPointerM(addOp.getOperand(i), builder);
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

struct AddFOpInterfaceReverse : public AutoDiffOpInterfaceReverse::ExternalModel<AddFOpInterfaceReverse, arith::AddFOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils) const {
    // Derivative of r = a + b -> dr = da + db
    auto addOp = cast<arith::AddFOp>(op);

    if(gutils->hasInvertPointer(addOp)){
      for (int i = 0; i < 2; i++) {
        if (!gutils->isConstantValue(addOp.getOperand(i))) {
          Value own_gradient = gutils->invertPointerM(addOp, builder);
          gutils->mapInvertPointer(addOp.getOperand(i), own_gradient, builder);
        }
      }
      //auto x = builder.create<enzyme::CreateCacheOp>(addOp.getLoc(), builder.getF64Type());
      //gutils->setDiffe(addOp, tmp, builder);
    }
    return success();
  }
};

struct MaxFOpInterface
    : public AutoDiffOpInterface::ExternalModel<MaxFOpInterface,
                                                arith::MaxFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    
    auto maxOp = cast<arith::MaxFOp>(op);

    Value tmp0 = gutils->getNewFromOriginal(maxOp.getOperand(0));
    Value tmp1 = gutils->getNewFromOriginal(maxOp.getOperand(1));

    if (!gutils->isConstantValue(maxOp.getOperand(0)) || !gutils->isConstantValue(maxOp.getOperand(1))){
      Value invert0 = gutils->invertPointerM(maxOp.getOperand(0), builder);
      Value invert1 = gutils->invertPointerM(maxOp.getOperand(1), builder);
      
      auto thenBuilder = [&](OpBuilder &nested, Location loc) {
        nested.create<scf::YieldOp>(loc, invert1);
      };

      auto elseBuilder = [&](OpBuilder &nested, Location loc) {
        nested.create<scf::YieldOp>(loc, invert0);
      };

      Value condition = builder.create<arith::CmpFOp>(maxOp.getLoc(), arith::CmpFPredicate::OLT, tmp0, tmp1);
      auto res = builder.create<scf::IfOp>(maxOp.getLoc(), tmp0.getType(), condition, thenBuilder, elseBuilder).getResults()[0];
      gutils->setDiffe(maxOp, res, builder);
    }
    else{
      Value res = gutils->invertPointerM(maxOp.getOperand(0), builder);
      gutils->setDiffe(maxOp, res, builder);
    }

    gutils->eraseIfUnused(op);
    return success();
  }
};

struct MinFOpInterface
    : public AutoDiffOpInterface::ExternalModel<MinFOpInterface,
                                                arith::MinFOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    
    auto minOp = cast<arith::MinFOp>(op);

    Value tmp0 = gutils->getNewFromOriginal(minOp.getOperand(0));
    Value tmp1 = gutils->getNewFromOriginal(minOp.getOperand(1));

    if (!gutils->isConstantValue(minOp.getOperand(0)) || !gutils->isConstantValue(minOp.getOperand(1))){
      Value invert0 = gutils->invertPointerM(minOp.getOperand(0), builder);
      Value invert1 = gutils->invertPointerM(minOp.getOperand(1), builder);
      
      auto thenBuilder = [&](OpBuilder &nested, Location loc) {
        nested.create<scf::YieldOp>(loc, invert0);
      };

      auto elseBuilder = [&](OpBuilder &nested, Location loc) {
        nested.create<scf::YieldOp>(loc, invert1);
      };

      Value condition = builder.create<arith::CmpFOp>(minOp.getLoc(), arith::CmpFPredicate::OLT, tmp0, tmp1);
      auto res = builder.create<scf::IfOp>(minOp.getLoc(), tmp0.getType(), condition, thenBuilder, elseBuilder).getResults()[0];
      gutils->setDiffe(minOp, res, builder);
    }
    else{
      Value res = gutils->invertPointerM(minOp.getOperand(0), builder);
      gutils->setDiffe(minOp, res, builder);
    }

    gutils->eraseIfUnused(op);
    return success();
  }
};

} 

void mlir::enzyme::registerArithDialectAutoDiffInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, arith::ArithDialect *) {
    arith::AddFOp::attachInterface<AddFOpInterfaceReverse>(*context);
    //arith::SubFOp::attachInterface<SubFOpInterface>(*context);
    //arith::MulFOp::attachInterface<MulFOpInterface>(*context);
    //arith::DivFOp::attachInterface<DivFOpInterface>(*context);
    //arith::MaxFOp::attachInterface<MaxFOpInterface>(*context);
    //arith::MinFOp::attachInterface<MinFOpInterface>(*context);
  });
}
