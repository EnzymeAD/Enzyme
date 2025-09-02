//===- MemRefAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR memref dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

// TODO: We need a way to zero out a memref (which linalg.fill does), but
// ideally we wouldn't depend on the linalg dialect.
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/MemRefDerivatives.inc"

struct LoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<LoadOpInterfaceReverse,
                                                       memref::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<memref::LoadOp>(op);
    Value memref = loadOp.getMemref();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(loadOp) &&
          !gutils->isConstantValue(memref)) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value memrefGradient = gutils->invertPointerM(memref, builder);

        SmallVector<Value> retrievedArguments;
        for (Value cache : caches) {
          Value retrievedValue = gutils->popCache(cache, builder);
          retrievedArguments.push_back(retrievedValue);
        }

        Value loadedGradient =
            builder.create<memref::LoadOp>(loadOp.getLoc(), memrefGradient,
                                           ArrayRef<Value>(retrievedArguments));
        Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                loadedGradient, gradient);
        builder.create<memref::StoreOp>(loadOp.getLoc(), addedGradient,
                                        memrefGradient,
                                        ArrayRef<Value>(retrievedArguments));
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<memref::LoadOp>(op);
    Value memref = loadOp.getMemref();
    ValueRange indices = loadOp.getIndices();
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(loadOp) &&
          !gutils->isConstantValue(memref)) {
        OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
        SmallVector<Value> caches;
        for (Value v : indices) {
          caches.push_back(gutils->initAndPushCache(
              gutils->getNewFromOriginal(v), cacheBuilder));
        }
        return caches;
      }
    }
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto loadOp = cast<memref::LoadOp>(op);
    // Value memref = loadOp.getMemref();
    // Value shadow = gutils->getShadowValue(memref);
    // Do nothing yet. In the future support memref<memref<...>>
  }
};

struct StoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<StoreOpInterfaceReverse,
                                                       memref::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<memref::StoreOp>(op);
    Value val = storeOp.getValue();
    Value memref = storeOp.getMemref();
    // ValueRange indices = storeOp.getIndices();

    auto iface = cast<AutoDiffTypeInterface>(val.getType());

    if (!gutils->isConstantValue(memref)) {
      OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));

      Value memrefGradient = gutils->invertPointerM(memref, builder);

      SmallVector<Value> retrievedArguments;
      for (Value cache : caches) {
        Value retrievedValue = gutils->popCache(cache, builder);
        retrievedArguments.push_back(retrievedValue);
      }

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient = builder.create<memref::LoadOp>(
              storeOp.getLoc(), memrefGradient,
              ArrayRef<Value>(retrievedArguments));
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        builder.create<memref::StoreOp>(storeOp.getLoc(), zero, memrefGradient,
                                        ArrayRef<Value>(retrievedArguments));
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<memref::StoreOp>(op);
    Value memref = storeOp.getMemref();
    ValueRange indices = storeOp.getIndices();
    Value val = storeOp.getValue();
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType())) {
      if (!gutils->isConstantValue(memref)) {
        OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
        SmallVector<Value> caches;
        for (Value v : indices) {
          caches.push_back(gutils->initAndPushCache(
              gutils->getNewFromOriginal(v), cacheBuilder));
        }
        return caches;
      }
    }
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto storeOp = cast<memref::StoreOp>(op);
    // Value memref = storeOp.getMemref();
    // Value shadow = gutils->getShadowValue(memref);
    // Do nothing yet. In the future support memref<memref<...>>
  }
};

struct SubViewOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          SubViewOpInterfaceReverse, memref::SubViewOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto subviewOp = cast<memref::SubViewOp>(op);
    auto newSubviewOp = cast<memref::SubViewOp>(gutils->getNewFromOriginal(op));
    if (!gutils->isConstantValue(subviewOp.getSource())) {
      Value shadow = builder.create<memref::SubViewOp>(
          op->getLoc(), newSubviewOp.getType(),
          gutils->invertPointerM(subviewOp.getSource(), builder),
          newSubviewOp.getMixedOffsets(), newSubviewOp.getMixedSizes(),
          newSubviewOp.getMixedStrides());
      gutils->setInvertedPointer(subviewOp, shadow);
    }
  }
};

class MemRefTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<MemRefTypeInterface,
                                                  MemRefType> {
public:
  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    llvm_unreachable("Cannot create null of memref (todo polygeist null)");
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    auto MT = cast<MemRefType>(self);
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(MT.getElementType())) {
      if (!iface.isMutable()) {
        Value zero = iface.createNullValue(builder, loc);
        builder.create<linalg::FillOp>(loc, zero, val);
      }
    } else {
      return failure();
    }
    return success();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute val) const { return false; }
};
} // namespace

void mlir::enzyme::registerMemRefDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    registerInterfaces(context);
    MemRefType::attachInterface<MemRefTypeInterface>(*context);

    memref::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    memref::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    memref::SubViewOp::attachInterface<SubViewOpInterfaceReverse>(*context);
  });
}
