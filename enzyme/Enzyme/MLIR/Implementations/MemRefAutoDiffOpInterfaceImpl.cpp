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

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/MemRefDerivatives.inc"

// Lets activity analysis treat memref.store generically via StoreLikeInterface.
struct MemRefStoreLike
    : public StoreLikeInterface::ExternalModel<MemRefStoreLike,
                                               memref::StoreOp> {
  Value getStoredValue(Operation *op) const {
    return cast<memref::StoreOp>(op).getValueToStore();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<memref::StoreOp>(op).getMemRef();
  }
};

struct LoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<LoadOpInterfaceReverse,
                                                       memref::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<memref::LoadOp>(op);
    Value memref = loadOp.getMemref();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      // A mutable type's derivative is a shadow, not a value to add into: there
      // is nothing to accumulate here, the same way the store adjoint below has
      // nothing to take back out. Loading a pointer is the case in hand.
      if (!gutils->isConstantValue(loadOp) &&
          !gutils->isConstantValue(memref) && !iface.isMutable()) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value memrefGradient = gutils->popCache(caches.front(), builder);

        SmallVector<Value> retrievedArguments;
        for (Value cache : ValueRange(caches).drop_front(1)) {
          Value retrievedValue = gutils->popCache(cache, builder);
          retrievedArguments.push_back(retrievedValue);
        }

        if (!gutils->AtomicAdd) {
          auto loadedGradientOp =
              memref::LoadOp::create(builder, loadOp.getLoc(), memrefGradient,
                                     ArrayRef<Value>(retrievedArguments));
          loadedGradientOp.setAlignmentAttr(loadOp.getAlignmentAttr());
          Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                  loadedGradientOp, gradient);
          auto storeGradientOp = memref::StoreOp::create(
              builder, loadOp.getLoc(), addedGradient, memrefGradient,
              ArrayRef<Value>(retrievedArguments));
          storeGradientOp.setAlignmentAttr(loadOp.getAlignmentAttr());
        } else {
          setDerivativeFastMath(enzyme::AtomicRMWOp::create(
              builder, loadOp.getLoc(), gradient.getType(),
              arith::AtomicRMWKind::addf, Ordering::monotonic, gradient,
              memrefGradient, retrievedArguments, loadOp.getAlignmentAttr()));
        }
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
        caches.push_back(gutils->initAndPushCache(
            gutils->invertPointerM(memref, cacheBuilder), cacheBuilder));
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
    auto loadOp = cast<memref::LoadOp>(op);
    Value memref = loadOp.getMemref();
    auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType());
    // What a load of a mutable type reads is itself a handle on active memory,
    // so what stands for it is the handle held at the same place in the shadow:
    // the same load, off the shadow memref. Reading it out is the whole of the
    // derivative -- see the adjoint above, which leaves it alone.
    if (!iface || !iface.isMutable())
      return;
    if (gutils->isConstantValue(loadOp) || gutils->isConstantValue(memref))
      return;
    Value memrefShadow = gutils->invertPointerM(memref, builder);
    auto newLoad = cast<memref::LoadOp>(gutils->getNewFromOriginal(op));
    auto shadowLoad = cast<memref::LoadOp>(builder.clone(*newLoad));
    shadowLoad.getMemrefMutable().assign(memrefShadow);
    gutils->setInvertedPointer(loadOp.getResult(), shadowLoad.getResult());
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
      Value memrefGradient = gutils->popCache(caches.front(), builder);

      SmallVector<Value> retrievedArguments;
      for (Value cache : ValueRange(caches).drop_front(1)) {
        Value retrievedValue = gutils->popCache(cache, builder);
        retrievedArguments.push_back(retrievedValue);
      }

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          auto loadedGradientOp =
              memref::LoadOp::create(builder, storeOp.getLoc(), memrefGradient,
                                     ArrayRef<Value>(retrievedArguments));
          loadedGradientOp.setAlignmentAttr(storeOp.getAlignmentAttr());
          gutils->addToDiffe(val, loadedGradientOp, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        auto zeroStoreOp = memref::StoreOp::create(
            builder, storeOp.getLoc(), zero, memrefGradient,
            ArrayRef<Value>(retrievedArguments));
        zeroStoreOp.setAlignmentAttr(storeOp.getAlignmentAttr());
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
        caches.push_back(gutils->initAndPushCache(
            gutils->invertPointerM(memref, cacheBuilder), cacheBuilder));
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
      Value shadow = memref::SubViewOp::create(
          builder, op->getLoc(), newSubviewOp.getType(),
          gutils->invertPointerM(subviewOp.getSource(), builder),
          newSubviewOp.getMixedOffsets(), newSubviewOp.getMixedSizes(),
          newSubviewOp.getMixedStrides());
      gutils->setInvertedPointer(subviewOp, shadow);
    }
  }
};

class MemRefClonableTypeInterface
    : public ClonableTypeInterface::ExternalModel<MemRefClonableTypeInterface,
                                                  MemRefType> {

public:
  mlir::Value cloneValue(mlir::Type self, OpBuilder &builder,
                         Value value) const {
    MemRefType MT = cast<MemRefType>(self);
    SmallVector<Value> dynamicSizes;

    for (auto [i, s] : llvm::enumerate(MT.getShape())) {
      if (s == ShapedType::kDynamic) {
        Value dim = arith::ConstantIndexOp::create(builder, value.getLoc(), i);
        dynamicSizes.push_back(
            memref::DimOp::create(builder, value.getLoc(), value, dim));
      }
    }

    // Pass the MemRefType, not `self`: with a plain Type the generic
    // (TypeRange, ValueRange) builder is selected, which appends the dynamic
    // sizes as operands without setting operandSegmentSizes -- producing an
    // invalid memref.alloc as soon as the type has a dynamic dimension.
    auto clone =
        memref::AllocOp::create(builder, value.getLoc(), MT, dynamicSizes);
    memref::CopyOp::create(builder, value.getLoc(), value, clone);

    return clone;
  }

  void copyValue(mlir::Type self, OpBuilder &builder, Value dst,
                 Value src) const {
    // memref.copy carries the extents in its operand types, so there is no
    // extent to recover here and dynamic shapes work unchanged.
    memref::CopyOp::create(builder, src.getLoc(), src, dst);
  }

  void freeClonedValue(mlir::Type self, OpBuilder &builder, Value value) const {
    memref::DeallocOp::create(builder, value.getLoc(), value);
  };
};

class MemRefAutoDiffTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<MemRefAutoDiffTypeInterface,
                                                  MemRefType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm_unreachable("Cannot create null of memref (todo polygeist null)");
  }
  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    // Create a memref of the given type with the required number of
    // dynamic dimensions, all set to 0
    MemRefType MT = cast<MemRefType>(self);
    unsigned numDynamicDims = MT.getNumDynamicDims();
    SmallVector<mlir::Value> dynamicSizes(numDynamicDims);
    for (unsigned i = 0; i < numDynamicDims; ++i) {
      dynamicSizes[i] = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    }
    return mlir::memref::AllocOp::create(builder, loc, MT, dynamicSizes);
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
        enzyme::FillZeroOp::create(builder, loc, val);
      }
    } else {
      return failure();
    }
    return success();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute val) const { return false; }
};

struct MemRefAllocOpInterface
    : public MultidimensionalAllocInterface::ExternalModel<
          MemRefAllocOpInterface, memref::AllocOp> {
  Value allocate(Operation *op, OpBuilder &rewriter, Location loc, Type newType,
                 ValueRange dynamicDims) const {
    return memref::AllocOp::create(rewriter, loc, cast<MemRefType>(newType),
                                   dynamicDims);
  }

  void deallocate(Operation *op, OpBuilder &rewriter, Location loc,
                  Value val) const {
    memref::DeallocOp::create(rewriter, loc, val);
  }

  bool isDeallocation(Operation *op, Operation *user) const {
    return isa<memref::DeallocOp>(user);
  }
};

} // namespace

void mlir::enzyme::registerMemRefDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    registerInterfaces(context);
    MemRefType::attachInterface<MemRefAutoDiffTypeInterface>(*context);
    MemRefType::attachInterface<MemRefClonableTypeInterface>(*context);

    memref::StoreOp::attachInterface<MemRefStoreLike>(*context);
    memref::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    memref::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    memref::SubViewOp::attachInterface<SubViewOpInterfaceReverse>(*context);
    memref::AllocOp::attachInterface<MemRefAllocOpInterface>(*context);
  });
}
