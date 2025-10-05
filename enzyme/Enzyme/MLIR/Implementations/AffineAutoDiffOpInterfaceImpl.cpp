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
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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

struct AffineForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AffineForOpInterfaceReverse, affine::AffineForOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto forOp = cast<affine::AffineForOp>(op);

    affine::AffineBound lb = forOp.getLowerBound();
    affine::AffineBound ub = forOp.getUpperBound();

    if (lb.getMap().getNumResults() != 1 || ub.getMap().getNumResults() != 1) {
      op->emitError() << "cannot differentiate loop with minmax bounds yet";
      return failure();
    }

    SmallVector<bool> operandsActive;
    for (auto [operand, result] : llvm::zip_equal(
             op->getOperands().slice(forOp.getNumControlOperands(),
                                     forOp->getNumOperands() -
                                         forOp.getNumControlOperands()),
             op->getResults())) {
      operandsActive.push_back(!gutils->isConstantValue(operand) ||
                               !gutils->isConstantValue(result));
    }

    SmallVector<Value> revLBOperands, revUBOperands, incomingGradients;

    for (int i = 0, e = lb.getNumOperands(); i < e; ++i) {
      revLBOperands.push_back(gutils->popCache(caches[i], builder));
    }

    for (int i = lb.getNumOperands(), e = forOp.getNumControlOperands(); i < e;
         ++i) {
      revUBOperands.push_back(gutils->popCache(caches[i], builder));
    }

    for (auto &&[active, res] :
         llvm::zip_equal(operandsActive, op->getResults())) {
      if (active) {
        incomingGradients.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    auto revFor = builder.create<affine::AffineForOp>(
        op->getLoc(), revLBOperands, lb.getMap(), revUBOperands, ub.getMap(),
        forOp.getStepAsInt(), incomingGradients);

    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), revFor->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());

        // Create implicit terminator if not present (when num results > 0)
        if (revBB.empty()) {
          bodyBuilder.create<affine::AffineYieldOp>(revFor->getLoc());
        }
        bodyBuilder.setInsertionPoint(revBB.getTerminator());

        // All values defined in the body should have no use outside this block
        // therefore we can set their diffe to zero upon entering the reverse
        // block to simplify the work of the remove-unnecessary-enzyme-ops pass.
        for (auto operand : oBB.getArguments().slice(1)) {
          if (!gutils->isConstantValue(operand)) {
            gutils->zeroDiffe(operand, bodyBuilder);
          }
        }

        for (auto &it : oBB.getOperations()) {
          for (auto res : it.getResults()) {
            if (!gutils->isConstantValue(res)) {
              gutils->zeroDiffe(res, bodyBuilder);
            }
          }
        }

        auto term = oBB.getTerminator();

        for (auto &&[active, arg, operand] :
             llvm::zip_equal(operandsActive, revBB.getArguments().slice(1),
                             term->getOperands())) {
          if (active) {
            // Set diffe here, not add because it should not accumulate across
            // iterations. Instead the new gradient for this operand is passed
            // in the return of the reverse for body.
            gutils->setDiffe(operand, arg, bodyBuilder);
          }
        }

        auto first = oBB.rbegin();
        first++; // skip terminator

        auto last = oBB.rend();

        for (auto it = first; it != last; ++it) {
          Operation *op = &*it;
          valid &=
              gutils->Logic.visitChild(op, bodyBuilder, gutils).succeeded();
        }

        SmallVector<Value> newResults;
        newResults.reserve(incomingGradients.size());

        for (auto &&[active, arg] :
             llvm::zip_equal(operandsActive, oBB.getArguments().slice(1))) {
          if (active) {
            newResults.push_back(gutils->diffe(arg, bodyBuilder));
            if (!gutils->isConstantValue(arg))
              gutils->zeroDiffe(arg, bodyBuilder);
          }
        }

        // yield new gradient values
        revBB.getTerminator()->setOperands(newResults);
      }
    }

    for (auto &&[active, res, arg] : llvm::zip_equal(
             operandsActive, revFor->getResults(), forOp.getInits())) {
      if (active) {
        if (!gutils->isConstantValue(arg))
          gutils->addToDiffe(arg, res, builder);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto forOp = cast<affine::AffineForOp>(op);

    SmallVector<Value> caches;
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    for (auto operand : forOp.getControlOperands()) {
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(operand), cacheBuilder));
    }

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct AffineLoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<AffineLoadOpInterfaceReverse,
                                                       affine::AffineLoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<affine::AffineLoadOp>(op);
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

        if (!gutils->AtomicAdd) {
          Value loadedGradient = builder.create<affine::AffineLoadOp>(
              loadOp.getLoc(), memrefGradient,
              loadOp.getAffineMap(),
              ArrayRef<Value>(retrievedArguments));
          Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                  loadedGradient, gradient);
          builder.create<affine::AffineStoreOp>(loadOp.getLoc(), addedGradient,
                                          memrefGradient,
                                          loadOp.getAffineMap(),
                                          ArrayRef<Value>(retrievedArguments));
        } else {
          auto idx = builder.create<affine::AffineApplyOp>(loadOp.getLoc(), loadOp.getAffineMap(), retrievedArguments);
          builder.create<memref::AtomicRMWOp>(
              loadOp.getLoc(), arith::AtomicRMWKind::addf, gradient,
              memrefGradient, idx->getResults());
        }
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<affine::AffineLoadOp>(op);
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

struct AffineStoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<AffineStoreOpInterfaceReverse,
                                                       affine::AffineStoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<affine::AffineStoreOp>(op);
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
          Value loadedGradient = builder.create<affine::AffineLoadOp>(
              storeOp.getLoc(), memrefGradient,
              ArrayRef<Value>(retrievedArguments));
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        builder.create<affine::AffineStoreOp>(storeOp.getLoc(), zero, memrefGradient,
                                        ArrayRef<Value>(retrievedArguments));
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<affine::AffineStoreOp>(op);
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

#include "Implementations/AffineDerivatives.inc"
} // namespace

void mlir::enzyme::registerAffineDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, affine::AffineDialect *) {
    registerInterfaces(context);
    affine::AffineLoadOp::attachInterface<AffineLoadOpInterfaceReverse>(*context);
    affine::AffineStoreOp::attachInterface<AffineStoreOpInterfaceReverse>(*context);
    affine::AffineForOp::attachInterface<AffineForOpInterfaceReverse>(*context);
  });
}
