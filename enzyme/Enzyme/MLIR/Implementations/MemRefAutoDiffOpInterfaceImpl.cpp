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
struct LoadOpInterface
    : public AutoDiffOpInterface::ExternalModel<LoadOpInterface,
                                                memref::LoadOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto loadOp = cast<memref::LoadOp>(op);
    if (!gutils->isConstantValue(loadOp)) {
      SmallVector<Value> inds;
      for (auto ind : loadOp.getIndices())
        inds.push_back(gutils->getNewFromOriginal(ind));
      mlir::Value res = builder.create<memref::LoadOp>(
          loadOp.getLoc(), gutils->invertPointerM(loadOp.getMemref(), builder),
          inds);
      gutils->setDiffe(loadOp, res, builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};


struct StoreOpInterface
    : public AutoDiffOpInterface::ExternalModel<StoreOpInterface,
                                                memref::StoreOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto storeOp = cast<memref::StoreOp>(op);
    if (!gutils->isConstantValue(storeOp.getMemref())) {
      SmallVector<Value> inds;
      for (auto ind : storeOp.getIndices())
        inds.push_back(gutils->getNewFromOriginal(ind));
      builder.create<memref::StoreOp>(
          storeOp.getLoc(), gutils->invertPointerM(storeOp.getValue(), builder),
          gutils->invertPointerM(storeOp.getMemref(), builder), inds);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct LoadOpInterfaceReverse : public AutoDiffOpInterfaceReverse::ExternalModel<LoadOpInterfaceReverse, memref::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder, MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<memref::LoadOp>(op);
    Value memref = loadOp.getMemref();
    ValueRange vr = loadOp.getIndices();
    
    if (auto iface = loadOp.getType().cast<AutoDiffTypeInterface>()) {      
      if(gutils->hasInvertPointer(loadOp) && gutils->hasInvertPointer(memref)){
        OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
        
        Value gradient = gutils->invertPointerM(loadOp, builder);
        Value memrefGradient = gutils->invertPointerM(memref, builder);

        SmallVector<Value> retrievedArguments;
        for (Value v : vr){
          Value cache = gutils->cacheForReverse(gutils->getNewFromOriginal(v), cacheBuilder);
          Value retrievedValue = gutils->popCache(cache, builder);
          retrievedArguments.push_back(retrievedValue);
        }

        Value loadedGradient = builder.create<memref::LoadOp>(loadOp.getLoc(), memrefGradient, ArrayRef<Value>(retrievedArguments));
        Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(), loadedGradient, gradient);
        builder.create<memref::StoreOp>(loadOp.getLoc(), addedGradient, memrefGradient, ArrayRef<Value>(retrievedArguments));
      }   
    }
    return success();
  }

  void clearGradient(Operation *op, SmallVector<OpBuilder *>builders, MGradientUtilsReverse *gutils) const {
  }
};

struct StoreOpInterfaceReverse : public AutoDiffOpInterfaceReverse::ExternalModel<StoreOpInterfaceReverse, memref::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder, MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<memref::StoreOp>(op);
    Value val = storeOp.getValue();
    Value memref = storeOp.getMemref();
    ValueRange vr = storeOp.getIndices();
    
    if (auto iface = val.getType().cast<AutoDiffTypeInterface>()){
      if(gutils->hasInvertPointer(memref)){
        OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
        
        Value memrefGradient = gutils->invertPointerM(memref, builder);

        SmallVector<Value> retrievedArguments;
        for (Value v : vr){
          Value cache = gutils->cacheForReverse(gutils->getNewFromOriginal(v), cacheBuilder);
          Value retrievedValue = gutils->popCache(cache, builder);
          retrievedArguments.push_back(retrievedValue);
        }

        Value loadedGradient = builder.create<memref::LoadOp>(storeOp.getLoc(), memrefGradient, ArrayRef<Value>(retrievedArguments));
        Value addedGradient = loadedGradient;
        if(gutils->hasInvertPointer(val)){
          Value gradient = gutils->invertPointerM(val, builder);
          addedGradient = iface.createAddOp(builder, storeOp.getLoc(), gradient, loadedGradient);
        }
        gutils->mapInvertPointer(val, addedGradient, builder);
      }   
    }

    return success();
  }

  void clearGradient(Operation *op, SmallVector<OpBuilder *>builders, MGradientUtilsReverse *gutils) const {
  }
};

struct AllocOpInterface
    : public AutoDiffOpInterface::ExternalModel<AllocOpInterface,
                                                memref::AllocOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto allocOp = cast<memref::AllocOp>(op);
    if (!gutils->isConstantValue(allocOp)) {
      Operation *nop = gutils->cloneWithNewOperands(builder, op);
      gutils->setDiffe(allocOp, nop->getResult(0), builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

class MemRefTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<MemRefTypeInterface,
                                                  MemRefType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    llvm_unreachable("Cannot create null of memref (todo polygeist null)");
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a, Value b) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isPointerType(Type self) const{
    return true;
  }
};
} // namespace

void mlir::enzyme::registerMemRefDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    memref::LoadOp::attachInterface<LoadOpInterface>(*context);
    memref::StoreOp::attachInterface<StoreOpInterface>(*context);
    memref::AllocOp::attachInterface<AllocOpInterface>(*context);
    MemRefType::attachInterface<MemRefTypeInterface>(*context);

    memref::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    memref::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
  });
}
