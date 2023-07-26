//===- TensorAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
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

// TODO: We need a way to zero out a memref (which linalg.fill does), but
// ideally we wouldn't depend on the linalg dialect.
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Value duplicateNullTensor(Value tensor, OpBuilder& builder){
  auto tensorType = tensor.getType().cast<TensorType>();
  auto tensorShape = tensorType.getShape();
  auto elementType = tensorType.getElementType();
  //Get dynamic sizes
  SmallVector<Value> dynamicSizes;
  for (auto dim : tensorShape) {
    if (dim == ShapedType::kDynamic) {
      dynamicSizes.push_back(builder.create<tensor::DimOp>(tensor.getLoc(), tensor, dynamicSizes.size()));
    }
  }
  //Create new tensor
  Value nullTensor = builder.create<tensor::EmptyOp>(tensor.getLoc(), tensorShape, elementType, dynamicSizes);
  //Fill with zeros
  auto zero = builder.create<arith::ConstantOp>(tensor.getLoc(), builder.getZeroAttr(elementType));
  auto fill = builder.create<linalg::FillOp>(tensor.getLoc(), ValueRange({zero}), ValueRange({nullTensor}));
  return fill.getResult(0);
}

struct ExtractOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ExtractOpInterfaceReverse,
                                                       tensor::ExtractOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto extractOp = cast<tensor::ExtractOp>(op);
    Value tensor = extractOp.getTensor();

    SmallVector<Value> retrievedArguments;
    for (Value cache : caches) {
      Value retrievedValue = gutils->popCache(cache, builder);
      retrievedArguments.push_back(retrievedValue);
    }
    
    auto iface = dyn_cast<AutoDiffTypeInterface>(extractOp.getType());
    if (!iface) {
      return;
    }
    if (!gutils->hasInvertPointer(extractOp)) {
      return;
    }
    Value extractGradient = gutils->invertPointerM(extractOp, builder);
    if (!gutils->hasInvertPointer(tensor)) {
      Value nullTensor = duplicateNullTensor(tensor, builder);
      gutils->mapInvertPointer(tensor, nullTensor, builder);
    }
    Value tensorGradient = gutils->invertPointerM(tensor, builder);
    Value currentGradient = builder.create<tensor::ExtractOp>(extractOp.getLoc(), tensorGradient, retrievedArguments);
    Value newGradient = iface.createAddOp(builder, extractOp.getLoc(), extractGradient, currentGradient);
    Value newTensorGradient = builder.create<tensor::InsertOp>(extractOp.getLoc(), newGradient, tensorGradient, retrievedArguments);
    gutils->mapInvertPointer(tensor, newTensorGradient, builder);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto extractOp = cast<tensor::ExtractOp>(op);
    OpBuilder builder(gutils->getNewFromOriginal(op));

    auto indices = extractOp.getIndices();
    SmallVector<Value> caches;
    for (Value index : indices) {
      caches.push_back(gutils->initAndPushCache(index, builder));
    }
    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {

  }
};
/*
struct StoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<StoreOpInterfaceReverse,
                                                       memref::StoreOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto storeOp = cast<memref::StoreOp>(op);
    Value val = storeOp.getValue();
    Value memref = storeOp.getMemref();
    ValueRange indices = storeOp.getIndices();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType())) {
      if (gutils->hasInvertPointer(memref)) {
        OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));

        Value memrefGradient = gutils->invertPointerM(memref, builder);

        SmallVector<Value> retrievedArguments;
        for (Value cache : caches) {
          Value retrievedValue = gutils->popCache(cache, builder);
          retrievedArguments.push_back(retrievedValue);
        }

        Value loadedGradient =
            builder.create<memref::LoadOp>(storeOp.getLoc(), memrefGradient,
                                           ArrayRef<Value>(retrievedArguments));
        Value addedGradient = loadedGradient;
        if (gutils->hasInvertPointer(val)) {
          Value gradient = gutils->invertPointerM(val, builder);
          addedGradient = iface.createAddOp(builder, storeOp.getLoc(), gradient,
                                            loadedGradient);
        }
        gutils->mapInvertPointer(val, addedGradient, builder);
      }
    }
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<memref::StoreOp>(op);
    Value memref = storeOp.getMemref();
    ValueRange indices = storeOp.getIndices();
    Value val = storeOp.getValue();
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType())) {
      if (gutils->hasInvertPointer(memref)) {
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
    auto storeOp = cast<memref::StoreOp>(op);
    Value memref = storeOp.getMemref();
    Value shadow = gutils->getShadowValue(memref);
    // Do nothing yet. In the future support memref<memref<...>>
  }
};

*/

class RankedTensorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<RankedTensorTypeInterface,
                                                  RankedTensorType> {
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

  bool requiresShadow(Type self) const { return false; }
};

class UnrankedTensorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<UnrankedTensorTypeInterface,
                                                  UnrankedTensorType> {
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

  bool requiresShadow(Type self) const { return false; }
};

} // namespace

void mlir::enzyme::registerTensorDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, tensor::TensorDialect *) {
    RankedTensorType::attachInterface<RankedTensorTypeInterface>(*context);
    UnrankedTensorType::attachInterface<UnrankedTensorTypeInterface>(*context);

    tensor::ExtractOp::attachInterface<ExtractOpInterfaceReverse>(*context);
    //memref::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
  });
}
