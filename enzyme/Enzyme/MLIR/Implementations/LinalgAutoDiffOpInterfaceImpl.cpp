//===- LinalgAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Shape/IR/ShapeOpsTypes.h.inc"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Value invertMemref(Value inp, OpBuilder &builder, Location loc) {
  MemRefType iType = cast<MemRefType>(inp.getType());
  SmallVector<Value> dims;
  SmallVector<Value> dimSubOnes;
  SmallVector<Value> strides;
  Value negOne = builder.create<arith::ConstantIndexOp>(loc, -1);
  int shapeDim = iType.getShape().size();
  for (int i = 0; i < shapeDim; i++) {
    Value dim = builder.create<memref::DimOp>(loc, inp, i);
    dims.push_back(dim);
    auto dimSubOne = builder.create<arith::AddIOp>(loc, dim, negOne);
    dimSubOnes.push_back(dimSubOne);
    strides.push_back(negOne);
  }
  Value view = builder.create<memref::SubViewOp>(
      loc, inp, ValueRange(dimSubOnes), ValueRange(dims), ValueRange(strides));
  return view;
}

Value invertTensor(Value inp, OpBuilder &builder, Location loc) {
  TensorType iType = cast<TensorType>(inp.getType());
  SmallVector<Value> dims;
  SmallVector<Value> dimSubOnes;
  SmallVector<Value> strides;
  Value negOne = builder.create<arith::ConstantIndexOp>(loc, -1);
  int shapeDim = iType.getShape().size();
  for (int i = 0; i < shapeDim; i++) {
    Value dim = builder.create<tensor::DimOp>(loc, inp, i);
    dims.push_back(dim);
    auto dimSubOne = builder.create<arith::AddIOp>(loc, dim, negOne);
    dimSubOnes.push_back(dimSubOne);
    strides.push_back(negOne);
  }
  Value view = builder.create<tensor::ExtractSliceOp>(
      loc, inp, ValueRange(dimSubOnes), ValueRange(dims), ValueRange(strides));
  return view;
}

SmallVector<AffineMap> getIndexingMapsArray(enzyme::GenericAdjointOp &op) {
  auto attr = op.getIndexingMapsAttr();
  SmallVector<AffineMap> indexingMaps;
  for (auto map : attr.getValue()) {
    indexingMaps.push_back(map.cast<AffineMapAttr>().getValue());
  }
  return indexingMaps;
}

void processDims(OpOperand * input, linalg::LinalgOp & newOp, SmallVector<Value> &dims,
                               OpBuilder &cacheBuilder, bool usesTensors){
  ArrayRef<int64_t> shape;
  if (!usesTensors){
    shape = cast<MemRefType>(input->get().getType()).getShape();
  } else {
    shape = cast<TensorType>(input->get().getType()).getShape();
  }

  for (unsigned i = 0; i < shape.size(); i++) {
    auto dimI =
        cacheBuilder.create<arith::ConstantIndexOp>(newOp->getLoc(), i);
    if (!usesTensors){
      auto dim = cacheBuilder.create<memref::DimOp>(newOp->getLoc(),
                                                    input->get(), dimI);
      dims.push_back(dim);
    } else {
      auto dim = cacheBuilder.create<tensor::DimOp>(newOp->getLoc(),
                                                    input->get(), dimI);
      dims.push_back(dim);
    }
  }
}

void collectDims(linalg::LinalgOp &newOp, SmallVector<Value> &dims,
                               OpBuilder &cacheBuilder, bool usesTensors) {
  for (OpOperand *input : newOp.getDpsInputOperands()) {
    processDims(input, newOp, dims, cacheBuilder, usesTensors);
  }
  for (OpOperand *output : newOp.getDpsInitOperands()) {
    processDims(output, newOp, dims, cacheBuilder, usesTensors);
  }
}

void processIterationDomains(AffineMap aMap, SmallVector<Value> dims,
                                 OpBuilder & cacheBuilder,
                                 linalg::LinalgOp & newOp,
                                 SmallVector<Value> &iterationDomains,
                                 SmallVector<int64_t> &shapes){
  for (unsigned int i = 0; i < aMap.getNumResults(); i++) {
    AffineMap subMap = aMap.getSubMap({i});
    Value domain = cacheBuilder.create<AffineApplyOp>(newOp->getLoc(), subMap,
                                                      ValueRange(dims));
    iterationDomains.push_back(domain);
    shapes.push_back(ShapedType::kDynamic);
  }
}

void prepareInputsAndOutputs(linalg::LinalgOp &linalgOp,
                            MGradientUtilsReverse *gutils,
                            OpBuilder &builder,
                            Operation *op,
                            SmallVector<Value> &inputs,
                            SmallVector<Value> &outputs,
                            SmallVector<AffineMap> &indexingMaps,
                            bool usesTensors){
  if (!usesTensors){
    for (OpOperand *output : linalgOp.getDpsInitOperands()) {
      if (!gutils->hasInvertPointer(output->get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(output));
      Value out = gutils->invertPointerM(output->get(), builder);
      Value view = invertMemref(out, builder, linalgOp->getLoc());
      outputs.push_back(view);
    }
  } else {
    for (unsigned i = 0; i < linalgOp.getNumDpsInits(); i++) {
      auto outValue = linalgOp->getOpResult(i);
      if (!gutils->hasInvertPointer(outValue)) {
        continue;
      }
      OpOperand *output = linalgOp.getDpsInitOperand(i);
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(output));

      Value out = gutils->invertPointerM(outValue, builder);
      Value view = invertTensor(out, builder, linalgOp->getLoc());
      outputs.push_back(view);
    }
  }

  for (OpOperand *input : linalgOp.getDpsInputOperands()) {
    if (!gutils->hasInvertPointer(input->get())) {
      continue;
    }
    indexingMaps.push_back(linalgOp.getMatchingIndexingMap(input));
    Value inp = gutils->invertPointerM(input->get(), builder);
    if (!usesTensors){
      Value view = invertMemref(inp, builder, linalgOp->getLoc());
      inputs.push_back(view);
    } else {
      inputs.push_back(inp); //Just Push anything of the correct shape. The value wont be used.
    }
  }
}

// TODO this might mess up your cacheBuilder
linalg::LinalgOp addResultToGenericOp(Value result, linalg::LinalgOp &newOp,
                          OpBuilder &cacheBuilder) {
  linalg::GenericOp genericOp = cast<linalg::GenericOp>(newOp);
  TypeRange newOpResultTypes = genericOp.getResultTypes();
  SmallVector<Type> newNewOpResultTypesVec = llvm::to_vector(newOpResultTypes);
  newNewOpResultTypesVec.push_back(result.getType());

  ValueRange inputs = genericOp.getInputs();
  ValueRange outputs = genericOp.getOutputs();

  cacheBuilder.setInsertionPoint(newOp);
  linalg::GenericOp newGenericOp = cacheBuilder.create<linalg::GenericOp>(
      newOp->getLoc(), newNewOpResultTypesVec, inputs, outputs, genericOp.getIndexingMaps(),
      genericOp.getIteratorTypes(), genericOp.getDocAttr(), genericOp.getLibraryCallAttr());
  
  newGenericOp.getRegion().takeBody(genericOp.getRegion());

  for (int i = 0; i < genericOp->getNumResults(); i++) {
    genericOp->getResult(i).replaceAllUsesWith(newGenericOp->getResult(i));
  }

  newOp.erase();
  
  return newGenericOp;
}


template <typename T_>
struct GenericOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GenericOpInterfaceReverse<T_>, T_> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    bool usesTensors = !linalgOp.hasBufferSemantics();

    linalg::LinalgOp newOp =
        cast<linalg::LinalgOp>(gutils->getNewFromOriginal(linalgOp));

    // Replace the op by a linalg.generic op if necessary
    // TODO : IRRewriter rewriter(builder.getContext()/*,
    // builder.getListener()*/);
    ConversionPatternRewriter rewriter(builder.getContext());
    auto failiureOrLinalgOp = generalizeNamedOp(rewriter, newOp);
    if (!failed(failiureOrLinalgOp)) {
      linalg::GenericOp replacement = failiureOrLinalgOp.value();
      auto scope = OpBuilder::InsertionGuard(builder);
      builder.setInsertionPointAfter(newOp);
      builder.insert(replacement);
      newOp.erase();
      newOp = replacement;
    }

    auto cacheBuilder = OpBuilder(newOp, builder.getListener());

    // Calculate the iteration domain
    SmallVector<Value> dims;
    collectDims(newOp, dims, cacheBuilder, usesTensors);

    AffineMap aMap = newOp.getShapesToLoopsMap();
    SmallVector<Value> iterationDomains;
    SmallVector<int64_t> shapes;
    processIterationDomains(aMap, dims, cacheBuilder, newOp, iterationDomains, shapes);

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> indexingMaps;
    prepareInputsAndOutputs(linalgOp, gutils, builder, op, inputs, outputs, indexingMaps, usesTensors);

    SmallVector<utils::IteratorType> iteratorTypes{
        linalgOp.getNumLoops(), utils::IteratorType::parallel};
    ArrayAttr indexingMapsArrayAttr =
        builder.getAffineMapArrayAttr(indexingMaps);
    ArrayAttr iteratorTypesArrayAttr =
        builder.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes, [&](utils::IteratorType iter) -> mlir::Attribute {
              return linalg::IteratorTypeAttr::get(builder.getContext(), iter);
            })));
    
    auto adjoint = builder.create<enzyme::GenericAdjointOp>(
        op->getLoc(), TypeRange(), ValueRange(outputs), ValueRange(inputs),
        indexingMapsArrayAttr, iteratorTypesArrayAttr, StringAttr(),
        StringAttr());

    int numInputs = inputs.size();
    auto buildFuncReturnOp = [numInputs, indexingMaps, &newOp, &adjoint,
                              &inputs](OpBuilder &builder, Location loc,
                                       SmallVector<Value> retargs) {
      builder.create<enzyme::AddToOp>(
          loc, ValueRange{retargs}.take_front(numInputs));
      return;
    };

    Region *newOpRegion = newOp.getBlock()->getParent();
    int numInputsNewOp = cast<linalg::GenericOp>(newOp).getInputs().size();
    Region *adjointRegion = &adjoint.getRegion();
    int numInputsAdjoint = adjoint.getInputs().size();
    Location loc = op->getLoc();
    int numCaches = 0;
    SmallVector<Value> pushCaches;

    auto hook = [newOpRegion, adjointRegion, loc, &numCaches = numCaches,
                 numInputsNewOp, numInputsAdjoint,
                 &pushCaches = pushCaches](Type t) {
      OpBuilder builder(newOpRegion);
      Value pushCache = builder.create<enzyme::InitOp>(loc, t);
      pushCaches.push_back(pushCache);
      newOpRegion->addArgument(t, loc);

      Value popCache =
          adjointRegion->insertArgument(numInputsAdjoint + numCaches, t, loc);
      numCaches++;
      return std::make_pair(pushCache, popCache);
    };

    gutils->Logic.differentiate(
        gutils, *linalgOp.getBlock()->getParent(), adjoint.getRegion(),
        /*parentRegion=*/false, buildFuncReturnOp, hook);

    auto newOpYield = cast<linalg::YieldOp>(
        cast<linalg::GenericOp>(newOp).getBodyRegion().front().getTerminator());
    for (Value pc : pushCaches) {
      newOpYield.getValuesMutable().append(pc);
    }

    Block *body = &(adjoint.getRegion().front());
    auto yieldOp = cast<enzyme::AddToOp>(body->getTerminator());
    for (auto opOperand : yieldOp.getOperands()) {
      body->addArgument(opOperand.getType(), opOperand.getLoc());
    }

    OpBuilder builderAdd(yieldOp);

    auto newIndexingMaps = newOp.getIndexingMapsArray();
    auto indexingMapsAdjoint = getIndexingMapsArray(adjoint);
    for (int i = 0; i < numCaches; i++) {
      Value cacheArg = body->getArgument(outputs.size() + i);

      Type ct = cacheArg.getType();
      Value cache;
      if (!usesTensors){
        Type type = MemRefType::get(shapes, ct);
        auto alloc = cacheBuilder.create<memref::AllocOp>(
            op->getLoc(), type, ValueRange(iterationDomains));
        cache = gutils->initAndPushCache(alloc, cacheBuilder);

        alloc->setAttr(
            alloc.getOperandSegmentSizesAttrName(),
            cacheBuilder.getDenseI32ArrayAttr({iterationDomains.size(), 0}));

        cast<linalg::GenericOp>(newOp).getOutputsMutable().append(
            ValueRange({alloc}));
      } else {
        Type type = RankedTensorType::get(shapes, ct);
        auto empty = cacheBuilder.create<tensor::EmptyOp>(
            op->getLoc(), type, ValueRange(iterationDomains));
        auto genericOpOutputsMutable = cast<linalg::GenericOp>(newOp).getOutputsMutable();
        genericOpOutputsMutable.append(ValueRange({empty}));
        newOp = addResultToGenericOp(empty, newOp, cacheBuilder);
        cacheBuilder.setInsertionPointAfter(newOp);
        cache = gutils->initAndPushCache(newOp->getResult(newOp->getNumResults()-1), cacheBuilder);
        cacheBuilder.setInsertionPoint(newOp);
      }

      newIndexingMaps.push_back(AffineMap::getMultiDimIdentityMap(
          iterationDomains.size(), cacheBuilder.getContext()));

      builderAdd.setInsertionPoint(adjoint);
      Value retrievedValue = gutils->popCache(cache, builderAdd);
      if (!usesTensors){
        retrievedValue = invertMemref(retrievedValue, builderAdd, op->getLoc());
      } else {
        retrievedValue = invertTensor(retrievedValue, builderAdd, op->getLoc());
      }
      adjoint.getInputsMutable().append(ValueRange({retrievedValue}));
      indexingMapsAdjoint.insert(
          indexingMapsAdjoint.begin() + numInputsAdjoint + i,
          AffineMap::getMultiDimIdentityMap(iterationDomains.size(),
                                            builderAdd.getContext()));
    }
    
    SmallVector<Attribute> indexingMapsAttr;
    SmallVector<Attribute> indexingMapsAttrAdjoint;
    for (auto &map : newIndexingMaps) {
      indexingMapsAttr.push_back(AffineMapAttr::get(map));
    }
    for (auto &map : indexingMapsAdjoint) {
      indexingMapsAttrAdjoint.push_back(AffineMapAttr::get(map));
    }
    cast<linalg::GenericOp>(newOp).setIndexingMapsAttr(
        cacheBuilder.getArrayAttr(indexingMapsAttr));
    adjoint->setAttr(adjoint.getIndexingMapsAttrName(),
                     builder.getArrayAttr(indexingMapsAttrAdjoint));
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};
} // namespace

template <typename... Ts> void attachAllInterfaces(MLIRContext *context) {
  (Ts::template attachInterface<GenericOpInterfaceReverse<Ts>>(*context), ...);
}

void mlir::enzyme::registerLinalgDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, linalg::LinalgDialect *) {
    attachAllInterfaces<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(context);
  });
}