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

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>

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

template <typename T_>
struct GenericOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GenericOpInterfaceReverse<T_>, T_> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    // TODO: currently in progress, this doesn't work
    auto linalgOp = cast<linalg::LinalgOp>(op);
    assert(linalgOp.hasBufferSemantics() &&
           "Linalg op with tensor semantics not yet supported");

    linalg::LinalgOp newOp =
        cast<linalg::LinalgOp>(gutils->getNewFromOriginal(linalgOp));

    ConversionPatternRewriter rewriter(builder.getContext());
    auto failiureOrLinalgOp = generalizeNamedOp(rewriter, newOp);
    if (!failed(failiureOrLinalgOp)) {
      linalg::GenericOp replacement = failiureOrLinalgOp.value();
      OpBuilder replacementBuilder(newOp);
      replacementBuilder.insert(replacement);
      newOp.erase();
      newOp = replacement;
    }

    auto cacheBuilder = OpBuilder(newOp);

    // get iteration domain
    AffineMap aMap = newOp.getShapesToLoopsMap();
    SmallVector<Value> dims;
    for (OpOperand *input : newOp.getInputOperands()) {
      auto shape = cast<MemRefType>(input->get().getType()).getShape();
      for (int i = 0; i < (int)shape.size(); i++) {
        auto dimI =
            cacheBuilder.create<arith::ConstantIndexOp>(op->getLoc(), i);
        auto dim = cacheBuilder.create<memref::DimOp>(op->getLoc(),
                                                      input->get(), dimI);
        dims.push_back(dim);
      }
    }

    SmallVector<Value> iterationDomains;
    SmallVector<int64_t> shapes;
    for (unsigned int i = 0; i < aMap.getNumResults(); i++) {
      AffineMap subMap = aMap.getSubMap({i});
      Value domain = cacheBuilder.create<AffineApplyOp>(op->getLoc(), subMap,
                                                        ValueRange(dims));
      iterationDomains.push_back(domain);
      shapes.push_back(ShapedType::kDynamicSize);
    }

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> indexingMaps;
    SmallVector<StringRef> iteratorTypes{linalgOp.getNumLoops(),
                                         getParallelIteratorTypeName()};

    for (OpOperand *output : linalgOp.getOutputOperands()) {
      if (!gutils->hasInvertPointer(output->get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(output));
      Value out = gutils->invertPointerM(output->get(), builder);
      Value view = invertMemref(out, builder, op->getLoc());
      outputs.push_back(view);
    }

    for (OpOperand *input : linalgOp.getInputOperands()) {
      if (!gutils->hasInvertPointer(input->get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(input));
      Value inp = gutils->invertPointerM(input->get(), builder);
      Value view = invertMemref(inp, builder, op->getLoc());
      inputs.push_back(view);
    }

    linalg::GenericOp adjoint = builder.create<linalg::GenericOp>(
        op->getLoc(), outputs, inputs, indexingMaps, iteratorTypes);

    int numInputs = inputs.size();
    auto buildFuncReturnOp = [numInputs](OpBuilder &builder, Location loc,
                                         SmallVector<Value> retargs) {
      builder.create<linalg::YieldOp>(
          loc, ValueRange{retargs}.take_front(numInputs));
      return;
    };

    Region *newOpRegion = newOp.getBlock()->getParent();
    int numInputsNewOp = cast<linalg::GenericOp>(newOp).getInputs().size();
    Region *adjointRegion = &adjoint.getBodyRegion();
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
        gutils, *linalgOp.getBlock()->getParent(), adjoint.getBodyRegion(),
        /*parentRegion=*/false, buildFuncReturnOp, hook);

    // TODO add pushCaches to the yield in newOp
    auto newOpYield = cast<linalg::YieldOp>(
        cast<linalg::GenericOp>(newOp).getBodyRegion().front().getTerminator());
    for (Value pc : pushCaches) {
      newOpYield.getValuesMutable().append(pc);
    }

    Block *body = &(adjoint.getBodyRegion().front());
    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());

    for (auto opOperand : yieldOp.getOperands()) {
      body->addArgument(opOperand.getType(), opOperand.getLoc());
    }
    OpBuilder builderAdd(yieldOp);
    for (auto &&[index, value] : llvm::enumerate(yieldOp.getOperands())) {
      Value arg = body->getArgument(outputs.size() + numCaches + index);
      auto diffeType = cast<AutoDiffTypeInterface>(arg.getType());
      Value grad =
          diffeType.createAddOp(builderAdd, value.getLoc(), arg, value);
      yieldOp.setOperand(index, grad);
    }

    auto newIndexingMaps = newOp.getIndexingMapsArray();
    auto indexingMapsAdjoint = adjoint.getIndexingMapsArray();
    for (int i = 0; i < numCaches; i++) {
      Value cacheArg = body->getArgument(outputs.size() + i);

      Type ct = cacheArg.getType();
      Type type = MemRefType::get(shapes, ct);
      auto alloc = cacheBuilder.create<memref::AllocOp>(
          op->getLoc(), type, ValueRange(iterationDomains));
      Value cache = gutils->initAndPushCache(alloc, cacheBuilder);
      alloc->setAttr(alloc.getOperandSegmentSizesAttrName(),
                     cacheBuilder.getDenseI32ArrayAttr({1, 0}));

      cast<linalg::GenericOp>(newOp).getOutputsMutable().append(
          ValueRange({alloc}));
      newIndexingMaps.push_back(AffineMap::getMultiDimIdentityMap(
          iterationDomains.size(), cacheBuilder.getContext()));

      builderAdd.setInsertionPoint(adjoint);
      Value retrievedValue = gutils->popCache(cache, builderAdd);
      retrievedValue = invertMemref(retrievedValue, builderAdd, op->getLoc());
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
    newOp->setAttr(cast<linalg::GenericOp>(newOp).getIndexingMapsAttrName(),
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