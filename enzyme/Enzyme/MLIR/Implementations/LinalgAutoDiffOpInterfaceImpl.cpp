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

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
  Value negOne = arith::ConstantIndexOp::create(builder, loc, -1);
  int shapeDim = iType.getShape().size();
  for (int i = 0; i < shapeDim; i++) {
    Value dim = memref::DimOp::create(builder, loc, inp, i);
    dims.push_back(dim);
    auto dimSubOne = arith::AddIOp::create(builder, loc, dim, negOne);
    dimSubOnes.push_back(dimSubOne);
    strides.push_back(negOne);
  }
  Value view =
      memref::SubViewOp::create(builder, loc, inp, ValueRange(dimSubOnes),
                                ValueRange(dims), ValueRange(strides));
  return view;
}

SmallVector<AffineMap> getIndexingMapsArray(enzyme::GenericAdjointOp &op) {
  auto attr = op.getIndexingMapsAttr();
  SmallVector<AffineMap> indexingMaps;
  for (auto map : attr.getValue()) {
    indexingMaps.push_back(cast<AffineMapAttr>(map).getValue());
  }
  return indexingMaps;
}

template <typename T_>
struct GenericOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GenericOpInterfaceReverse<T_>, T_> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    if (!linalgOp.hasPureBufferSemantics()) {
      llvm::errs() << "Linalg op with tensor semantics not yet supported\n";
      return failure();
    }

    linalg::LinalgOp newOp =
        cast<linalg::LinalgOp>(gutils->getNewFromOriginal(linalgOp));

    // Replace the op by a linalg.generic op if necessary
    IRRewriter rewriter(builder.getContext(), builder.getListener());
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
    AffineMap aMap = newOp.getShapesToLoopsMap();
    SmallVector<Value> dims;
    for (OpOperand *input : newOp.getDpsInputOperands()) {
      auto shape = cast<MemRefType>(input->get().getType()).getShape();
      for (unsigned i = 0; i < shape.size(); i++) {
        auto dimI =
            arith::ConstantIndexOp::create(cacheBuilder, op->getLoc(), i);
        auto dim = memref::DimOp::create(cacheBuilder, op->getLoc(),
                                         input->get(), dimI);
        dims.push_back(dim);
      }
    }
    for (Value output : newOp.getDpsInits()) {
      auto shape = cast<MemRefType>(output.getType()).getShape();
      for (unsigned i = 0; i < shape.size(); i++) {
        auto dimI =
            arith::ConstantIndexOp::create(cacheBuilder, op->getLoc(), i);
        auto dim =
            memref::DimOp::create(cacheBuilder, op->getLoc(), output, dimI);
        dims.push_back(dim);
      }
    }

    SmallVector<Value> iterationDomains;
    SmallVector<int64_t> shapes;
    for (unsigned int i = 0; i < aMap.getNumResults(); i++) {
      AffineMap subMap = aMap.getSubMap({i});
      Value domain = affine::AffineApplyOp::create(cacheBuilder, op->getLoc(),
                                                   subMap, ValueRange(dims));
      iterationDomains.push_back(domain);
      shapes.push_back(ShapedType::kDynamic);
    }
    //

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes{
        linalgOp.getNumLoops(), utils::IteratorType::parallel};

    for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
      if (gutils->isConstantValue(output.get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(&output));
      Value out = gutils->invertPointerM(output.get(), builder);
      Value view = invertMemref(out, builder, op->getLoc());
      outputs.push_back(view);
    }

    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      if (gutils->isConstantValue(input->get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(input));
      Value inp = gutils->invertPointerM(input->get(), builder);
      Value view = invertMemref(inp, builder, op->getLoc());
      inputs.push_back(view);
    }

    ArrayAttr indexingMapsArrayAttr =
        builder.getAffineMapArrayAttr(indexingMaps);
    ArrayAttr iteratorTypesArrayAttr =
        builder.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes, [&](utils::IteratorType iter) -> mlir::Attribute {
              return linalg::IteratorTypeAttr::get(builder.getContext(), iter);
            })));
    auto adjoint = enzyme::GenericAdjointOp::create(
        builder, op->getLoc(), TypeRange(), ValueRange(outputs),
        ValueRange(inputs), indexingMapsArrayAttr, iteratorTypesArrayAttr,
        StringAttr(), StringAttr());

    int numInputs = inputs.size();
    auto buildFuncReturnOp = [&gutils, numInputs](OpBuilder &builder,
                                                  Block *oBB) {
      auto loc = oBB->rbegin()->getLoc();
      SmallVector<Value> retargs;
      for (auto arg : oBB->getArguments()) {
        retargs.push_back(gutils->invertPointerM(arg, builder));
      }
      enzyme::AddToOp::create(builder, loc,
                              ValueRange{retargs}.take_front(numInputs));
      return;
    };

    Region *newOpRegion = newOp.getBlock()->getParent();
    Region *adjointRegion = &adjoint.getRegion();
    int numInputsAdjoint = adjoint.getInputs().size();
    Location loc = op->getLoc();
    int numCaches = 0;
    SmallVector<Value> pushCaches;

    auto hook = [newOpRegion, adjointRegion, loc, &numCaches = numCaches,
                 numInputsAdjoint, &pushCaches = pushCaches](Type t) {
      OpBuilder builder(newOpRegion);
      Value pushCache = enzyme::InitOp::create(builder, loc, t);
      pushCaches.push_back(pushCache);
      newOpRegion->addArgument(t, loc);

      Value popCache =
          adjointRegion->insertArgument(numInputsAdjoint + numCaches, t, loc);
      numCaches++;
      return std::make_pair(pushCache, popCache);
    };

    auto sub = gutils->Logic.differentiate(
        gutils, *linalgOp.getBlock()->getParent(), adjoint.getRegion(),
        buildFuncReturnOp, hook);
    if (!sub.succeeded())
      return sub;

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
      Type type = MemRefType::get(shapes, ct);
      auto alloc = memref::AllocOp::create(cacheBuilder, op->getLoc(), type,
                                           ValueRange(iterationDomains));
      Value cache = gutils->initAndPushCache(alloc, cacheBuilder);
      // TODO use higher level API
      alloc->setAttr(alloc.getOperandSegmentSizesAttrName(),
                     cacheBuilder.getDenseI32ArrayAttr(
                         {static_cast<int32_t>(iterationDomains.size()), 0}));

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
    cast<linalg::GenericOp>(newOp).setIndexingMapsAttr(
        cacheBuilder.getArrayAttr(indexingMapsAttr));
    adjoint->setAttr(adjoint.getIndexingMapsAttrName(),
                     builder.getArrayAttr(indexingMapsAttrAdjoint));
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

class GenericFwd
    : public AutoDiffOpInterface::ExternalModel<GenericFwd, linalg::GenericOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {

    auto op = cast<linalg::GenericOp>(orig);

    // For all active results, add shadow types.
    // For now, assuming all results are relevant.
    Operation *newOp = gutils->getNewFromOriginal(op);
    SmallVector<Type> newOpResultTypes;
    newOpResultTypes.reserve(op->getNumResults() * 2);
    for (auto &&[result, init] :
         llvm::zip_equal(op->getResults(), op.getOutputs())) {
      newOpResultTypes.push_back(result.getType());
      if (gutils->isConstantValue(result) && gutils->isConstantValue(init)) {
        continue;
      }
      auto typeIface = dyn_cast<AutoDiffTypeInterface>(result.getType());
      if (!typeIface) {
        op->emitError() << " AutoDiffTypeInterface not implemented for "
                        << result.getType() << "\n";
        return failure();
      }
      newOpResultTypes.push_back(typeIface.getShadowType(gutils->width));
    }

    SmallVector<Value> newInputs;
    SmallVector<Value> newOutputs;
    SmallVector<AffineMap> indexingMaps;
    {
      size_t idx = 0;
      for (Value operand : op.getInputs()) {
        newInputs.push_back(gutils->getNewFromOriginal(operand));
        indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        if (!gutils->isConstantValue(operand)) {
          newInputs.push_back(gutils->invertPointerM(operand, builder));
          indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        }
        idx++;
      }
      for (auto &&[operand, res, oarg] :
           llvm::zip_equal(op.getOutputs(), op->getResults(),
                           op.getRegion().front().getArguments().slice(
                               op.getInputs().size()))) {
        newOutputs.push_back(gutils->getNewFromOriginal(operand));
        indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        bool shadow = false;
        if (!gutils->isConstantValue(operand)) {
          shadow = true;
          newOutputs.push_back(gutils->invertPointerM(operand, builder));
          indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        } else if (!gutils->isConstantValue(res)) {
          auto typeIface = dyn_cast<AutoDiffTypeInterface>(operand.getType());
          shadow = true;
          newOutputs.push_back(
              typeIface.createNullValue(builder, operand.getLoc()));
          indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        }

        if (shadow && gutils->isConstantValue(oarg)) {
          auto typeIface = dyn_cast<AutoDiffTypeInterface>(oarg.getType());
          auto newBA = cast<BlockArgument>(gutils->getNewFromOriginal(oarg));
          newBA.getOwner()->insertArgument(newBA.getArgNumber() + 1,
                                           typeIface.getShadowType(),
                                           newBA.getLoc());
        }

        idx++;
      }
    }
    // We are assuming the op can forward additional operands, listed
    // immediately after the original operands, to the same regions.
    // ^^
    // Our interface guarantees this.
    // We also assume that the region-holding op returns all of the values
    // yielded by terminators, and only those values.

    auto replacement = linalg::GenericOp::create(
        builder, op.getLoc(), newOpResultTypes, newInputs, newOutputs,
        indexingMaps, op.getIteratorTypesArray(),
        /*doc*/ "",
        /*libraryCall*/ "");

    assert(replacement->getNumResults() == newOpResultTypes.size());
    for (auto &&[region, replacementRegion] :
         llvm::zip(newOp->getRegions(), replacement->getRegions())) {
      replacementRegion.takeBody(region);
    }

    // Inject the mapping for the new results into GradientUtil's shadow
    // table.
    SmallVector<Value> reps;
    size_t idx = 0;
    for (OpResult r : op->getResults()) {
      // TODO only if used
      reps.push_back(replacement->getResult(idx));
      idx++;
      if (!gutils->isConstantValue(r)) {
        auto inverted = gutils->invertedPointers.lookupOrNull(r);
        assert(inverted);
        gutils->invertedPointers.map(r, replacement->getResult(idx));
        inverted.replaceAllUsesWith(replacement->getResult(idx));
        gutils->erase(inverted.getDefiningOp());
        idx++;
      }
    }

    // Differentiate body.
    for (auto &origRegion : op->getRegions()) {
      for (auto &origBlock : origRegion) {
        for (Operation &o : origBlock) {
          if (failed(gutils->visitChild(&o))) {
            return failure();
          }
        }
      }
    }

    // Replace all uses of original results
    gutils->replaceOrigOpWith(op, reps);
    gutils->erase(newOp);
    gutils->originalToNewFnOps[op] = replacement;

    return success();
  }
};

#include "Implementations/LinalgDerivatives.inc"
} // namespace

template <typename... Ts> void attachAllInterfaces(MLIRContext *context) {
  (Ts::template attachInterface<GenericOpInterfaceReverse<Ts>>(*context), ...);
}

void mlir::enzyme::registerLinalgDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, linalg::LinalgDialect *) {
    registerInterfaces(context);
    linalg::GenericOp::attachInterface<GenericFwd>(*context);
    attachAllInterfaces<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(context);
  });
}
