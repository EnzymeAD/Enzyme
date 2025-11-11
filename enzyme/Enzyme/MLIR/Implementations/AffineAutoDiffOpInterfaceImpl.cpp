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
#include "Passes/RemovalUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::affine;

namespace {
affine::AffineForOp
createAffineForWithShadows(Operation *op, OpBuilder &builder,
                           MGradientUtils *gutils, Operation *original,
                           ValueRange remappedOperands, TypeRange rettys) {
  affine::AffineForOpAdaptor adaptor(remappedOperands,
                                     cast<affine::AffineForOp>(original));
  auto repFor = affine::AffineForOp::create(
      builder, original->getLoc(), adaptor.getLowerBoundOperands(),
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
  return affine::AffineIfOp::create(
      builder, original->getLoc(), rettys, original.getIntegerSet(),
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

    auto revFor = affine::AffineForOp::create(
        builder, op->getLoc(), revLBOperands, lb.getMap(), revUBOperands,
        ub.getMap(), forOp.getStepAsInt(), incomingGradients);

    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), revFor->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());

        // Create implicit terminator if not present (when num results > 0)
        if (revBB.empty()) {
          affine::AffineYieldOp::create(bodyBuilder, revFor->getLoc());
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
              auto iface = dyn_cast<AutoDiffTypeInterface>(res.getType());
              if (iface && !iface.isMutable())
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

    return success(valid);
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

struct AffineParallelOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AffineParallelOpInterfaceReverse, affine::AffineParallelOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto parOp = cast<affine::AffineParallelOp>(op);
    if (!parOp.getReductions().empty()) {
      return parOp.emitError() << "parallel reductions not yet implemented";
    }
    if (parOp.hasMinMaxBounds()) {
      return parOp.emitError() << "minmax bounds not yet supported";
    }

    SmallVector<Value> bounds = llvm::map_to_vector(
        caches, [&](Value cache) { return gutils->popCache(cache, builder); });
    auto revPar = affine::AffineParallelOp::create(
        builder, op->getLoc(), parOp.getResultTypes(), parOp.getReductions(),
        parOp.getLowerBoundsMap(), parOp.getLowerBoundsGroups(),
        parOp.getUpperBoundsMap(), parOp.getUpperBoundsGroups(),
        parOp.getSteps(), bounds);

    // Create the body block and terminator
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Type> ivTypes(parOp.getIVs().size(), builder.getIndexType());
    SmallVector<Location> ivLocs(parOp.getIVs().size(), parOp.getLoc());
    builder.createBlock(&revPar.getBodyRegion(), revPar.getBodyRegion().begin(),
                        ivTypes, ivLocs);
    affine::AffineYieldOp::create(builder, parOp.getLoc());

    bool valid = true;
    bool wasAtomic = gutils->AtomicAdd;
    gutils->AtomicAdd = true;
    std::function<Value(Location, Type)> gradientCreator = [&](Location loc,
                                                               Type t) {
      auto shadowty = getShadowType(t);
      OpBuilder builder(t.getContext());
      // Gradients of values defined within the parallel body should be local to
      // each iteration
      builder.setInsertionPointToStart(revPar.getBody());

      auto shadow = enzyme::InitOp::create(
          builder, loc, enzyme::GradientType::get(t.getContext(), shadowty));
      auto toset =
          cast<AutoDiffTypeInterface>(shadowty).createNullValue(builder, loc);
      enzyme::SetOp::create(builder, loc, shadow, toset);
      return shadow;
    };

    {
      Block *oBB = parOp.getBody();
      Block *rBB = revPar.getBody();

      OpBuilder bodyBuilder = revPar.getBodyBuilder();

      bodyBuilder.setInsertionPointToStart(revPar.getBody());
      mlir::enzyme::localizeGradients(bodyBuilder, gutils, oBB);

      bodyBuilder.setInsertionPoint(rBB->getTerminator());

      auto first = oBB->rbegin();
      first++; // skip terminator

      auto last = oBB->rend();

      for (auto it = first; it != last; ++it) {
        Operation *op = &*it;
        valid &= gutils->Logic.visitChild(op, bodyBuilder, gutils).succeeded();
      }
    }

    gutils->AtomicAdd = wasAtomic;
    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto parOp = cast<affine::AffineParallelOp>(op);

    SmallVector<Value> caches;
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    for (auto operand : parOp.getMapOperands()) {
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(operand), cacheBuilder));
    }
    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct AffineParallelOpEnzymeOpsRemover
    : public ForLikeEnzymeOpsRemover<AffineParallelOpEnzymeOpsRemover,
                                     affine::AffineParallelOp> {
  static SmallVector<IntOrValue, 1>
  getDimensionBounds(OpBuilder &builder, affine::AffineParallelOp parOp) {
    SmallVector<IntOrValue, 1> bounds;
    auto ranges = parOp.getConstantRanges();
    if (ranges) {
      for (auto &&[r, step] : llvm::zip(*ranges, parOp.getSteps())) {
        bounds.push_back(r / step);
      }
    } else {
      for (auto &&[dim, step] : llvm::enumerate(parOp.getSteps())) {
        auto lb = AffineApplyOp::create(builder, parOp.getLoc(),
                                        parOp.getLowerBoundMap(dim),
                                        parOp.getLowerBoundsOperands());
        auto ub = AffineApplyOp::create(builder, parOp.getLoc(),
                                        parOp.getUpperBoundMap(dim),
                                        parOp.getUpperBoundsOperands());
        Value diff = arith::SubIOp::create(builder, parOp.getLoc(), ub, lb);
        if (step != 1) {
          Value stepVal =
              arith::ConstantIndexOp::create(builder, parOp.getLoc(), step);
          diff = arith::DivUIOp::create(builder, parOp.getLoc(), diff, stepVal);
        }
        bounds.push_back(diff);
      }
    }
    return bounds;
  }

  static SmallVector<Value> computeReversedIndices(
      PatternRewriter &rewriter, affine::AffineParallelOp parOp,
      ArrayRef<Value> otherInductionVariable, ArrayRef<IntOrValue> bounds) {
    return SmallVector<Value>(otherInductionVariable);
  }

  static SmallVector<Value>
  getCanonicalLoopIVs(OpBuilder &builder, affine::AffineParallelOp parOp) {
    SmallVector<Value> ivs(parOp.getIVs());
    for (auto &&[dim, step] : llvm::enumerate(parOp.getSteps())) {
      Value iv = ivs[dim];
      auto lbMap = parOp.getLowerBoundMap(dim);
      if (!(lbMap.isSingleConstant() && lbMap.getSingleConstantResult() == 0)) {
        auto lb = AffineApplyOp::create(builder, parOp.getLoc(), lbMap,
                                        parOp.getLowerBoundsOperands());
        iv = arith::SubIOp::create(builder, parOp.getLoc(), iv, lb);
      }

      if (step != 1) {
        auto stepVal =
            arith::ConstantIndexOp::create(builder, parOp.getLoc(), step);
        iv = arith::DivUIOp::create(builder, parOp.getLoc(), iv, stepVal);
      }

      ivs[dim] = iv;
    }
    return ivs;
  }

  static IRMapping createArgumentMap(PatternRewriter &rewriter,
                                     affine::AffineParallelOp parOp,
                                     ArrayRef<Value> indPar,
                                     affine::AffineParallelOp otherParOp,
                                     ArrayRef<Value> indOther) {
    IRMapping map;
    for (auto &&[f, o] : llvm::zip_equal(indPar, indOther))
      map.map(f, o);

    for (auto &&[fiv, oiv] :
         llvm::zip_equal(parOp.getIVs(), otherParOp.getIVs())) {
      if (!map.contains(fiv)) {
        assert(parOp.getLowerBoundsMap() == otherParOp.getLowerBoundsMap());
        for (auto &&[f, o] :
             llvm::zip_equal(parOp.getLowerBoundsOperands(),
                             otherParOp.getLowerBoundsOperands()))
          assert(Equivalent(f, o));
        for (auto [fstep, ostep] :
             llvm::zip_equal(parOp.getSteps(), otherParOp.getSteps()))
          assert(fstep == ostep);
        map.map(fiv, oiv);
      }
    }
    return map;
  }

  static affine::AffineParallelOp
  replaceWithNewOperands(PatternRewriter &rewriter,
                         affine::AffineParallelOp otherParOp,
                         ArrayRef<Value> operands) {
    SmallVector<mlir::Attribute> reductionKinds(
        otherParOp.getReductions().begin(), otherParOp.getReductions().end());

    for (unsigned i = otherParOp->getNumOperands(); i < operands.size(); i++) {
      reductionKinds.push_back(arith::AtomicRMWKindAttr::get(
          otherParOp.getContext(), arith::AtomicRMWKind::addf));
    }

    ValueRange operands_(operands);
    auto newOtherParOp = affine::AffineParallelOp::create(
        rewriter, otherParOp.getLoc(), operands_.getTypes(),
        ArrayAttr::get(otherParOp.getContext(), reductionKinds),
        otherParOp.getLowerBoundsMap(), otherParOp.getLowerBoundsGroups(),
        otherParOp.getUpperBoundsMap(), otherParOp.getUpperBoundsGroups(),
        otherParOp.getSteps(), otherParOp.getMapOperands());

    newOtherParOp.getRegion().takeBody(otherParOp.getRegion());
    rewriter.replaceOp(otherParOp, newOtherParOp->getResults().slice(
                                       0, otherParOp->getNumResults()));
    return newOtherParOp;
  }

  static ValueRange getInits(affine::AffineParallelOp parOp) {
    return parOp.getInits();
  }

  static bool mustPostAdd(affine::AffineParallelOp forOp) { return true; }

  static Value initialValueInBlock(OpBuilder &builder, Block *body,
                                   Value grad) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    return cast<AutoDiffTypeInterface>(
               cast<enzyme::GradientType>(grad.getType()).getBasetype())
        .createNullValue(builder, grad.getLoc());
  }
};

static void computeAffineIndices(OpBuilder &builder, Location loc,
                                 AffineMap map, ValueRange operands,
                                 SmallVectorImpl<Value> &indices) {
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    indices.push_back(
        AffineApplyOp::create(builder, loc, map.getSubMap({i}), operands));
  }
}

struct AffineLoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AffineLoadOpInterfaceReverse, affine::AffineLoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<affine::AffineLoadOp>(op);
    Value memref = loadOp.getMemref();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(loadOp) &&
          !gutils->isConstantValue(memref)) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value memrefGradient = gutils->popCache(caches.front(), builder);

        SmallVector<Value> retrievedArguments;
        for (Value cache : ValueRange(caches).drop_front(1)) {
          Value retrievedValue = gutils->popCache(cache, builder);
          retrievedArguments.push_back(retrievedValue);
        }

        if (!gutils->AtomicAdd) {
          bool hasIndex = loadOp.getAffineMap().getNumDims() > 0;
          // if index had to be cached, the pop is not necessarily a valid index
          if (hasIndex) {
            SmallVector<Value> indices;
            computeAffineIndices(builder, loadOp.getLoc(),
                                 loadOp.getAffineMap(), retrievedArguments,
                                 indices);

            Value loadedGradient = memref::LoadOp::create(
                builder, loadOp.getLoc(), memrefGradient, indices);
            Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                    loadedGradient, gradient);
            memref::StoreOp::create(builder, loadOp.getLoc(), addedGradient,
                                    memrefGradient, indices);
          } else {
            Value loadedGradient = affine::AffineLoadOp::create(
                builder, loadOp.getLoc(), memrefGradient, loadOp.getAffineMap(),
                ArrayRef<Value>(retrievedArguments));
            Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                    loadedGradient, gradient);
            affine::AffineStoreOp::create(
                builder, loadOp.getLoc(), addedGradient, memrefGradient,
                loadOp.getAffineMap(), ArrayRef<Value>(retrievedArguments));
          }
        } else {
          SmallVector<Value> indices;
          computeAffineIndices(builder, loadOp.getLoc(), loadOp.getAffineMap(),
                               retrievedArguments, indices);
          memref::AtomicRMWOp::create(builder, loadOp.getLoc(),
                                      arith::AtomicRMWKind::addf, gradient,
                                      memrefGradient, indices);
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
    // auto loadOp = cast<memref::LoadOp>(op);
    // Value memref = loadOp.getMemref();
    // Value shadow = gutils->getShadowValue(memref);
    // Do nothing yet. In the future support memref<memref<...>>
  }
};

struct AffineStoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AffineStoreOpInterfaceReverse, affine::AffineStoreOp> {
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

      Value memrefGradient = gutils->popCache(caches.front(), builder);

      SmallVector<Value> retrievedArguments;
      for (Value cache : ValueRange(caches).drop_front(1)) {
        Value retrievedValue = gutils->popCache(cache, builder);
        retrievedArguments.push_back(retrievedValue);
      }

      bool hasIndex = storeOp.getAffineMap().getNumDims() > 0;

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient;
          if (hasIndex) {
            SmallVector<Value> indices;
            computeAffineIndices(builder, storeOp.getLoc(),
                                 storeOp.getAffineMap(), retrievedArguments,
                                 indices);
            loadedGradient = memref::LoadOp::create(builder, storeOp.getLoc(),
                                                    memrefGradient, indices);
          } else {
            loadedGradient = affine::AffineLoadOp::create(
                builder, storeOp.getLoc(), memrefGradient,
                storeOp.getAffineMap(), ArrayRef<Value>(retrievedArguments));
          }
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        // if index had to be cached, the pop is not necessarily a valid index
        if (hasIndex) {
          SmallVector<Value> indices;
          computeAffineIndices(builder, storeOp.getLoc(),
                               storeOp.getAffineMap(), retrievedArguments,
                               indices);
          memref::StoreOp::create(builder, storeOp.getLoc(), zero,
                                  memrefGradient, indices);
        } else {
          affine::AffineStoreOp::create(builder, storeOp.getLoc(), zero,
                                        memrefGradient, storeOp.getAffineMap(),
                                        ArrayRef<Value>(retrievedArguments));
        }
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

struct AffineForOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<AffineForOpADDataFlow,
                                                  affine::AffineForOp> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   OpResult res) const {
    auto forOp = cast<affine::AffineForOp>(op);
    return {
        forOp.getInits()[res.getResultNumber()],
        forOp.getBody()->getTerminator()->getOperand(res.getResultNumber())};
  }
  SmallVector<Value> getPotentialIncomingValuesArg(Operation *op,
                                                   BlockArgument arg) const {
    auto forOp = cast<affine::AffineForOp>(op);
    if (arg.getArgNumber() < 1) {
      return {};
    }
    auto idx = arg.getArgNumber() - 1;
    return {forOp.getInits()[idx],
            forOp.getBody()->getTerminator()->getOperand(idx)};
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value val) const {
    auto forOp = cast<affine::AffineForOp>(op);
    SmallVector<Value> sv;

    for (auto &&[res, arg, barg] :
         llvm::zip_equal(forOp->getResults(), term->getOperands(),
                         forOp.getRegionIterArgs())) {
      if (arg == val) {
        sv.push_back(res);
        sv.push_back(barg);
      }
    }

    return sv;
  }
};

struct AffineForOpEnzymeOpsRemover
    : public ForLikeEnzymeOpsRemover<AffineForOpEnzymeOpsRemover,
                                     affine::AffineForOp> {
public:
  // TODO: support non constant number of iteration by using unknown dimensions
  static std::optional<int64_t>
  getConstantNumberOfIterations(affine::AffineForOp forOp) {
    if (!forOp.hasConstantLowerBound())
      return std::nullopt;
    if (!forOp.hasConstantUpperBound())
      return std::nullopt;
    return (forOp.getConstantUpperBound() - forOp.getConstantLowerBound()) /
           forOp.getStepAsInt();
  }

  static SmallVector<IntOrValue, 1>
  getDimensionBounds(OpBuilder &builder, affine::AffineForOp forOp) {
    auto iters = getConstantNumberOfIterations(forOp);
    if (iters) {
      return {IntOrValue(*iters)};
    } else {
      auto lb = AffineApplyOp::create(builder, forOp.getLoc(),
                                      forOp.getLowerBoundMap(),
                                      forOp.getLowerBoundOperands());
      auto ub = AffineApplyOp::create(builder, forOp.getLoc(),
                                      forOp.getUpperBoundMap(),
                                      forOp.getUpperBoundOperands());

      Value diff = arith::SubIOp::create(builder, forOp->getLoc(), ub, lb);
      if (forOp.getStepAsInt() != 1) {
        auto step = arith::ConstantIntOp::create(
            builder, forOp->getLoc(), diff.getType(), forOp.getStepAsInt());
        diff = arith::DivUIOp::create(builder, forOp->getLoc(), diff, step);
      }
      return {IntOrValue(diff)};
    }
  }

  static SmallVector<Value> getCanonicalLoopIVs(OpBuilder &builder,
                                                affine::AffineForOp forOp) {
    Value val = forOp.getBody()->getArgument(0);
    if (!forOp.hasConstantLowerBound() || forOp.getConstantLowerBound() != 0) {
      auto lb = AffineApplyOp::create(builder, forOp.getLoc(),
                                      forOp.getLowerBoundMap(),
                                      forOp.getLowerBoundOperands());
      val = arith::SubIOp::create(builder, forOp->getLoc(), val, lb);
    }

    if (forOp.getStepAsInt() != 1) {
      auto step = arith::ConstantIntOp::create(
          builder, forOp->getLoc(), val.getType(), forOp.getStepAsInt());
      val = arith::DivUIOp::create(builder, forOp->getLoc(), val, step);
    }
    return {val};
  }

  static IRMapping createArgumentMap(PatternRewriter &rewriter,
                                     affine::AffineForOp forOp,
                                     ArrayRef<Value> indFor,
                                     affine::AffineForOp otherForOp,
                                     ArrayRef<Value> indOther) {
    IRMapping map;
    for (auto &&[f, o] : llvm::zip_equal(indFor, indOther))
      map.map(f, o);

    Value canIdx = forOp.getBody()->getArgument(0);
    if (!map.contains(canIdx)) {
      assert(forOp.getLowerBoundMap() == otherForOp.getLowerBoundMap());
      for (auto &&[f, o] : llvm::zip_equal(forOp.getLowerBoundOperands(),
                                           otherForOp.getLowerBoundOperands()))
        assert(Equivalent(f, o));
      assert(forOp.getStep() == otherForOp.getStep());
      map.map(forOp.getBody()->getArgument(0),
              otherForOp.getBody()->getArgument(0));
    }
    return map;
  }

  static affine::AffineForOp
  replaceWithNewOperands(PatternRewriter &rewriter,
                         affine::AffineForOp otherForOp,
                         ArrayRef<Value> operands) {
    auto newOtherForOp = affine::AffineForOp::create(
        rewriter, otherForOp->getLoc(), otherForOp.getLowerBoundOperands(),
        otherForOp.getLowerBoundMap(), otherForOp.getUpperBoundOperands(),
        otherForOp.getUpperBoundMap(), otherForOp.getStepAsInt(), operands);

    newOtherForOp.getRegion().takeBody(otherForOp.getRegion());
    rewriter.replaceOp(otherForOp, newOtherForOp->getResults().slice(
                                       0, otherForOp->getNumResults()));
    return newOtherForOp;
  }

  static ValueRange getInits(affine::AffineForOp forOp) {
    return forOp.getInits();
  }

  static bool mustPostAdd(affine::AffineForOp forOp) { return false; }

  static Value initialValueInBlock(OpBuilder &builder, Block *body,
                                   Value grad) {
    auto Ty = cast<enzyme::GradientType>(grad.getType()).getBasetype();
    return body->addArgument(Ty, grad.getLoc());
  }
};

#include "Implementations/AffineDerivatives.inc"
} // namespace

void mlir::enzyme::registerAffineDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, affine::AffineDialect *) {
    registerInterfaces(context);
    affine::AffineLoadOp::attachInterface<AffineLoadOpInterfaceReverse>(
        *context);
    affine::AffineStoreOp::attachInterface<AffineStoreOpInterfaceReverse>(
        *context);
    affine::AffineForOp::attachInterface<AffineForOpInterfaceReverse>(*context);
    affine::AffineForOp::attachInterface<AffineForOpEnzymeOpsRemover>(*context);
    affine::AffineForOp::attachInterface<AffineForOpADDataFlow>(*context);
    affine::AffineParallelOp::attachInterface<AffineParallelOpInterfaceReverse>(
        *context);
    affine::AffineParallelOp::attachInterface<AffineParallelOpEnzymeOpsRemover>(
        *context);
  });
}
