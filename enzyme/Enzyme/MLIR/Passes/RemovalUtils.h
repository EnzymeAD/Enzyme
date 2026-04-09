//===- RemovalUtils.h - Utilities to remove Enzyme ops -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/IR/Matchers.h"

namespace mlir {
namespace enzyme {

constexpr static llvm::StringLiteral kPreserveCacheAttrName = "preserve_cache";

/// Information about a cache, each cache init should have one corresponding
/// push and pop.
struct CacheInfo {
  enzyme::InitOp initOp;
  enzyme::PushOp pushOp;
  enzyme::PopOp popOp;

  CacheInfo() {
    initOp = nullptr;
    pushOp = nullptr;
    popOp = nullptr;
  }
  CacheInfo(Value cache) {
    initOp = cache.getDefiningOp<enzyme::InitOp>();
    unsigned nusers = 0;
    for (auto user : cache.getUsers()) {
      nusers++;
      if (!popOp)
        popOp = dyn_cast<enzyme::PopOp>(user);
      if (!pushOp)
        pushOp = dyn_cast<enzyme::PushOp>(user);
    }
    (void)nusers;
    assert(nusers == 2); // TODO: support more uses
  }

  Value pushedValue() { return pushOp.getValue(); }
  Type cachedType() {
    return cast<enzyme::CacheType>(initOp.getResult().getType()).getType();
  }

  // Pushed values must be the same
  CacheInfo merge(CacheInfo other, PatternRewriter &rewriter);
};

// Tries to limit the amount of values cache from block `forward` to `reverse`
// using a mincut algorithm and heuristics based on the size of values.
// All pushes must go after `lastFwd`, if non null
void minCutCache(Block *forward, Block *reverse, SmallVector<CacheInfo> &caches,
                 PatternRewriter &rewriter, const IRMapping &fwdrevmap,
                 Operation *lastFwd = nullptr);

enum class LoopCacheType { TENSOR, MEMREF };

static LoopCacheType getCacheType(Operation *op) {
  LoopCacheType cacheType = LoopCacheType::MEMREF;
  if (op->hasAttr("enzyme.cache_use_tensor")) {
    cacheType = LoopCacheType::TENSOR;
  }
  return cacheType;
}

static bool hasMinCut(Operation *op) {
  return !op->hasAttr("enzyme.disable_mincut");
}

static bool hasLICM(Operation *op) {
  return !op->hasAttr("enzyme.disable_licm");
}

template <typename FinalClass, typename OpName>
struct ForLikeEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<FinalClass, OpName> {
private:
public:
  struct IntOrValue {
    size_t ival;
    mlir::Value vval;
    IntOrValue(mlir::Value vval) : ival(0), vval(vval) {}
    IntOrValue(size_t ival) : ival(ival), vval(nullptr) {}
  };

  static bool Equivalent(Value lhs, Value rhs) {
    if (lhs == rhs)
      return true;
    Attribute la, ra;
    if (matchPattern(lhs, m_Constant(&la)) &&
        matchPattern(rhs, m_Constant(&ra)))
      return true;
    auto pop = rhs.getDefiningOp<enzyme::PopOp>();
    if (!pop)
      return false;
    auto init = pop.getOperand().getDefiningOp<enzyme::InitOp>();
    if (!init)
      return false;
    for (auto u : init->getResult(0).getUsers()) {
      if (u == pop)
        continue;
      auto push = dyn_cast<enzyme::PushOp>(u);
      if (!push)
        continue;
      return push.getValue() == lhs;
    }
    return false;
  }

  static llvm::SmallVector<mlir::Value>
  computeReversedIndices(PatternRewriter &rewriter, OpName op,
                         llvm::ArrayRef<mlir::Value> otherInductionVariable,
                         llvm::ArrayRef<IntOrValue> bounds) {
    llvm::SmallVector<mlir::Value> results;
    for (auto &&[bound, iv] : llvm::zip_equal(bounds, otherInductionVariable)) {
      Value boundv;
      if (bound.vval) {
        Value c1;
        if (iv.getType().isIndex())
          c1 = arith::ConstantIndexOp::create(rewriter, op->getLoc(), 1);
        else
          c1 = arith::ConstantIntOp::create(rewriter, op->getLoc(),
                                            iv.getType(), 1);
        boundv = arith::SubIOp::create(rewriter, op->getLoc(), bound.vval, c1);
      } else {
        if (iv.getType().isIndex())
          boundv = arith::ConstantIndexOp::create(rewriter, op->getLoc(),
                                                  bound.ival - 1);
        else
          boundv = arith::ConstantIntOp::create(rewriter, op->getLoc(),
                                                iv.getType(), bound.ival - 1);
      }
      Value result = arith::SubIOp::create(rewriter, op->getLoc(), boundv, iv);
      results.push_back(result);
    }
    return results;
  }

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto forOp = cast<OpName>(op);

    if (hasLICM(op)) { // perform licm
      auto loopLike = dyn_cast<LoopLikeOpInterface>(op);

      if (loopLike) {
        mlir::moveLoopInvariantCode(loopLike);
      }
    }

    OpName otherForOp = nullptr; // where caches pops are

    // There is support for two push/pop removal modes, one is using immutable
    // tensors, the other uses memrefs. memref is the default, but tensor can be
    // enabled with enzyme.cache_use_tensor
    auto cacheType = getCacheType(op);

    // Gradients whose values need to be passed as iteration variables.
    llvm::SetVector<Value> updatedGradients;

    llvm::MapVector<Value, CacheInfo> cachesMap;
    SmallVector<CacheInfo> toDelete;

    Block *body = forOp.getBody();

    body->walk([&](Operation *op) {
      if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
        if (!body->getParent()->isAncestor(
                setOp.getGradient().getDefiningOp()->getParentRegion()))
          updatedGradients.insert(setOp.getGradient());
      }

      if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
        CacheInfo info(pushOp.getCache());

        Value pushedValue = info.pushedValue();
        if (cachesMap.contains(pushedValue)) {
          info = info.merge(cachesMap.lookup(pushedValue), rewriter);
        }

        if (info.pushOp->getBlock() == body && info.popOp->getBlock() == body &&
            info.pushOp->isBeforeInBlock(info.popOp)) {
          toDelete.push_back(info);
          return;
        }
        cachesMap[pushedValue] = info;

        if (isa<OpName>(info.popOp->getParentOp())) {
          otherForOp = cast<OpName>(info.popOp->getParentOp());
        }
      }
    });

    while (!toDelete.empty()) {
      CacheInfo info = toDelete.pop_back_val();
      rewriter.replaceAllUsesWith(info.popOp.getResult(),
                                  info.pushOp.getValue());
      rewriter.eraseOp(info.pushOp);
      rewriter.eraseOp(info.popOp);
      rewriter.eraseOp(info.initOp);
    }

    SmallVector<CacheInfo> caches0 =
        llvm::map_to_vector(cachesMap, [](auto p) { return std::get<1>(p); });

    SmallVector<CacheInfo> caches = caches0;

    // nothing to do
    if (updatedGradients.empty() && caches.empty())
      return success();

    DenseMap<Value, llvm::SmallVector<Operation *>> updatedGradientUsers;

    for (auto &it : llvm::make_early_inc_range(*body)) {
      Operation *op = &it;

      auto getOp = dyn_cast<enzyme::GetOp>(op);

      if (getOp && updatedGradients.contains(getOp.getGradient())) {
        updatedGradientUsers[getOp.getGradient()].push_back(getOp);
      } else if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
        updatedGradientUsers[setOp.getGradient()].push_back(setOp);
      }

      if (!getOp || updatedGradients.contains(getOp.getGradient()))
        continue;

      auto outerGet = enzyme::GetOp::create(rewriter, getOp->getLoc(),
                                            getOp.getResult().getType(),
                                            getOp.getGradient());

      rewriter.replaceAllUsesWith(getOp.getResult(), outerGet.getResult());
      rewriter.eraseOp(getOp);
    }

    // postadd means that the loops init is zero and that the result
    // is added with the previous grad after the loop.
    bool postAdd = FinalClass::mustPostAdd(forOp);

    auto term = body->getTerminator();

    SmallVector<Value> newOperands(FinalClass::getInits(forOp));
    for (auto grad : updatedGradients) {
      auto Ty = cast<enzyme::GradientType>(grad.getType()).getBasetype();

      Value newInit;

      if (!postAdd) {
        newInit = enzyme::GetOp::create(rewriter, grad.getLoc(), Ty, grad);
      } else {
        newInit = cast<AutoDiffTypeInterface>(Ty).createNullValue(
            rewriter, grad.getLoc());
      }

      newOperands.push_back(newInit);

      // here we do a primitive form of mem2reg within the loop. We have a
      // sorted (by instruction number) list of all users of the
      // instruction.
      Value val = FinalClass::initialValueInBlock(rewriter, body, grad);
      for (auto user : updatedGradientUsers[grad]) {
        if (auto getOp = dyn_cast<enzyme::GetOp>(user)) {
          rewriter.replaceOp(getOp, val);
        } else {
          auto setOp = cast<enzyme::SetOp>(user);
          val = setOp.getValue();
          rewriter.eraseOp(setOp);
        }
      }

      term->insertOperands(term->getNumOperands(), ValueRange(val));
    }

    IRMapping fwdrevmap;
    bool mincut = false;

    // [0,..., N - 1] counter
    SmallVector<Value> inductionVariable;
    SmallVector<Value> otherInductionVariable;
    SmallVector<Value> reversedIndex;

    SmallVector<IntOrValue> revNumIters;
    SmallVector<IntOrValue> fwdNumIters;

    if (!fwdNumIters.size()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(forOp);
      fwdNumIters = FinalClass::getDimensionBounds(rewriter, forOp);
    }

    Operation *lastFwd = nullptr;
    if (caches.size()) {
      rewriter.setInsertionPointToStart(forOp.getBody());
      inductionVariable = FinalClass::getCanonicalLoopIVs(rewriter, forOp);
      if (rewriter.getInsertionPoint() != forOp.getBody()->begin()) {
        lastFwd = rewriter.getInsertionPoint()->getPrevNode();
      }

      rewriter.setInsertionPointToStart(otherForOp.getBody());
      otherInductionVariable =
          FinalClass::getCanonicalLoopIVs(rewriter, otherForOp);

      // The reverse iteration count may not be known at this point, as it may
      // be cached via a push/pop, use the fwd count in that case.
      if (!revNumIters.size()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(otherForOp);
        revNumIters = FinalClass::getDimensionBounds(rewriter, otherForOp);
        for (auto &&[rev, fwd] : llvm::zip_equal(revNumIters, fwdNumIters)) {
          if (!fwd.vval && rev.vval) {
            rev.vval = nullptr;
            rev.ival = fwd.ival;
          }
        }
      }

      reversedIndex = FinalClass::computeReversedIndices(
          rewriter, otherForOp, otherInductionVariable, revNumIters);
      fwdrevmap = FinalClass::createArgumentMap(
          rewriter, forOp, inductionVariable, otherForOp, reversedIndex);
      for (auto v : inductionVariable) {
        if (auto op = v.getDefiningOp()) {
          op->setAttr("enzyme.no_erase", rewriter.getUnitAttr());
        }
      }
      for (auto v : otherInductionVariable) {
        if (auto op = v.getDefiningOp()) {
          op->setAttr("enzyme.no_erase", rewriter.getUnitAttr());
        }
      }
    }

    if (hasMinCut(forOp) && caches.size()) {
      mincut = true;
      mlir::enzyme::minCutCache(forOp.getBody(), otherForOp.getBody(), caches,
                                rewriter, fwdrevmap, lastFwd);
    }
    for (auto v : inductionVariable) {
      if (auto op = v.getDefiningOp()) {
        op->removeAttr("enzyme.no_erase");
      }
    }
    for (auto v : otherInductionVariable) {
      if (auto op = v.getDefiningOp()) {
        op->removeAttr("enzyme.no_erase");
      }
    }
    auto revIP = rewriter.saveInsertionPoint();

    SmallVector<Value> newPushValues;

    unsigned numNewValuePushes = 0;

    if (lastFwd)
      rewriter.setInsertionPointAfter(lastFwd);
    else
      rewriter.setInsertionPointToStart(forOp.getBody());
    for (auto &info : caches) {

      Value pushedValue = info.pushedValue();
      if (mincut)
        assert(forOp.getRegion().isAncestor(pushedValue.getParentRegion()));

      // Otherwise, add a new variable to keep track.
      if (!inductionVariable.size()) {
        Value zero = arith::ConstantOp::create(rewriter, forOp->getLoc(),
                                               rewriter.getIndexAttr(0));
        newOperands.push_back(zero);

        inductionVariable = {
            body->addArgument(zero.getType(), forOp->getLoc())};
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(term);

          auto one = arith::ConstantOp::create(rewriter, forOp->getLoc(),
                                               rewriter.getIndexAttr(1));
          auto newInductionVar = arith::AddIOp::create(
              rewriter, forOp->getLoc(), inductionVariable[0], one);
          term->insertOperands(term->getNumOperands(),
                               ValueRange(newInductionVar));
        }
      }

      SmallVector<int64_t> newShape;
      SmallVector<Value> dynamicDims;
      for (const auto &dim : fwdNumIters) {
        if (dim.vval) {
          newShape.push_back(mlir::ShapedType::kDynamic);
          dynamicDims.push_back(dim.vval);
        } else {
          newShape.push_back(dim.ival);
        }
      }

      auto ET = info.cachedType();
      ShapedType NT;

      bool multiDim = false;
      if (auto ST = dyn_cast<ShapedType>(ET)) {
        auto allocOp = pushedValue.getDefiningOp<memref::AllocOp>();
        if (cacheType == LoopCacheType::MEMREF && allocOp &&
            allocOp.getSymbolOperands().empty() &&
            llvm::all_of(allocOp.getDynamicSizes(), [&](Value dynSize) {
              return !forOp.getRegion().isAncestor(dynSize.getParentRegion());
            })) {
          multiDim = true;

          dynamicDims.append(allocOp.getDynamicSizes().begin(),
                             allocOp.getDynamicSizes().end());

        } else if (llvm::all_of(ST.getShape(), [](int64_t dim) {
                     return dim != ShapedType::kDynamic;
                   })) {
          multiDim = true;
        }

        if (multiDim) {
          newShape.append(ST.getShape().begin(), ST.getShape().end());
          ET = ST.getElementType();
        }
      }

      auto newType = cacheType == LoopCacheType::TENSOR
                         ? cast<ShapedType>(RankedTensorType::get(newShape, ET))
                         : cast<ShapedType>(MemRefType::get(newShape, ET));

      if (cacheType == LoopCacheType::TENSOR) {
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(forOp);
          Value initValue = tensor::EmptyOp::create(
              rewriter, info.initOp->getLoc(), newType, dynamicDims);

          newOperands.push_back(initValue);
        }

        auto cacheValue = body->addArgument(newType, info.pushOp->getLoc());

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.pushOp);

          Value newCacheValue;
          if (auto TT = dyn_cast<TensorType>(info.cachedType())) {
            auto shape = TT.getShape();

            SmallVector<int64_t> offsets(shape.size() + 1, 0);
            offsets[0] = ShapedType::kDynamic;

            SmallVector<int64_t> sizes;
            sizes.reserve(shape.size() + 1);
            sizes.push_back(1);
            sizes.append(shape.begin(), shape.end());

            SmallVector<int64_t> strides(shape.size() + 1, 1);

            newCacheValue = tensor::InsertSliceOp::create(
                rewriter, info.pushOp->getLoc(), info.pushOp.getValue(),
                cacheValue, inductionVariable, ValueRange(), ValueRange(),
                rewriter.getDenseI64ArrayAttr(offsets),
                rewriter.getDenseI64ArrayAttr(sizes),
                rewriter.getDenseI64ArrayAttr(strides));
          } else {
            newCacheValue = tensor::InsertOp::create(
                rewriter, info.pushOp->getLoc(), info.pushOp.getValue(),
                cacheValue, inductionVariable);
          }

          term->insertOperands(term->getNumOperands(),
                               ValueRange(newCacheValue));

          numNewValuePushes++;
        }
      } else if (cacheType == LoopCacheType::MEMREF) {
        Value initValue;
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(forOp);
          initValue =
              memref::AllocOp::create(rewriter, info.initOp->getLoc(),
                                      cast<MemRefType>(newType), dynamicDims);
          newPushValues.push_back(initValue);
        }

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.pushOp);

          auto MT = dyn_cast<MemRefType>(info.cachedType());
          if (multiDim && MT) {
            auto memref = info.pushOp.getValue();
            auto shape = MT.getShape();

            SmallVector<int64_t> offsets(newShape.size(), 0);
            SmallVector<int64_t> sizes;
            for (auto [i, _] : llvm::enumerate(inductionVariable)) {
              offsets[i] = ShapedType::kDynamic;
              sizes.push_back(1);
            }

            SmallVector<Value> dynSizes;
            for (size_t i = inductionVariable.size(); i < dynamicDims.size();
                 ++i) {
              dynSizes.push_back(dynamicDims[i]);
            }

            sizes.append(shape.begin(), shape.end());

            SmallVector<int64_t> strides(newShape.size(), 1);

            auto RT = memref::SubViewOp::inferRankReducedResultType(
                MT.getShape(), cast<MemRefType>(initValue.getType()), offsets,
                sizes, strides);

            rewriter.setInsertionPoint(memref.getDefiningOp());
            rewriter.replaceOpWithNewOp<memref::SubViewOp>(
                memref.getDefiningOp(), RT, initValue,
                /*offsets*/ inductionVariable,
                /*sizes*/ dynSizes,
                /*strides*/ ValueRange(),
                /*static_offsets*/ rewriter.getDenseI64ArrayAttr(offsets),
                /*static_sizes*/ rewriter.getDenseI64ArrayAttr(sizes),
                /*static_strides*/ rewriter.getDenseI64ArrayAttr(strides));

          } else {
            memref::StoreOp::create(rewriter, info.pushOp->getLoc(),
                                    info.pushOp.getValue(), initValue,
                                    inductionVariable);
          }
        }
      }
    }

    auto numInitArgs = FinalClass::getInits(forOp).size();
    rewriter.setInsertionPoint(forOp);

    forOp = FinalClass::replaceWithNewOperands(rewriter, forOp, newOperands);
    if (cacheType == LoopCacheType::TENSOR) {
      for (size_t i = 0; i < numNewValuePushes; ++i)
        newPushValues.push_back(
            forOp->getResult(forOp->getNumResults() - numNewValuePushes + i));
    }

    rewriter.setInsertionPointAfter(forOp);

    unsigned resultIdx = numInitArgs;
    for (auto grad : updatedGradients) {
      // set the updated gradient after the new for op.

      Value incoming = forOp->getResult(resultIdx);
      Value outgoing;
      if (!postAdd) {
        outgoing = incoming;
      } else {
        auto T = cast<AutoDiffTypeInterface>(incoming.getType());
        Value current = enzyme::GetOp::create(rewriter, grad.getLoc(), T, grad);
        outgoing = T.createAddOp(rewriter, grad.getLoc(), incoming, current);
      }
      enzyme::SetOp::create(rewriter, grad.getLoc(), grad, outgoing);
      ++resultIdx;
    }

    int pushedValueIdx = 0;

    if (caches.size()) {
      if (otherInductionVariable.size()) {
        rewriter.restoreInsertionPoint(revIP);
      } else
        rewriter.setInsertionPointToStart(otherForOp.getBody());
    }
    for (auto &info : caches) {
      if (mincut)
        assert(
            forOp.getRegion().isAncestor(info.pushedValue().getParentRegion()));

      Value cache = info.initOp.getResult();

      // The reverse iteration count may not be known at this point, as it may
      // be cached via a push/pop, use the fwd count in that case.
      if (!revNumIters.size()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(otherForOp);
        revNumIters = FinalClass::getDimensionBounds(rewriter, otherForOp);
        for (auto &&[rev, fwd] : llvm::zip_equal(revNumIters, fwdNumIters)) {
          if (!fwd.vval && rev.vval) {
            rev.vval = nullptr;
            rev.ival = fwd.ival;
          }
        }
      }

      // First, try to get canonical vars from looking up directly
      if (otherInductionVariable.size() && !reversedIndex.size()) {
        reversedIndex = FinalClass::computeReversedIndices(
            rewriter, otherForOp, otherInductionVariable, revNumIters);
      }

      // Otherwise, add a new variable to keep track.
      if (!otherInductionVariable.size()) {
        Value zero = arith::ConstantOp::create(rewriter, otherForOp->getLoc(),
                                               rewriter.getIndexAttr(0));
        SmallVector<Value> newOperands =
            llvm::to_vector(FinalClass::getInits(otherForOp));
        newOperands.push_back(zero);

        otherInductionVariable = {
            body->addArgument(zero.getType(), otherForOp->getLoc())};
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(term);

          auto one = arith::ConstantOp::create(rewriter, forOp->getLoc(),
                                               rewriter.getIndexAttr(1));
          auto newInductionVar = arith::AddIOp::create(
              rewriter, forOp->getLoc(), otherInductionVariable[0], one);
          term->insertOperands(term->getNumOperands(),
                               ValueRange(newInductionVar));
        }

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(otherForOp);
          otherForOp = FinalClass::replaceWithNewOperands(rewriter, otherForOp,
                                                          newOperands);
        }

        reversedIndex = FinalClass::computeReversedIndices(
            rewriter, otherForOp, otherInductionVariable, revNumIters);
      }

      SmallVector<int64_t> newShape;
      for (const auto &dim : revNumIters) {
        if (dim.vval) {
          newShape.push_back(mlir::ShapedType::kDynamic);
        } else {
          newShape.push_back(dim.ival);
        }
      }

      auto ET = info.cachedType();
      ShapedType NT;

      bool multiDim = false;
      if (auto ST = dyn_cast<ShapedType>(ET)) {
        auto svOp = info.pushedValue().getDefiningOp<memref::SubViewOp>();
        if (cacheType == LoopCacheType::MEMREF && svOp) {
          multiDim = true;
        } else if (llvm::all_of(ST.getShape(), [](int64_t dim) {
                     return dim != ShapedType::kDynamic;
                   })) {
          multiDim = true;
        }

        if (multiDim) {
          newShape.append(ST.getShape().begin(), ST.getShape().end());
          ET = ST.getElementType();
        }
      }

      auto newType = cacheType == LoopCacheType::TENSOR
                         ? cast<ShapedType>(RankedTensorType::get(newShape, ET))
                         : cast<ShapedType>(MemRefType::get(newShape, ET));
      enzyme::InitOp newInit = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.initOp);

        enzyme::InitOp::create(
            rewriter, info.initOp->getLoc(),
            enzyme::CacheType::get(cache.getContext(), newType));
      });
      info.pushOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(forOp);
        auto newPush = enzyme::PushOp::create(rewriter, cache.getLoc(),
                                              newInit.getResult(),
                                              newPushValues[pushedValueIdx]);
        rewriter.eraseOp(info.pushOp);
        newPush;
      });

      pushedValueIdx++;

      OpBuilder::InsertionGuard guard(rewriter);

      rewriter.setInsertionPoint(otherForOp);

      auto popNewValue = enzyme::PopOp::create(rewriter, info.popOp->getLoc(),
                                               newType, newInit.getResult());

      rewriter.setInsertionPoint(info.popOp);

      Value popValue;
      if (cacheType == LoopCacheType::TENSOR) {
        if (auto TT = dyn_cast<TensorType>(info.cachedType())) {
          auto shape = TT.getShape();
          SmallVector<int64_t> offsets(shape.size() + 1, 0);
          offsets[0] = ShapedType::kDynamic;

          SmallVector<int64_t> sizes;
          sizes.reserve(shape.size() + 1);
          sizes.push_back(1);
          sizes.append(shape.begin(), shape.end());

          SmallVector<int64_t> strides(shape.size() + 1, 1);

          popValue = tensor::ExtractSliceOp::create(
                         rewriter, info.popOp->getLoc(), TT, popNewValue,
                         reversedIndex, ValueRange(), ValueRange(),
                         rewriter.getDenseI64ArrayAttr(offsets),
                         rewriter.getDenseI64ArrayAttr(sizes),
                         rewriter.getDenseI64ArrayAttr(strides))
                         .getResult();
        } else {
          popValue = tensor::ExtractOp::create(rewriter, info.popOp->getLoc(),
                                               popNewValue, reversedIndex)
                         .getResult();
        }
      } else if (cacheType == LoopCacheType::MEMREF) {

        auto MT = dyn_cast<MemRefType>(info.cachedType());
        if (multiDim && MT) {
          auto shape = MT.getShape();

          SmallVector<int64_t> offsets(newShape.size(), 0);
          SmallVector<int64_t> sizes;
          for (auto [i, _] : llvm::enumerate(reversedIndex)) {
            offsets[i] = ShapedType::kDynamic;
            sizes.push_back(1);
          }

          sizes.append(shape.begin(), shape.end());

          SmallVector<Value> dynSizes;
          for (size_t i = reversedIndex.size(); i < newShape.size(); ++i) {
            // we use memref.dim here to know the size, hopefully further
            // optimization/canonicalizations can just forward the right size
            // here.
            if (newShape[i] == ShapedType::kDynamic)
              dynSizes.push_back(memref::DimOp::create(
                  rewriter, popNewValue.getLoc(), popNewValue,
                  arith::ConstantIndexOp::create(rewriter, popNewValue.getLoc(),
                                                 i)));
          }

          SmallVector<int64_t> strides(shape.size() + 1, 1);

          auto RT = memref::SubViewOp::inferRankReducedResultType(
              MT.getShape(), cast<MemRefType>(popNewValue.getType()), offsets,
              sizes, strides);

          popValue = memref::SubViewOp::create(
              rewriter, info.popOp->getLoc(), RT, popNewValue,
              /*offsets*/ reversedIndex,
              /*sizes*/ dynSizes,
              /*strides*/ ValueRange(),
              /*static_offsets*/ rewriter.getDenseI64ArrayAttr(offsets),
              /*static_sizes*/ rewriter.getDenseI64ArrayAttr(sizes),
              /*static_strides*/ rewriter.getDenseI64ArrayAttr(strides));

          for (auto user :
               llvm::make_early_inc_range(info.popOp.getResult().getUsers())) {
            if (isa<memref::DeallocOp>(user))
              rewriter.eraseOp(user);
          }
        } else {
          popValue = memref::LoadOp::create(rewriter, info.popOp->getLoc(),
                                            popNewValue, reversedIndex);
        }

        // this memref was allocated on push, dealloc it
        rewriter.setInsertionPointAfter(otherForOp);
        memref::DeallocOp::create(rewriter, info.initOp->getLoc(), popNewValue);
      }

      rewriter.replaceAllUsesWith(info.popOp.getResult(), popValue);
      rewriter.eraseOp(info.popOp);
    }

    return success();
  }
};

// All values defined in fwd should have no use outside this block
// therefore we can localize their differential to only the rev block in order
// to simplify the work of the remove-unnecessary-enzyme-ops pass.
//
// The builder insertion point should be at the start of the corresponding rev
// block.
void localizeGradients(OpBuilder &builder, MGradientUtilsReverse *gutils,
                       Block *fwd);

void removalBlockExplore(Block *block, IRMapping &mapping,
                         PatternRewriter &rewriter,
                         llvm::SetVector<Value> &gradients,
                         llvm::MapVector<Value, CacheInfo> &caches);

template <typename FinalClass, typename OpName>
struct IfLikeEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<FinalClass, OpName> {
  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto ifOp = cast<OpName>(op);
    // Gradients:
    //
    //  For each set in a branch, we instead set after the if by using the
    //  return value.
    //
    //  if %pred {
    //    enzyme.set %grad, %2
    //  } else {
    //  }
    //
    //  %0 = enzyme.get %grad
    //  %1 = if %pred {
    //    return %2
    //  } else {
    //    return %0
    //  }
    //  enzyme.set %grad, %1
    //
    //  For each get in a branch, we get before and use that instead of the
    //  get.

    // Caches:
    //
    // For each push, push after the if instead add a dummy value in the other
    // branch.
    //
    // For each pop in the reverse if, pop before the if instead of inside a
    // branch.

    Block *trueBlock = FinalClass::getThenBlock(ifOp, rewriter),
          *falseBlock = FinalClass::getElseBlock(ifOp, rewriter);

    // Gradients whose value is set in either branches.
    llvm::SetVector<Value> gradients;

    // We assume pushes are exclusive.
    llvm::MapVector<Value, CacheInfo> pushedCaches;

    // Grad to value
    IRMapping trueMapping, falseMapping;

    removalBlockExplore(trueBlock, trueMapping, rewriter, gradients,
                        pushedCaches);
    removalBlockExplore(falseBlock, falseMapping, rewriter, gradients,
                        pushedCaches);

    if (gradients.empty() && pushedCaches.empty())
      return success();
    bool removeCaches = !op->hasAttr(kPreserveCacheAttrName);

    Operation *trueTerm = trueBlock->getTerminator();
    Operation *falseTerm = falseBlock->getTerminator();

    for (auto grad : gradients) {
      auto trueValue = trueMapping.lookupOrNull(grad);
      if (!trueValue) {
        trueValue = enzyme::GetOp::create(
            rewriter, grad.getLoc(),
            cast<enzyme::GradientType>(grad.getType()).getBasetype(), grad);
      }
      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));

      auto falseValue = falseMapping.lookupOrNull(grad);
      if (!falseValue) {
        falseValue = enzyme::GetOp::create(
            rewriter, grad.getLoc(),
            cast<enzyme::GradientType>(grad.getType()).getBasetype(), grad);
      }
      falseTerm->insertOperands(falseTerm->getNumOperands(),
                                ValueRange(falseValue));
    }

    if (removeCaches) {
      for (auto &[pushedValue, info] : pushedCaches) {
        Value dummy = FinalClass::getDummyValue(rewriter, pushedValue.getLoc(),
                                                pushedValue.getType());

        Value trueValue =
            pushedValue.getParentBlock() == trueBlock ? pushedValue : dummy;
        Value falseValue =
            pushedValue.getParentBlock() == falseBlock ? pushedValue : dummy;

        trueTerm->insertOperands(trueTerm->getNumOperands(),
                                 ValueRange(trueValue));
        falseTerm->insertOperands(falseTerm->getNumOperands(),
                                  ValueRange(falseValue));
      }
    }

    size_t idx = ifOp->getNumResults();
    ifOp = FinalClass::replace(rewriter, ifOp, trueTerm->getOperandTypes());

    for (auto grad : gradients) {
      enzyme::SetOp::create(rewriter, grad.getLoc(), grad,
                            ifOp->getResult(idx));
      idx++;
    }

    if (removeCaches) {
      for (auto &[pushedValue, info] : pushedCaches) {
        enzyme::PushOp::create(rewriter, info.pushOp->getLoc(),
                               info.initOp.getResult(), ifOp->getResult(idx));
        rewriter.eraseOp(info.pushOp);

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.popOp->getParentOp());

        auto newPop = enzyme::PopOp::create(rewriter, info.popOp->getLoc(),
                                            info.popOp.getResult().getType(),
                                            info.popOp.getCache());
        rewriter.replaceAllUsesWith(info.popOp.getResult(), newPop);
        rewriter.eraseOp(info.popOp);

        idx++;
      }
    }

    return success();
  }
};

} // namespace enzyme
} // namespace mlir
