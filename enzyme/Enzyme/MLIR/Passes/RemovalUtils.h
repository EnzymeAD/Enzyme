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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/IR/Matchers.h"

namespace mlir {
namespace enzyme {

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
                 PatternRewriter &rewriter, const IRMapping &fwdrevmap, Operation *lastFwd=nullptr);

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
    if (matchPattern(lhs, m_Constant(&la)) && matchPattern(rhs, m_Constant(&ra)))
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
          c1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
        else
          c1 = rewriter.create<arith::ConstantIntOp>(op->getLoc(), iv.getType(),
                                                     1);
        boundv = rewriter.create<arith::SubIOp>(op->getLoc(), bound.vval, c1);
      } else {
        if (iv.getType().isIndex())
          boundv = rewriter.create<arith::ConstantIndexOp>(op->getLoc(),
                                                           bound.ival - 1);
        else
          boundv = rewriter.create<arith::ConstantIntOp>(
              op->getLoc(), iv.getType(), bound.ival - 1);
      }
      Value result = rewriter.create<arith::SubIOp>(op->getLoc(), boundv, iv);
      results.push_back(result);
    }
    return results;
  }

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto forOp = cast<OpName>(op);

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

    for (auto &it : *body) {
      Operation *op = &it;

      if (auto setOp = dyn_cast<enzyme::SetOp>(op))
        updatedGradients.insert(setOp.getGradient());

      if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
        CacheInfo info(pushOp.getCache());

        Value pushedValue = info.pushedValue();
        if (cachesMap.contains(pushedValue)) {
          info = info.merge(cachesMap.lookup(pushedValue), rewriter);
        }

        if (info.pushOp->getBlock() == body && info.popOp->getBlock() == body &&
            info.pushOp->isBeforeInBlock(info.popOp)) {
          toDelete.push_back(info);
          continue;
        }
        cachesMap[pushedValue] = info;

        otherForOp = cast<OpName>(info.popOp->getParentOp());
      }
    }

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

    llvm::map_to_vector(cachesMap, [](auto p) { return std::get<1>(p); });

    // nothing to do
    if (updatedGradients.empty() && caches.empty())
      return success();

    DenseMap<Value, llvm::SmallVector<Operation *>> updatedGradientUsers;

    for (auto &it : *body) {
      Operation *op = &it;

      auto getOp = dyn_cast<enzyme::GetOp>(op);

      if (getOp && updatedGradients.contains(getOp.getGradient())) {
        updatedGradientUsers[getOp.getGradient()].push_back(getOp);
      } else if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
        updatedGradientUsers[setOp.getGradient()].push_back(setOp);
      }

      if (!getOp || updatedGradients.contains(getOp.getGradient()))
        continue;

      auto outerGet = rewriter.create<enzyme::GetOp>(
          getOp->getLoc(),
          cast<enzyme::GradientType>(getOp.getResult().getType()).getBasetype(),
          getOp.getGradient());

      rewriter.replaceAllUsesWith(getOp.getResult(), outerGet.getResult());
      rewriter.eraseOp(getOp);
    }

    auto term = body->getTerminator();

    SmallVector<Value> newOperands(FinalClass::getInits(forOp));
    for (auto grad : updatedGradients) {
      auto Ty = cast<enzyme::GradientType>(grad.getType()).getBasetype();
      auto outerGet = rewriter.create<enzyme::GetOp>(grad.getLoc(), Ty, grad);

      newOperands.push_back(outerGet.getResult());
      auto newArg = body->addArgument(Ty, grad.getLoc());

      {
        OpBuilder::InsertionGuard guard(rewriter);
        // here we do a primitive form of mem2reg within the loop. We have a
        // sorted (by instruction number) list of all users of the instruction.
        Value val = newArg;
        for (auto user : updatedGradientUsers[grad]) {
          if (auto getOp = dyn_cast<enzyme::GetOp>(user)) {
            rewriter.replaceOp(getOp, val);
          } else {
            auto setOp = cast<enzyme::SetOp>(user);
            val = setOp.getValue();
            rewriter.eraseOp(setOp);
          }
        }
        // rewriter.setInsertionPointToStart(body);
        // rewriter.create<enzyme::SetOp>(grad.getLoc(), grad, newArg);

        // rewriter.setInsertionPoint(term);

        // auto outputVal =
        //     rewriter.create<enzyme::GetOp>(grad.getLoc(), Ty,
        //     grad).getResult();
        term->insertOperands(term->getNumOperands(), ValueRange(val));
      }
    }
    
    IRMapping fwdrevmap;
    bool mincut = false;

    // [0,..., N - 1] counter
    SmallVector<Value> inductionVariable;
    SmallVector<Value> otherInductionVariable;

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
      fwdrevmap =
          FinalClass::createArgumentMap(rewriter, forOp, inductionVariable,
                                        otherForOp, otherInductionVariable);
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

    SmallVector<IntOrValue> fwdNumIters;

    if (lastFwd)
      rewriter.setInsertionPointAfter(lastFwd);
    else
      rewriter.setInsertionPointToStart(forOp.getBody());
    for (auto &info : caches) {

      if (mincut)
        assert(info.pushedValue().getParentRegion() == &forOp.getRegion());

      // Otherwise, add a new variable to keep track.
      if (!inductionVariable.size()) {
        Value zero = rewriter.create<arith::ConstantOp>(
            forOp->getLoc(), rewriter.getIndexAttr(0));
        newOperands.push_back(zero);

        inductionVariable = {
            body->addArgument(zero.getType(), forOp->getLoc())};
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(term);

          auto one = rewriter.create<arith::ConstantOp>(
              forOp->getLoc(), rewriter.getIndexAttr(1));
          auto newInductionVar = rewriter.create<arith::AddIOp>(
              forOp->getLoc(), inductionVariable[0], one);
          term->insertOperands(term->getNumOperands(),
                               ValueRange(newInductionVar));
        }
      }

      SmallVector<int64_t> newShape;
      if (!fwdNumIters.size()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(forOp);
        fwdNumIters = FinalClass::getDimensionBounds(rewriter, forOp);
      }
      for (const auto &dim : fwdNumIters) {
        if (dim.vval) {
          newShape.push_back(mlir::ShapedType::kDynamic);
        } else {
          newShape.push_back(dim.ival);
        }
      }

      auto ET = info.cachedType();
      ShapedType NT;

      if (auto ST = dyn_cast<ShapedType>(ET)) {
        newShape.append(ST.getShape().begin(), ST.getShape().end());
        ET = ST.getElementType();
      }

      auto newType = cacheType == LoopCacheType::TENSOR
                         ? cast<ShapedType>(RankedTensorType::get(newShape, ET))
                         : cast<ShapedType>(MemRefType::get(newShape, ET));

      SmallVector<Value> dynamicDims;
      for (const auto &dim : fwdNumIters) {
        if (dim.vval) {
          dynamicDims.push_back(dim.vval);
        }
      }

      for (size_t i = fwdNumIters.size(); i < newShape.size(); i++) {
        if (newShape[i] == mlir::ShapedType::kDynamic) {
          return info.initOp->emitError()
                 << "Cached type uses dynamic index, unsupported presently "
                    "from forhandler";
        }
      }

      if (cacheType == LoopCacheType::TENSOR) {
	      {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(forOp);
        Value initValue = rewriter.create<tensor::EmptyOp>(
            info.initOp->getLoc(), newType, dynamicDims);

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

            newCacheValue = rewriter.create<tensor::InsertSliceOp>(
                info.pushOp->getLoc(), info.pushOp.getValue(), cacheValue,
                inductionVariable, ValueRange(), ValueRange(),
                rewriter.getDenseI64ArrayAttr(offsets),
                rewriter.getDenseI64ArrayAttr(sizes),
                rewriter.getDenseI64ArrayAttr(strides));
          } else {
            newCacheValue = rewriter.create<tensor::InsertOp>(
                info.pushOp->getLoc(), info.pushOp.getValue(), cacheValue,
                inductionVariable);
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
        initValue = rewriter.create<memref::AllocOp>(
            info.initOp->getLoc(), cast<MemRefType>(newType), dynamicDims);
        newPushValues.push_back(initValue);
	      }

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.pushOp);

          if (auto MT = dyn_cast<MemRefType>(info.cachedType())) {
            auto memref = info.pushOp.getValue();
            auto shape = MT.getShape();

            SmallVector<int64_t> offsets(shape.size() + 1, 0);
            offsets[0] = ShapedType::kDynamic;

            SmallVector<int64_t> sizes;
            sizes.push_back(1);
            sizes.append(shape.begin(), shape.end());

            SmallVector<int64_t> strides(shape.size() + 1, 1);

            auto RT = memref::SubViewOp::inferRankReducedResultType(
                MT.getShape(), cast<MemRefType>(initValue.getType()), offsets,
                sizes, strides);

            rewriter.setInsertionPoint(memref.getDefiningOp());
            rewriter.replaceOpWithNewOp<memref::SubViewOp>(
                memref.getDefiningOp(), RT, initValue,
                /*offsets*/ inductionVariable,
                /*sizes*/ ValueRange(),
                /*strides*/ ValueRange(),
                /*static_offsets*/ rewriter.getDenseI64ArrayAttr(offsets),
                /*static_sizes*/ rewriter.getDenseI64ArrayAttr(sizes),
                /*static_strides*/ rewriter.getDenseI64ArrayAttr(strides));

          } else {
            rewriter.create<memref::StoreOp>(info.pushOp->getLoc(),
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
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.create<enzyme::SetOp>(grad.getLoc(), grad,
                                     forOp->getResult(resultIdx));
      ++resultIdx;
    }

    int pushedValueIdx = 0;

    SmallVector<Value> reversedIndex;

    SmallVector<IntOrValue> revNumIters;

    if (caches.size()) {
    if (otherInductionVariable.size()) {
      // We move before any pops that might have been created by mincut, since we need
      // the reversedIdx to come right after other induction var computations.
      while (revIP.getBlock()->begin() != revIP.getPoint() && isa<enzyme::PopOp>(--revIP.getPoint())) {
        revIP = OpBuilder::InsertPoint(revIP.getBlock(), --revIP.getPoint());
      }
      rewriter.restoreInsertionPoint(revIP);
    } else
      rewriter.setInsertionPointToStart(otherForOp.getBody());
    }
    for (auto &info : caches) {
      if (mincut)
        assert(info.pushedValue().getParentRegion() == &forOp.getRegion());

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
        Value zero = rewriter.create<arith::ConstantOp>(
            otherForOp->getLoc(), rewriter.getIndexAttr(0));
        SmallVector<Value> newOperands =
            llvm::to_vector(FinalClass::getInits(otherForOp));
        newOperands.push_back(zero);

        otherInductionVariable = {
            body->addArgument(zero.getType(), otherForOp->getLoc())};
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(term);

          auto one = rewriter.create<arith::ConstantOp>(
              forOp->getLoc(), rewriter.getIndexAttr(1));
          auto newInductionVar = rewriter.create<arith::AddIOp>(
              forOp->getLoc(), otherInductionVariable[0], one);
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

      if (auto ST = dyn_cast<ShapedType>(ET)) {
        newShape.append(ST.getShape().begin(), ST.getShape().end());
        ET = ST.getElementType();
      }

      auto newType = cacheType == LoopCacheType::TENSOR
                         ? cast<ShapedType>(RankedTensorType::get(newShape, ET))
                         : cast<ShapedType>(MemRefType::get(newShape, ET));
      enzyme::InitOp newInit = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.initOp);

        rewriter.create<enzyme::InitOp>(
            info.initOp->getLoc(),
            enzyme::CacheType::get(cache.getContext(), newType));
      });
      info.pushOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(forOp);
        auto newPush = rewriter.create<enzyme::PushOp>(
            cache.getLoc(), newInit.getResult(), newPushValues[pushedValueIdx]);
        rewriter.eraseOp(info.pushOp);
        newPush;
      });

      pushedValueIdx++;

      OpBuilder::InsertionGuard guard(rewriter);

      rewriter.setInsertionPoint(otherForOp);

      auto popNewValue = rewriter.create<enzyme::PopOp>(
          info.popOp->getLoc(), newType, newInit.getResult());

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

          popValue = rewriter
                         .create<tensor::ExtractSliceOp>(
                             info.popOp->getLoc(), TT, popNewValue,
                             reversedIndex, ValueRange(), ValueRange(),
                             rewriter.getDenseI64ArrayAttr(offsets),
                             rewriter.getDenseI64ArrayAttr(sizes),
                             rewriter.getDenseI64ArrayAttr(strides))
                         .getResult();
        } else {
          popValue = rewriter
                         .create<tensor::ExtractOp>(info.popOp->getLoc(),
                                                    popNewValue, reversedIndex)
                         .getResult();
        }
      } else if (cacheType == LoopCacheType::MEMREF) {

        if (auto MT = dyn_cast<MemRefType>(info.cachedType())) {
          auto shape = MT.getShape();

          SmallVector<int64_t> offsets(shape.size() + 1, 0);
          offsets[0] = ShapedType::kDynamic;

          SmallVector<int64_t> sizes;
          sizes.reserve(shape.size() + 1);
          sizes.push_back(1);
          sizes.append(shape.begin(), shape.end());

          SmallVector<int64_t> strides(shape.size() + 1, 1);

          auto RT = memref::SubViewOp::inferRankReducedResultType(
              MT.getShape(), cast<MemRefType>(popNewValue.getType()), offsets,
              sizes, strides);

          popValue = rewriter.create<memref::SubViewOp>(
              info.popOp->getLoc(), RT, popNewValue,
              /*offsets*/ reversedIndex,
              /*sizes*/ ValueRange(),
              /*strides*/ ValueRange(),
              /*static_offsets*/ rewriter.getDenseI64ArrayAttr(offsets),
              /*static_sizes*/ rewriter.getDenseI64ArrayAttr(sizes),
              /*static_strides*/ rewriter.getDenseI64ArrayAttr(strides));

          for (auto user : info.popOp.getResult().getUsers()) {
            if (isa<memref::DeallocOp>(user))
              rewriter.eraseOp(user);
          }
        } else {
          popValue = rewriter.create<memref::LoadOp>(
              info.popOp->getLoc(), popNewValue, reversedIndex);
        }

        // this memref was allocated on push, dealloc it
        rewriter.setInsertionPointAfter(otherForOp);
        rewriter.create<memref::DeallocOp>(info.initOp->getLoc(), popNewValue);
      }

      rewriter.replaceAllUsesWith(info.popOp.getResult(), popValue);
      rewriter.eraseOp(info.popOp);
    }

    return success();
  }
};

} // namespace enzyme
} // namespace mlir
