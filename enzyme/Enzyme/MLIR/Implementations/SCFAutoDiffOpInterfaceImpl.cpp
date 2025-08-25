//===- SCFAutoDiffOpInterfaceImpl.cpp - Interface external model ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/EnzymeLogic.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Passes/RemovalUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/SCFDerivatives.inc"

// TODO: support non constant number of iteration by using unknown dimensions
static std::optional<int64_t> getConstantNumberOfIterations(scf::ForOp forOp) {
  auto lb = forOp.getLowerBound();
  auto ub = forOp.getUpperBound();
  auto step = forOp.getStep();

  IntegerAttr lbAttr, ubAttr, stepAttr;
  if (!matchPattern(lb, m_Constant(&lbAttr)))
    return std::nullopt;
  if (!matchPattern(ub, m_Constant(&ubAttr)))
    return std::nullopt;
  if (!matchPattern(step, m_Constant(&stepAttr)))
    return std::nullopt;

  int64_t lbI = lbAttr.getInt(), ubI = ubAttr.getInt(),
          stepI = stepAttr.getInt();

  return (ubI - lbI) / stepI;
}

static Value getNumberOfIterations(OpBuilder &builder, scf::ForOp forOp) {
  Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound(),
        step = forOp.getStep();
  Value diff = builder.create<arith::SubIOp>(forOp->getLoc(), ub, lb);
  Value nSteps = builder.create<arith::DivUIOp>(forOp->getLoc(), diff, step);
  return nSteps;
}

struct ForOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<ForOpEnzymeOpsRemover,
                                                        scf::ForOp> {
private:
  enum CacheType { TENSOR, MEMREF };

public:
  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto forOp = cast<scf::ForOp>(op);
    scf::ForOp otherForOp = nullptr; // where caches pops are

    // There is support for two push/pop removal modes, one is using immutable
    // tensors, the other uses memrefs. memref is the default, but tensor can be
    // enabled with enzyme.cache_use_tensor
    enum CacheType cacheType = MEMREF;
    if (op->hasAttr("enzyme.cache_use_tensor")) {
      cacheType = TENSOR;
    }

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

        otherForOp = cast<scf::ForOp>(info.popOp->getParentOp());
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

    SmallVector<CacheInfo> caches =
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

    SmallVector<Value> newOperands(forOp.getInitArgs());
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

    auto numIters = getConstantNumberOfIterations(forOp);
    Value inductionVariable; // [0,..., N - 1] counter

    if (matchPattern(forOp.getLowerBound(), m_Zero()) &&
        matchPattern(forOp.getStep(), m_One())) {
      inductionVariable = body->getArgument(0);
    }

    if (forOp->hasAttr("enzyme.enable_mincut")) {
      mlir::enzyme::minCutCache(forOp.getBody(), otherForOp.getBody(), caches,
                                rewriter);
    }

    SmallVector<Value> newPushValues;

    unsigned numNewValuePushes = 0;

    for (auto &info : caches) {
      Value cache = info.initOp.getResult();

      // push does not depend on a value inside the loop, we can hoist the
      // push/pop before the for loops.
      if (info.pushedValue().getParentRegion() != forOp.getRegion()) {
        auto newPush = rewriter.create<enzyme::PushOp>(cache.getLoc(), cache,
                                                       info.pushedValue());
        rewriter.eraseOp(info.pushOp);
        info.pushOp = newPush;

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.popOp->getParentOp());

          auto popVal = info.popOp.getResult();
          auto newPop = rewriter.create<enzyme::PopOp>(cache.getLoc(),
                                                       popVal.getType(), cache);
          rewriter.replaceAllUsesWith(popVal, newPop.getResult());
          rewriter.eraseOp(info.popOp);
          info.popOp = newPop;
        }

        continue;
      }

      if (!inductionVariable) {
        Value zero = rewriter.create<arith::ConstantOp>(
            forOp->getLoc(), rewriter.getIndexAttr(0));
        newOperands.push_back(zero);

        inductionVariable = body->addArgument(zero.getType(), forOp->getLoc());
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(term);

          auto one = rewriter.create<arith::ConstantOp>(
              forOp->getLoc(), rewriter.getIndexAttr(1));
          auto newInductionVar = rewriter.create<arith::AddIOp>(
              forOp->getLoc(), inductionVariable, one);
          term->insertOperands(term->getNumOperands(),
                               ValueRange(newInductionVar));
        }
      }

      SmallVector<int64_t> newShape;
      newShape.push_back(numIters.value_or(mlir::ShapedType::kDynamic));

      auto ET = info.cachedType();
      ShapedType NT;

      if (auto ST = dyn_cast<ShapedType>(ET)) {
        newShape.append(ST.getShape().begin(), ST.getShape().end());
        ET = ST.getElementType();
      }

      auto newType = cacheType == TENSOR
                         ? cast<ShapedType>(RankedTensorType::get(newShape, ET))
                         : cast<ShapedType>(MemRefType::get(newShape, ET));

      SmallVector<Value> dynamicDims;

      for (auto it : llvm::enumerate(newType.getShape())) {
        if (ShapedType::isDynamic(it.value())) {
          if (it.index() == 0)
            dynamicDims.push_back(getNumberOfIterations(rewriter, forOp));
          else
            return failure(); // TODO: find dynamic dims within the body.
        }
      }

      if (cacheType == TENSOR) {
        Value initValue = rewriter.create<tensor::EmptyOp>(
            info.initOp->getLoc(), newType, dynamicDims);

        newOperands.push_back(initValue);

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
                ValueRange(inductionVariable), ValueRange(), ValueRange(),
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
      } else if (cacheType == MEMREF) {
        Value initValue = rewriter.create<memref::AllocOp>(
            info.initOp->getLoc(), cast<MemRefType>(newType), dynamicDims);
        newPushValues.push_back(initValue);

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
                /*offsets*/ ValueRange(inductionVariable),
                /*sizes*/ ValueRange(),
                /*strides*/ ValueRange(),
                /*static_offsets*/ rewriter.getDenseI64ArrayAttr(offsets),
                /*static_sizes*/ rewriter.getDenseI64ArrayAttr(sizes),
                /*static_strides*/ rewriter.getDenseI64ArrayAttr(strides));

          } else {
            rewriter.create<memref::StoreOp>(info.pushOp->getLoc(),
                                             info.pushOp.getValue(), initValue,
                                             ValueRange(inductionVariable));
          }
        }
      }
    }

    auto numInitArgs = forOp.getInitArgs().size();
    rewriter.setInsertionPoint(forOp);
    auto newFor = rewriter.create<scf::ForOp>(
        op->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newOperands);

    newFor.getRegion().takeBody(forOp.getRegion());

    for (auto &&[res, newRes] :
         llvm::zip(forOp->getResults(), newFor->getResults())) {
      rewriter.replaceAllUsesWith(res, newRes);
    }

    if (cacheType == TENSOR) {
      for (int i = 0; i < numNewValuePushes; ++i)
        newPushValues.push_back(
            newFor->getResult(newFor->getNumResults() - numNewValuePushes + i));
    }

    rewriter.eraseOp(forOp);
    forOp = newFor;
    rewriter.setInsertionPointAfter(forOp);

    unsigned resultIdx = numInitArgs;
    for (auto grad : updatedGradients) {
      // set the updated gradient after the new for op.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.create<enzyme::SetOp>(grad.getLoc(), grad,
                                     newFor->getResult(resultIdx));
      ++resultIdx;
    }

    if (inductionVariable && !caches.empty()) {
      if (isa<BlockArgument>(inductionVariable) &&
          cast<BlockArgument>(inductionVariable).getArgNumber() != 0)
        resultIdx++;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(otherForOp);
      SmallVector<Value> operands(otherForOp.getInitArgs().begin(),
                                  otherForOp.getInitArgs().end());
      operands.push_back(numIters.has_value()
                             ? rewriter.create<arith::ConstantOp>(
                                   otherForOp->getLoc(),
                                   rewriter.getIndexAttr(numIters.value() - 1))
                             : getNumberOfIterations(rewriter, forOp));

      Block *otherBody = otherForOp.getBody();
      Value otherInductionVariable =
          otherBody->addArgument(rewriter.getIndexType(), otherForOp->getLoc());
      auto otherTerm = otherBody->getTerminator();

      rewriter.setInsertionPoint(otherTerm);

      otherInductionVariable =
          rewriter
              .create<arith::SubIOp>(
                  otherForOp->getLoc(), otherInductionVariable,
                  rewriter
                      .create<arith::ConstantOp>(otherForOp->getLoc(),
                                                 rewriter.getIndexAttr(1))
                      .getResult())
              .getResult();
      otherTerm->insertOperands(otherTerm->getNumOperands(),
                                ValueRange(otherInductionVariable));

      rewriter.setInsertionPoint(otherForOp);
      auto newOtherForOp = rewriter.create<scf::ForOp>(
          otherForOp->getLoc(), otherForOp.getLowerBound(),
          otherForOp.getUpperBound(), otherForOp.getStep(), operands);

      for (auto &&[res, newRes] :
           llvm::zip(otherForOp->getResults(), newOtherForOp->getResults())) {
        rewriter.replaceAllUsesWith(res, newRes);
      }
      newOtherForOp.getRegion().takeBody(otherForOp.getRegion());

      rewriter.eraseOp(otherForOp);
      otherForOp = newOtherForOp;
    }

    int pushedValueIdx = 0;
    for (auto &info : caches) {
      if (info.pushedValue().getParentRegion() != newFor.getRegion())
        continue;

      Value cache = info.initOp.getResult();

      SmallVector<int64_t> newShape;
      newShape.push_back(numIters.value_or(mlir::ShapedType::kDynamic));

      auto ET = info.cachedType();
      ShapedType NT;

      if (auto ST = dyn_cast<ShapedType>(ET)) {
        newShape.append(ST.getShape().begin(), ST.getShape().end());
        ET = ST.getElementType();
      }

      auto newType = cacheType == TENSOR
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
        rewriter.setInsertionPointAfter(newFor);
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

      Block *popBody = otherForOp.getBody();
      rewriter.setInsertionPoint(info.popOp);

      Value newInductionVariable =
          popBody->getArgument(popBody->getNumArguments() - 1);

      Value popValue;
      if (cacheType == TENSOR) {
        if (auto TT = dyn_cast<TensorType>(info.cachedType())) {
          auto shape = TT.getShape();
          SmallVector<int64_t> offsets(shape.size() + 1, 0);
          offsets[0] = ShapedType::kDynamic;

          SmallVector<int64_t> sizes;
          sizes.reserve(shape.size() + 1);
          sizes.push_back(1);
          sizes.append(shape.begin(), shape.end());

          SmallVector<int64_t> strides(shape.size() + 1, 1);

          popValue =
              rewriter
                  .create<tensor::ExtractSliceOp>(
                      info.popOp->getLoc(), TT, popNewValue,
                      ValueRange(newInductionVariable), ValueRange(),
                      ValueRange(), rewriter.getDenseI64ArrayAttr(offsets),
                      rewriter.getDenseI64ArrayAttr(sizes),
                      rewriter.getDenseI64ArrayAttr(strides))
                  .getResult();
        } else {
          popValue =
              rewriter
                  .create<tensor::ExtractOp>(info.popOp->getLoc(), popNewValue,
                                             newInductionVariable)
                  .getResult();
        }
      } else if (cacheType == MEMREF) {

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
              /*offsets*/ ValueRange(newInductionVariable),
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
              info.popOp->getLoc(), popNewValue, newInductionVariable);
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

struct ForOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<ForOpADDataFlow, scf::ForOp> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   OpResult res) const {
    auto forOp = cast<scf::ForOp>(op);
    return {
        forOp->getOperand(res.getResultNumber() + 3),
        forOp.getBody()->getTerminator()->getOperand(res.getResultNumber())};
  }
  SmallVector<Value> getPotentialIncomingValuesArg(Operation *op,
                                                   BlockArgument arg) const {
    auto forOp = cast<scf::ForOp>(op);
    auto idx = arg.getArgNumber() - forOp.getNumInductionVars();
    return {forOp->getOperand(idx + 3),
            forOp.getBody()->getTerminator()->getOperand(idx)};
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value val) const {
    auto forOp = cast<scf::ForOp>(op);
    SmallVector<Value> sv;

    for (auto &&[res, arg, barg] :
         llvm::zip_equal(forOp->getResults(), term->getOperands(),
                         forOp.getBody()->getArguments().drop_front())) {
      if (arg == val) {
        sv.push_back(res);
        sv.push_back(barg);
      }
    }

    return sv;
  }
};

struct ForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ForOpInterfaceReverse,
                                                       scf::ForOp> {
private:
  static Value makeIntConstant(Location loc, OpBuilder builder, int64_t val,
                               Type ty) {
    return builder.create<arith::ConstantOp>(loc, IntegerAttr::get(ty, val))
        .getResult();
  };

  static void preserveAttributesButCheckpointing(Operation *newOp,
                                                 Operation *oldOp) {
    for (auto attr : oldOp->getDiscardableAttrs()) {
      if (attr.getName() != "enzyme.enable_checkpointing")
        newOp->setAttr(attr.getName(), attr.getValue());
    }
  }

  static bool needsCheckpointing(scf::ForOp forOp) {
    return forOp->hasAttrOfType<BoolAttr>("enzyme.enable_checkpointing") &&
           forOp->getAttrOfType<BoolAttr>("enzyme.enable_checkpointing")
               .getValue() &&
           getConstantNumberOfIterations(forOp).has_value();
  }

public:
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    // SCF ForOp has 3 more operands than results (lb, ub, step).
    // Its body has 1 more argument than yielded values (the induction
    // variable).

    auto forOp = cast<scf::ForOp>(op);

    SmallVector<bool> operandsActive(forOp.getNumOperands() - 3, false);
    for (int i = 0, e = operandsActive.size(); i < e; ++i) {
      operandsActive[i] = !gutils->isConstantValue(op->getOperand(i + 3)) ||
                          !gutils->isConstantValue(op->getResult(i));
    }

    SmallVector<Value> incomingGradients;
    for (auto &&[active, res] :
         llvm::zip_equal(operandsActive, op->getResults())) {
      if (active) {
        incomingGradients.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    if (needsCheckpointing(forOp)) {
      int64_t numIters = getConstantNumberOfIterations(forOp).value();
      int64_t nInner = std::sqrt(numIters), nOuter = nInner;
      int64_t trailingIters = numIters - nInner * nOuter;

      bool hasTrailing = trailingIters > 0;

      auto numIterArgs = forOp.getNumRegionIterArgs();

      SetVector<Value> outsideRefs;
      getUsedValuesDefinedAbove(op->getRegions(), outsideRefs);

      IRMapping &mapping = gutils->originalToNewFn;

      assert(outsideRefs.size() == caches.size() - numIterArgs);

      SmallVector<Value> cachedOutsideRefs;
      for (auto [i, ref] : llvm::enumerate(outsideRefs)) {
        Value refVal = gutils->popCache(caches[numIterArgs + i], builder);
        cachedOutsideRefs.push_back(refVal);
        mapping.map(ref, refVal);
      }

      auto ivTy = forOp.getLowerBound().getType();
      Value outerUB = makeIntConstant(forOp.getLowerBound().getLoc(), builder,
                                      nOuter + hasTrailing, ivTy);
      auto revOuter = builder.create<scf::ForOp>(
          op->getLoc(),
          makeIntConstant(forOp.getLowerBound().getLoc(), builder, 0, ivTy),
          outerUB,
          makeIntConstant(forOp.getLowerBound().getLoc(), builder, 1, ivTy),
          incomingGradients);
      preserveAttributesButCheckpointing(revOuter, forOp);

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(revOuter.getBody());

      Location loc = forOp.getInductionVar().getLoc();
      Value currentOuterStep = builder.create<arith::SubIOp>(
          loc, makeIntConstant(loc, builder, nOuter, ivTy),
          revOuter.getInductionVar());

      SmallVector<Value> initArgs(numIterArgs, nullptr);
      for (int i = 0; i < numIterArgs; ++i) {
        initArgs[i] = gutils->popCache(caches[i], builder);
      }

      auto nInnerCst = makeIntConstant(forOp.getLowerBound().getLoc(), builder,
                                       nInner, ivTy);
      Value zero = makeIntConstant(forOp.getLowerBound().getLoc(), builder, 0,
                                   ivTy),
            one = makeIntConstant(forOp.getLowerBound().getLoc(), builder, 1,
                                  ivTy);

      Value nInnerUB = nInnerCst;
      if (trailingIters > 0) {
        // this is the first reverse iteration
        Location loc = forOp.getUpperBound().getLoc();
        nInnerUB = builder.create<arith::SelectOp>(
            loc,
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          revOuter.getInductionVar(), zero),
            makeIntConstant(loc, builder, trailingIters, ivTy), nInnerCst);
      }

      auto revInner = builder.create<scf::ForOp>(forOp.getLoc(), zero, nInnerUB,
                                                 one, initArgs);
      preserveAttributesButCheckpointing(revInner, forOp);

      revInner->setAttrs(op->getAttrs());
      revInner->removeAttr("enzyme.enable_checkpointing");

      llvm::APInt stepI;
      if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepI))) {
        op->emitError() << "step size is not known constant\n";
        return failure();
      }

      llvm::APInt startI;
      if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&startI))) {
        op->emitError() << "lower bound is not known constant\n";
        return failure();
      }

      builder.setInsertionPointToEnd(revInner.getBody());

      Value currentIV = builder.create<arith::AddIOp>(
          loc,
          builder.create<arith::MulIOp>(
              loc,
              builder.create<arith::AddIOp>(
                  loc,
                  builder.create<arith::MulIOp>(loc, currentOuterStep,
                                                nInnerCst),
                  revInner.getInductionVar()),
              builder.create<arith::ConstantOp>(loc,
                                                IntegerAttr::get(ivTy, stepI))),
          builder.create<arith::ConstantOp>(loc,
                                            IntegerAttr::get(ivTy, startI)));

      for (auto [oldArg, newArg] :
           llvm::zip_equal(forOp.getBody()->getArguments(),
                           revInner.getBody()->getArguments()))
        mapping.map(oldArg, newArg);
      mapping.map(forOp.getInductionVar(), currentIV);

      for (auto &it : *forOp.getBody()) {
        auto newOp = builder.clone(it, mapping);
        gutils->originalToNewFnOps[&it] = newOp;
      }

      builder.setInsertionPointToEnd(revOuter.getBody());

      auto revLoop = builder.create<scf::ForOp>(
          forOp.getLoc(), zero, nInnerUB, one,
          revOuter.getBody()->getArguments().drop_front());
      preserveAttributesButCheckpointing(revLoop, forOp);

      Block *revLoopBody = revLoop.getBody();
      builder.setInsertionPointToEnd(revLoopBody);

      int revIdx = 1;
      for (auto &&[active, operand] :
           llvm::zip_equal(operandsActive,
                           forOp.getBody()->getTerminator()->getOperands())) {
        if (active) {
          gutils->addToDiffe(operand, revLoopBody->getArgument(revIdx),
                             builder);
          revIdx++;
        }
      }

      Block *origBody = forOp.getBody();

      bool valid = true;

      auto first = origBody->rbegin();
      first++; // skip terminator

      auto last = origBody->rend();

      for (auto it = first; it != last; ++it) {
        Operation *op = &*it;
        valid &= gutils->Logic.visitChild(op, builder, gutils).succeeded();
      }

      SmallVector<Value> newResults;
      for (auto &&[active, arg] : llvm::zip_equal(
               operandsActive, origBody->getArguments().drop_front())) {
        if (active) {
          newResults.push_back(gutils->diffe(arg, builder));
          if (!gutils->isConstantValue(arg))
            gutils->zeroDiffe(arg, builder);
        }
      }

      builder.setInsertionPointToEnd(revLoopBody);
      builder.create<scf::YieldOp>(forOp.getBody()->getTerminator()->getLoc(),
                                   newResults);

      builder.setInsertionPointToEnd(revOuter.getBody());
      builder.create<scf::YieldOp>(forOp.getBody()->getTerminator()->getLoc(),
                                   revLoop.getResults());

      builder.setInsertionPointAfter(revOuter);

      revIdx = 0;
      for (auto &&[active, arg] : llvm::zip_equal(
               operandsActive,
               op->getOperands().slice(3, op->getNumOperands() - 3))) {
        if (active) {
          if (!gutils->isConstantValue(arg)) {
            gutils->addToDiffe(arg, revOuter->getResult(revIdx), builder);
          }
          revIdx++;
        }
      }

      return success(valid);
    }

    auto start = gutils->popCache(caches[0], builder);
    auto end = gutils->popCache(caches[1], builder);
    auto step = gutils->popCache(caches[2], builder);

    auto repFor = builder.create<scf::ForOp>(forOp.getLoc(), start, end, step,
                                             incomingGradients);
    preserveAttributesButCheckpointing(repFor, forOp);

    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), repFor->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());

        // Create implicit terminator if not present (when num results > 0)
        if (revBB.empty()) {
          bodyBuilder.create<scf::YieldOp>(repFor->getLoc());
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
             operandsActive, repFor->getResults(), forOp.getInitArgs())) {
      if (active) {
        if (!gutils->isConstantValue(arg))
          gutils->addToDiffe(arg, res, builder);
      }
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto forOp = cast<scf::ForOp>(op);
    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);

    if (needsCheckpointing(forOp)) {
      int64_t numIters = getConstantNumberOfIterations(forOp).value();
      int64_t nInner = std::sqrt(numIters), nOuter = nInner;
      int64_t trailingIters = numIters - nInner * nOuter;
      bool hasTrailing = trailingIters > 0;

      SetVector<Value> outsideRefs;
      getUsedValuesDefinedAbove(op->getRegions(), outsideRefs);

      SmallVector<Value> caches;

      scf::ForOp newForOp = cast<scf::ForOp>(gutils->getNewFromOriginal(op));

      Type ty = forOp.getLowerBound().getType();
      auto outerFwd = cacheBuilder.create<scf::ForOp>(
          op->getLoc(),
          makeIntConstant(forOp.getLowerBound().getLoc(), cacheBuilder, 0, ty),
          makeIntConstant(forOp.getUpperBound().getLoc(), cacheBuilder,
                          nInner * (nOuter + hasTrailing), ty),
          makeIntConstant(forOp.getStep().getLoc(), cacheBuilder, nInner, ty),
          newForOp.getInitArgs());
      preserveAttributesButCheckpointing(outerFwd, forOp);

      cacheBuilder.setInsertionPointToStart(outerFwd.getBody());
      auto nInnerCst = makeIntConstant(forOp.getUpperBound().getLoc(),
                                       cacheBuilder, nInner, ty);

      Value nInnerUB = nInnerCst;
      if (trailingIters > 0) {
        // if this is the last iteration, then the inner
        // loop will only make trailingIters iterations
        Location loc = forOp.getUpperBound().getLoc();
        nInnerUB = cacheBuilder.create<arith::SelectOp>(
            loc,
            cacheBuilder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, outerFwd.getInductionVar(),
                makeIntConstant(loc, cacheBuilder, nInner * nOuter, ty)),
            makeIntConstant(loc, cacheBuilder, trailingIters, ty), nInnerCst);
      }

      auto innerFwd = cacheBuilder.create<scf::ForOp>(
          op->getLoc(),
          makeIntConstant(forOp.getLowerBound().getLoc(), cacheBuilder, 0, ty),
          nInnerUB,
          makeIntConstant(forOp.getStep().getLoc(), cacheBuilder, 1, ty),
          outerFwd.getBody()->getArguments().drop_front());
      preserveAttributesButCheckpointing(innerFwd, forOp);

      cacheBuilder.setInsertionPointToEnd(innerFwd.getBody());
      IRMapping &mapping = gutils->originalToNewFn;

      Location loc = forOp.getInductionVar().getLoc();
      auto currentIV = cacheBuilder.create<arith::MulIOp>(
          loc,
          cacheBuilder.create<arith::AddIOp>(
              loc,
              cacheBuilder.create<arith::MulIOp>(
                  loc, outerFwd.getInductionVar(), nInnerCst),
              innerFwd.getInductionVar()),
          newForOp.getStep());

      for (auto [oldArg, newArg] :
           llvm::zip_equal(forOp.getBody()->getArguments(),
                           innerFwd.getBody()->getArguments()))
        mapping.map(oldArg, newArg);
      mapping.map(forOp.getInductionVar(), currentIV);

      for (auto &it : *forOp.getBody())
        cacheBuilder.clone(it, mapping);

      cacheBuilder.setInsertionPointToEnd(outerFwd.getBody());
      for (auto initArg : innerFwd.getInitArgs())
        caches.push_back(gutils->initAndPushCache(initArg, cacheBuilder));

      cacheBuilder.create<scf::YieldOp>(
          forOp.getBody()->getTerminator()->getLoc(), innerFwd->getResults());

      cacheBuilder.setInsertionPointAfter(outerFwd);

      for (auto ref : outsideRefs)
        caches.push_back(gutils->initAndPushCache(mapping.lookupOrDefault(ref),
                                                  cacheBuilder));

      gutils->replaceOrigOpWith(op, outerFwd.getResults());
      gutils->erase(newForOp);
      gutils->originalToNewFnOps[op] = outerFwd;

      return caches;
    }

    SmallVector<Value> caches;

    Value cacheLB = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getLowerBound()), cacheBuilder);
    caches.push_back(cacheLB);

    Value cacheUB = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getUpperBound()), cacheBuilder);
    caches.push_back(cacheUB);

    Value cacheStep = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getStep()), cacheBuilder);
    caches.push_back(cacheStep);

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto forOp = cast<scf::ForOp>(op);
  }
};

} // namespace

void mlir::enzyme::registerSCFDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, scf::SCFDialect *) {
    registerInterfaces(context);
    scf::ForOp::attachInterface<ForOpInterfaceReverse>(*context);
    scf::ForOp::attachInterface<ForOpEnzymeOpsRemover>(*context);
    scf::ForOp::attachInterface<ForOpADDataFlow>(*context);
  });
}
