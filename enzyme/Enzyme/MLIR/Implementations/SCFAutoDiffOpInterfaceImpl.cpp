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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
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

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto forOp = cast<scf::ForOp>(op);
    scf::ForOp otherForOp; // where caches pops are

    // Gradients whose values need to be passed as iteration variables.
    llvm::SetVector<Value> updatedGradients;

    llvm::MapVector<Value, CacheInfo> cachesMap;

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
        cachesMap[pushedValue] = info;

        otherForOp = cast<scf::ForOp>(info.popOp->getParentOp());
      }
    }

    SmallVector<CacheInfo> caches;
    caches.reserve(cachesMap.size());
    for (auto &&[_, info] : cachesMap) {
      caches.push_back(info);
    }

    // nothing to do
    if (updatedGradients.empty() && caches.empty())
      return success();

    for (auto &it : *body) {
      Operation *op = &it;

      auto getOp = dyn_cast<enzyme::GetOp>(op);
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

        rewriter.setInsertionPointToStart(body);
        rewriter.create<enzyme::SetOp>(grad.getLoc(), grad, newArg);

        rewriter.setInsertionPoint(term);

        auto outputVal =
            rewriter.create<enzyme::GetOp>(grad.getLoc(), Ty, grad).getResult();
        term->insertOperands(term->getNumOperands(), ValueRange(outputVal));
      }
    }

    auto numIters = getConstantNumberOfIterations(forOp);
    Value inductionVariable; // [0,..., N - 1] counter

    if (matchPattern(forOp.getLowerBound(), m_Zero()) &&
        matchPattern(forOp.getStep(), m_One())) {
      inductionVariable = body->getArgument(0);
    }

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

      auto newType =
          cast<ShapedType>(cast<AutoDiffTypeInterface>(info.cachedType())
              .getShadowType(numIters.value_or(mlir::ShapedType::kDynamic)));

      SmallVector<Value> dynamicDims;

      for (auto it : llvm::enumerate(newType.getShape())) {
        if (ShapedType::isDynamic(it.value())) {
          if (it.index() == 0)
            dynamicDims.push_back(getNumberOfIterations(rewriter, forOp));
          else
            return failure(); // TODO: find dynamic dims within the body.
        }
      }

      Value initValue = rewriter.create<tensor::EmptyOp>(info.initOp->getLoc(),
                                                         newType, dynamicDims);

      // cast<AutoDiffTypeInterface>(newType).createNullValue(
      // rewriter, info.initOp->getLoc());

      newOperands.push_back(initValue);

      auto cacheValue = body->addArgument(newType, info.pushOp->getLoc());

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.pushOp);

        // TODO: if type is tensor, use insert_slice instead
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

        term->insertOperands(term->getNumOperands(), ValueRange(newCacheValue));
      }
    }

    auto numInitArgs = forOp.getInitArgs().size();
    auto newFor = rewriter.create<scf::ForOp>(
        op->getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newOperands);

    newFor.getRegion().takeBody(forOp.getRegion());

    for (auto &&[res, newRes] :
         llvm::zip(forOp->getResults(), newFor->getResults())) {
      rewriter.replaceAllUsesWith(res, newRes);
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

    for (auto &info : caches) {
      if (info.pushedValue().getParentRegion() != newFor.getRegion())
        continue;

      Value cache = info.initOp.getResult();

      auto newType =
          cast<AutoDiffTypeInterface>(info.cachedType()).getShadowType(
              numIters.value_or(ShapedType::kDynamic));
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
            cache.getLoc(), newInit.getResult(), newFor->getResult(resultIdx));
        rewriter.eraseOp(info.pushOp);
        newPush;
      });

      resultIdx++;

      {
        OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPoint(otherForOp);

        auto popNewValue = rewriter.create<enzyme::PopOp>(
            info.popOp->getLoc(), newType, newInit.getResult());

        Block *popBody = otherForOp.getBody();
        rewriter.setInsertionPoint(info.popOp);

        Value newInductionVariable =
            popBody->getArgument(popBody->getNumArguments() - 1);

        Value popValue;
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

        rewriter.replaceAllUsesWith(info.popOp.getResult(), popValue);
        rewriter.eraseOp(info.popOp);
      }
    }

    return success();
  }
};

struct ForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ForOpInterfaceReverse,
                                                       scf::ForOp> {
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

    auto start = gutils->popCache(caches[0], builder);
    auto end = gutils->popCache(caches[1], builder);
    auto step = gutils->popCache(caches[2], builder);

    SmallVector<Value> incomingGradients;
    for (auto &&[active, res] :
         llvm::zip_equal(operandsActive, op->getResults())) {
      if (active) {
        incomingGradients.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    auto repFor = builder.create<scf::ForOp>(forOp.getLoc(), start, end, step,
                                             incomingGradients);

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
  });
}
