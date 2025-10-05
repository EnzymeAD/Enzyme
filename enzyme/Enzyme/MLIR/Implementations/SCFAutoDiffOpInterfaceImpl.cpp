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


struct ForOpEnzymeOpsRemover : public ForLikeEnzymeOpsRemover<ForOpEnzymeOpsRemover, scf::ForOp> {
public:
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

  static bool isCanonicalLoop(scf::ForOp forOp) {
    return matchPattern(forOp.getLowerBound(), m_Zero()) &&
          matchPattern(forOp.getStep(), m_One());
  }

  static scf::ForOp replaceWithNewOperands(PatternRewriter &rewriter, scf::ForOp otherForOp, ArrayRef<Value> operands) {
    auto newOtherForOp = rewriter.create<scf::ForOp>(
            otherForOp->getLoc(),
            otherForOp.getLowerBound(), 
            otherForOp.getUpperBound(), 
            otherForOp.getStep(), operands);

    newOtherForOp.getRegion().takeBody(otherForOp.getRegion());
    rewriter.replaceOp(otherForOp, newOtherForOp);
    return newOtherForOp;
  }

  static ValueRange getInits(scf::ForOp forOp) {
    return forOp.getInitArgs();
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
           ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).has_value();
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
      int64_t numIters = ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).value();
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
      int64_t numIters = ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).value();
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

struct IfOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<IfOpInterfaceReverse,
                                                       scf::IfOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto ifOp = cast<scf::IfOp>(op);
    bool hasElse = ifOp.elseBlock() != nullptr;
    Value cond = gutils->popCache(caches[0], builder);

    SmallVector<bool> resultsActive(ifOp.getNumResults(), false);
    for (int i = 0, e = resultsActive.size(); i < e; ++i) {
      resultsActive[i] = !gutils->isConstantValue(ifOp.getResult(i));
    }

    SmallVector<Value> incomingGradients;
    for (auto &&[active, res] :
         llvm::zip_equal(resultsActive, ifOp.getResults())) {
      if (active) {
        incomingGradients.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    auto revIf =
        builder.create<scf::IfOp>(ifOp.getLoc(), TypeRange{}, cond, hasElse);
    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), revIf->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());
        bodyBuilder.setInsertionPoint(revBB.getTerminator());

        // All values defined in the body should have no use outside this block
        // therefore we can set their diffe to zero upon entering the reverse
        // block to simplify the work of the remove-unnecessary-enzyme-ops pass.
        for (auto &it : oBB.getOperations()) {
          for (auto res : it.getResults()) {
            if (!gutils->isConstantValue(res)) {
              gutils->zeroDiffe(res, bodyBuilder);
            }
          }
        }

        auto term = oBB.getTerminator();
        // Align incomingGradients with their corresponding yield operands.
        SmallVector<Value> activeTermOperands;
        activeTermOperands.reserve(incomingGradients.size());
        for (auto &&[resultActive, operand] :
             llvm::zip_equal(resultsActive, term->getOperands())) {
          if (resultActive)
            activeTermOperands.push_back(operand);
        }

        for (auto &&[arg, operand] :
             llvm::zip_equal(incomingGradients, activeTermOperands)) {
          // Check activity of the argument separately from the result. If some
          // branches yield inactive values while others yield active values,
          // the result will be active, but this operand may still be inactive
          // (and we cannot addToDiffe)
          if (!gutils->isConstantValue(operand)) {
            gutils->addToDiffe(operand, arg, bodyBuilder);
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
      }
    }
    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto ifOp = cast<scf::IfOp>(op);

    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);
    Value cacheCond = gutils->initAndPushCache(
        gutils->getNewFromOriginal(ifOp.getCondition()), cacheBuilder);
    return SmallVector<Value>{cacheCond};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
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
    if (arg.getArgNumber() < forOp.getNumInductionVars())
      return {};
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
                         forOp.getRegionIterArgs())) {
      if (arg == val) {
        sv.push_back(res);
        sv.push_back(barg);
      }
    }

    return sv;
  }
};

} // namespace

void mlir::enzyme::registerSCFDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, scf::SCFDialect *) {
    registerInterfaces(context);
    scf::IfOp::attachInterface<IfOpInterfaceReverse>(*context);
    scf::ForOp::attachInterface<ForOpInterfaceReverse>(*context);
    scf::ForOp::attachInterface<ForOpEnzymeOpsRemover>(*context);
    scf::ForOp::attachInterface<ForOpADDataFlow>(*context);
  });
}
