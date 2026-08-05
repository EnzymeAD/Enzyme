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
#include "Implementations/LoopCheckpointing.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/EnzymeLogic.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Passes/RemovalUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include <array>
#include <functional>

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/SCFDerivatives.inc"

struct ForOpEnzymeOpsRemover
    : public ForLikeEnzymeOpsRemover<ForOpEnzymeOpsRemover, scf::ForOp> {
public:
  // TODO: support non constant number of iteration by using unknown dimensions
  static std::optional<int64_t>
  getConstantNumberOfIterations(scf::ForOp forOp) {
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

  static SmallVector<IntOrValue, 1> getDimensionBounds(OpBuilder &builder,
                                                       scf::ForOp forOp) {
    auto iters = getConstantNumberOfIterations(forOp);
    if (iters) {
      return {IntOrValue(*iters)};
    } else {
      Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound(),
            step = forOp.getStep();
      Value diff = arith::SubIOp::create(builder, forOp->getLoc(), ub, lb);
      Value nSteps =
          arith::DivUIOp::create(builder, forOp->getLoc(), diff, step);
      return {IntOrValue(nSteps)};
    }
  }

  static SmallVector<Value> getCanonicalLoopIVs(OpBuilder &builder,
                                                scf::ForOp forOp) {

    Value val = forOp.getBody()->getArgument(0);
    if (!matchPattern(forOp.getLowerBound(), m_Zero())) {
      val = arith::SubIOp::create(builder, forOp->getLoc(), val,
                                  forOp.getLowerBound());
    }

    if (!matchPattern(forOp.getStep(), m_One())) {
      val = arith::DivUIOp::create(builder, forOp->getLoc(), val,
                                   forOp.getStep());
    }
    return {val};
  }

  static IRMapping createArgumentMap(PatternRewriter &rewriter,
                                     scf::ForOp forOp, ArrayRef<Value> indFor,
                                     scf::ForOp otherForOp,
                                     ArrayRef<Value> reversedOther) {
    IRMapping map;
    for (auto &&[f, o] : llvm::zip_equal(indFor, reversedOther)) {
      map.map(f, o);
    }

    Value canIdx = forOp.getBody()->getArgument(0);
    if (!map.contains(canIdx)) {
      assert(Equivalent(forOp.getLowerBound(), otherForOp.getLowerBound()));
      assert(Equivalent(forOp.getStep(), otherForOp.getStep()));

      Location loc = forOp.getLoc();
      // The reverse IV can be computed as (lb + ub - 1 - iv)
      Value revIV =
          arith::AddIOp::create(rewriter, loc, otherForOp.getLowerBound(),
                                otherForOp.getUpperBound());
      Value c1 = arith::ConstantOp::create(
          rewriter, loc, IntegerAttr::get(revIV.getType(), 1));
      revIV = arith::SubIOp::create(rewriter, loc, revIV, c1);
      revIV = arith::SubIOp::create(rewriter, loc, revIV,
                                    otherForOp.getBody()->getArgument(0));
      map.map(forOp.getBody()->getArgument(0), revIV);
    }
    return map;
  }

  static scf::ForOp replaceWithNewOperands(PatternRewriter &rewriter,
                                           scf::ForOp otherForOp,
                                           ArrayRef<Value> operands) {
    auto newOtherForOp = scf::ForOp::create(
        rewriter, otherForOp->getLoc(), otherForOp.getLowerBound(),
        otherForOp.getUpperBound(), otherForOp.getStep(), operands);

    // The rebuilt loop is the same loop with extra iteration arguments, so it
    // keeps everything that was set on it. Without this, anything the caller
    // put there -- enzyme.disable_mincut in particular -- is dropped the moment
    // the removal pass needs to widen the loop.
    newOtherForOp->setDiscardableAttrs(
        otherForOp->getDiscardableAttrDictionary());

    newOtherForOp.getRegion().takeBody(otherForOp.getRegion());
    rewriter.replaceOp(otherForOp, newOtherForOp->getResults().slice(
                                       0, otherForOp->getNumResults()));
    return newOtherForOp;
  }

  static ValueRange getInits(scf::ForOp forOp) { return forOp.getInitArgs(); }

  static bool mustPostAdd(scf::ForOp forOp) { return false; }

  static Value initialValueInBlock(OpBuilder &builder, Block *body,
                                   Value grad) {
    auto Ty = cast<enzyme::GradientType>(grad.getType()).getBasetype();
    return body->addArgument(Ty, grad.getLoc());
  }
};

struct ForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ForOpInterfaceReverse,
                                                       scf::ForOp>,
      public LoopCheckpointing<ForOpInterfaceReverse, scf::ForOp> {
  // ---- hooks required by LoopCheckpointing<ForOpInterfaceReverse, scf::ForOp> ----

  static std::optional<int64_t> getConstantNumberOfIterations(scf::ForOp forOp) {
    return ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp);
  }

  static Value materializeLowerBound(OpBuilder &, Location, scf::ForOp forOp,
                                     MGradientUtilsReverse *gutils) {
    return gutils->getNewFromOriginal(forOp.getLowerBound());
  }

  static Value materializeUpperBound(OpBuilder &, Location, scf::ForOp forOp,
                                     MGradientUtilsReverse *gutils) {
    return gutils->getNewFromOriginal(forOp.getUpperBound());
  }

  static Value materializeStep(OpBuilder &, Location, scf::ForOp forOp,
                               MGradientUtilsReverse *gutils) {
    return gutils->getNewFromOriginal(forOp.getStep());
  }

  static int64_t getConstantStart(scf::ForOp forOp) {
    llvm::APInt v;
    (void)matchPattern(forOp.getLowerBound(), m_ConstantInt(&v));
    return v.getSExtValue();
  }

  static int64_t getConstantStep(scf::ForOp forOp) {
    llvm::APInt v;
    (void)matchPattern(forOp.getStep(), m_ConstantInt(&v));
    return v.getSExtValue();
  }

  static LogicalResult requireSingleResultBounds(scf::ForOp) {
    return success();
  }

  static void cloneOp(OpBuilder &builder, Operation &op, IRMapping &mapping) {
    builder.clone(op, mapping);
  }

  // ---- periodic-scaffold hooks (see LoopCheckpointing.h doc comment) ----
  // All of these reproduce this file's pre-existing periodic-checkpointing
  // formulas verbatim (including the known reverse-formula defect noted in
  // the header) -- this is a pure refactor for scf.for, not a behavior
  // change.

  static scf::ForOp createConstantScaffoldLoop(OpBuilder &builder, Location loc,
                                               int64_t lb, int64_t ub,
                                               int64_t step, ValueRange inits) {
    // Creation order (step, then ub, then lb) matches the original
    // outerFwd construction exactly: canonicalize's constant placement is
    // sensitive to it (constants aren't freely reordered), so this order
    // must be preserved for scf.for's output to stay byte-identical.
    Value stepV = arith::ConstantIndexOp::create(builder, loc, step);
    Value ubV = arith::ConstantIndexOp::create(builder, loc, ub);
    Value lbV = arith::ConstantIndexOp::create(builder, loc, lb);
    return scf::ForOp::create(builder, loc, lbV, ubV, stepV, inits);
  }

  // The forward outer loop steps *by* nInner, so its induction variable is the
  // segment's base iteration directly. With a dynamic trip count the bound is
  // the trip count rounded up to a whole number of segments.
  static scf::ForOp createForwardOuterLoop(OpBuilder &builder, Location loc,
                                           const PeriodicSchedule &sched,
                                           ValueRange inits) {
    if (!sched.isDynamic())
      return createConstantScaffoldLoop(builder, loc, 0,
                                        sched.nInner * sched.numSegments(),
                                        sched.nInner, inits);

    Value stepV = arith::ConstantIndexOp::create(builder, loc, sched.nInner);
    Value ubV = arith::MulIOp::create(builder, loc, sched.nOuterV, stepV);
    Value lbV = arith::ConstantIndexOp::create(builder, loc, 0);
    return scf::ForOp::create(builder, loc, lbV, ubV, stepV, inits);
  }

  // Unchanged shape: a plain sequential reverse-iteration counter.
  static scf::ForOp createReverseOuterLoop(OpBuilder &builder, Location loc,
                                           const PeriodicSchedule &sched,
                                           ValueRange inits) {
    if (!sched.isDynamic())
      return createConstantScaffoldLoop(builder, loc, 0, sched.numSegments(), 1,
                                        inits);

    Value stepV = arith::ConstantIndexOp::create(builder, loc, 1);
    Value lbV = arith::ConstantIndexOp::create(builder, loc, 0);
    return scf::ForOp::create(builder, loc, lbV, sched.nOuterV, stepV, inits);
  }

  static SmallVector<Value>
  computeForwardSegmentHint(OpBuilder &builder, Location loc, Value outerIV,
                            const PeriodicSchedule &sched) {
    Value nInnerCst =
        arith::ConstantIndexOp::create(builder, loc, sched.nInner);
    // The last segment is short whenever the trip count is not a whole
    // multiple of the period, which for a dynamic trip count is not something
    // that can be decided here: clamp unconditionally.
    if (sched.isDynamic())
      return {arith::MinUIOp::create(
          builder, loc, nInnerCst,
          arith::SubIOp::create(builder, loc, sched.numItersV, outerIV))};

    Value nInnerUB = nInnerCst;
    if (sched.hasTrailing()) {
      // if this is the last iteration, then the inner loop will only make
      // trailingIters iterations
      Value trailingCst =
          arith::ConstantIndexOp::create(builder, loc, sched.trailingIters);
      Value lastOuterIter = arith::ConstantIndexOp::create(
          builder, loc, sched.nInner * sched.nOuter);
      Value isLastFwdIter = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, outerIV, lastOuterIter);
      nInnerUB = arith::SelectOp::create(builder, loc, isLastFwdIter,
                                         trailingCst, nInnerCst);
    }
    return {nInnerUB};
  }

  static scf::ForOp createForwardSegmentLoop(OpBuilder &builder, Location loc,
                                             Value, ArrayRef<Value> fwdHint,
                                             const PeriodicSchedule &,
                                             ValueRange inits) {
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return scf::ForOp::create(builder, loc, zero, fwdHint[0], one, inits);
  }

  static scf::ForOp createReverseSegmentLoop(OpBuilder &builder, Location loc,
                                             Value outerIV,
                                             ArrayRef<Value> revHint,
                                             const PeriodicSchedule &sched,
                                             ValueRange inits) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value nInnerCst =
        arith::ConstantIndexOp::create(builder, loc, sched.nInner);
    Value nInnerUB = nInnerCst;
    if (sched.isDynamic()) {
      // Same clamp as the forward direction, against this segment's own base.
      // `precomputedBound` is the segment index counted from the end.
      Value base = arith::MulIOp::create(builder, loc, revHint[0], nInnerCst);
      nInnerUB = arith::MinUIOp::create(
          builder, loc, nInnerCst,
          arith::SubIOp::create(builder, loc, sched.numItersV, base));
    } else if (sched.hasTrailing()) {
      // this is the first reverse iteration
      Value trailingCst =
          arith::ConstantIndexOp::create(builder, loc, sched.trailingIters);
      Value isFirstRevIter = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, outerIV, zero);
      nInnerUB = arith::SelectOp::create(builder, loc, isFirstRevIter,
                                         trailingCst, nInnerCst);
    }
    return scf::ForOp::create(builder, loc, zero, nInnerUB, one, inits);
  }

  static scf::ForOp createLoopWithSameBounds(OpBuilder &builder, Location loc,
                                             scf::ForOp templateLoop,
                                             ValueRange inits) {
    return scf::ForOp::create(builder, loc, templateLoop.getLowerBound(),
                              templateLoop.getUpperBound(),
                              templateLoop.getStep(), inits);
  }

  static Value computeForwardSegmentIV(OpBuilder &builder, Location loc,
                                       scf::ForOp forOp, Value outerIV,
                                       Value localIV,
                                       const PeriodicSchedule &sched,
                                       ArrayRef<Value> /*fwdHint*/) {
    Value flatIV = arith::AddIOp::create(builder, loc, outerIV, localIV);
    if (sched.isDynamic()) {
      flatIV = castToType(builder, loc, flatIV, sched.stepV.getType());
      return arith::AddIOp::create(
          builder, loc, sched.startV,
          arith::MulIOp::create(builder, loc, sched.stepV, flatIV));
    }
    Value stepCst =
        arith::ConstantIndexOp::create(builder, loc, getConstantStep(forOp));
    return arith::MulIOp::create(builder, loc, flatIV, stepCst);
  }

  // The index of the segment being replayed, counted from the end.
  static SmallVector<Value>
  computeReverseSegmentHint(OpBuilder &builder, Location loc, Value outerIV,
                            const PeriodicSchedule &sched) {
    if (sched.isDynamic()) {
      Value last = arith::SubIOp::create(
          builder, loc, sched.nOuterV,
          arith::ConstantIndexOp::create(builder, loc, 1));
      return {arith::SubIOp::create(builder, loc, last, outerIV)};
    }
    // numSegments() - 1, not nOuter: the two coincide only when there *is* a
    // trailing segment. Without one (a period that divides the trip count, or a
    // perfect-square sqrt split) nOuter - j names a segment one past the end,
    // and the replayed segment's induction variable comes out shifted by a
    // whole period -- silently, since the value is dead whenever the loop body
    // does not read its induction variable, which is why no test caught it.
    Value lastSegment =
        arith::ConstantIndexOp::create(builder, loc, sched.numSegments() - 1);
    return {arith::SubIOp::create(builder, loc, lastSegment, outerIV)};
  }

  static Value computeReverseSegmentIV(OpBuilder &builder, Location loc,
                                       scf::ForOp forOp, Value outerIV,
                                       Value localIV,
                                       const PeriodicSchedule &sched,
                                       ArrayRef<Value> revHint) {
    Value currentOuterStep = revHint[0];
    Value nInnerCst =
        arith::ConstantIndexOp::create(builder, loc, sched.nInner);
    Value flatIV = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, currentOuterStep, nInnerCst),
        localIV);
    if (sched.isDynamic()) {
      flatIV = castToType(builder, loc, flatIV, sched.stepV.getType());
      return arith::AddIOp::create(
          builder, loc, sched.startV,
          arith::MulIOp::create(builder, loc, sched.stepV, flatIV));
    }
    Value startCst =
        arith::ConstantIndexOp::create(builder, loc, getConstantStart(forOp));
    Value stepCst =
        arith::ConstantIndexOp::create(builder, loc, getConstantStep(forOp));
    return arith::AddIOp::create(
        builder, loc, arith::MulIOp::create(builder, loc, flatIV, stepCst),
        startCst);
  }

  static void createScaffoldYield(OpBuilder &builder, Location loc,
                                  ValueRange operands) {
    scf::YieldOp::create(builder, loc, operands);
  }

  // preserveAttributesButCheckpointing is inherited from LoopCheckpointing.

public:
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    // SCF ForOp has 3 more operands than results (lb, ub, step).
    // Its body has 1 more argument than yielded values (the induction
    // variable).

    auto forOp = cast<scf::ForOp>(op);
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    SmallVector<bool> operandsActive(forOp.getNumOperands() - 3, false);
    for (int i = 0, e = operandsActive.size(); i < e; ++i) {
      operandsActive[i] = !gutils->isConstantValue(op->getOperand(i + 3)) ||
                          !gutils->isConstantValue(op->getResult(i)) ||
                          !gutils->isConstantValue(yieldOp.getOperand(i));
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

    if (auto r = tryCreateReverseModeAdjoint(forOp, op, builder, gutils,
                                             caches, operandsActive,
                                             incomingGradients))
      return *r;

    auto start = gutils->popCache(caches[0], builder);
    auto end = gutils->popCache(caches[1], builder);
    auto step = gutils->popCache(caches[2], builder);

    auto repFor = scf::ForOp::create(builder, forOp.getLoc(), start, end, step,
                                     incomingGradients);
    preserveAttributesButCheckpointing(repFor, forOp);

    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), repFor->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());

        // Create implicit terminator if not present (when num results > 0)
        if (revBB.empty()) {
          scf::YieldOp::create(bodyBuilder, repFor->getLoc());
        }

        bodyBuilder.setInsertionPointToStart(&revBB);
        mlir::enzyme::localizeGradients(bodyBuilder, gutils, &oBB);

        bodyBuilder.setInsertionPoint(revBB.getTerminator());

        auto term = oBB.getTerminator();

        for (auto &&[active, operand] :
             llvm::zip_equal(operandsActive, term->getOperands())) {
          if (active) {
            // Zero the diffe at the start of each iteration because it should
            // not accumulate across iterations. The new gradient is passed as
            // an iter_arg in the reverse for.
            gutils->zeroDiffe(operand, bodyBuilder);
          }
        }

        unsigned argIdx = 1; // Skip over the reversed IV
        for (auto &&[active, operand] :
             llvm::zip_equal(operandsActive, term->getOperands())) {
          if (active) {
            // If the same value is yielded multiple times in the original, the
            // gradients must be accumulated.
            gutils->addToDiffe(operand, revBB.getArgument(argIdx), bodyBuilder);
            argIdx++;
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

    unsigned resIdx = 0;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, forOp.getInitArgs())) {
      if (active) {
        if (!gutils->isConstantValue(arg)) {
          gutils->addToDiffe(arg, repFor.getResult(resIdx), builder);
          resIdx++;
        }
      }
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto forOp = cast<scf::ForOp>(op);

    if (auto r = tryCacheValues(forOp, op, gutils))
      return *r;

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

struct ParallelOpEnzymeOpsRemover
    : public ForLikeEnzymeOpsRemover<ParallelOpEnzymeOpsRemover,
                                     scf::ParallelOp> {
  static std::optional<int64_t>
  getConstantNumberOfIterations(Value lb, Value ub, Value step) {
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

  static SmallVector<IntOrValue, 1> getDimensionBounds(OpBuilder &builder,
                                                       scf::ParallelOp parOp) {
    SmallVector<IntOrValue, 1> bounds;
    bounds.reserve(parOp.getNumLoops());
    for (auto &&[lb, ub, step] : llvm::zip_equal(
             parOp.getLowerBound(), parOp.getUpperBound(), parOp.getStep())) {
      auto iters = getConstantNumberOfIterations(lb, ub, step);
      if (iters) {
        bounds.push_back(IntOrValue(*iters));
      } else {
        Value diff = arith::SubIOp::create(builder, parOp.getLoc(), ub, lb);
        Value nSteps =
            arith::DivUIOp::create(builder, parOp.getLoc(), diff, step);
        bounds.push_back(IntOrValue(nSteps));
      }
    }
    return bounds;
  }

  static SmallVector<Value>
  computeReversedIndices(PatternRewriter &rewriter, scf::ParallelOp parOp,
                         ArrayRef<Value> otherInductionVariable,
                         ArrayRef<IntOrValue> bounds) {
    return SmallVector<Value>(otherInductionVariable);
  }

  static SmallVector<Value> getCanonicalLoopIVs(OpBuilder &builder,
                                                scf::ParallelOp parOp) {
    SmallVector<Value> canonicalIVs;
    canonicalIVs.reserve(parOp.getNumLoops());
    for (auto &&[iv, lb, step] :
         llvm::zip_equal(parOp.getInductionVars(), parOp.getLowerBound(),
                         parOp.getStep())) {
      Value val = iv;
      if (!matchPattern(lb, m_Zero())) {
        val = arith::SubIOp::create(builder, parOp.getLoc(), val, lb);
      }

      if (!matchPattern(step, m_One())) {
        val = arith::DivUIOp::create(builder, parOp.getLoc(), val, step);
      }
      canonicalIVs.push_back(val);
    }
    return canonicalIVs;
  }

  static IRMapping createArgumentMap(PatternRewriter &rewriter,
                                     scf::ParallelOp parOp,
                                     ArrayRef<Value> indPar,
                                     scf::ParallelOp otherParOp,
                                     ArrayRef<Value> indOther) {
    IRMapping map;
    for (auto &&[f, o] : llvm::zip_equal(indPar, indOther))
      map.map(f, o);

    for (auto &&[iv, oiv, lb, olb, step, ostep] : llvm::zip_equal(
             parOp.getInductionVars(), otherParOp.getInductionVars(),
             parOp.getLowerBound(), otherParOp.getLowerBound(), parOp.getStep(),
             otherParOp.getStep())) {
      if (!map.contains(iv)) {
        assert(Equivalent(lb, olb));
        assert(Equivalent(step, ostep));
        map.map(iv, oiv);
      }
    }
    return map;
  }

  static scf::ParallelOp replaceWithNewOperands(PatternRewriter &rewriter,
                                                scf::ParallelOp otherParallelOp,
                                                ArrayRef<Value> operands) {
    auto newOtherParOp = scf::ParallelOp::create(
        rewriter, otherParallelOp.getLoc(), otherParallelOp.getLowerBound(),
        otherParallelOp.getUpperBound(), otherParallelOp.getStep(), operands);

    newOtherParOp->setDiscardableAttrs(
        otherParallelOp->getDiscardableAttrDictionary());

    newOtherParOp.getRegion().takeBody(otherParallelOp.getRegion());
    rewriter.replaceOp(
        otherParallelOp,
        newOtherParOp.getResults().slice(0, otherParallelOp.getNumResults()));

    if (operands.size() >= 1) {
      OpBuilder::InsertionGuard guard(rewriter);
      Operation *oldTerm = newOtherParOp.getBody()->getTerminator();
      rewriter.setInsertionPointToEnd(newOtherParOp.getBody());
      auto term = scf::ReduceOp::create(rewriter, newOtherParOp.getLoc(),
                                        oldTerm->getOperands());

      for (auto [reg, operand] :
           llvm::zip_equal(term->getRegions(), operands)) {
        Block *b = &reg.front();
        rewriter.setInsertionPointToEnd(b);

        auto Ty = cast<AutoDiffTypeInterface>(operand.getType());
        Value reduced = Ty.createAddOp(rewriter, operand.getLoc(),
                                       b->getArgument(0), b->getArgument(1));
        scf::ReduceReturnOp::create(rewriter, reduced.getLoc(), reduced);
      }

      oldTerm->erase();
    }

    return newOtherParOp;
  }

  static ValueRange getInits(scf::ParallelOp parallelOp) {
    return parallelOp.getInitVals();
  }

  static bool mustPostAdd(scf::ParallelOp forOp) { return false; }

  static Value initialValueInBlock(OpBuilder &builder, Block *body,
                                   Value grad) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    return cast<AutoDiffTypeInterface>(
               cast<enzyme::GradientType>(grad.getType()).getBasetype())
        .createNullValue(builder, grad.getLoc());
  }
};

struct ParallelOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          ParallelOpInterfaceReverse, scf::ParallelOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto parallelOp = cast<scf::ParallelOp>(op);
    if (parallelOp.getNumReductions() != 0) {
      return parallelOp.emitError()
             << "parallel reductions not yet implemented\n";
    }

    unsigned loopCount = parallelOp.getNumLoops();
    SmallVector<Value> bounds = llvm::map_to_vector(
        caches, [&](Value cache) { return gutils->popCache(cache, builder); });

    auto revPar = scf::ParallelOp::create(
        builder, op->getLoc(),
        /*lowerBounds=*/ValueRange(bounds).slice(0, loopCount),
        /*upperBounds=*/ValueRange(bounds).slice(loopCount, loopCount),
        /*steps=*/ValueRange(bounds).slice(loopCount * 2, loopCount));

    bool valid = true;
    bool wasAtomic = gutils->AtomicAdd;
    gutils->AtomicAdd = true;

    {
      Block *oBB = parallelOp.getBody();
      Block *revBB = revPar.getBody();

      OpBuilder bodyBuilder(revBB, revBB->end());

      bodyBuilder.setInsertionPointToStart(revBB);
      mlir::enzyme::localizeGradients(bodyBuilder, gutils, oBB);

      bodyBuilder.setInsertionPoint(revBB->getTerminator());

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
    auto parallelOp = cast<scf::ParallelOp>(op);
    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);
    SmallVector<Value> caches;
    for (Value lb : parallelOp.getLowerBound())
      caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(lb),
                                                cacheBuilder));
    for (Value ub : parallelOp.getUpperBound())
      caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(ub),
                                                cacheBuilder));
    for (Value step : parallelOp.getStep())
      caches.push_back(gutils->initAndPushCache(
          gutils->getNewFromOriginal(step), cacheBuilder));

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct IfOpEnzymeOpsRemover
    : public IfLikeEnzymeOpsRemover<IfOpEnzymeOpsRemover, scf::IfOp> {
  static Block *getThenBlock(scf::IfOp ifOp, OpBuilder &builder) {
    return ifOp.thenBlock();
  }

  static Block *getElseBlock(scf::IfOp ifOp, OpBuilder &builder) {
    // Ensure the if has an else block
    if (ifOp.getElseRegion().empty()) {
      OpBuilder::InsertionGuard guard(builder);
      Block &newBlock = ifOp.getElseRegion().emplaceBlock();
      builder.setInsertionPointToStart(&newBlock);
      scf::YieldOp::create(builder, ifOp.getLoc());
    }

    return ifOp.elseBlock();
  }

  static Value getDummyValue(OpBuilder &builder, Location loc, Type dummyType) {
    return cast<AutoDiffTypeInterface>(dummyType).createNullValue(builder, loc);
  }

  static scf::IfOp replace(PatternRewriter &rewriter, scf::IfOp otherIfOp,
                           TypeRange resultTypes) {
    auto newIf = scf::IfOp::create(rewriter, otherIfOp->getLoc(), resultTypes,
                                   otherIfOp.getCondition());

    newIf.getThenRegion().takeBody(otherIfOp.getThenRegion());
    newIf.getElseRegion().takeBody(otherIfOp.getElseRegion());

    rewriter.replaceAllUsesWith(
        otherIfOp->getResults(),
        newIf->getResults().slice(0, otherIfOp->getNumResults()));
    rewriter.eraseOp(otherIfOp);
    return newIf;
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
      auto result = ifOp.getResult(i);
      auto iface = dyn_cast<AutoDiffTypeInterface>(result.getType());
      bool needsGrad = iface && !iface.isMutable();
      resultsActive[i] = needsGrad && !gutils->isConstantValue(result);
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
        scf::IfOp::create(builder, ifOp.getLoc(), TypeRange{}, cond, hasElse);
    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), revIf->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());
        bodyBuilder.setInsertionPoint(revBB.getTerminator());

        // All values defined in the body should have no use outside this
        // block therefore we can set their diffe to zero upon entering the
        // reverse block to simplify the work of the
        // remove-unnecessary-enzyme-ops pass.
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
          // Check activity of the argument separately from the result. If
          // some branches yield inactive values while others yield active
          // values, the result will be active, but this operand may still be
          // inactive (and we cannot addToDiffe)
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
                          MGradientUtilsReverse *gutils) const {
    // TODO: consider making this generic for RegionBranchOpInterface
    auto ifOp = cast<scf::IfOp>(op);
    if (ifOp.getNumResults() == 0)
      return;

    auto newIf = cast<scf::IfOp>(gutils->getNewFromOriginal(ifOp));
    SmallVector<Type> newResultTypes;
    SmallVector<bool> needsShadow(op->getNumResults());
    for (auto result : op->getResults()) {
      newResultTypes.push_back(result.getType());
      auto iface = dyn_cast<AutoDiffTypeInterface>(result.getType());
      if (iface && iface.isMutable() && !gutils->isConstantValue(result)) {
        newResultTypes.push_back(result.getType());
        needsShadow[result.getResultNumber()] = true;
      } else {
        needsShadow[result.getResultNumber()] = false;
      }
    }

    // Replace the new op with an augmented op
    auto augmentedOp =
        scf::IfOp::create(builder, op->getLoc(), newResultTypes,
                          gutils->getNewFromOriginal(ifOp.getCondition()),
                          /*withElseRegion=*/true);

    for (auto &&[oldReg, newReg, augReg] :
         llvm::zip(op->getRegions(), newIf->getRegions(),
                   augmentedOp->getRegions())) {
      augReg.takeBody(newReg);
      for (auto &&[oldBlk, augBlk] : llvm::zip(oldReg, augReg)) {
        Operation *oldYield = oldBlk.getTerminator();
        Operation *augYield = augBlk.getTerminator();

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(augYield);
        SmallVector<Value> newOperands;
        for (auto &&[oldOperand, augOperand] :
             llvm::zip(oldYield->getOpOperands(), augYield->getOpOperands())) {
          newOperands.push_back(augOperand.get());
          if (needsShadow[oldOperand.getOperandNumber()]) {
            newOperands.push_back(
                gutils->invertPointerM(oldOperand.get(), builder));
          }
        }

        scf::YieldOp::create(builder, oldYield->getLoc(), newOperands);
        augYield->erase();
      }
    }

    // Determine which returns correspond to the primal
    SmallVector<Value> augmentedResults;
    unsigned resIdx = 0;
    for (auto res : ifOp.getResults()) {
      augmentedResults.push_back(augmentedOp.getResult(resIdx));
      resIdx++;
      if (needsShadow[res.getResultNumber()]) {
        gutils->setInvertedPointer(res, augmentedOp.getResult(resIdx));
        resIdx++;
      }
    }
    newIf.replaceAllUsesWith(augmentedResults);
    newIf.erase();
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

struct ParallelOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<ParallelOpADDataFlow,
                                                  scf::ParallelOp> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   OpResult res) const {
    auto parOp = cast<scf::ParallelOp>(op);
    const size_t num_lower = parOp.getLowerBound().size();
    const size_t num_upper = parOp.getUpperBound().size();
    const size_t num_step = parOp.getStep().size();
    const size_t init_vals_offset = num_lower + num_upper + num_step;
    return {parOp->getOperand(res.getResultNumber() + init_vals_offset),
            parOp.getBody()
                ->getTerminator()
                ->getRegion(res.getResultNumber())
                .front()
                .getTerminator()
                ->getOperand(0)};
  }
  SmallVector<Value> getPotentialIncomingValuesArg(Operation *op,
                                                   BlockArgument arg) const {
    // TO DO:  do we need this?
    assert(0);
    return SmallVector<Value>();
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value val) const {
    SmallVector<Value> sv;

    for (auto [idx, arg] : llvm::enumerate(term->getOperands())) {
      if (arg == val) {
        sv.push_back(term->getRegion(idx).front().getArgument(0));
      }
    }

    return sv;
  }
};

struct ReduceOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<ReduceOpADDataFlow,
                                                  scf::ReduceOp> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   OpResult res) const {
    // ReduceOp's have no results
    return SmallVector<Value>();
  }
  SmallVector<Value> getPotentialIncomingValuesArg(Operation *op,
                                                   BlockArgument arg) const {
    // The op here is the parent of the block, which is a ReduceOp
    // All but the last block arguments match up with the corresponding operand
    // of the reduce op.  The last matches up with terminator operand as well as
    // the initial value.  If this is the ith block, it is the ith initial value

    auto redOp = cast<scf::ReduceOp>(op);
    mlir::Block *ownerBlock = arg.getOwner();
    auto num_args = ownerBlock->getNumArguments();
    auto arg_idx = arg.getArgNumber();
    auto region_idx = ownerBlock->getParent()->getRegionNumber();
    if (arg_idx == num_args - 1) {
      auto parOp = cast<scf::ParallelOp>(redOp->getParentOp());
      auto num_lb = parOp.getLowerBound().size();
      auto num_ub = parOp.getUpperBound().size();
      auto num_st = parOp.getStep().size();
      return {parOp->getOperand(num_lb + num_ub + num_st + region_idx),
              ownerBlock->getTerminator()->getOperand(0)};
    } else {
      return {redOp->getOperand(region_idx)};
    }
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value val) const {
    auto redOp = cast<scf::ReduceOp>(op);
    auto parOp = cast<scf::ParallelOp>(redOp->getParentOp());
    mlir::Block *ownerBlock = term->getBlock();
    auto region_idx = ownerBlock->getParent()->getRegionNumber();

    return {parOp->getResult(region_idx), ownerBlock->getArgument(1)};
  }
};

class SCFReduceAutoDiffOpInterface
    : public AutoDiffOpInterface::ExternalModel<SCFReduceAutoDiffOpInterface,
                                                scf::ReduceOp> {
public:
  LogicalResult createForwardModeTangent(Operation *origTerminator,
                                         OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto parentOp = origTerminator->getParentOp();
    if (!isa<scf::ParallelOp>(parentOp)) {
      origTerminator->emitError()
          << " createForwardModeTangent called with invalid parent" << *parentOp
          << "\n";
      return failure();
    }

    // Note, this works for scf::ReduceOp because it has the same number of
    // operands as the parent (scf::ParallelOp) has results
    assert(parentOp->getNumResults() == origTerminator->getNumOperands());
    llvm::SmallDenseSet<unsigned> operandsToShadow;
    for (auto res : parentOp->getResults()) {
      if (!gutils->isConstantValue(res))
        operandsToShadow.insert(res.getResultNumber());
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(origTerminator->getNumOperands() +
                        operandsToShadow.size());
    for (OpOperand &operand : origTerminator->getOpOperands()) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
      if (operandsToShadow.contains(operand.getOperandNumber()))
        newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
    }

    // Assuming shadows following the originals are fine.
    // TODO: consider extending to have a ShadowableTerminatorOpInterface
    Operation *replTerminator = gutils->getNewFromOriginal(origTerminator);
    replTerminator->setOperands(newOperands);

    // Differentiate the body of the reducer
    for (auto &origRegion : origTerminator->getRegions()) {
      for (auto &origBlock : origRegion) {
        for (Operation &o : origBlock) {
          if (failed(gutils->visitChild(&o))) {
            replTerminator->emitError() << " Differentiating reducer block "
                                        << *replTerminator << " failed!\n";
          }
        }
      }
    }

    // Delete the primal operations in each differentiated reducer block by
    // building a map of the operations that are ultimately used by starting
    // from the shadow operands of the terminator (scf::ReduceReturnOp). Then
    // erase all of the operations that aren't used.  Note that from above, all
    // operands for the terminator are shadow operands.
    for (auto &region : replTerminator->getRegions()) {
      for (auto &block : region) {
        std::map<Operation *, bool> used;
        std::vector<Operation *> op_list;

        // Initialize all operations as not used
        for (Operation &o : block) {
          used[&o] = false;
          op_list.push_back(&o);
        }

        // Recursively mark operations that are used starting from the
        // terminator
        auto mark_used = [&used](const auto &self, Operation *op) -> void {
          if (op != nullptr) {
            assert(used.find(op) != used.end());
            used[op] = true;
            for (auto v : op->getOperands())
              self(self, v.getDefiningOp());
          }
        };
        mark_used(mark_used, block.getTerminator());

        // Delete the unused operations squentially, starting from the last so
        // that all users of an operation are erased before the operation itself
        for (auto it = op_list.rbegin(); it != op_list.rend(); ++it) {
          if (!used[*it]) {
            (*it)->erase();
          }
        }

        // Delete the primal arguments from the block.  We have to go backwards
        // starting from the second-to-last as the args will shift forward after
        // erasing.
        for (int i = block.getNumArguments() - 2; i >= 0; i -= 2) {
          block.eraseArgument(i);
        }
      }
    }

    // Create a new terminator combining the regions of differentiated and
    // original terminators. We clone the original region so that it still
    // exists for the undifferentiated reducer but we can take the region from
    // the originally differentiated one because we delete it later
    mlir::OpBuilder term_builder(replTerminator);
    mlir::IRMapping mapper;
    OperationState state(replTerminator->getLoc(),
                         scf::ReduceOp::getOperationName());
    state.addOperands(newOperands);
    size_t num_regions = origTerminator->getNumRegions();
    for (size_t i = 0; i < num_regions; ++i) {
      Region *new_orig_region = state.addRegion();
      Region *new_diff_region = state.addRegion();
      origTerminator->getRegion(i).cloneInto(new_orig_region, mapper);
      new_diff_region->takeBody(replTerminator->getRegion(i));
    }
    Operation *new_terminator_op = term_builder.create(state);
    gutils->erase(replTerminator);
    gutils->originalToNewFnOps[origTerminator] = new_terminator_op;

    return success();
  }
};

class SCFReduceReturnAutoDiffOpInterface
    : public AutoDiffOpInterface::ExternalModel<
          SCFReduceReturnAutoDiffOpInterface, scf::ReduceReturnOp> {
public:
  LogicalResult createForwardModeTangent(Operation *origTerminator,
                                         OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto parentOp = origTerminator->getParentOp();
    if (!isa<scf::ReduceOp>(parentOp)) {
      origTerminator->emitError()
          << " createForwardModeTangent called with invalid parent" << *parentOp
          << "\n";
      return failure();
    }

    // ReduceOp has no direct results, instead the result of the ith reducer
    // block within the ReduceOp matches up with the ith result of the parent
    // ParallelOp of the ReduceOp.  Therefore the terminator must have exactly 1
    // operand and we will shadow it
    auto reducer_index =
        origTerminator->getBlock()->getParent()->getRegionNumber();
    assert(reducer_index < parentOp->getParentOp()->getNumResults());
    assert(origTerminator->getNumOperands() == 1);
    llvm::SmallDenseSet<unsigned> operandsToShadow;
    if (!gutils->isConstantValue(
            parentOp->getParentOp()->getResult(reducer_index)))
      operandsToShadow.insert(0);

    // For scf::ReduceReturnOp only add the
    // shadows as operands since the primal reducer will be in a different
    // region with its own scf::ReduceReturnOp
    SmallVector<Value> newOperands;
    newOperands.reserve(operandsToShadow.size());
    for (OpOperand &operand : origTerminator->getOpOperands()) {
      if (operandsToShadow.contains(operand.getOperandNumber()))
        newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
    }

    // Special handling for scf::ReduceOp where the assumption that shadows
    // follow originals is violated. Here the shadow operations need to be put
    // in a shadow region.  It isn't clear how to do that directly, so instead
    // we will create the shadows as normal and then create a new scf::ReduceOp
    // terminator that combines the regions from the original and
    // differentiated.  We then erase the primal operations from the derivative
    // reducer region(s).
    Operation *replTerminator = gutils->getNewFromOriginal(origTerminator);
    replTerminator->setOperands(newOperands);

    return success();
  }
};

} // namespace

void mlir::enzyme::registerSCFDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, scf::SCFDialect *) {
    registerInterfaces(context);
    scf::IfOp::attachInterface<IfOpInterfaceReverse>(*context);
    scf::IfOp::attachInterface<IfOpEnzymeOpsRemover>(*context);
    scf::ParallelOp::attachInterface<ParallelOpInterfaceReverse>(*context);
    scf::ParallelOp::attachInterface<ParallelOpEnzymeOpsRemover>(*context);
    scf::ParallelOp::attachInterface<ParallelOpADDataFlow>(*context);
    scf::ReduceOp::attachInterface<ReduceOpADDataFlow>(*context);
    scf::ReduceOp::attachInterface<SCFReduceAutoDiffOpInterface>(*context);
    scf::ReduceReturnOp::attachInterface<SCFReduceReturnAutoDiffOpInterface>(
        *context);
    scf::ForOp::attachInterface<ForOpInterfaceReverse>(*context);
    scf::ForOp::attachInterface<ForOpEnzymeOpsRemover>(*context);
    scf::ForOp::attachInterface<ForOpADDataFlow>(*context);
  });
}
