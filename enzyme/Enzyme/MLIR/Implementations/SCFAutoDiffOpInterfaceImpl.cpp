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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
  // ---- hooks required by LoopCheckpointing<ForOpInterfaceReverse, scf::ForOp>
  // ----

  static std::optional<int64_t>
  getConstantNumberOfIterations(scf::ForOp forOp) {
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
    // Creation order (step, then ub, then lb): canonicalize's constant
    // placement is sensitive to it, since constants aren't freely reordered,
    // so keeping one order here is what keeps the two scaffolds' constants
    // laid out the same way.
    Value stepV = arith::ConstantIndexOp::create(builder, loc, step);
    Value ubV = arith::ConstantIndexOp::create(builder, loc, ub);
    Value lbV = arith::ConstantIndexOp::create(builder, loc, lb);
    return scf::ForOp::create(builder, loc, lbV, ubV, stepV, inits);
  }

  // Both outer loops count segments: one iteration per checkpoint, so the trip
  // count is the stated period. It stays a compile-time constant even when the
  // trip count of the loop being differentiated is not -- that is what reading
  // the period as a segment count buys -- which is what lets the checkpoint
  // storage these loops push one entry to per iteration be sized statically.
  static scf::ForOp createForwardOuterLoop(OpBuilder &builder, Location loc,
                                           const PeriodicSchedule &sched,
                                           ValueRange inits) {
    return createConstantScaffoldLoop(builder, loc, 0, sched.numSegments(), 1,
                                      inits);
  }

  static scf::ForOp createReverseOuterLoop(OpBuilder &builder, Location loc,
                                           const PeriodicSchedule &sched,
                                           ValueRange inits) {
    return createConstantScaffoldLoop(builder, loc, 0, sched.numSegments(), 1,
                                      inits);
  }

  // {segment base, segment length}, for the segment `outerIV` indexes. Both are
  // emitted once here, at the top of the outer body, and handed back to the
  // hooks that need them -- which is also what keeps the base from being
  // computed twice, once for the segment loop's bound and once for the
  // induction variable derived from it.
  static SmallVector<Value>
  computeForwardSegmentHint(OpBuilder &builder, Location loc, Value outerIV,
                            const PeriodicSchedule &sched) {
    Value base = segmentBase(builder, loc, sched, outerIV);
    return {base, segmentLength(builder, loc, sched, base)};
  }

  // The reverse replays the segments back to front, so its own counter names
  // segment numSegments() - 1 - i. Same two values as the forward direction,
  // from there on.
  //
  // numSegments() - 1, not nOuter: the two coincide only when there *is* a
  // trailing segment. Without one (a period that divides the trip count, or a
  // perfect-square sqrt split) nOuter - i names a segment one past the end, and
  // the replayed segment's induction variable comes out shifted by a whole
  // period -- silently, since the value is dead whenever the loop body does not
  // read its induction variable, which is why no test caught it.
  static SmallVector<Value>
  computeReverseSegmentHint(OpBuilder &builder, Location loc, Value outerIV,
                            const PeriodicSchedule &sched) {
    Value lastSegment =
        arith::ConstantIndexOp::create(builder, loc, sched.numSegments() - 1);
    Value index = arith::SubIOp::create(builder, loc, lastSegment, outerIV);
    Value base = segmentBase(builder, loc, sched, index);
    return {base, segmentLength(builder, loc, sched, base)};
  }

  static scf::ForOp createSegmentLoop(OpBuilder &builder, Location loc,
                                      ArrayRef<Value> hint, ValueRange inits) {
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    return scf::ForOp::create(builder, loc, zero, hint[1], one, inits);
  }

  static scf::ForOp createForwardSegmentLoop(OpBuilder &builder, Location loc,
                                             Value, ArrayRef<Value> fwdHint,
                                             const PeriodicSchedule &,
                                             ValueRange inits) {
    return createSegmentLoop(builder, loc, fwdHint, inits);
  }

  static scf::ForOp createReverseSegmentLoop(OpBuilder &builder, Location loc,
                                             Value, ArrayRef<Value> revHint,
                                             const PeriodicSchedule &,
                                             ValueRange inits) {
    return createSegmentLoop(builder, loc, revHint, inits);
  }

  static scf::ForOp createLoopWithSameBounds(OpBuilder &builder, Location loc,
                                             scf::ForOp templateLoop,
                                             ValueRange inits) {
    return scf::ForOp::create(builder, loc, templateLoop.getLowerBound(),
                              templateLoop.getUpperBound(),
                              templateLoop.getStep(), inits);
  }

  // start + step * (segment base + localIV): the original loop's own induction
  // variable at the iteration this replay step stands for. Both directions
  // count iterations from the same place, so both ask for it the same way.
  static Value segmentIV(OpBuilder &builder, Location loc, scf::ForOp forOp,
                         const PeriodicSchedule &sched, ArrayRef<Value> hint,
                         Value localIV) {
    Value flatIV = arith::AddIOp::create(builder, loc, hint[0], localIV);
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

  static Value computeForwardSegmentIV(OpBuilder &builder, Location loc,
                                       scf::ForOp forOp, Value /*outerIV*/,
                                       Value localIV,
                                       const PeriodicSchedule &sched,
                                       ArrayRef<Value> fwdHint) {
    return segmentIV(builder, loc, forOp, sched, fwdHint, localIV);
  }

  static Value computeReverseSegmentIV(OpBuilder &builder, Location loc,
                                       scf::ForOp forOp, Value /*outerIV*/,
                                       Value localIV,
                                       const PeriodicSchedule &sched,
                                       ArrayRef<Value> revHint) {
    return segmentIV(builder, loc, forOp, sched, revHint, localIV);
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

    if (auto r = tryCreateReverseModeAdjoint(forOp, op, builder, gutils, caches,
                                             operandsActive, incomingGradients))
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

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    // auto forOp = cast<scf::ForOp>(op);
    return success();
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

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
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

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    // TODO: consider making this generic for RegionBranchOpInterface
    auto ifOp = cast<scf::IfOp>(op);
    if (ifOp.getNumResults() == 0)
      return success();

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

//===----------------------------------------------------------------------===//
// WhileOpInterfaceReverse — online checkpointing for scf.while
//
// Unlike for-loops, scf.while has no a priori bound; the trip count is
// discovered during the forward pass.  This makes online (Stumm-Walther)
// checkpointing the natural fit: OnlineProgressOp drives STORE/FORWARD
// decisions at each step, and the scaffold scf.while exits when the primal's
// own termination condition first returns false.
//
// For simplicity this initial implementation:
//   - Supports only immutable outside references (no ClonableTypeInterface
//     mutable-ref snapshotting; that is left as a TODO matching the for-loop
//     path's mutableRefs machinery).
//   - Differentiates both the before block and the after block of the primal.
//     Gradients flow: after-block adjoint → before-block adjoint.
//===----------------------------------------------------------------------===//

struct WhileOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<WhileOpInterfaceReverse,
                                                       scf::WhileOp> {

  // Scaffold helpers are accessed via the already-instantiated for-loop CRTP
  // so we avoid multiple inheritance issues with ExternalModel's vtable.
  using H = LoopCheckpointing<ForOpInterfaceReverse, scf::ForOp>;

  // Attribute name helpers (replicated from LoopCheckpointing defaults).
  static bool needsOnlineCheckpointing(scf::WhileOp op) {
    auto enabled = op->template getAttrOfType<BoolAttr>(
        "enzyme.enable_checkpointing");
    return enabled && enabled.getValue() &&
           op->hasAttr("enzyme.online_checkpointing");
  }
  static std::optional<int64_t> getCheckpointBudget(scf::WhileOp op) {
    if (auto a = op->template getAttrOfType<IntegerAttr>(
            "enzyme.checkpoint_period"))
      return a.getInt();
    return std::nullopt;
  }
  static void preserveAttributesButCheckpointing(Operation *newOp,
                                                 Operation *oldOp) {
    H::preserveAttributesButCheckpointing(newOp, oldOp);
  }

  // ── Cache layout for while-loop online checkpointing ─────────────────────
  // Same as OnlineCacheLayout but no start/step entries (while loops have
  // no bounds).  Layout: [ckptBufs | idxBuf | immutableRefs | numFwdBuf | spBuf]
  struct WhileOnlineCacheLayout {
    size_t numCarried = 0;
    size_t numImmutable = 0;

    size_t ckptBuf(size_t i) const { return i; }
    size_t idxBuf() const { return numCarried; }
    size_t immutable(size_t i) const { return numCarried + 1 + i; }
    size_t numFwdBuf() const { return immutable(numImmutable); }
    size_t size() const { return immutable(numImmutable) + 1; }
  };

  // ── Helper: copy a block's ops (without terminator) into the builder ──────
  // Equivalent to LoopCheckpointing::copyBlockWithoutTerminator but also maps
  // outside values (defined outside the block's parent regions) to their
  // new-function counterparts via gutils->getNewFromOriginal().
  static void copyWhileBlock(OpBuilder &builder, Block *b,
                              MGradientUtilsReverse *gutils,
                              IRMapping &mapping) {
    for (auto &op : b->without_terminator()) {
      // Map each operand that is defined outside the block (and not yet in
      // the mapping) to its new-function version.
      for (Value operand : op.getOperands()) {
        if (!mapping.contains(operand)) {
          // Use lookupOrNull on originalToNewFn (not the crashing
          // getNewFromOriginal) so that block arguments of the primal's own
          // while regions — which are explicitly mapped beforehand — are
          // silently skipped rather than triggering an unreachable.
          if (Value nv = gutils->originalToNewFn.lookupOrNull(operand))
            mapping.map(operand, nv);
        }
      }
      builder.clone(op, mapping);
    }
    for (auto [oldVal, newVal] : mapping.getValueMap())
      gutils->originalToNewFn.map(oldVal, newVal);
    for (auto [oldOp, newOp] : mapping.getOperationMap())
      gutils->originalToNewFnOps[oldOp] = newOp;
  }

  // ── Forward pass: online checkpointing of scf.while ──────────────────────
  //
  // Scaffold scf.while carries:
  //   before: (sp, numFwd, 10 SM scalars, done, primalCarried...)
  //   after:  (sp, numFwd, action, 10 SM scalars, done, primalCarried...)
  //             [action injected at index 2 by OnlineProgressOp in before]
  //
  // Loop condition: !done  (done=1 when the primal's before block returns false)
  // STORE action: checkpoint primalCarried at slot sp, sp++
  // FORWARD action: run primal before → if true run primal after, numFwd++
  //                 else set done=1 (primal terminated)
  static SmallVector<Value> cacheWhileOnline(scf::WhileOp whileOp,
                                              int64_t budget,
                                              MGradientUtilsReverse *gutils) {
    Location loc = whileOp.getLoc();
    auto newWhileOp =
        cast<scf::WhileOp>(gutils->getNewFromOriginal(whileOp));
    OpBuilder builder(newWhileOp);
    Type idxTy = builder.getIndexType();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
    Value budgetV = arith::ConstantIndexOp::create(builder, loc, budget);

    // Primal carried state = init values (before-block arg types).
    ValueRange primalInits = newWhileOp.getInits();
    size_t numCarried = primalInits.size();

    // Checkpoint stores: one per carried value + one index store.
    // Allocate budget+1 slots (not budget) so the inner Revolve reconstruction
    // in the reverse pass can safely use slot `budget` as a scratch slot.
    // The forward pass only ever writes to slots 0..budget-1 (via new_check),
    // but the inner Revolve increments acapo past budget-1 before checking the
    // while exit condition, potentially storing at slot budget.  The extra slot
    // absorbs that write without a buffer overflow.
    SmallVector<Value> stores;
    for (auto init : primalInits)
      stores.push_back(memref::AllocOp::create(
          builder, loc, MemRefType::get({budget + 1}, init.getType())));
    stores.push_back(memref::AllocOp::create(
        builder, loc, MemRefType::get({budget + 1}, idxTy)));
    const size_t idxStore = stores.size() - 1;

    // Online_r2 SM arrays (freed after the forward pass).
    auto smArrTy = MemRefType::get({budget}, idxTy);
    Value chBuf     = memref::AllocOp::create(builder, loc, smArrTy);
    Value ordChBuf  = memref::AllocOp::create(builder, loc, smArrTy);
    Value numRepBuf = memref::AllocOp::create(builder, loc, smArrTy);

    // Scaffold init: (numFwd=0, smCheck=-1, 9×smScalar=0, primalInit...)
    // sp is deliberately OMITTED from the before-block carry.  The reverse
    // pass uses a slot-scan (Julia's storemap) and the constant `budget` to
    // find the top checkpoint, so it never needs sp.  Carrying sp caused
    // --canonicalize to drop it as a trivial pass-through in the outer reverse
    // for loop, leaving the scaffold while result for numFwd at the wrong index.
    Value smCheckInit = arith::ConstantIndexOp::create(builder, loc, -1);
    SmallVector<Value> whileInit;
    whileInit.push_back(c0);           // numFwd
    whileInit.push_back(smCheckInit);  // check = -1
    for (int i = 0; i < 9; ++i)       // capo..offset (9 more SM scalars)
      whileInit.push_back(c0);
    // (no 'done' — primal termination is detected locally in the before-block)
    whileInit.append(primalInits.begin(), primalInits.end());

    SmallVector<Type> beforeTypes;
    for (Value v : whileInit)
      beforeTypes.push_back(v.getType());

    // After-block types: numFwd, action, 10 SM scalars, primalCarried
    // (sp and done both omitted; the reverse uses budget constant + scan)
    SmallVector<Type> afterTypes;
    afterTypes.push_back(idxTy); // numFwd
    afterTypes.push_back(idxTy); // action
    for (int i = 0; i < 10; ++i)
      afterTypes.push_back(idxTy); // SM scalars
    for (auto init : primalInits)
      afterTypes.push_back(init.getType()); // primalCarried

    SmallVector<Location> befLocs(beforeTypes.size(), loc);
    SmallVector<Location> aftLocs(afterTypes.size(), loc);

    // Create scaffold scf.while (results typed by afterTypes, inits by beforeTypes).
    auto scaffoldWhile =
        scf::WhileOp::create(builder, loc, afterTypes, whileInit);

    // ── Before block ────────────────────────────────────────────────────────
    {
      OpBuilder::InsertionGuard g(builder);
      Block *before = builder.createBlock(&scaffoldWhile.getBefore(), {},
                                          beforeTypes, befLocs);
      builder.setInsertionPointToEnd(before);
      auto ba = before->getArguments();
      Value numFwd = ba[0];
      // SM scalars at positions 1..10; primalCarried at 11+  (no sp, no done)
      Value smCheck = ba[1], smCapo = ba[2], smOldcapoO = ba[3];
      Value smT = ba[4], smIncr = ba[5], smIter = ba[6];
      Value smOldind = ba[7], smInd = ba[8], smOldF = ba[9], smOffset = ba[10];
      auto primalCarried = ba.drop_front(11);

      auto prog = enzyme::OnlineProgressOp::create(
          builder, loc,
          smCheck, smCapo, smOldcapoO, smT, smIncr, smIter,
          smOldind, smInd, smOldF, smOffset, budgetV,
          chBuf, ordChBuf, numRepBuf);

      // Loop terminates when the PRIMAL while terminates (primalCond=false).
      // We do NOT check action != 0 here: when Online_r2 transitions from the
      // online phase to the offline Revolve it returns action=0 (TERMINATE),
      // but the primal may still have unrun steps.  Gating on action would exit
      // the scaffold early and under-count numFwd.  When action=0 in the
      // after-block, isStore=false so we fall through to the FORWARD branch
      // and continue advancing the primal.

      // Run the primal's condition check on the CURRENT carried state.
      Block *primalBeforeBlk = &newWhileOp.getBefore().front();
      IRMapping condCheckMap;
      for (auto &&[old, nw] :
           llvm::zip_equal(primalBeforeBlk->getArguments(), primalCarried))
        condCheckMap.map(old, nw);
      copyWhileBlock(builder, primalBeforeBlk, gutils, condCheckMap);
      auto primalCondOp =
          cast<scf::ConditionOp>(primalBeforeBlk->getTerminator());
      Value primalCond =
          condCheckMap.lookupOrDefault(primalCondOp.getCondition());

      // Loop terminates when the primal while terminates (primalCond=false).
      //
      // When the Online_r2 online phase completes (action=0), Julia's
      // next_action! transparently hands off to offline Revolve and keeps
      // returning STORE/FORWARD; the forward loop only exits when condition()
      // is false.  We match that by using primalCond as the sole termination
      // criterion — the remaining advance steps after action=0 are handled
      // by the offline Revolve reconstruction in the reverse pass.
      //
      // However, Online_r2 with `snaps` checkpoints can only handle loops of
      // at most β(snaps) = (snaps+2)*(snaps+1)/2 - 1 iterations.  Beyond
      // that the repetition counters in num_rep are exhausted and the
      // reconstruction is invalid.  Julia raises an explicit error in that
      // case.  We assert at runtime so the error is surfaced clearly rather
      // than silently producing wrong gradients.
      {
        int64_t beta = (int64_t)(budget + 2) * (budget + 1) / 2 - 1;
        Value betaV  = arith::ConstantIndexOp::create(builder, loc, beta);
        // Assert: action==0 implies numFwd < beta  (i.e. within capacity).
        // Fires only when action is TERMINATE and we've already exceeded the
        // algorithm's supported range, matching Julia's error condition.
        Value actionIsZero = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, prog.getAction(), c0);
        Value belowBeta = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sle, numFwd, betaV);
        // valid = !actionIsZero || belowBeta
        Value notActionZero = arith::XOrIOp::create(
            builder, loc, actionIsZero,
            arith::ConstantIntOp::create(builder, loc, 1, 1));
        Value capacityOk = arith::OrIOp::create(builder, loc, notActionZero, belowBeta);
        cf::AssertOp::create(builder, loc, capacityOk,
            "Online_r2: loop exceeds maximum supported iterations for this "
            "checkpoint budget.  Increase enzyme.checkpoint_period.");
      }
      Value shouldContinue = primalCond;  // exit only when primal terminates

      SmallVector<Value> condVals;
      condVals.push_back(numFwd);
      condVals.push_back(prog.getAction());
      condVals.push_back(prog.getNewCheck());
      condVals.push_back(prog.getNewCapo());
      condVals.push_back(prog.getNewOldcapoO());
      condVals.push_back(prog.getNewT());
      condVals.push_back(prog.getNewIncr());
      condVals.push_back(prog.getNewIter());
      condVals.push_back(prog.getNewOldind());
      condVals.push_back(prog.getNewInd());
      condVals.push_back(prog.getNewOldF());
      condVals.push_back(prog.getNewOffset());
      // (no 'done' in condVals — it was the source of the canonicalize bug)
      condVals.append(primalCarried.begin(), primalCarried.end());
      scf::ConditionOp::create(builder, loc, shouldContinue, condVals);
    }

    // ── After block ─────────────────────────────────────────────────────────
    {
      OpBuilder::InsertionGuard g(builder);
      Block *after = builder.createBlock(&scaffoldWhile.getAfter(), {},
                                         afterTypes, aftLocs);
      builder.setInsertionPointToEnd(after);
      auto aa = after->getArguments();
      Value numFwd = aa[0], action = aa[1];
      // aa[2..11]: updated SM scalars (new_check..new_offset); aa[12+]: primalCarried
      auto primalCarried = aa.drop_front(12);

      // Dispatch on action == 1 (STORE) vs action == 2 (FORWARD).
      Value storeVal = arith::ConstantIndexOp::create(builder, loc, 1);
      Value isStore = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, action, storeVal);

      // Branch result types: (newNumFwd, newPrimalCarried...)
      SmallVector<Type> brTys = {idxTy};
      for (auto init : primalInits)
        brTys.push_back(init.getType());

      auto dispatch = scf::IfOp::create(builder, loc, brTys, isStore,
                                        /*withElse=*/true);

      // ── STORE branch ───────────────────────────────────────────────────
      // Mirror Julia's checkpoint_struct_while:
      //   check = next_action.cpnum + 1   (1-indexed)
      //   model_check[check] = deepcopy(model)
      //
      // Here aa[3] = new_check = prog.getNewCheck(), which is the 0-indexed
      // equivalent of Julia's cpnum.  The algorithm keeps new_check in
      // [0, budget-1], so the fixed-size buffer never overflows.
      // sp is set to new_check + 1 so the reverse finds the top checkpoint
      // at capo = sp - 1 = new_check.
      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(dispatch.thenBlock());
        Value checkSlot = aa[2]; // new_check at aa[2] (aa[0]=numFwd, aa[1]=action)
        for (size_t i = 0; i < numCarried; ++i)
          memref::StoreOp::create(builder, loc, primalCarried[i],
                                  stores[i], ValueRange{checkSlot});
        memref::StoreOp::create(builder, loc, numFwd, stores[idxStore],
                                ValueRange{checkSlot});
        SmallVector<Value> thenY;
        thenY.push_back(numFwd); // unchanged
        thenY.append(primalCarried.begin(), primalCarried.end());
        scf::YieldOp::create(builder, loc, thenY);
      }

      // ── FORWARD branch ─────────────────────────────────────────────────
      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(dispatch.elseBlock());

        Block *primalBefore = &newWhileOp.getBefore().front();
        Block *primalAfter  = &newWhileOp.getAfter().front();

        // Run primal before block with current carried state.
        IRMapping beforeMap;
        for (auto &&[old, nw] :
             llvm::zip_equal(primalBefore->getArguments(), primalCarried))
          beforeMap.map(old, nw);
        copyWhileBlock(builder, primalBefore, gutils, beforeMap);

        auto condOp = cast<scf::ConditionOp>(primalBefore->getTerminator());
        Value primalCond = beforeMap.lookupOrDefault(condOp.getCondition());
        SmallVector<Value> mappedCondVals;
        for (auto arg : condOp.getArgs())
          mappedCondVals.push_back(beforeMap.lookupOrDefault(arg));

        // Run primal after block (we know primal_cond is true here because
        // the before-block only passes action=FORWARD when primalCond holds).
        // Branch types: (newCarried..., stepTaken)
        SmallVector<Type> cBrTys;
        for (auto init : primalInits)
          cBrTys.push_back(init.getType());
        cBrTys.push_back(idxTy); // stepTaken = 1

        auto condBranch = scf::IfOp::create(builder, loc, cBrTys, primalCond,
                                            /*withElse=*/true);
        {
          // Primal continues: run one step.
          OpBuilder::InsertionGuard g3(builder);
          builder.setInsertionPointToStart(condBranch.thenBlock());
          IRMapping afterMap;
          for (auto &&[old, nw] :
               llvm::zip_equal(primalAfter->getArguments(), mappedCondVals))
            afterMap.map(old, nw);
          copyWhileBlock(builder, primalAfter, gutils, afterMap);
          auto yieldOp = cast<scf::YieldOp>(primalAfter->getTerminator());
          SmallVector<Value> thenY2;
          for (auto op : yieldOp.getOperands())
            thenY2.push_back(afterMap.lookupOrDefault(op));
          thenY2.push_back(c1); // stepTaken = 1
          scf::YieldOp::create(builder, loc, thenY2);
        }
        {
          // Primal terminated (fallback; shouldn't fire since the before-block
          // condition prevents FORWARD when primal is done, but safe to have).
          OpBuilder::InsertionGuard g3(builder);
          builder.setInsertionPointToStart(condBranch.elseBlock());
          SmallVector<Value> elseY2(primalCarried.begin(), primalCarried.end());
          elseY2.push_back(c0); // stepTaken = 0
          scf::YieldOp::create(builder, loc, elseY2);
        }

        auto cbRes       = condBranch.getResults();
        auto newCarried  = cbRes.drop_back(1);
        Value stepTaken  = cbRes.back();
        Value newNumFwd  = arith::AddIOp::create(builder, loc, numFwd, stepTaken);

        SmallVector<Value> elseY;
        elseY.push_back(newNumFwd);
        elseY.append(newCarried.begin(), newCarried.end());
        scf::YieldOp::create(builder, loc, elseY);
      }

      // Yield updated carry back to the before block.
      // before-block order: numFwd, SM(1..10), primalCarried  (no sp, no done)
      auto dr = dispatch.getResults();
      Value newNumFwd  = dr[0];
      auto  newCarried = dr.drop_front(1);

      SmallVector<Value> whileY;
      whileY.push_back(newNumFwd);
      // SM scalars: pass through from the after-block args (aa[2..11]).
      for (int i = 2; i <= 11; ++i)
        whileY.push_back(aa[i]);
      whileY.append(newCarried.begin(), newCarried.end());
      scf::YieldOp::create(builder, loc, whileY);
    }

    builder.setInsertionPointAfter(scaffoldWhile);
    auto wr = scaffoldWhile.getResults();
    // wr[0]=numFwd, wr[1]=action, wr[2..11]=SM scalars — all unneeded post-loop
    // except numFwd.  wr[12+] = finalPrimalState.
    Value finalNumFwd = wr[0];
    auto finalPrimalState = wr.drop_front(12);

    // Free SM arrays (their job is done after the forward pass).
    memref::DeallocOp::create(builder, loc, chBuf);
    memref::DeallocOp::create(builder, loc, ordChBuf);
    memref::DeallocOp::create(builder, loc, numRepBuf);

    // Collect immutable outside references.
    SmallVector<Value> immutableRefs;
    {
      SetVector<Value> outside;
      mlir::getUsedValuesDefinedAbove(whileOp->getRegions(), outside);
      for (auto ref : outside)
        if (!isa<ClonableTypeInterface>(ref.getType()))
          immutableRefs.push_back(ref);
    }

    // Build caches in WhileOnlineCacheLayout order.
    SmallVector<Value> caches;
    for (auto store : stores)
      caches.push_back(gutils->initAndPushCache(store, builder));
    for (auto ref : immutableRefs)
      caches.push_back(
          gutils->initAndPushCache(gutils->getNewFromOriginal(ref), builder));
    // Online-specific: total primal steps (sp is gone, reverse uses budget constant).
    caches.push_back(gutils->initAndPushCache(finalNumFwd, builder));

    // Replace the primal while op with the scaffold's final primal state.
    gutils->replaceOrigOpWith(whileOp,
                              SmallVector<Value>(finalPrimalState.begin(),
                                                 finalPrimalState.end()));
    gutils->erase(newWhileOp);
    gutils->originalToNewFnOps[whileOp] = scaffoldWhile;

    return caches;
  }

  // ── Forward pass: full-tape caching of scf.while ─────────────────────────
  //
  // Used when no checkpointing attributes are present.  The forward scaffold
  // pushes the before-block args (= loop-carried state) to a dynamic tape
  // (enzyme.push) at the start of every iteration, and counts iterations.
  //
  // Scaffold before-block carries: (primalCarried..., numIters)
  // Scaffold after-block carries:  (condArgs..., numIters)
  //
  // After the while: final numIters is pushed to its own cache.
  // Returns pop-caches: [numIters_pop, tape_pop_0, tape_pop_1, ...]
  //                      (one tape entry per active before-block arg)
  static SmallVector<Value> cacheWhileFullTape(scf::WhileOp whileOp,
                                               MGradientUtilsReverse *gutils) {
    Location loc = whileOp.getLoc();
    auto newWhileOp = cast<scf::WhileOp>(gutils->getNewFromOriginal(whileOp));
    OpBuilder builder(newWhileOp);

    Block *primalBefore = &newWhileOp.getBefore().front();
    Block *primalAfter  = &newWhileOp.getAfter().front();
    size_t numCarried   = whileOp.getNumOperands();

    // Determine which carried values are active.
    SmallVector<bool> operandsActive;
    for (size_t i = 0; i < numCarried; ++i) {
      bool active = !gutils->isConstantValue(whileOp.getOperand(i)) ||
                    !gutils->isConstantValue(whileOp.getResult(i));
      operandsActive.push_back(active);
    }

    // Create one tape cache per active carried value + one for numIters.
    // getNewCache places the enzyme.init at the initialization block (outside
    // any loops), so it is safe to call from here (before the new while).
    SmallVector<Value> tapePushCaches, tapePopCaches;
    for (size_t i = 0; i < numCarried; ++i) {
      if (operandsActive[i]) {
        auto ct = gutils->getCacheType(primalBefore->getArgument(i).getType());
        auto [push, pop] = gutils->getNewCache(ct);
        tapePushCaches.push_back(push);
        tapePopCaches.push_back(pop);
      }
    }
    auto niCt = gutils->getCacheType(builder.getIndexType());
    auto [niPushCache, niPopCache] = gutils->getNewCache(niCt);

    // ── Build scaffold scf.while ──────────────────────────────────────────
    // Inits: (primalInits..., numIters=0)
    ValueRange primalInits = newWhileOp.getInits();
    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    SmallVector<Value> scaffoldInit(primalInits.begin(), primalInits.end());
    scaffoldInit.push_back(c0); // numIters = 0

    // Result types = after-block types = (primalCarried..., numIters)
    SmallVector<Type> afterTypes;
    for (auto v : primalInits)
      afterTypes.push_back(v.getType());
    afterTypes.push_back(builder.getIndexType());

    auto scaffoldWhile =
        scf::WhileOp::create(builder, loc, afterTypes, scaffoldInit);

    // Before block: push active args to tape, then run primal condition.
    {
      Block &bb = scaffoldWhile.getBefore().emplaceBlock();
      for (size_t i = 0; i < numCarried; ++i)
        bb.addArgument(primalBefore->getArgument(i).getType(), loc);
      bb.addArgument(builder.getIndexType(), loc); // numIters
      OpBuilder bld(&bb, bb.end());

      auto bbArgs = bb.getArguments();
      Value numItersArg = bbArgs.back();
      auto primalArgs   = bbArgs.drop_back(1);

      // Push active carried values (= loop-carried state entering this iter).
      size_t tapeIdx = 0;
      for (size_t i = 0; i < numCarried; ++i) {
        if (operandsActive[i])
          enzyme::PushOp::create(bld, loc, tapePushCaches[tapeIdx++], primalArgs[i]);
      }

      // Copy primal before-block ops (condition computation).
      IRMapping bmap;
      for (auto &&[old, nw] :
           llvm::zip_equal(primalBefore->getArguments(), primalArgs))
        bmap.map(old, nw);
      copyWhileBlock(bld, primalBefore, gutils, bmap);

      auto condOp = cast<scf::ConditionOp>(primalBefore->getTerminator());
      Value condV = bmap.lookupOrDefault(condOp.getCondition());
      SmallVector<Value> condArgs;
      for (auto v : condOp.getArgs())
        condArgs.push_back(bmap.lookupOrDefault(v));
      condArgs.push_back(numItersArg); // pass numIters to after block
      scf::ConditionOp::create(bld, loc, condV, condArgs);
    }

    // After block: run primal body, yield (newCarried..., numIters+1).
    {
      Block &ab = scaffoldWhile.getAfter().emplaceBlock();
      // After-block args match the condition's passed values + numIters.
      // Types must match afterTypes.
      for (size_t i = 0; i < numCarried; ++i)
        ab.addArgument(primalAfter->getArgument(i).getType(), loc);
      ab.addArgument(builder.getIndexType(), loc); // numIters
      OpBuilder bld(&ab, ab.end());

      auto abArgs     = ab.getArguments();
      Value numIters2 = abArgs.back();
      auto condPassed = abArgs.drop_back(1);

      IRMapping amap;
      for (auto &&[old, nw] :
           llvm::zip_equal(primalAfter->getArguments(), condPassed))
        amap.map(old, nw);
      copyWhileBlock(bld, primalAfter, gutils, amap);

      auto yieldOp = cast<scf::YieldOp>(primalAfter->getTerminator());
      SmallVector<Value> yields;
      for (auto v : yieldOp.getOperands())
        yields.push_back(amap.lookupOrDefault(v));
      yields.push_back(arith::AddIOp::create(bld, loc, numIters2, c1));
      scf::YieldOp::create(bld, loc, yields);
    }

    // After the scaffold while: push the final iteration count.
    builder.setInsertionPointAfter(scaffoldWhile);
    auto scaffoldResults = scaffoldWhile.getResults();
    Value finalNumIters  = scaffoldResults.back();
    auto primalResults   = scaffoldResults.drop_back(1);
    enzyme::PushOp::create(builder, loc, niPushCache, finalNumIters);

    // Replace the primal while op with the scaffold's primal results.
    gutils->replaceOrigOpWith(whileOp, primalResults);
    gutils->erase(newWhileOp);
    gutils->originalToNewFnOps[whileOp] = scaffoldWhile;

    // Return pop-caches: [numIters_pop, tapePop_0, tapePop_1, ...]
    SmallVector<Value> resultCaches;
    resultCaches.push_back(niPopCache);
    resultCaches.append(tapePopCaches.begin(), tapePopCaches.end());
    return resultCaches;
  }

  // ── Reverse pass: full-tape reverse for scf.while ────────────────────────
  //
  // Pops numIters, then runs scf.for i = 0..numIters.  Each iteration:
  //   - Pops the before-block args (loop-carried state) from tape (LIFO =
  //     reverse chronological order, which is what the reverse sweep needs).
  //   - Re-runs the before+after blocks forward to recompute intermediates.
  //   - Seeds adjoints from the outer gradient iter-args.
  //   - Visits the after block then the before block in reverse (adjoint).
  //   - Yields the updated gradient iter-args.
  static LogicalResult
  reverseWhileFullTape(scf::WhileOp whileOp, OpBuilder &builder,
                       MGradientUtilsReverse *gutils,
                       SmallVector<Value> caches, ArrayRef<bool> operandsActive,
                       ArrayRef<Value> incomingGradients) {
    Location loc = whileOp.getLoc();
    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    // caches[0] = numIters pop-cache; caches[1..] = tape pop-caches.
    Value numIters = gutils->popCache(caches[0], builder);

    Block *primalBefore = &whileOp.getBefore().front();
    Block *primalAfter  = &whileOp.getAfter().front();

    auto revFor = H::createScaffoldForLoop(
        builder, loc, c0, numIters, c1,
        SmallVector<Value>(incomingGradients.begin(), incomingGradients.end()));

    bool valid = true;
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(revFor.body);

      auto adjArgs = revFor.args();

      // Pop active before-block args (= loop-carried state at start of this
      // forward iteration) from the tape in LIFO order.
      // Inactive args must also be explicitly mapped to prevent copyWhileBlock
      // from falling back to stale gutils->originalToNewFn entries (which
      // point into the now-erased primal while's regions).  For inactive args
      // we use the new-function version of the init operand — an approximation
      // that is safe for gradient correctness because inactive values carry no
      // gradient and copyWhileBlock only needs a valid dominating SSA value.
      IRMapping mapping;
      size_t tapeIdx = 1;
      for (auto &&[active, arg, initOp] :
           llvm::zip_equal(operandsActive, primalBefore->getArguments(),
                           whileOp.getInits())) {
        if (active) {
          Value popped = gutils->popCache(caches[tapeIdx++], builder);
          mapping.map(arg, popped);
          gutils->originalToNewFn.map(arg, popped);
        } else {
          // Map inactive args to the combined-function's version of the while
          // op's init value.  This is a constant or function argument that
          // dominates the reverse loop and won't introduce any gradient.
          if (Value nv = gutils->originalToNewFn.lookupOrNull(initOp))
            mapping.map(arg, nv);
        }
      }

      // Re-run before-block forward to recompute condition and intermediates.
      copyWhileBlock(builder, primalBefore, gutils, mapping);

      // Map after-block args to the condition's passed values.
      auto condOp = cast<scf::ConditionOp>(primalBefore->getTerminator());
      IRMapping afterMap;
      for (auto &&[arg, condVal] :
           llvm::zip_equal(primalAfter->getArguments(), condOp.getArgs()))
        afterMap.map(arg, mapping.lookupOrDefault(condVal));

      // Re-run after-block forward to recompute intermediates.
      copyWhileBlock(builder, primalAfter, gutils, afterMap);

      // Seed adjoints: each yield operand gets the gradient from the outer
      // reverse-for iter-arg.
      auto yieldOp = cast<scf::YieldOp>(primalAfter->getTerminator());
      unsigned revIdx = 0;
      for (auto &&[active, operand] :
           llvm::zip_equal(operandsActive, yieldOp.getOperands())) {
        if (active) {
          gutils->addToDiffe(operand, adjArgs[revIdx], builder);
          revIdx++;
        }
      }

      // Visit after-block then before-block in reverse (adjoint rules).
      // visitChild accumulates gradient into dffe(primalAfter->getArguments())
      // (the after-block args, which map to the same popped values as the
      // before-block args when scf.condition passes them through unchanged).
      {
        H::SegmentCacheCreatorGuard cacheGuard(gutils, revFor.op);
        auto visitBlock = [&](Block *b) {
          for (auto it = b->rbegin(); it != b->rend(); ++it) {
            if (&*it == b->getTerminator())
              continue;
            valid &=
                gutils->Logic.visitChild(&*it, builder, gutils).succeeded();
          }
        };
        visitBlock(primalAfter);
        visitBlock(primalBefore);
      }

      // Collect new adjoints from after-block args.
      // visitChild accumulates into dffe(primalAfter->getArguments()), not
      // dffe(primalBefore->getArguments()), because after-block args are where
      // the body ops run.  In the typical scf.while pattern the condition
      // passes the before-block args unchanged, so after-block args and
      // before-block args map to the same popped values and share the gradient.
      SmallVector<Value> newAdjoints;
      for (auto &&[active, arg] :
           llvm::zip_equal(operandsActive, primalAfter->getArguments())) {
        if (active) {
          newAdjoints.push_back(gutils->diffe(arg, builder));
          if (!gutils->isConstantValue(arg))
            gutils->zeroDiffe(arg, builder);
        }
      }
      H::finalizeScaffoldLoop(builder, loc, revFor, newAdjoints);
    }

    builder.setInsertionPointAfter(revFor.op);

    // Propagate final gradients to the while op's init operands.
    unsigned revIdx = 0;
    auto revForResults = revFor.results();
    for (auto &&[active, initArg] :
         llvm::zip_equal(operandsActive, whileOp.getInits())) {
      if (active) {
        if (!gutils->isConstantValue(initArg))
          gutils->addToDiffe(initArg, revForResults[revIdx], builder);
        revIdx++;
      }
    }
    return success(valid);
  }

  // ── Reverse pass: Revolve reconstruction for scf.while ───────────────────
  //
  // Structurally identical to reverseOnline (for scf.for) but the inner remat
  // loop advances the primal by running the while op's before+after blocks
  // instead of computing an IV and calling a single body block.
  static LogicalResult
  reverseWhileOnline(scf::WhileOp whileOp, int64_t budget,
                     OpBuilder &builder, MGradientUtilsReverse *gutils,
                     SmallVector<Value> caches,
                     ArrayRef<bool> operandsActive,
                     ArrayRef<Value> incomingGradients) {
    Location loc = whileOp.getLoc();
    size_t numCarried = whileOp.getNumOperands();

    // Build the cache layout and pop all entries.
    WhileOnlineCacheLayout layout;
    layout.numCarried = numCarried;

    SmallVector<Value> immutableRefs;
    {
      SetVector<Value> outside;
      mlir::getUsedValuesDefinedAbove(whileOp->getRegions(), outside);
      for (auto ref : outside)
        if (!isa<ClonableTypeInterface>(ref.getType()))
          immutableRefs.push_back(ref);
    }
    layout.numImmutable = immutableRefs.size();

    SmallVector<Value> stores;
    for (size_t j = 0; j < numCarried; ++j)
      stores.push_back(gutils->popCache(caches[layout.ckptBuf(j)], builder));
    stores.push_back(gutils->popCache(caches[layout.idxBuf()], builder));
    const size_t idxStore = stores.size() - 1;

    IRMapping mapping;
    for (auto &&[i, ref] : llvm::enumerate(immutableRefs)) {
      Value cached = gutils->popCache(caches[layout.immutable(i)], builder);
      mapping.map(ref, cached);
    }

    Value numItersV = gutils->popCache(caches[layout.numFwdBuf()], builder);
    // sp was removed from the scaffold while entirely, so there is no spBuf.
    // The reverse uses the constant `budget` for slot scans and budgetRem.

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
    Value totalBudget = arith::ConstantIndexOp::create(builder, loc, budget);
    Value negOne = arith::ConstantIndexOp::create(builder, loc, -1);

    // Outer reverse loop: carries only the gradient adjoints (no sp).
    SmallVector<Value> outerInit;
    outerInit.append(incomingGradients.begin(), incomingGradients.end());

    auto revOuter = H::createScaffoldForLoop(builder, loc, c0, numItersV, c1,
                                             outerInit);
    preserveAttributesButCheckpointing(revOuter.op, whileOp);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(revOuter.body);

      Value ivO = revOuter.getIV();
      auto adjArgs = revOuter.args();  // all iter-args = gradients

      SmallVector<Value> liveStores(stores);

      Value currentRevStep = H::emitSub(builder, loc, numItersV, ivO);
      // targetStep = currentRevStep - 1: the step we need state just BEFORE.
      Value targetStep = H::emitSub(builder, loc, currentRevStep, c1);

      // ── Step→slot lookup (Julia's storemap) ─────────────────────────────
      // Find the checkpoint slot whose stored step is the HIGHEST value
      // ≤ targetStep.  This is the slot from which Revolve reconstruction
      // should start — the equivalent of Julia's
      //   storemap[next_action.iteration - 1]
      // The scan costs O(budget) per outer iteration; for small budgets
      // (2–10) this is negligible.
      //
      //   bestCapo = argmax{ liveStores[idxStore][s]
      //                      | s in 0..budget-1, step[s] ≤ targetStep }
      auto scanLoop = scf::ForOp::create(builder, loc, c0, totalBudget, c1,
                                         SmallVector<Value>{c0, negOne});
      if (!scanLoop.getBody()->empty())
        scanLoop.getBody()->front().erase();
      {
        OpBuilder::InsertionGuard sg(builder);
        builder.setInsertionPointToStart(scanLoop.getBody());
        Value slot     = scanLoop.getInductionVar();
        Value bestCapo = scanLoop.getRegionIterArgs()[0];
        Value bestStep = scanLoop.getRegionIterArgs()[1];
        Value stepAtSlot = H::loadSlot(builder, loc, liveStores[idxStore],
                                       slot, builder.getIndexType());
        Value withinTarget = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sle, stepAtSlot, targetStep);
        Value isBetter = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sgt, stepAtSlot, bestStep);
        Value update = arith::AndIOp::create(builder, loc, withinTarget, isBetter);
        Value newBestCapo = arith::SelectOp::create(builder, loc, update, slot, bestCapo);
        Value newBestStep = arith::SelectOp::create(builder, loc, update, stepAtSlot, bestStep);
        scf::YieldOp::create(builder, loc, ValueRange{newBestCapo, newBestStep});
      }
      Value capo = scanLoop.getResult(0);  // slot of the best checkpoint

      // Load top checkpoint: carried state + the primal step it was taken at.
      SmallVector<Value> ckptState;
      Block *primalBefore = &whileOp.getBefore().front();
      Block *primalAfter  = &whileOp.getAfter().front();
      for (auto &&[store, arg] : llvm::zip_equal(
               ArrayRef<Value>(liveStores).drop_back(),
               primalBefore->getArguments()))
        ckptState.push_back(
            H::loadSlot(builder, loc, store, capo, arg.getType()));
      Value ckptStep =
          H::loadSlot(builder, loc, liveStores[idxStore], capo,
                      builder.getIndexType());

      // Inner Revolve reconstruction: advance from ckptStep to currentRevStep-1.
      SmallVector<Value> whileInitInner;
      whileInitInner.push_back(ckptStep);
      whileInitInner.push_back(capo);
      whileInitInner.append(ckptState.begin(), ckptState.end());

      auto revWhile = H::createScaffoldWhileLoop(
          builder, loc, whileInitInner,
          [&](OpBuilder &b, Location l, ValueRange args) {
            Value posPlus1 = H::emitAdd(b, l, args[0], c1);
            return H::emitCmpLT(b, l, posPlus1, currentRevStep);
          });

      {
        H::setInsertionPointToBodyEnd(builder, revWhile.body);
        auto wargs = revWhile.args();
        Value pos    = wargs[0];
        Value acapo  = wargs[1];
        auto astate  = wargs.slice(2, numCarried);

        Value remaining = H::emitSub(builder, loc, currentRevStep, pos);
        // Use the constant budget (not budgetV=sp which may be < budget after
        // r=2 STORE reuses a lower slot) so Revolve knows how many checkpoint
        // slots the buffer actually holds.
        Value totalBudget = arith::ConstantIndexOp::create(builder, loc, budget);
        Value budgetRem = H::emitSub(builder, loc, totalBudget, acapo);
        budgetRem = H::emitMin(builder, loc, budgetRem, remaining);
        Value split = enzyme::BinomialProgressOp::create(
            builder, loc, builder.getIndexType(), remaining, budgetRem);

        // Checkpoint current astate at slot acapo.
        for (auto &&[i, val] : llvm::enumerate(astate))
          H::storeSlot(builder, loc, liveStores[i], acapo, val);
        H::storeSlot(builder, loc, liveStores[idxStore], acapo, pos);

        // Advance the primal by `split` steps from `pos`.
        // Each "step" = run primal before block → run primal after block.
        Value posPlusSplit = H::emitAdd(builder, loc, pos, split);
        Value isLast = H::emitCmpEQ(builder, loc, posPlusSplit, currentRevStep);
        Value rematUB = H::emitSelect(builder, loc, isLast,
                                      H::emitSub(builder, loc, posPlusSplit, c1),
                                      posPlusSplit);

        // Inner remat: scf.for i = pos to rematUB, carrying (primal state...)
        auto innerRemat = H::createScaffoldForLoop(
            builder, loc, pos, rematUB, c1,
            SmallVector<Value>(astate.begin(), astate.end()));
        preserveAttributesButCheckpointing(innerRemat.op, whileOp);

        {
          OpBuilder::InsertionGuard g2(builder);
          builder.setInsertionPointToStart(innerRemat.body);

          // One primal step: run before block → run after block.
          IRMapping stepMap;
          for (auto &&[old, nw] :
               llvm::zip_equal(primalBefore->getArguments(), innerRemat.args()))
            stepMap.map(old, nw);
          copyWhileBlock(builder, primalBefore, gutils, stepMap);

          auto condOp = cast<scf::ConditionOp>(primalBefore->getTerminator());
          SmallVector<Value> condVals;
          for (auto arg : condOp.getArgs())
            condVals.push_back(stepMap.lookupOrDefault(arg));

          IRMapping afterStepMap;
          for (auto &&[old, nw] :
               llvm::zip_equal(primalAfter->getArguments(), condVals))
            afterStepMap.map(old, nw);
          copyWhileBlock(builder, primalAfter, gutils, afterStepMap);

          auto yieldOp = cast<scf::YieldOp>(primalAfter->getTerminator());
          SmallVector<Value> stepYields;
          for (auto op : yieldOp.getOperands())
            stepYields.push_back(afterStepMap.lookupOrDefault(op));
          H::finalizeScaffoldLoop(builder, loc, innerRemat, stepYields);
        }

        Value newCapo = H::emitAdd(builder, loc, acapo, c1);
        SmallVector<Value> afterYields;
        afterYields.push_back(posPlusSplit);
        afterYields.push_back(newCapo);
        auto rematResults = innerRemat.results();
        afterYields.append(rematResults.begin(), rematResults.end());
        H::finalizeScaffoldLoop(builder, loc, revWhile, afterYields);
      }

      H::setInsertionPointToBodyEnd(builder, revOuter.body);
      auto whileResults = revWhile.results();
      // whileResults[0]=finalPos, [1]=finalCapo (not used), [2+]=reconState
      auto reconState  = whileResults.slice(2, numCarried);

      // Adjoint of one primal step at (currentRevStep - 1).
      mapping = IRMapping();
      for (auto &&[old, nw] :
           llvm::zip_equal(primalBefore->getArguments(), reconState))
        mapping.map(old, nw);
      copyWhileBlock(builder, primalBefore, gutils, mapping);

      auto condOp = cast<scf::ConditionOp>(primalBefore->getTerminator());
      SmallVector<Value> condVals;
      for (auto arg : condOp.getArgs())
        condVals.push_back(mapping.lookupOrDefault(arg));

      IRMapping afterAdjMap;
      for (auto &&[old, nw] :
           llvm::zip_equal(primalAfter->getArguments(), condVals))
        afterAdjMap.map(old, nw);
      copyWhileBlock(builder, primalAfter, gutils, afterAdjMap);

      // Seed adjoints from the outer carried gradients.
      auto yieldOp = cast<scf::YieldOp>(primalAfter->getTerminator());
      unsigned revIdx = 0;
      for (auto &&[active, operand] :
           llvm::zip_equal(operandsActive, yieldOp.getOperands())) {
        if (active) {
          gutils->addToDiffe(operand, adjArgs[revIdx], builder);
          revIdx++;
        }
      }

      // Visit after block ops in reverse (adjoint), then before block ops.
      bool valid = true;
      {
        H::SegmentCacheCreatorGuard cacheGuard(gutils, revWhile.op);
        auto visitBlock = [&](Block *b) {
          for (auto it = b->rbegin(); it != b->rend(); ++it) {
            if (&*it == b->getTerminator())
              continue;
            valid &= gutils->Logic.visitChild(&*it, builder, gutils).succeeded();
          }
        };
        visitBlock(primalAfter);
        visitBlock(primalBefore);
      }

      // Collect from after-block args: visitChild accumulates into
      // dffe(primalAfter->getArguments()), which map to the same reconstructed
      // state as primalBefore->getArguments() when condition passes them through.
      SmallVector<Value> newAdjoints;
      for (auto &&[active, arg] :
           llvm::zip_equal(operandsActive, primalAfter->getArguments())) {
        if (active) {
          newAdjoints.push_back(gutils->diffe(arg, builder));
          if (!gutils->isConstantValue(arg))
            gutils->zeroDiffe(arg, builder);
        }
      }

      SmallVector<Value> outerYields;
      outerYields.append(newAdjoints.begin(), newAdjoints.end());
      H::finalizeScaffoldLoop(builder, loc, revOuter, outerYields);

      builder.setInsertionPointAfter(revOuter.op);

      // Propagate gradients back to the while op's init operands.
      revIdx = 0;
      auto revOuterResults = revOuter.results();
      auto whileOpInits    = whileOp.getInits();
      for (auto &&[active, initArg] :
           llvm::zip_equal(operandsActive, whileOpInits)) {
        if (active) {
          if (!gutils->isConstantValue(initArg))
            gutils->addToDiffe(initArg, revOuterResults[revIdx], builder);
          revIdx++;
        }
      }

      // Release checkpoint stores.
      for (auto store : stores)
        H::destroyStore(builder, loc, store);
    }

    return success(true);
  }

  // ── Interface methods ─────────────────────────────────────────────────────

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto whileOp = cast<scf::WhileOp>(op);
    if (needsOnlineCheckpointing(whileOp)) {
      auto budget = getCheckpointBudget(whileOp);
      if (budget && *budget > 1)
        return cacheWhileOnline(whileOp, *budget, gutils);
      op->emitError()
          << "online checkpointing requires enzyme.checkpoint_period > 1";
      return {};
    }
    // No checkpointing annotation: full-tape mode.
    return cacheWhileFullTape(whileOp, gutils);
  }

  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto whileOp = cast<scf::WhileOp>(op);

    // Build operandsActive and incomingGradients from the while op's results.
    SmallVector<bool> operandsActive;
    SmallVector<Value> incomingGradients;
    for (auto &&[init, res] :
         llvm::zip_equal(whileOp.getInits(), op->getResults())) {
      bool active = !gutils->isConstantValue(init) ||
                    !gutils->isConstantValue(res);
      operandsActive.push_back(active);
      if (active) {
        incomingGradients.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    if (needsOnlineCheckpointing(whileOp)) {
      auto budget = getCheckpointBudget(whileOp);
      if (!budget || *budget <= 1)
        return op->emitError()
               << "online checkpointing requires enzyme.checkpoint_period > 1";
      return reverseWhileOnline(whileOp, *budget, builder, gutils, caches,
                                operandsActive, incomingGradients);
    }

    // No checkpointing annotation: full-tape mode.
    return reverseWhileFullTape(whileOp, builder, gutils, caches,
                                operandsActive, incomingGradients);
  }

  LogicalResult createShadowValues(Operation *, OpBuilder &,
                                   MGradientUtilsReverse *) const {
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
    scf::WhileOp::attachInterface<WhileOpInterfaceReverse>(*context);
  });
}
