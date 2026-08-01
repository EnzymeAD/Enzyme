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
                                                       scf::ForOp> {
private:
  static Value makeIntConstant(Location loc, OpBuilder builder, int64_t val,
                               Type ty) {
    return arith::ConstantOp::create(builder, loc, IntegerAttr::get(ty, val))
        .getResult();
  };

  static void preserveAttributesButCheckpointing(Operation *newOp,
                                                 Operation *oldOp) {
    for (auto attr : oldOp->getDiscardableAttrs()) {
      auto name = attr.getName();
      if (name != "enzyme.enable_checkpointing" &&
          name != "enzyme.binomial_checkpointing" &&
          name != "enzyme.checkpoint_period")
        newOp->setAttr(name, attr.getValue());
    }
  }

  static bool hasBinomialAttr(scf::ForOp forOp) {
    return forOp->hasAttr("enzyme.binomial_checkpointing");
  }

  static bool needsCheckpointing(scf::ForOp forOp) {
    return forOp->hasAttrOfType<BoolAttr>("enzyme.enable_checkpointing") &&
           forOp->getAttrOfType<BoolAttr>("enzyme.enable_checkpointing")
               .getValue() &&
           !hasBinomialAttr(forOp) &&
           ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp)
               .has_value();
  }

  static bool needsBinomialCheckpointing(scf::ForOp forOp) {
    return forOp->hasAttrOfType<BoolAttr>("enzyme.enable_checkpointing") &&
           forOp->getAttrOfType<BoolAttr>("enzyme.enable_checkpointing")
               .getValue() &&
           hasBinomialAttr(forOp);
  }

  static Value getNumIterationsValue(OpBuilder &builder, Location loc,
                                     scf::ForOp forOp,
                                     MGradientUtilsReverse *gutils) {
    Value lb = gutils->getNewFromOriginal(forOp.getLowerBound());
    Value ub = gutils->getNewFromOriginal(forOp.getUpperBound());
    Value step = gutils->getNewFromOriginal(forOp.getStep());
    Value diff = arith::SubIOp::create(builder, loc, ub, lb);
    return arith::DivUIOp::create(builder, loc, diff, step);
  }

  static Value castToType(OpBuilder &builder, Location loc, Value v,
                          Type targetType) {
    if (v.getType() == targetType)
      return v;
    assert(targetType.isIndex() || v.getType().isIndex());
    return arith::IndexCastOp::create(builder, loc, targetType, v);
  }

  static std::optional<int64_t> getCheckpointBudget(scf::ForOp forOp) {
    if (auto a = forOp->getAttrOfType<IntegerAttr>("enzyme.checkpoint_period"))
      return a.getInt();
    return std::nullopt;
  }

  static MemRefType checkpointBufferType(int64_t budget, Type t) {
    if (auto mt = dyn_cast<MemRefType>(t)) {
      SmallVector<int64_t> shape;
      shape.push_back(budget);
      shape.append(mt.getShape().begin(), mt.getShape().end());
      return MemRefType::get(shape, mt.getElementType(),
                             MemRefLayoutAttrInterface{}, mt.getMemorySpace());
    }
    return MemRefType::get({budget}, t);
  }

  static Value checkpointRow(OpBuilder &b, Location loc, Value buf, Value slot,
                             MemRefType rowTy) {
    auto bufTy = cast<MemRefType>(buf.getType());
    SmallVector<OpFoldResult> offsets, sizes, strides;
    offsets.push_back(slot);
    sizes.push_back(b.getIndexAttr(1));
    strides.push_back(b.getIndexAttr(1));
    for (int64_t i = 0, e = rowTy.getRank(); i < e; ++i) {
      offsets.push_back(b.getIndexAttr(0));
      if (rowTy.isDynamicDim(i))
        sizes.push_back(memref::DimOp::create(b, loc, buf, i + 1).getResult());
      else
        sizes.push_back(b.getIndexAttr(rowTy.getDimSize(i)));
      strides.push_back(b.getIndexAttr(1));
    }
    auto resTy = memref::SubViewOp::inferRankReducedResultType(
        rowTy.getShape(), bufTy, offsets, sizes, strides);
    return memref::SubViewOp::create(b, loc, cast<MemRefType>(resTy), buf,
                                     offsets, sizes, strides);
  }

  static void storeCheckpoint(OpBuilder &b, Location loc, Value buf, Value slot,
                              Value val) {
    if (auto mt = dyn_cast<MemRefType>(val.getType())) {
      Value row = checkpointRow(b, loc, buf, slot, mt);
      memref::CopyOp::create(b, loc, val, row);
    } else {
      memref::StoreOp::create(b, loc, val, buf, ValueRange{slot});
    }
  }

  // Read a snapshot from checkpoint buffer slot `slot`. For scalars returns the
  // loaded value; for memrefs returns a fresh alloc initialized from the row
  // (the caller is responsible for deallocating it).
  static Value loadCheckpoint(OpBuilder &b, Location loc, Value buf, Value slot,
                              Type valTy) {
    if (auto mt = dyn_cast<MemRefType>(valTy)) {
      Value row = checkpointRow(b, loc, buf, slot, mt);
      SmallVector<Value> dynSizes;
      for (int64_t i = 0, e = mt.getRank(); i < e; ++i)
        if (mt.isDynamicDim(i))
          dynSizes.push_back(memref::DimOp::create(b, loc, row, i));
      Value fresh = memref::AllocOp::create(b, loc, mt, dynSizes);
      memref::CopyOp::create(b, loc, row, fresh);
      return fresh;
    }
    return memref::LoadOp::create(b, loc, buf, ValueRange{slot});
  }

  static void copyBlockWithoutTerminator(OpBuilder &builder, Block *b,
                                         MGradientUtilsReverse *gutils,
                                         IRMapping &mapping) {
    for (auto &it : b->without_terminator()) {
      OpBuilder::InsertionGuard g(builder);
      builder.clone(it, mapping);
    }

    for (auto [oldVal, newVal] : mapping.getValueMap())
      gutils->originalToNewFn.map(oldVal, newVal);

    for (auto [oldBlock, newBlock] : mapping.getBlockMap()) {
      gutils->originalToNewFn.map(oldBlock, newBlock);
      for (auto [oldArg, newArg] :
           llvm::zip(oldBlock->getArguments(), newBlock->getArguments()))
        gutils->originalToNewFn.map(oldArg, newArg);
    }

    for (auto [oldOp, newOp] : mapping.getOperationMap())
      gutils->originalToNewFnOps[oldOp] = newOp;
  }

  // Splits the values `forOp` reads from above into the ones holding mutable
  // memory, which has to be snapshotted for a replay, and the ones that can
  // just be forwarded. cacheBinomial and reverseBinomial must produce the same
  // order, which they do: getUsedValuesDefinedAbove is deterministic for a
  // fixed IR and both walk the *original* forOp.
  static void splitOutsideRefs(scf::ForOp forOp,
                               SmallVectorImpl<Value> &mutableRefs,
                               SmallVectorImpl<Value> &immutableRefs) {
    SetVector<Value> outsideRefs;
    getUsedValuesDefinedAbove(forOp->getRegions(), outsideRefs);
    for (auto ref : outsideRefs) {
      if (isa<ClonableTypeInterface>(ref.getType()))
        mutableRefs.push_back(ref);
      else
        immutableRefs.push_back(ref);
    }
  }

  // Index layout of the `caches` vector handed from cacheBinomial to
  // reverseBinomial:
  //
  //   [ ckptBufs (one per iter arg) | idxBuf
  //   | per mutable ref: clone buffer, + shadow if the ref is active
  //   | immutableRefs
  //   | numIters, start, step -- only when the loop bounds are dynamic ]
  //
  // The per-ref region has variable width. That used to be re-derived by hand
  // at each use, with one site testing the predicate on the wrong value;
  // computing it once here is what keeps the two sides from drifting.
  struct BinomialCacheLayout {
    size_t numIterArgs = 0;
    SmallVector<bool> mutableActive;
    size_t numImmutable = 0;
    bool isDynamic = false;

    static BinomialCacheLayout get(scf::ForOp forOp,
                                   ArrayRef<Value> mutableRefs,
                                   ArrayRef<Value> immutableRefs,
                                   bool isDynamic,
                                   MGradientUtilsReverse *gutils) {
      BinomialCacheLayout l;
      l.numIterArgs = forOp.getNumRegionIterArgs();
      for (auto ref : mutableRefs)
        l.mutableActive.push_back(!gutils->isConstantValue(ref));
      l.numImmutable = immutableRefs.size();
      l.isDynamic = isDynamic;
      return l;
    }

    size_t ckptBuf(size_t i) const { return i; }
    size_t idxBuf() const { return numIterArgs; }

    size_t mutableBegin() const { return numIterArgs + 1; }
    size_t mutableWidth() const {
      size_t w = 0;
      for (bool active : mutableActive)
        w += 1 + active;
      return w;
    }
    size_t cloneBuf(size_t r) const {
      size_t idx = mutableBegin();
      for (size_t i = 0; i < r; ++i)
        idx += 1 + mutableActive[i];
      return idx;
    }
    size_t shadow(size_t r) const {
      assert(mutableActive[r] && "inactive ref has no shadow cache");
      return cloneBuf(r) + 1;
    }
    size_t immutable(size_t i) const {
      return mutableBegin() + mutableWidth() + i;
    }
    size_t numIters() const {
      assert(isDynamic && "bounds are static; not cached");
      return immutable(numImmutable);
    }
    size_t start() const { return numIters() + 1; }
    size_t step() const { return numIters() + 2; }
    size_t size() const {
      return immutable(numImmutable) + (isDynamic ? 3 : 0);
    }
  };

  // A `budget`-slot buffer of clone *handles* for one mutable ref. Not
  // checkpointBufferType: what lives here is the identity of a snapshot
  // allocation (slot j pairs with ckptBufs slot j), not its contents -- for a
  // bare pointer the extent of the contents is not in the type at all.
  static MemRefType cloneBufferType(int64_t budget, Type refTy) {
    return MemRefType::get({budget}, refTy);
  }

  static Value cloneSlot(OpBuilder &b, Location loc, Value buf, Value slot) {
    return memref::LoadOp::create(b, loc, buf, ValueRange{slot});
  }

  // Give every slot of `buf` its own clone of `proto`. Deliberately unrolled:
  // `budget` is static, and an allocation sitting in a loop body is both a
  // hoisting candidate and (for pointers, under a raised alloca threshold) an
  // alloca-promotion candidate -- either would silently make all slots alias.
  static void fillCloneSlots(OpBuilder &b, Location loc, int64_t budget,
                             Value buf, Value proto,
                             ClonableTypeInterface iface) {
    for (int64_t j = 0; j < budget; ++j) {
      Value slot = arith::ConstantIndexOp::create(b, loc, j);
      Value clone = iface.cloneValue(b, proto);
      memref::StoreOp::create(b, loc, clone, buf, ValueRange{slot});
    }
  }

  // Free each slot's clone, then the handle buffer itself. A loop is fine here:
  // a free has side effects so it cannot be hoisted, and nothing can alias.
  static void freeCloneSlots(OpBuilder &b, Location loc, int64_t budget,
                             Value buf, ClonableTypeInterface iface) {
    Value lb = arith::ConstantIndexOp::create(b, loc, 0);
    Value ub = arith::ConstantIndexOp::create(b, loc, budget);
    Value step = arith::ConstantIndexOp::create(b, loc, 1);
    auto loop = scf::ForOp::create(b, loc, lb, ub, step);
    {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(loop.getBody());
      iface.freeClonedValue(b, cloneSlot(b, loc, buf, loop.getInductionVar()));
    }
    memref::DeallocOp::create(b, loc, buf);
  }

  // Forward augmentation for binomial (Revolve) checkpointing. Builds an outer
  // loop of `budget` iterations that snapshots the loop state into memref
  // checkpoint buffers at Revolve-scheduled positions, advancing the primal in
  // an inner recompute loop between snapshots. Returns the caches (buffer
  // handles + index buffer + outside refs) transported to the reverse pass;
  // see BinomialCacheLayout for their order.
  static SmallVector<Value> cacheBinomial(scf::ForOp forOp, int64_t budget,
                                          MGradientUtilsReverse *gutils) {
    Location loc = forOp.getLoc();
    bool isDynamic =
        !ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp)
             .has_value();

    auto newForOp = cast<scf::ForOp>(gutils->getNewFromOriginal(forOp));
    OpBuilder builder(newForOp);
    Type idxTy = builder.getIndexType();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    // Loop trip count / lower bound / step as index values (constant-folded
    // when the bounds are constant).
    Value numItersV, startV, stepV;
    if (isDynamic) {
      startV = gutils->getNewFromOriginal(forOp.getLowerBound());
      stepV = gutils->getNewFromOriginal(forOp.getStep());
      numItersV = getNumIterationsValue(builder, loc, forOp, gutils);
      numItersV = castToType(builder, loc, numItersV, idxTy);
    } else {
      int64_t numIters =
          ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).value();
      llvm::APInt startI, stepI;
      (void)matchPattern(forOp.getLowerBound(), m_ConstantInt(&startI));
      (void)matchPattern(forOp.getStep(), m_ConstantInt(&stepI));
      numItersV = arith::ConstantIndexOp::create(builder, loc, numIters);
      startV =
          arith::ConstantIndexOp::create(builder, loc, startI.getSExtValue());
      stepV =
          arith::ConstantIndexOp::create(builder, loc, stepI.getSExtValue());
    }

    // Effective budget = min(requested budget, trip count): never keep more
    // checkpoints than there are iterations. Buffers stay sized by the (static)
    // requested budget; the effective budget bounds the loops at runtime.
    //
    // Unconditional: with a budget above the trip count this loop would run
    // more iterations than there are steps, and the slots past the end get
    // recorded at a step beyond the last one -- holding the final state instead
    // of a checkpoint, which the reverse pass then replays from. This used to
    // be gated behind enzyme.use_safe_budgeting, an attribute nothing ever
    // sets.
    Value budgetV = arith::MinUIOp::create(
        builder, loc, arith::ConstantIndexOp::create(builder, loc, budget),
        numItersV);

    SmallVector<Value> immutableRefs, mutableRefs;
    splitOutsideRefs(forOp, mutableRefs, immutableRefs);
    auto layout = BinomialCacheLayout::get(forOp, mutableRefs, immutableRefs,
                                           isDynamic, gutils);

    IRMapping mapping;
    SmallVector<Value> caches;

    // Allocate one checkpoint buffer per iter arg + the step-index buffer.
    SmallVector<Value> ckptBufs;
    for (auto arg : newForOp.getInitArgs()) {
      auto bufTy = checkpointBufferType(budget, arg.getType());
      ckptBufs.push_back(memref::AllocOp::create(builder, loc, bufTy));
    }
    Value idxBuf =
        memref::AllocOp::create(builder, loc, MemRefType::get({budget}, idxTy));

    // One clone buffer per mutable ref, filled up front so that taking a
    // checkpoint is a pure copy into an existing allocation. Slot j of these
    // buffers holds the ref's content at forward step idxBuf[j] -- the same
    // instant as ckptBufs[*][j]; the reverse pass indexes all three by the
    // checkpoint stack pointer.
    SmallVector<Value> mutBufs;
    for (auto ref : mutableRefs) {
      auto iface = cast<ClonableTypeInterface>(ref.getType());
      Value buf = memref::AllocOp::create(
          builder, loc, cloneBufferType(budget, ref.getType()));
      fillCloneSlots(builder, loc, budget, buf, gutils->getNewFromOriginal(ref),
                     iface);
      mutBufs.push_back(buf);
    }

    // Outer checkpoint-placement loop: for %k = 0 to budgetV carrying
    // (stepCtr, state...).
    SmallVector<Value> outerInit;
    outerInit.push_back(c0);
    outerInit.append(newForOp.getInitArgs().begin(),
                     newForOp.getInitArgs().end());
    auto outerFwd =
        scf::ForOp::create(builder, loc, c0, budgetV, c1, outerInit);
    preserveAttributesButCheckpointing(outerFwd, forOp);

    builder.setInsertionPointToStart(outerFwd.getBody());
    Value k = outerFwd.getInductionVar();
    Value stepCtr = outerFwd.getBody()->getArgument(1);
    auto state = outerFwd.getBody()->getArguments().drop_front(2);

    for (auto &&[buf, val] : llvm::zip_equal(ckptBufs, state))
      storeCheckpoint(builder, loc, buf, k, val);
    memref::StoreOp::create(builder, loc, stepCtr, idxBuf, ValueRange{k});

    Value numStepsRem = arith::SubIOp::create(builder, loc, numItersV, stepCtr);
    Value budgetRem = arith::SubIOp::create(builder, loc, budgetV, k);
    // Never use more checkpoints than remaining steps (binomial_progress is
    // degenerate for budget > steps).
    budgetRem = arith::MinUIOp::create(builder, loc, budgetRem, numStepsRem);
    Value split = enzyme::BinomialProgressOp::create(builder, loc, idxTy,
                                                     numStepsRem, budgetRem);

    // Snapshot each mutable ref into slot `k`, reusing the clone already there.
    // Stays here, before innerFwd, so the snapshot precedes the advance.
    for (auto &&[ref, buf] : llvm::zip_equal(mutableRefs, mutBufs)) {
      auto iface = cast<ClonableTypeInterface>(ref.getType());
      iface.copyValue(builder, cloneSlot(builder, loc, buf, k),
                      gutils->getNewFromOriginal(ref));
    }

    // Inner recompute loop: advance the primal `split` steps.
    auto innerFwd =
        scf::ForOp::create(builder, loc, c0, split, c1,
                           SmallVector<Value>(state.begin(), state.end()));
    preserveAttributesButCheckpointing(innerFwd, forOp);

    // Remove scf.yield automatically added when there are no carried values
    if (!innerFwd.getBody()->empty())
      innerFwd.getBody()->front().erase();

    builder.setInsertionPointToStart(innerFwd.getBody());
    Value i = innerFwd.getInductionVar();
    Value globalStep = arith::AddIOp::create(builder, loc, stepCtr, i);
    globalStep = castToType(builder, loc, globalStep, stepV.getType());
    Value iv = arith::AddIOp::create(
        builder, loc, startV,
        arith::MulIOp::create(builder, loc, stepV, globalStep));

    for (auto &&[oldArg, newArg] :
         llvm::zip_equal(newForOp.getBody()->getArguments().drop_front(),
                         innerFwd.getBody()->getArguments().drop_front()))
      mapping.map(oldArg, newArg);
    mapping.map(newForOp.getInductionVar(), iv);

    copyBlockWithoutTerminator(builder, newForOp.getBody(), gutils, mapping);

    SmallVector<Value> innerYields;
    for (auto operand : newForOp.getBody()->getTerminator()->getOperands())
      innerYields.push_back(mapping.lookupOrDefault(operand));
    scf::YieldOp::create(builder, loc, innerYields);

    builder.setInsertionPointToEnd(outerFwd.getBody());
    SmallVector<Value> outerYields;
    outerYields.push_back(arith::AddIOp::create(builder, loc, stepCtr, split));
    outerYields.append(innerFwd.getResults().begin(),
                       innerFwd.getResults().end());
    scf::YieldOp::create(builder, loc, outerYields);

    builder.setInsertionPointAfter(outerFwd);

    // Cache buffer handles + index buffer + outside refs (single push each).
    for (auto buf : ckptBufs)
      caches.push_back(gutils->initAndPushCache(buf, builder));
    caches.push_back(gutils->initAndPushCache(idxBuf, builder));

    // One push per mutable ref: the clone buffer, plus the shadow, which is
    // loop-invariant (it is the accumulating gradient buffer, not a snapshot)
    // and so does not belong in a per-checkpoint slot.
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs)) {
      caches.push_back(gutils->initAndPushCache(mutBufs[r], builder));
      if (layout.mutableActive[r])
        caches.push_back(gutils->initAndPushCache(
            gutils->invertPointerM(ref, builder), builder));
    }

    for (auto ref : immutableRefs)
      caches.push_back(
          gutils->initAndPushCache(gutils->getNewFromOriginal(ref), builder));

    // For dynamic bounds the reverse pass cannot recover the trip count / lower
    // bound / step from constants, so cache them (as the trailing entries).
    if (isDynamic) {
      caches.push_back(gutils->initAndPushCache(numItersV, builder));
      caches.push_back(gutils->initAndPushCache(startV, builder));
      caches.push_back(gutils->initAndPushCache(stepV, builder));
    }
    assert(caches.size() == layout.size() && "binomial cache layout mismatch");

    // The primal result of the loop is the final state.
    gutils->replaceOrigOpWith(forOp, outerFwd.getResults().drop_front());
    // NB: hoisting placeholders to function scope here has not been revalidated
    // since the mutableRef shadows moved out of revOuter's body (they are now
    // popped before the loop, so the old dominance objection no longer
    // applies). hoistPlaceholdersBefore(newForOp, newForOp);
    gutils->erase(newForOp);
    gutils->originalToNewFnOps[forOp] = outerFwd;

    return caches;
  }

  // Every checkpointing scheme re-clones forOp's body one or more times (to
  // replay a segment of the primal, or to re-materialize a single step for the
  // reverse visitor). Such a clone needs exactly two things mapped up front:
  // the values the body reads from above -- to their counterparts in the new
  // function -- and the body's own block arguments, to this clone's induction
  // variable and iter args. Everything else the cloner discovers as it goes,
  // mapping each op result and each nested block argument to the fresh copy it
  // just made.
  //
  // Seeding from gutils->originalToNewFn instead (whether by reference or as a
  // copy) drags in an entry for every value inside forOp left over from the
  // whole-function forward-augmentation clone -- crucially including block
  // arguments of nested regions, e.g. an affine.parallel's induction variable.
  // Region::cloneInto only creates a fresh block argument when
  // `!mapper.contains(arg)`, so each of those inherited entries silently wires
  // the new clone back to the augmented-forward copy, which checkpointing
  // erases as redundant. A copy also leaks the reverse: the clone writes its
  // own nested block arguments into the copy, and (for the long-lived
  // originalToNewFn) a later Value allocated at a since-freed address collides
  // with the stale entry, corrupting an unrelated clone's use-def chain.
  //
  // Starting from an empty mapping sidesteps both directions, and is cheaper
  // than copying a whole-function map per clone.
  static IRMapping bodyCloneMapping(scf::ForOp forOp, ValueRange iterArgs,
                                    Value iv, const IRMapping &outer) {
    IRMapping mapping;

    SetVector<Value> refs;
    getUsedValuesDefinedAbove(forOp->getRegions(), refs);
    for (Value ref : refs)
      mapping.map(ref, outer.lookupOrDefault(ref));

    mapping.map(forOp.getInductionVar(), iv);
    for (auto &&[oldArg, newArg] : llvm::zip_equal(
             forOp.getBody()->getArguments().drop_front(), iterArgs))
      mapping.map(oldArg, newArg);

    return mapping;
  }

  // Clone one primal step of forOp -- its body minus the terminator -- at
  // `builder`'s insertion point, ending the new body with the corresponding
  // scf.yield.
  static void cloneStepWithYield(OpBuilder &builder, Location loc,
                                 scf::ForOp forOp, IRMapping &mapping) {
    for (Operation &op : forOp.getBody()->without_terminator())
      builder.clone(op, mapping);

    SmallVector<Value> yields = llvm::map_to_vector(
        forOp.getBody()->getTerminator()->getOperands(),
        [&](Value v) { return mapping.lookupOrDefault(v); });
    scf::YieldOp::create(builder, loc, yields);
  }

  // originalToNewFnOps (unlike originalToNewFn) is a plain std::map maintained
  // entirely by hand -- MLIR's cloner has no idea it exists. Ops nested inside
  // a cloned op therefore keep whatever entry they had from the whole-function
  // augmentation clone, which for a checkpointed loop points into the copy
  // gutils->erase()s; a later getNewFromOriginal(op) -- how a nested rule's
  // cacheValues() finds where to put its pushes -- then hands out a dangling
  // pointer. Republishing straight off the map the cloner filled in as it went
  // covers nested ops for free and cannot fall out of sync with what was
  // actually cloned.
  static void publishClonedOps(const IRMapping &mapping,
                               MGradientUtilsReverse *gutils) {
    for (auto &&[oldOp, newOp] : mapping.getOperationMap())
      gutils->originalToNewFnOps[oldOp] = newOp;
  }

  // Hand a re-materialized step to gutils as forOp's body, so the reverse
  // visitor resolves the step's primal through getNewFromOriginal. Only for the
  // clone the visitor actually runs against: one that merely replays the primal
  // publishes its ops but must keep its values private, or it would claim to be
  // the new counterpart of forOp's body for the rest of the pass.
  //
  // The values published are the body's own arguments and its top-level ops'
  // results -- precisely the ones a body op's reverse can name as an operand.
  // Values nested deeper stay private to the clone: nothing outside resolves
  // them, and parking them in the long-lived originalToNewFn is what lets a
  // later Value allocated at a since-freed address collide with a stale entry.
  static void publishClonedStep(scf::ForOp forOp, const IRMapping &mapping,
                                MGradientUtilsReverse *gutils) {
    for (BlockArgument arg : forOp.getBody()->getArguments())
      gutils->originalToNewFn.map(arg, mapping.lookup(arg));

    for (Operation &op : forOp.getBody()->without_terminator())
      for (auto &&[oldVal, newVal] :
           llvm::zip_equal(op.getResults(), mapping.lookup(&op)->getResults()))
        gutils->originalToNewFn.map(oldVal, newVal);

    publishClonedOps(mapping, gutils);
  }

  // forceAugmentedReturns() (called once, early, over the whole original
  // function) plants an enzyme.placeholder for every mutable-typed value's
  // shadow, including ones nested inside a checkpointed forOp's augmented
  // copy (e.g. an affine.parallel's own pointer2memref result). Those
  // placeholders are meant to survive until the reverse pass visits their
  // defining op and replaces them via setInvertedPointer. But checkpointing
  // discards the whole augmented-forward copy of forOp (it's redundant once
  // the checkpoint-buffer reconstruction takes over) via gutils->erase(...),
  // which -- if it happens before that replacement runs -- destroys the
  // still-referenced placeholder out from under invertedPointers, leaving a
  // dangling entry that only crashes later, whenever something finally reads
  // it. Hoist any not-yet-replaced placeholders out of the doomed subtree
  // first so they survive.
  static void hoistPlaceholdersBefore(Operation *root, Operation *before) {
    SmallVector<enzyme::PlaceholderOp> placeholders;
    root->walk([&](enzyme::PlaceholderOp p) { placeholders.push_back(p); });
    for (auto p : placeholders)
      p->moveBefore(before);
  }

  // Reverse pass for binomial (Revolve) checkpointing. Iterates all N steps in
  // reverse; for each step it reconstructs the state just before that step from
  // the top checkpoint (recursively re-placing finer checkpoints during the
  // remat), then emits the adjoint of a single body step.
  static LogicalResult reverseBinomial(scf::ForOp forOp, int64_t budget,
                                       OpBuilder &builder,
                                       MGradientUtilsReverse *gutils,
                                       SmallVector<Value> caches,
                                       ArrayRef<bool> operandsActive,
                                       ArrayRef<Value> incomingGradients) {
    Location loc = forOp.getLoc();
    bool isDynamic =
        !ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp)
             .has_value();
    auto numIterArgs = forOp.getNumRegionIterArgs();

    SmallVector<Value> immutableRefs, mutableRefs;
    splitOutsideRefs(forOp, mutableRefs, immutableRefs);
    auto layout = BinomialCacheLayout::get(forOp, mutableRefs, immutableRefs,
                                           isDynamic, gutils);
    assert(caches.size() == layout.size() && "binomial cache layout mismatch");

    IRMapping mapping;

    // Below, each mutableRef's shadow is re-bound to the popped shadow handle.
    // That binding must not outlive this function: ops *outside* the loop are
    // visited after it (reverse order) and expect the caller's shadow, not
    // ours. Restore the previous bindings on the way out.
    SmallVector<std::pair<Value, Value>> savedMutableShadows;
    for (auto ref : mutableRefs)
      savedMutableShadows.emplace_back(
          ref, gutils->invertedPointers.lookupOrNull(ref));
    auto restoreMutableShadows = llvm::make_scope_exit([&]() {
      for (auto &[ref, prev] : savedMutableShadows) {
        if (prev)
          gutils->invertedPointers.map(ref, prev);
        else
          gutils->invertedPointers.erase(ref);
      }
    });

    // Pop cached handles (indices from the shared layout).
    SmallVector<Value> ckptBufs;
    for (size_t j = 0; j < numIterArgs; ++j)
      ckptBufs.push_back(gutils->popCache(caches[layout.ckptBuf(j)], builder));
    Value idxBuf = gutils->popCache(caches[layout.idxBuf()], builder);

    SmallVector<Value> immutableRefsCaches;
    for (auto &&[i, ref] : llvm::enumerate(immutableRefs)) {
      Value cached = gutils->popCache(caches[layout.immutable(i)], builder);
      mapping.map(ref, cached);
      immutableRefsCaches.push_back(cached);
    }

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    // Loop trip count / lower bound / step as index values. For dynamic bounds
    // these were cached by cacheBinomial (trailing entries, same order).
    Value numItersV, startV, stepV;
    if (isDynamic) {
      numItersV = gutils->popCache(caches[layout.numIters()], builder);
      startV = gutils->popCache(caches[layout.start()], builder);
      stepV = gutils->popCache(caches[layout.step()], builder);
    } else {
      int64_t numIters =
          ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).value();
      llvm::APInt startI, stepI;
      (void)matchPattern(forOp.getLowerBound(), m_ConstantInt(&startI));
      (void)matchPattern(forOp.getStep(), m_ConstantInt(&stepI));
      numItersV = arith::ConstantIndexOp::create(builder, loc, numIters);
      startV =
          arith::ConstantIndexOp::create(builder, loc, startI.getSExtValue());
      stepV =
          arith::ConstantIndexOp::create(builder, loc, stepI.getSExtValue());
    }

    // Effective budget = min(requested budget, trip count); must match
    // cacheBinomial, including being unconditional.
    Value budgetV = arith::MinUIOp::create(
        builder, loc, arith::ConstantIndexOp::create(builder, loc, budget),
        numItersV);

    // Clone buffers and shadows are single-entry caches, so they are popped
    // once, here, outside the loop. Each ref also gets a working clone the
    // reverse pass owns: the replay writes through the ref, so it must not
    // write into a checkpoint slot -- slot `capo` is re-read on the next
    // iteration whenever the remat placed finer checkpoints above it.
    SmallVector<Value> mutBufs, workClones;
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs)) {
      auto iface = cast<ClonableTypeInterface>(ref.getType());
      mutBufs.push_back(gutils->popCache(caches[layout.cloneBuf(r)], builder));
      workClones.push_back(
          iface.cloneValue(builder, gutils->getNewFromOriginal(ref)));
      mapping.map(ref, workClones.back());
      if (layout.mutableActive[r])
        gutils->invertedPointers.map(
            ref, gutils->popCache(caches[layout.shadow(r)], builder));
    }

    // Outer reverse loop over all N steps; carries (sp, adjoints...).
    SmallVector<Value> outerInit;
    outerInit.push_back(budgetV); // live checkpoint count
    outerInit.append(incomingGradients.begin(), incomingGradients.end());

    auto revOuter =
        scf::ForOp::create(builder, loc, c0, numItersV, c1, outerInit);
    preserveAttributesButCheckpointing(revOuter, forOp);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(revOuter.getBody());

    Value ivO = revOuter.getInductionVar();
    Value sp = revOuter.getBody()->getArgument(1);
    auto adjArgs = revOuter.getBody()->getArguments().drop_front(2);

    Value capo = arith::SubIOp::create(builder, loc, sp, c1);
    Value currentRevStep = arith::SubIOp::create(builder, loc, numItersV, ivO);

    // Load the top checkpoint state + its forward step.
    SmallVector<Value> ckptState;
    for (auto &&[buf, arg] : llvm::zip_equal(
             ckptBufs, forOp.getBody()->getArguments().drop_front()))
      ckptState.push_back(
          loadCheckpoint(builder, loc, buf, capo, arg.getType()));
    Value ckptStep =
        memref::LoadOp::create(builder, loc, idxBuf, ValueRange{capo});

    // Re-prime each working clone from the snapshot paired with slot `capo`, so
    // the replay below starts from the mutable memory as it was at ckptStep.
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs))
      cast<ClonableTypeInterface>(ref.getType())
          .copyValue(builder, workClones[r],
                     cloneSlot(builder, loc, mutBufs[r], capo));

    // Inner remat scf.while: reconstruct state at (currentRevStep - 1),
    // carrying (pos, capo, state...).
    SmallVector<Value> whileInit;
    whileInit.push_back(ckptStep);
    whileInit.push_back(capo);
    whileInit.append(ckptState.begin(), ckptState.end());
    SmallVector<Type> whileTypes =
        llvm::to_vector(ValueRange(whileInit).getTypes());
    SmallVector<Location> whileLocs(whileInit.size(), loc);

    auto revWhile = scf::WhileOp::create(builder, loc, whileTypes, whileInit);
    {
      Block *before =
          builder.createBlock(&revWhile.getBefore(), {}, whileTypes, whileLocs);
      builder.setInsertionPointToEnd(before);
      Value pos = before->getArgument(0);
      Value posPlus1 = arith::AddIOp::create(builder, loc, pos, c1);
      Value cond = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::slt, posPlus1, currentRevStep);
      scf::ConditionOp::create(builder, loc, cond, before->getArguments());
    }
    {
      Block *after =
          builder.createBlock(&revWhile.getAfter(), {}, whileTypes, whileLocs);
      builder.setInsertionPointToEnd(after);
      Value pos = after->getArgument(0);
      Value acapo = after->getArgument(1);
      auto astate = after->getArguments().drop_front(2);

      Value remaining =
          arith::SubIOp::create(builder, loc, currentRevStep, pos);
      Value budgetRem = arith::SubIOp::create(builder, loc, budgetV, acapo);
      // Never use more checkpoints than remaining steps (binomial_progress is
      // degenerate for budget > steps).
      budgetRem = arith::MinUIOp::create(builder, loc, budgetRem, remaining);
      Value split = enzyme::BinomialProgressOp::create(
          builder, loc, builder.getIndexType(), remaining, budgetRem);

      // Place a checkpoint at slot `acapo`. The mutable-ref snapshot has to go
      // with it: the working clones currently hold the content at step `pos`
      // (innerRemat below advances them only after this store), and a slot
      // whose state and snapshot came from different steps replays wrongly.
      for (auto &&[buf, val] : llvm::zip_equal(ckptBufs, astate))
        storeCheckpoint(builder, loc, buf, acapo, val);
      memref::StoreOp::create(builder, loc, pos, idxBuf, ValueRange{acapo});
      for (auto &&[r, ref] : llvm::enumerate(mutableRefs))
        cast<ClonableTypeInterface>(ref.getType())
            .copyValue(builder, cloneSlot(builder, loc, mutBufs[r], acapo),
                       workClones[r]);

      Value posPlusSplit = arith::AddIOp::create(builder, loc, pos, split);
      Value isLast = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, posPlusSplit, currentRevStep);
      Value rematUB = arith::SelectOp::create(
          builder, loc, isLast,
          arith::SubIOp::create(builder, loc, posPlusSplit, c1), posPlusSplit);

      // Recompute the primal from `pos` to `rematUB`.
      auto innerRemat =
          scf::ForOp::create(builder, loc, pos, rematUB, c1,
                             SmallVector<Value>(astate.begin(), astate.end()));
      preserveAttributesButCheckpointing(innerRemat, forOp);
      if (!innerRemat.getBody()->empty())
        innerRemat.getBody()->front().erase();

      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(innerRemat.getBody());
        Value idx = castToType(builder, loc, innerRemat.getInductionVar(),
                               stepV.getType());
        Value iv = arith::AddIOp::create(
            builder, loc, startV,
            arith::MulIOp::create(builder, loc, stepV, idx));

        for (auto &&[oldArg, newArg] :
             llvm::zip_equal(forOp.getBody()->getArguments().drop_front(),
                             innerRemat.getBody()->getArguments().drop_front()))
          mapping.map(oldArg, newArg);
        mapping.map(forOp.getInductionVar(), iv);

        copyBlockWithoutTerminator(builder, forOp.getBody(), gutils, mapping);

        SmallVector<Value> yields;
        for (auto operand : forOp.getBody()->getTerminator()->getOperands())
          yields.push_back(mapping.lookupOrDefault(operand));
        scf::YieldOp::create(builder, loc, yields);
      }

      Value newCapo = arith::AddIOp::create(builder, loc, acapo, c1);
      SmallVector<Value> afterYields;
      afterYields.push_back(posPlusSplit);
      afterYields.push_back(newCapo);
      afterYields.append(innerRemat.getResults().begin(),
                         innerRemat.getResults().end());
      scf::YieldOp::create(builder, loc, afterYields);
    }

    builder.setInsertionPointToEnd(revOuter.getBody());
    Value newSp = revWhile.getResult(1);
    auto reconState = revWhile.getResults().drop_front(2);

    // Adjoint of a single body step at (currentRevStep - 1).
    Value stepAdj = arith::SubIOp::create(builder, loc, currentRevStep, c1);
    Value stepAdjC = castToType(builder, loc, stepAdj, stepV.getType());
    Value ivAdj = arith::AddIOp::create(
        builder, loc, startV,
        arith::MulIOp::create(builder, loc, stepV, stepAdjC));

    mapping = IRMapping();

    for (auto [ref, cached] :
         llvm::zip_equal(immutableRefs, immutableRefsCaches))
      mapping.map(ref, cached);

    // Re-bind after the mapping reset above; the shadows were bound once,
    // before revOuter, and do not need rebinding per iteration.
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs))
      mapping.map(ref, workClones[r]);

    for (auto &&[oldArg, newArg] : llvm::zip_equal(
             forOp.getBody()->getArguments().drop_front(), reconState))
      mapping.map(oldArg, newArg);
    mapping.map(forOp.getInductionVar(), ivAdj);

    // Re-materialize primal ops of this step for the reverse visitor.
    copyBlockWithoutTerminator(builder, forOp.getBody(), gutils, mapping);

    // forceAugmentedReturns() seeded invertedPointers with one PlaceholderOp
    // per active mutable value, positioned in the single augmented primal --
    // which cacheBinomial has since erased, leaving those entries dangling (a
    // later invertPointerM() would hand out freed IR). Re-seed a placeholder
    // per body value here, in this per-step reconstruction: that is where the
    // shadow that replaces it legitimately lives, and where the per-iteration
    // popCache shadows of the outside refs (bound above) dominate it. Hoisting
    // the originals out of the loop instead cannot work, precisely because
    // those source shadows are per-iteration values inside revOuter.
    SmallVector<Value> reseededShadowKeys;
    {
      OpBuilder::InsertionGuard g4(builder);
      forOp.getBody()->walk([&](Operation *inner) {
        for (Value res : inner->getResults()) {
          if (gutils->isConstantValue(res))
            continue;
          Type shadowTy = gutils->getShadowType(res.getType());
          auto iface = dyn_cast<AutoDiffTypeInterface>(shadowTy);
          if (!iface || !iface.isMutable())
            continue;
          Value newRes = mapping.lookupOrNull(res);
          if (!newRes)
            continue;
          if (Operation *defOp = newRes.getDefiningOp())
            builder.setInsertionPointAfter(defOp);
          else
            continue;
          auto ph =
              enzyme::PlaceholderOp::create(builder, res.getLoc(), shadowTy);
          gutils->invertedPointers.map(res, ph);
          reseededShadowKeys.push_back(res);
        }
      });
    }

    // Reset every (non-mutable) intermediate gradient slot to zero at the start
    // of each reverse step and zero the diffe of the yielded operands; the
    // loop-carried gradient is supplied via the outer carried adjoints. Without
    // this, scalar gradient slots (e.g. the diffe of a value loaded from an
    // enzyme_dup'ed memref) leak across reverse iterations and over-accumulate
    // into the external shadow. Mirrors the non-checkpointed reverse path.
    auto term = forOp.getBody()->getTerminator();
    {
      OpBuilder::InsertionGuard g3(builder);
      builder.setInsertionPointToStart(revOuter.getBody());
      mlir::enzyme::localizeGradients(builder, gutils, forOp.getBody());
    }
    for (auto &&[active, operand] :
         llvm::zip_equal(operandsActive, term->getOperands())) {
      if (active)
        gutils->zeroDiffe(operand, builder);
    }

    // Seed adjoints of the yielded operands from the outer carried gradients.
    unsigned revIdx = 0;
    for (auto &&[active, operand] :
         llvm::zip_equal(operandsActive, term->getOperands())) {
      if (active) {
        gutils->addToDiffe(operand, adjArgs[revIdx], builder);
        revIdx++;
      }
    }

    bool valid = true;
    auto first = forOp.getBody()->rbegin();
    first++; // skip terminator
    auto last = forOp.getBody()->rend();
    for (auto it = first; it != last; ++it)
      valid &= gutils->Logic.visitChild(&*it, builder, gutils).succeeded();

    // Placeholders re-seeded above are consumed by setInvertedPointer (which
    // RAUWs and erases them) only for values some rule actually asked to
    // invert. Drop the unused remainder rather than leaving enzyme.placeholder
    // litter in the output -- keyed by the original value, so invertedPointers
    // never keeps an entry pointing at IR we just erased. A resolved entry no
    // longer names a PlaceholderOp, which is what distinguishes it (its op is
    // already gone, so we must not dereference the recorded pointer).
    for (Value orig : reseededShadowKeys) {
      Value cur = gutils->invertedPointers.lookupOrNull(orig);
      if (!cur)
        continue;
      auto ph = cur.getDefiningOp<enzyme::PlaceholderOp>();
      if (ph && ph->use_empty()) {
        gutils->invertedPointers.erase(orig);
        ph->erase();
      }
    }

    SmallVector<Value> newAdjoints;
    for (auto &&[active, arg] : llvm::zip_equal(
             operandsActive, forOp.getBody()->getArguments().drop_front())) {
      if (active) {
        newAdjoints.push_back(gutils->diffe(arg, builder));
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, builder);
      }
    }

    SmallVector<Value> outerYields;
    outerYields.push_back(newSp);
    outerYields.append(newAdjoints.begin(), newAdjoints.end());
    scf::YieldOp::create(builder, loc, outerYields);

    builder.setInsertionPointAfter(revOuter);

    revIdx = 0;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, forOp.getInitArgs())) {
      if (active) {
        if (!gutils->isConstantValue(arg))
          gutils->addToDiffe(arg, revOuter.getResult(revIdx + 1), builder);
        revIdx++;
      }
    }

    // Free checkpoint buffers, index buffer, and cloned mutable refs.
    for (auto buf : ckptBufs)
      memref::DeallocOp::create(builder, loc, buf);
    memref::DeallocOp::create(builder, loc, idxBuf);
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs)) {
      auto iface = cast<ClonableTypeInterface>(ref.getType());
      iface.freeClonedValue(builder, workClones[r]);
      freeCloneSlots(builder, loc, budget, mutBufs[r], iface);
    }

    return success(valid);
  }

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

    if (needsBinomialCheckpointing(forOp)) {
      auto budget = getCheckpointBudget(forOp);
      if (!budget || *budget <= 1) {
        op->emitError() << "binomial checkpointing requires a "
                           "enzyme.checkpoint_period attribute greater than 1";
        return failure();
      }
      return reverseBinomial(forOp, *budget, builder, gutils, caches,
                             operandsActive, incomingGradients);
    }

    if (needsCheckpointing(forOp)) {
      int64_t numIters =
          ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).value();
      int64_t nInner = std::sqrt(numIters), nOuter = nInner;
      int64_t trailingIters = numIters - nInner * nOuter;

      bool hasTrailing = trailingIters > 0;

      auto numIterArgs = forOp.getNumRegionIterArgs();

      SetVector<Value> outsideRefs;
      getUsedValuesDefinedAbove(op->getRegions(), outsideRefs);

      SmallVector<Value> immutableRefs;
      SmallVector<Value> mutableRefs;

      for (auto ref : outsideRefs) {
        if (isa<ClonableTypeInterface>(ref.getType()))
          mutableRefs.push_back(ref);
        else
          immutableRefs.push_back(ref);
      }

      IRMapping &mapping = gutils->originalToNewFn;

      assert(outsideRefs.size() == caches.size() - numIterArgs);

      for (auto [i, ref] : llvm::enumerate(immutableRefs)) {
        Value refVal = gutils->popCache(
            caches[numIterArgs + mutableRefs.size() + i], builder);
        mapping.map(ref, refVal);
      }

      auto ivTy = forOp.getLowerBound().getType();
      Value outerUB = makeIntConstant(forOp.getLowerBound().getLoc(), builder,
                                      nOuter + hasTrailing, ivTy);
      auto revOuter = scf::ForOp::create(
          builder, op->getLoc(),
          makeIntConstant(forOp.getLowerBound().getLoc(), builder, 0, ivTy),
          outerUB,
          makeIntConstant(forOp.getLowerBound().getLoc(), builder, 1, ivTy),
          incomingGradients);
      preserveAttributesButCheckpointing(revOuter, forOp);

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(revOuter.getBody());

      SmallVector<Value> cachedOutsideRefs;
      for (auto [i, ref] : llvm::enumerate(mutableRefs)) {
        Value refVal = gutils->popCache(caches[numIterArgs + i], builder);
        cachedOutsideRefs.push_back(refVal);
        mapping.map(ref, refVal);
      }

      Location loc = forOp.getInductionVar().getLoc();
      Value currentOuterStep = arith::SubIOp::create(
          builder, loc, makeIntConstant(loc, builder, nOuter, ivTy),
          revOuter.getInductionVar());

      SmallVector<Value> initArgs(numIterArgs, nullptr);
      for (size_t i = 0; i < numIterArgs; ++i) {
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
        nInnerUB = arith::SelectOp::create(
            builder, loc,
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                  revOuter.getInductionVar(), zero),
            makeIntConstant(loc, builder, trailingIters, ivTy), nInnerCst);
      }

      auto revInner = scf::ForOp::create(builder, forOp.getLoc(), zero,
                                         nInnerUB, one, initArgs);
      preserveAttributesButCheckpointing(revInner, forOp);

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

      Value currentIV = arith::AddIOp::create(
          builder, loc,
          arith::MulIOp::create(
              builder, loc,
              arith::AddIOp::create(builder, loc,
                                    arith::MulIOp::create(builder, loc,
                                                          currentOuterStep,
                                                          nInnerCst),
                                    revInner.getInductionVar()),
              arith::ConstantOp::create(builder, loc,
                                        IntegerAttr::get(ivTy, stepI))),
          arith::ConstantOp::create(builder, loc,
                                    IntegerAttr::get(ivTy, startI)));

      // Re-materialize this segment's primal for the reverse visitor (below,
      // in revLoop), and hand it that clone as forOp's body. The terminator is
      // cloned along with the rest: it terminates revInner.
      IRMapping segmentMap = bodyCloneMapping(
          forOp, revInner.getBody()->getArguments().drop_front(), currentIV,
          mapping);
      for (Operation &it : *forOp.getBody())
        builder.clone(it, segmentMap);
      publishClonedStep(forOp, segmentMap, gutils);

      builder.setInsertionPointToEnd(revOuter.getBody());

      for (auto outsideRef : cachedOutsideRefs) {
        if (auto cachableT =
                dyn_cast<ClonableTypeInterface>(outsideRef.getType())) {
          cachableT.freeClonedValue(builder, outsideRef);
        }
      }

      auto revLoop =
          scf::ForOp::create(builder, forOp.getLoc(), zero, nInnerUB, one,
                             revOuter.getBody()->getArguments().drop_front());
      preserveAttributesButCheckpointing(revLoop, forOp);

      Block *revLoopBody = revLoop.getBody();
      Block *origBody = forOp.getBody();

      // Reset every (non-mutable) intermediate gradient slot to zero at the
      // start of each reverse iteration and zero the diffe of the yielded
      // operands: the loop-carried gradient is supplied via the iter_arg. This
      // mirrors the non-checkpointed reverse path below. Without it, scalar
      // gradient slots such as the diffe of a value loaded from an
      // enzyme_dup'ed memref leak across reverse iterations, get promoted to
      // loop-carried iter_args, and over-accumulate into the external shadow.
      builder.setInsertionPointToStart(revLoopBody);
      mlir::enzyme::localizeGradients(builder, gutils, origBody);

      builder.setInsertionPointToEnd(revLoopBody);
      for (auto &&[active, operand] : llvm::zip_equal(
               operandsActive, origBody->getTerminator()->getOperands())) {
        if (active)
          gutils->zeroDiffe(operand, builder);
      }

      int revIdx = 1;
      for (auto &&[active, operand] : llvm::zip_equal(
               operandsActive, origBody->getTerminator()->getOperands())) {
        if (active) {
          gutils->addToDiffe(operand, revLoopBody->getArgument(revIdx),
                             builder);
          revIdx++;
        }
      }

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
      scf::YieldOp::create(builder, forOp.getBody()->getTerminator()->getLoc(),
                           newResults);

      builder.setInsertionPointToEnd(revOuter.getBody());
      scf::YieldOp::create(builder, forOp.getBody()->getTerminator()->getLoc(),
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
    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);

    if (getenv("ENZYME_DEBUG_REVERSE_BINOMIAL"))
      llvm::errs() << "[ForOpInterfaceReverse::cacheValues] ENTER op=" << op
                   << " needsBinomial=" << needsBinomialCheckpointing(forOp)
                   << " needsPeriodic=" << needsCheckpointing(forOp) << "\n";

    if (needsBinomialCheckpointing(forOp)) {
      auto budget = getCheckpointBudget(forOp);
      if (!budget || *budget <= 1) {
        // Error is reported in createReverseModeAdjoint; fall back to caching
        // the bounds so the reverse pass can proceed to emit the diagnostic.
        SmallVector<Value> caches;
        caches.push_back(gutils->initAndPushCache(
            gutils->getNewFromOriginal(forOp.getLowerBound()), cacheBuilder));
        caches.push_back(gutils->initAndPushCache(
            gutils->getNewFromOriginal(forOp.getUpperBound()), cacheBuilder));
        caches.push_back(gutils->initAndPushCache(
            gutils->getNewFromOriginal(forOp.getStep()), cacheBuilder));
        return caches;
      }
      return cacheBinomial(forOp, *budget, gutils);
    }

    if (needsCheckpointing(forOp)) {
      int64_t numIters =
          ForOpEnzymeOpsRemover::getConstantNumberOfIterations(forOp).value();
      int64_t nInner = std::sqrt(numIters), nOuter = nInner;
      int64_t trailingIters = numIters - nInner * nOuter;
      bool hasTrailing = trailingIters > 0;

      SetVector<Value> outsideRefs;
      getUsedValuesDefinedAbove(op->getRegions(), outsideRefs);

      SmallVector<Value> immutableRefs;
      SmallVector<Value> mutableRefs;

      for (auto ref : outsideRefs) {
        if (isa<ClonableTypeInterface>(ref.getType()))
          mutableRefs.push_back(ref);
        else
          immutableRefs.push_back(ref);
      }

      SmallVector<Value> caches;

      scf::ForOp newForOp = cast<scf::ForOp>(gutils->getNewFromOriginal(op));

      Type ty = forOp.getLowerBound().getType();
      auto outerFwd = scf::ForOp::create(
          cacheBuilder, op->getLoc(),
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
        nInnerUB = arith::SelectOp::create(
            cacheBuilder, loc,
            arith::CmpIOp::create(
                cacheBuilder, loc, arith::CmpIPredicate::eq,
                outerFwd.getInductionVar(),
                makeIntConstant(loc, cacheBuilder, nInner * nOuter, ty)),
            makeIntConstant(loc, cacheBuilder, trailingIters, ty), nInnerCst);
      }

      IRMapping &mapping = gutils->originalToNewFn;

      SmallVector<Value> mutableRefsCaches;
      for (auto ref : mutableRefs) {
        auto iface = cast<ClonableTypeInterface>(ref.getType());
        auto clone =
            iface.cloneValue(cacheBuilder, mapping.lookupOrDefault(ref));
        mutableRefsCaches.push_back(
            gutils->initAndPushCache(clone, cacheBuilder));
      }

      auto innerFwd = scf::ForOp::create(
          cacheBuilder, op->getLoc(),
          makeIntConstant(forOp.getLowerBound().getLoc(), cacheBuilder, 0, ty),
          nInnerUB,
          makeIntConstant(forOp.getStep().getLoc(), cacheBuilder, 1, ty),
          outerFwd.getBody()->getArguments().drop_front());
      preserveAttributesButCheckpointing(innerFwd, forOp);

      cacheBuilder.setInsertionPointToEnd(innerFwd.getBody());

      Location loc = forOp.getInductionVar().getLoc();
      auto currentIV = arith::MulIOp::create(
          cacheBuilder, loc,
          arith::AddIOp::create(cacheBuilder, loc, outerFwd.getInductionVar(),
                                innerFwd.getInductionVar()),
          newForOp.getStep());

      // As in cacheBinomial: a forward simulation, so publish its ops (theirs
      // would otherwise stay pointing into the copy erased below) but not its
      // values -- the reverse re-clones the segment it works against. The
      // terminator is cloned along with the rest: it terminates innerFwd.
      IRMapping fwdMap = bodyCloneMapping(
          forOp, innerFwd.getBody()->getArguments().drop_front(), currentIV,
          mapping);
      for (Operation &it : *forOp.getBody())
        cacheBuilder.clone(it, fwdMap);
      publishClonedOps(fwdMap, gutils);

      cacheBuilder.setInsertionPointToEnd(outerFwd.getBody());
      for (auto initArg : innerFwd.getInitArgs())
        caches.push_back(gutils->initAndPushCache(initArg, cacheBuilder));

      scf::YieldOp::create(cacheBuilder,
                           forOp.getBody()->getTerminator()->getLoc(),
                           innerFwd->getResults());

      cacheBuilder.setInsertionPointAfter(outerFwd);

      caches.append(mutableRefsCaches);

      for (auto ref : immutableRefs)
        caches.push_back(gutils->initAndPushCache(mapping.lookupOrDefault(ref),
                                                  cacheBuilder));

      gutils->replaceOrigOpWith(op, outerFwd.getResults());
      hoistPlaceholdersBefore(newForOp, newForOp);
      gutils->erase(newForOp);
      gutils->originalToNewFnOps[op] = outerFwd;

      // caches is composed of:
      // [
      //  <caches of iter args>...,
      //  <caches of mutable values>...,
      //  <caches of immutable values>...,
      // ]
      //
      // TODO: we don't need to cache refs of arith.constants
      // ....  which we can "clone" just before the inner forward
      // ....  in the reverse pass.
      // ....  create an interface that mincut can also use?

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
