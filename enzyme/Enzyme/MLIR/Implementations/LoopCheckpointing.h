//===- LoopCheckpointing.h - Generic loop checkpointing --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Op generic implementation of checkpointing.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_IMPLEMENTATIONS_LOOPCHECKPOINTING_H
#define ENZYME_MLIR_IMPLEMENTATIONS_LOOPCHECKPOINTING_H

#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Passes/RemovalUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <array>
#include <cmath>
#include <optional>

namespace mlir {
namespace enzyme {

template <typename FinalClass, typename OpName> struct LoopCheckpointing {
  // A loop created while differentiating `oldOp` is still doing that loop's
  // work, so it inherits what was set on it. The checkpointing directives are
  // left behind, since the rewrite has already acted on them.
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

  static bool hasBinomialAttr(OpName forOp) {
    return forOp->hasAttr("enzyme.binomial_checkpointing");
  }

  static bool needsCheckpointing(OpName forOp) {
    return forOp->template hasAttrOfType<BoolAttr>(
               "enzyme.enable_checkpointing") &&
           forOp
               ->template getAttrOfType<BoolAttr>("enzyme.enable_checkpointing")
               .getValue() &&
           !hasBinomialAttr(forOp) &&
           FinalClass::getConstantNumberOfIterations(forOp).has_value();
  }

  static bool needsBinomialCheckpointing(OpName forOp) {
    return forOp->template hasAttrOfType<BoolAttr>(
               "enzyme.enable_checkpointing") &&
           forOp
               ->template getAttrOfType<BoolAttr>("enzyme.enable_checkpointing")
               .getValue() &&
           hasBinomialAttr(forOp);
  }

  static std::optional<int64_t> getCheckpointBudget(OpName forOp) {
    if (auto a = forOp->template getAttrOfType<IntegerAttr>(
            "enzyme.checkpoint_period"))
      return a.getInt();
    return std::nullopt;
  }

  // The "init" operands used as initialization values for the loop's
  // iter_args. No FinalClass hook needed: scf::ForOp exposes this correctly
  // through LoopLikeOpInterface (it overrides getInitsMutable), and
  // affine::AffineForOp exposes its own `$inits`-derived accessor of the
  // same name, which -- being declared directly on the op -- takes priority
  // over (and happens to be the one actually wired up, unlike) the
  // interface's own default. Both resolve correctly when called directly on
  // a concrete OpName.
  static ValueRange getInits(OpName forOp) { return forOp.getInits(); }

  static Value getNumIterationsValue(OpBuilder &builder, Location loc,
                                     OpName forOp,
                                     MGradientUtilsReverse *gutils) {
    Value lb = FinalClass::materializeLowerBound(builder, loc, forOp, gutils);
    Value ub = FinalClass::materializeUpperBound(builder, loc, forOp, gutils);
    Value step = FinalClass::materializeStep(builder, loc, forOp, gutils);
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

  // Read a snapshot from checkpoint buffer slot `slot`. For scalars returns
  // the loaded value; for memrefs returns a fresh alloc initialized from the
  // row (the caller is responsible for deallocating it).
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
      FinalClass::cloneOp(builder, it, mapping);
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
  // just be forwarded. cacheBinomial and reverseBinomial must produce the
  // same order, which they do: getUsedValuesDefinedAbove is deterministic for
  // a fixed IR and both walk the *original* forOp.
  static void splitOutsideRefs(OpName forOp,
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

    static BinomialCacheLayout get(OpName forOp, ArrayRef<Value> mutableRefs,
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

  static Value cloneSlot(OpBuilder &b, Location loc, Value buf, Value slot) {
    return memref::LoadOp::create(b, loc, buf, ValueRange{slot});
  }

  // A `budget`-slot buffer of clone *handles* for one mutable ref, with a
  // clone of `proto` already in every slot. Not checkpointBufferType: what
  // lives here is the identity of a snapshot allocation (slot j pairs with
  // ckptBufs slot j), not its contents -- for a bare pointer the extent of
  // the contents is not in the type at all.
  //
  // The clones are made first because they are what the buffer is typed
  // from: a clone does not have to have the type of what it clones (a
  // pointer whose size hint annotates a memory space is cloned into that
  // space), and a buffer built from the ref's type could then neither hold
  // nor free a handle.
  //
  // Deliberately unrolled: `budget` is static, and an allocation sitting in a
  // loop body is both a hoisting candidate and (for pointers, under a raised
  // alloca threshold) an alloca-promotion candidate -- either would silently
  // make all slots alias.
  static Value allocCloneSlots(OpBuilder &b, Location loc, int64_t budget,
                               Value proto, ClonableTypeInterface iface) {
    SmallVector<Value> clones;
    for (int64_t j = 0; j < budget; ++j)
      clones.push_back(iface.cloneValue(b, proto));

    Value buf = memref::AllocOp::create(
        b, loc, MemRefType::get({budget}, clones.front().getType()));
    for (auto &&[j, clone] : llvm::enumerate(clones)) {
      Value slot = arith::ConstantIndexOp::create(b, loc, j);
      memref::StoreOp::create(b, loc, clone, buf, ValueRange{slot});
    }
    return buf;
  }

  // Free each slot's clone, then the handle buffer itself. A loop is fine
  // here: a free has side effects so it cannot be hoisted, and nothing can
  // alias.
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

  // Forward augmentation for binomial (Revolve) checkpointing. Builds an
  // outer loop of `budget` iterations that snapshots the loop state into
  // memref checkpoint buffers at Revolve-scheduled positions, advancing the
  // primal in an inner recompute loop between snapshots. Returns the caches
  // (buffer handles + index buffer + outside refs) transported to the
  // reverse pass; see BinomialCacheLayout for their order.
  static SmallVector<Value> cacheBinomial(OpName forOp, int64_t budget,
                                          MGradientUtilsReverse *gutils) {
    Location loc = forOp.getLoc();
    bool isDynamic =
        !FinalClass::getConstantNumberOfIterations(forOp).has_value();

    auto newForOp = cast<OpName>(gutils->getNewFromOriginal(forOp));
    OpBuilder builder(newForOp);
    Type idxTy = builder.getIndexType();

    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

    // Loop trip count / lower bound / step as index values (constant-folded
    // when the bounds are constant).
    Value numItersV, startV, stepV;
    if (isDynamic) {
      startV = FinalClass::materializeLowerBound(builder, loc, forOp, gutils);
      stepV = FinalClass::materializeStep(builder, loc, forOp, gutils);
      numItersV = getNumIterationsValue(builder, loc, forOp, gutils);
      numItersV = castToType(builder, loc, numItersV, idxTy);
    } else {
      int64_t numIters =
          FinalClass::getConstantNumberOfIterations(forOp).value();
      numItersV = arith::ConstantIndexOp::create(builder, loc, numIters);
      startV = arith::ConstantIndexOp::create(
          builder, loc, FinalClass::getConstantStart(forOp));
      stepV = arith::ConstantIndexOp::create(
          builder, loc, FinalClass::getConstantStep(forOp));
    }

    // Effective budget = min(requested budget, trip count): never keep more
    // checkpoints than there are iterations. Buffers stay sized by the
    // (static) requested budget; the effective budget bounds the loops at
    // runtime.
    //
    // Unconditional: with a budget above the trip count this loop would run
    // more iterations than there are steps, and the slots past the end get
    // recorded at a step beyond the last one -- holding the final state
    // instead of a checkpoint, which the reverse pass then replays from.
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
    for (auto arg : getInits(newForOp)) {
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
      Value buf = allocCloneSlots(builder, loc, budget,
                                  gutils->getNewFromOriginal(ref), iface);
      mutBufs.push_back(buf);
    }

    // Outer checkpoint-placement loop: for %k = 0 to budgetV carrying
    // (stepCtr, state...).
    SmallVector<Value> outerInit;
    outerInit.push_back(c0);
    auto newForOpInits = getInits(newForOp);
    outerInit.append(newForOpInits.begin(), newForOpInits.end());
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

    // Snapshot each mutable ref into slot `k`, reusing the clone already
    // there. Stays here, before innerFwd, so the snapshot precedes the
    // advance.
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
    // loop-invariant (it is the accumulating gradient buffer, not a
    // snapshot) and so does not belong in a per-checkpoint slot.
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs)) {
      caches.push_back(gutils->initAndPushCache(mutBufs[r], builder));
      if (layout.mutableActive[r])
        caches.push_back(gutils->initAndPushCache(
            gutils->invertPointerM(ref, builder), builder));
    }

    for (auto ref : immutableRefs)
      caches.push_back(
          gutils->initAndPushCache(gutils->getNewFromOriginal(ref), builder));

    // For dynamic bounds the reverse pass cannot recover the trip count /
    // lower bound / step from constants, so cache them (as the trailing
    // entries).
    if (isDynamic) {
      caches.push_back(gutils->initAndPushCache(numItersV, builder));
      caches.push_back(gutils->initAndPushCache(startV, builder));
      caches.push_back(gutils->initAndPushCache(stepV, builder));
    }
    assert(caches.size() == layout.size() && "binomial cache layout mismatch");

    // The primal result of the loop is the final state.
    gutils->replaceOrigOpWith(forOp, outerFwd.getResults().drop_front());
    gutils->erase(newForOp);
    gutils->originalToNewFnOps[forOp] = outerFwd;

    return caches;
  }

  // forceAugmentedReturns() (called once, early, over the whole original
  // function) plants an enzyme.placeholder for every mutable-typed value's
  // shadow, including ones nested inside a checkpointed forOp's augmented
  // copy. Those placeholders are meant to survive until the reverse pass
  // visits their defining op and replaces them via setInvertedPointer. But
  // checkpointing discards the whole augmented-forward copy of forOp (it's
  // redundant once the checkpoint-buffer reconstruction takes over) via
  // gutils->erase(...), which -- if it happens before that replacement runs
  // -- destroys the still-referenced placeholder out from under
  // invertedPointers, leaving a dangling entry that only crashes later,
  // whenever something finally reads it. Hoist any not-yet-replaced
  // placeholders out of the doomed subtree first so they survive.
  static void hoistPlaceholdersBefore(Operation *root, Operation *before) {
    SmallVector<enzyme::PlaceholderOp> placeholders;
    root->walk([&](enzyme::PlaceholderOp p) { placeholders.push_back(p); });
    for (auto p : placeholders)
      p->moveBefore(before);
  }

  // Reverse pass for binomial (Revolve) checkpointing. Iterates all N steps
  // in reverse; for each step it reconstructs the state just before that
  // step from the top checkpoint (recursively re-placing finer checkpoints
  // during the remat), then emits the adjoint of a single body step.
  static LogicalResult reverseBinomial(OpName forOp, int64_t budget,
                                       OpBuilder &builder,
                                       MGradientUtilsReverse *gutils,
                                       SmallVector<Value> caches,
                                       ArrayRef<bool> operandsActive,
                                       ArrayRef<Value> incomingGradients) {
    Location loc = forOp.getLoc();
    bool isDynamic =
        !FinalClass::getConstantNumberOfIterations(forOp).has_value();
    auto numIterArgs = forOp.getNumRegionIterArgs();

    SmallVector<Value> immutableRefs, mutableRefs;
    splitOutsideRefs(forOp, mutableRefs, immutableRefs);
    auto layout = BinomialCacheLayout::get(forOp, mutableRefs, immutableRefs,
                                           isDynamic, gutils);
    assert(caches.size() == layout.size() && "binomial cache layout mismatch");

    IRMapping mapping;

    // Below, each mutableRef's shadow is re-bound to the popped shadow
    // handle. That binding must not outlive this function: ops *outside* the
    // loop are visited after it (reverse order) and expect the caller's
    // shadow, not ours. Restore the previous bindings on the way out.
    SmallVector<std::pair<Value, Value>> savedMutableShadows;
    for (auto ref : mutableRefs)
      savedMutableShadows.emplace_back(
          ref, gutils->invertedPointers.lookupOrNull(ref));
    auto restoreMutableShadows = llvm::scope_exit([&]() {
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

    // Loop trip count / lower bound / step as index values. For dynamic
    // bounds these were cached by cacheBinomial (trailing entries, same
    // order).
    Value numItersV, startV, stepV;
    if (isDynamic) {
      numItersV = gutils->popCache(caches[layout.numIters()], builder);
      startV = gutils->popCache(caches[layout.start()], builder);
      stepV = gutils->popCache(caches[layout.step()], builder);
    } else {
      int64_t numIters =
          FinalClass::getConstantNumberOfIterations(forOp).value();
      numItersV = arith::ConstantIndexOp::create(builder, loc, numIters);
      startV = arith::ConstantIndexOp::create(
          builder, loc, FinalClass::getConstantStart(forOp));
      stepV = arith::ConstantIndexOp::create(
          builder, loc, FinalClass::getConstantStep(forOp));
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

    // Re-prime each working clone from the snapshot paired with slot `capo`,
    // so the replay below starts from the mutable memory as it was at
    // ckptStep.
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
      // Never use more checkpoints than remaining steps (binomial_progress
      // is degenerate for budget > steps).
      budgetRem = arith::MinUIOp::create(builder, loc, budgetRem, remaining);
      Value split = enzyme::BinomialProgressOp::create(
          builder, loc, builder.getIndexType(), remaining, budgetRem);

      // Place a checkpoint at slot `acapo`. The mutable-ref snapshot has to
      // go with it: the working clones currently hold the content at step
      // `pos` (innerRemat below advances them only after this store), and a
      // slot whose state and snapshot came from different steps replays
      // wrongly.
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
    // which cacheBinomial has since erased, leaving those entries dangling
    // (a later invertPointerM() would hand out freed IR). Re-seed a
    // placeholder per body value here, in this per-step reconstruction: that
    // is where the shadow that replaces it legitimately lives, and where the
    // per-iteration popCache shadows of the outside refs (bound above)
    // dominate it. Hoisting the originals out of the loop instead cannot
    // work, precisely because those source shadows are per-iteration values
    // inside revOuter.
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

    // Reset every (non-mutable) intermediate gradient slot to zero at the
    // start of each reverse step and zero the diffe of the yielded
    // operands; the loop-carried gradient is supplied via the outer carried
    // adjoints. Without this, scalar gradient slots (e.g. the diffe of a
    // value loaded from an enzyme_dup'ed memref) leak across reverse
    // iterations and over-accumulate into the external shadow. Mirrors the
    // non-checkpointed reverse path.
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

    // Seed adjoints of the yielded operands from the outer carried
    // gradients.
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
    // invert. Drop the unused remainder rather than leaving
    // enzyme.placeholder litter in the output -- keyed by the original
    // value, so invertedPointers never keeps an entry pointing at IR we just
    // erased. A resolved entry no longer names a PlaceholderOp, which is
    // what distinguishes it (its op is already gone, so we must not
    // dereference the recorded pointer).
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
    auto forOpInits = getInits(forOp);
    for (auto &&[active, arg] : llvm::zip_equal(operandsActive, forOpInits)) {
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

  //===--------------------------------------------------------------------===//
  // Periodic (sqrt-decomposition) checkpointing: decompose the N-iteration
  // loop into ~sqrt(N) outer segments, each holding down ~sqrt(N) primal
  // steps in a checkpoint; the reverse pass replays one segment at a time.
  //===--------------------------------------------------------------------===//

  static SmallVector<Value> cachePeriodic(OpName forOp, Operation *op,
                                          MGradientUtilsReverse *gutils) {
    int64_t numIters = FinalClass::getConstantNumberOfIterations(forOp).value();
    int64_t nInner = std::sqrt(numIters), nOuter = nInner;
    int64_t trailingIters = numIters - nInner * nOuter;
    bool hasTrailing = trailingIters > 0;

    Operation *newOpBase = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOpBase);
    Location loc = forOp.getLoc();

    SetVector<Value> outsideRefs;
    getUsedValuesDefinedAbove(op->getRegions(), outsideRefs);

    SmallVector<Value> immutableRefs, mutableRefs;
    splitOutsideRefs(forOp, mutableRefs, immutableRefs);

    SmallVector<Value> caches;

    OpName newForOp = cast<OpName>(newOpBase);

    auto newForOpInits = getInits(newForOp);
    auto outerFwd = FinalClass::createConstantScaffoldLoop(
        cacheBuilder, loc, 0, nInner * (nOuter + hasTrailing), nInner,
        newForOpInits);
    preserveAttributesButCheckpointing(outerFwd, forOp);

    cacheBuilder.setInsertionPointToStart(outerFwd.getBody());

    IRMapping mapping;

    // The bound computation (for scf.for) must happen before the
    // mutable-ref cloning loop below, and the loop itself must be created
    // after it, matching the original code's exact order: canonicalize does
    // not freely reorder ops, so this keeps scf.for's output byte-identical.
    Value fwdBoundHint = FinalClass::computeForwardSegmentBound(
        cacheBuilder, loc, outerFwd.getInductionVar(), nInner, nOuter,
        trailingIters);

    SmallVector<Value> mutableRefsCaches;
    for (auto ref : mutableRefs) {
      auto iface = cast<ClonableTypeInterface>(ref.getType());
      auto clone =
          iface.cloneValue(cacheBuilder, gutils->getNewFromOriginal(ref));
      mutableRefsCaches.push_back(
          gutils->initAndPushCache(clone, cacheBuilder));
    }

    auto innerFwd = FinalClass::createForwardSegmentLoop(
        cacheBuilder, loc, outerFwd.getInductionVar(), fwdBoundHint, nInner,
        nOuter, trailingIters, outerFwd.getBody()->getArguments().drop_front());
    preserveAttributesButCheckpointing(innerFwd, forOp);

    cacheBuilder.setInsertionPointToEnd(innerFwd.getBody());

    Value currentIV = FinalClass::computeForwardSegmentIV(
        cacheBuilder, loc, forOp, outerFwd.getInductionVar(),
        innerFwd.getInductionVar());

    for (auto [oldArg, newArg] :
         llvm::zip_equal(newForOp.getBody()->getArguments(),
                         innerFwd.getBody()->getArguments()))
      mapping.map(oldArg, newArg);

    mapping.map(newForOp.getInductionVar(), currentIV);

    copyBlockWithoutTerminator(cacheBuilder, newForOp.getBody(), gutils,
                               mapping);

    Operation *fwdTerm = newForOp.getBody()->getTerminator();
    SmallVector<Value> fwdYields;
    for (auto operand : fwdTerm->getOperands())
      fwdYields.push_back(mapping.lookupOrDefault(operand));
    FinalClass::createScaffoldYield(cacheBuilder, fwdTerm->getLoc(), fwdYields);

    cacheBuilder.setInsertionPointToEnd(outerFwd.getBody());
    for (auto initArg : getInits(innerFwd))
      caches.push_back(gutils->initAndPushCache(initArg, cacheBuilder));

    FinalClass::createScaffoldYield(cacheBuilder,
                                    forOp.getBody()->getTerminator()->getLoc(),
                                    innerFwd->getResults());

    cacheBuilder.setInsertionPointAfter(outerFwd);

    caches.append(mutableRefsCaches);

    for (auto ref : immutableRefs)
      caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(ref),
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
    return caches;
  }

  static LogicalResult reversePeriodic(OpName forOp, Operation *op,
                                       OpBuilder &builder,
                                       MGradientUtilsReverse *gutils,
                                       SmallVector<Value> caches,
                                       ArrayRef<bool> operandsActive,
                                       ArrayRef<Value> incomingGradients) {
    int64_t numIters = FinalClass::getConstantNumberOfIterations(forOp).value();
    int64_t nInner = std::sqrt(numIters), nOuter = nInner;
    int64_t trailingIters = numIters - nInner * nOuter;
    bool hasTrailing = trailingIters > 0;

    auto numIterArgs = forOp.getNumRegionIterArgs();

    SetVector<Value> outsideRefs;
    getUsedValuesDefinedAbove(op->getRegions(), outsideRefs);

    SmallVector<Value> immutableRefs, mutableRefs;
    splitOutsideRefs(forOp, mutableRefs, immutableRefs);

    IRMapping mapping;

    assert(outsideRefs.size() == caches.size() - numIterArgs);

    for (auto [i, ref] : llvm::enumerate(immutableRefs)) {
      Value refVal = gutils->popCache(
          caches[numIterArgs + mutableRefs.size() + i], builder);
      mapping.map(ref, refVal);
    }

    Location loc = forOp.getLoc();
    auto [revOuterLB, revOuterUB, revOuterStep] =
        FinalClass::getReverseOuterScaffoldBounds(nInner, nOuter,
                                                  trailingIters);
    auto revOuter = FinalClass::createConstantScaffoldLoop(
        builder, loc, revOuterLB, revOuterUB, revOuterStep, incomingGradients);
    preserveAttributesButCheckpointing(revOuter, forOp);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(revOuter.getBody());

    SmallVector<Value> cachedOutsideRefs;
    for (auto [i, ref] : llvm::enumerate(mutableRefs)) {
      Value refVal = gutils->popCache(caches[numIterArgs + i], builder);
      cachedOutsideRefs.push_back(refVal);
      mapping.map(ref, refVal);
    }

    // Must happen here, before the initArgs pop loop below, matching the
    // original code's exact order (canonicalize does not freely reorder
    // ops, so this keeps scf.for's output byte-identical).
    Value revBoundHint = FinalClass::computeReverseSegmentBound(
        builder, loc, revOuter.getInductionVar(), nInner, nOuter,
        trailingIters);

    SmallVector<Value> initArgs(numIterArgs, nullptr);
    for (size_t i = 0; i < numIterArgs; ++i) {
      initArgs[i] = gutils->popCache(caches[i], builder);
    }

    auto revInner = FinalClass::createReverseSegmentLoop(
        builder, loc, revOuter.getInductionVar(), nInner, nOuter, trailingIters,
        initArgs);
    preserveAttributesButCheckpointing(revInner, forOp);

    builder.setInsertionPointToEnd(revInner.getBody());

    Value currentIV = FinalClass::computeReverseSegmentIV(
        builder, loc, forOp, revOuter.getInductionVar(),
        revInner.getInductionVar(), nInner, nOuter, trailingIters,
        revBoundHint);

    for (auto [oldArg, newArg] :
         llvm::zip_equal(forOp.getBody()->getArguments(),
                         revInner.getBody()->getArguments()))
      mapping.map(oldArg, newArg);

    mapping.map(forOp.getInductionVar(), currentIV);

    copyBlockWithoutTerminator(builder, forOp.getBody(), gutils, mapping);
    Operation *segTerm = forOp.getBody()->getTerminator();
    SmallVector<Value> segYields;
    for (auto operand : segTerm->getOperands())
      segYields.push_back(mapping.lookupOrDefault(operand));
    FinalClass::createScaffoldYield(builder, segTerm->getLoc(), segYields);

    builder.setInsertionPointToEnd(revOuter.getBody());

    for (auto outsideRef : cachedOutsideRefs) {
      if (auto cachableT =
              dyn_cast<ClonableTypeInterface>(outsideRef.getType())) {
        cachableT.freeClonedValue(builder, outsideRef);
      }
    }

    auto revLoop = FinalClass::createLoopWithSameBounds(
        builder, loc, revInner,
        revOuter.getBody()->getArguments().drop_front());
    preserveAttributesButCheckpointing(revLoop, forOp);

    Block *revLoopBody = revLoop.getBody();
    Block *origBody = forOp.getBody();

    // Reset every (non-mutable) intermediate gradient slot to zero at the
    // start of each reverse iteration and zero the diffe of the yielded
    // operands: the loop-carried gradient is supplied via the iter_arg.
    // Without this, scalar gradient slots such as the diffe of a value
    // loaded from an enzyme_dup'ed memref leak across reverse iterations,
    // get promoted to loop-carried iter_args, and over-accumulate into the
    // external shadow.
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
        gutils->addToDiffe(operand, revLoopBody->getArgument(revIdx), builder);
        revIdx++;
      }
    }

    bool valid = true;

    auto first = origBody->rbegin();
    first++; // skip terminator

    auto last = origBody->rend();

    for (auto it = first; it != last; ++it) {
      Operation *o = &*it;
      valid &= gutils->Logic.visitChild(o, builder, gutils).succeeded();
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
    FinalClass::createScaffoldYield(
        builder, forOp.getBody()->getTerminator()->getLoc(), newResults);

    builder.setInsertionPointToEnd(revOuter.getBody());
    FinalClass::createScaffoldYield(builder,
                                    forOp.getBody()->getTerminator()->getLoc(),
                                    revLoop.getResults());

    builder.setInsertionPointAfter(revOuter);

    revIdx = 0;
    auto forOpInits = getInits(forOp);
    for (auto &&[active, arg] : llvm::zip_equal(operandsActive, forOpInits)) {
      if (active) {
        if (!gutils->isConstantValue(arg)) {
          gutils->addToDiffe(arg, revOuter->getResult(revIdx), builder);
        }
        revIdx++;
      }
    }

    return success(valid);
  }

  //===--------------------------------------------------------------------===//
  // Entry points
  //===--------------------------------------------------------------------===//

  static std::optional<SmallVector<Value>>
  tryCacheValues(OpName forOp, Operation *op, MGradientUtilsReverse *gutils) {
    if (!needsBinomialCheckpointing(forOp) && !needsCheckpointing(forOp))
      return std::nullopt;

    // Both schemes build their bookkeeping scaffold (budget/recompute loops,
    // the remat scf.while) out of scf ops regardless of which dialect the
    // differentiated loop itself belongs to. Loading the scf dialect this
    // late (mid-pass, possibly under a multi-threaded PassManager) is not
    // safe -- MLIR requires it to be declared in the triggering pass's
    // `dependentDialects` instead (see DifferentiatePass/
    // DifferentiateWrapperPass in Passes.td), so it's preloaded up front.

    // Bound shapes materializeLowerBound/UpperBound can't handle (e.g.
    // affine min/max multi-result maps) are rejected up front, before any
    // bound materialization is attempted below; trivial success() for
    // dialects whose bounds are always single-valued (scf.for). The error is
    // emitted here so it surfaces even though cacheValues runs before
    // createReverseModeAdjoint.
    if (failed(FinalClass::requireSingleResultBounds(forOp)))
      return SmallVector<Value>();

    if (needsBinomialCheckpointing(forOp)) {
      auto budget = getCheckpointBudget(forOp);
      if (!budget || *budget <= 1) {
        // Error is reported in tryCreateReverseModeAdjoint; fall back to
        // caching the bounds so the reverse pass can proceed to emit the
        // diagnostic (mirrors the plain-loop path's 3-cache convention).
        Operation *newOp = gutils->getNewFromOriginal(op);
        OpBuilder cacheBuilder(newOp);
        Location loc = forOp.getLoc();
        SmallVector<Value> dummyCaches;
        dummyCaches.push_back(gutils->initAndPushCache(
            FinalClass::materializeLowerBound(cacheBuilder, loc, forOp, gutils),
            cacheBuilder));
        dummyCaches.push_back(gutils->initAndPushCache(
            FinalClass::materializeUpperBound(cacheBuilder, loc, forOp, gutils),
            cacheBuilder));
        dummyCaches.push_back(gutils->initAndPushCache(
            FinalClass::materializeStep(cacheBuilder, loc, forOp, gutils),
            cacheBuilder));
        return dummyCaches;
      }
      return cacheBinomial(forOp, *budget, gutils);
    }

    if (needsCheckpointing(forOp))
      return cachePeriodic(forOp, op, gutils);

    return std::nullopt;
  }

  static std::optional<LogicalResult> tryCreateReverseModeAdjoint(
      OpName forOp, Operation *op, OpBuilder &builder,
      MGradientUtilsReverse *gutils, SmallVector<Value> caches,
      ArrayRef<bool> operandsActive, ArrayRef<Value> incomingGradients) {
    if (!needsBinomialCheckpointing(forOp) && !needsCheckpointing(forOp))
      return std::nullopt;

    // Mirrors the guard in tryCacheValues; already reported there, but
    // createReverseModeAdjoint must still fail (not fall through to the
    // plain-loop path, whose cache layout wouldn't match what cacheValues
    // actually cached).
    if (failed(FinalClass::requireSingleResultBounds(forOp)))
      return failure();

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

    if (needsCheckpointing(forOp))
      return reversePeriodic(forOp, op, builder, gutils, caches, operandsActive,
                             incomingGradients);

    return std::nullopt;
  }
};

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_LOOPCHECKPOINTING_H
