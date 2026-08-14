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
#include <functional>
#include <optional>
#include <utility>

namespace mlir {
namespace enzyme {

template <typename FinalClass, typename OpName> struct LoopCheckpointing {
  // How the trip count is decomposed for periodic checkpointing: `nOuter`
  // segments of `nInner` iterations, plus a shorter trailing segment of
  // `trailingIters`.
  //
  // A stated period budgets the *outer* half of that: it is the number of
  // segments, hence of checkpoints, hence both the trip count of the scaffold's
  // outer loops and the size of the one allocation that outlives the forward
  // pass. The segment length is what is derived from it, as
  // ceil(numIters / nOuter). Reading the period as the segment length instead
  // would leave the checkpoint storage growing with the trip count --
  // ceil(numIters / period) live checkpoints -- which is the opposite of what
  // stating a budget is for.
  //
  // Every dialect's outer loops therefore count segments, one iteration per
  // checkpoint, and derive the segment's base iteration as nInner * index.
  struct PeriodicSchedule {
    // Iterations per full segment, or -1 when only `nInnerV` knows it.
    int64_t nInner = 0;
    // Number of full segments. Always known at compile time.
    int64_t nOuter = -1;
    // Iterations in the short trailing segment, 0 when the full segments
    // divide the trip count evenly -- and always 0 for a dynamic trip count,
    // where every segment is instead clamped against the trip count at
    // runtime. That clamp covers both the short segment and the empty ones
    // that a rounded-up segment length can leave at the end.
    int64_t trailingIters = 0;

    // Non-null exactly when the trip count is dynamic. numIters/start/step are
    // materialized before the scaffold (and cached for the reverse pass);
    // nOuterV is the segment count as a value, which only the dynamic schedule
    // needs (a static one has numSegments() for that).
    Value numItersV, nOuterV, nInnerV, startV, stepV;

    bool isDynamic() const { return numItersV != nullptr; }
    bool hasTrailing() const { return trailingIters > 0; }
    // Total number of segments, the trailing one included.
    int64_t numSegments() const { return nOuter + hasTrailing(); }
  };

  // Names of the attributes the checkpointing directives are read from. A
  // downstream project differentiating its own loop op spells them with its
  // own prefix (Enzyme-JAX uses `enzymexla.*`), so every access below goes
  // through these three hooks rather than a literal.
  static StringRef enableCheckpointingAttrName() {
    return "enzyme.enable_checkpointing";
  }

  static StringRef binomialCheckpointingAttrName() {
    return "enzyme.binomial_checkpointing";
  }

  // Both schemes read the period as a budget on the number of live
  // checkpoints: the size of the binomial checkpoint table, or the number of
  // segments a periodic decomposition is cut into. See PeriodicSchedule.
  static StringRef checkpointPeriodAttrName() {
    return "enzyme.checkpoint_period";
  }

  // A loop created while differentiating `oldOp` is still doing that loop's
  // work, so it inherits what was set on it. The checkpointing directives are
  // left behind, since the rewrite has already acted on them.
  static void preserveAttributesButCheckpointing(Operation *newOp,
                                                 Operation *oldOp) {
    for (auto attr : oldOp->getDiscardableAttrs()) {
      auto name = attr.getName();
      if (name != FinalClass::enableCheckpointingAttrName() &&
          name != FinalClass::binomialCheckpointingAttrName() &&
          name != FinalClass::checkpointPeriodAttrName())
        newOp->setAttr(name, attr.getValue());
    }
  }

  static bool hasBinomialAttr(OpName forOp) {
    return forOp->hasAttr(FinalClass::binomialCheckpointingAttrName());
  }

  static bool checkpointingEnabled(OpName forOp) {
    auto a = forOp->template getAttrOfType<BoolAttr>(
        FinalClass::enableCheckpointingAttrName());
    return a && a.getValue();
  }

  // Whether periodic checkpointing can be built for a loop whose trip count is
  // only known at runtime. Dialects whose scaffold bounds must be compile-time
  // constants (affine.for, whose bounds are AffineMaps) say no and fall back to
  // the plain reverse path, exactly as they did before dynamic support existed.
  static bool supportsDynamicPeriodic() { return true; }

  static bool needsCheckpointing(OpName forOp) {
    if (!FinalClass::checkpointingEnabled(forOp) ||
        FinalClass::hasBinomialAttr(forOp))
      return false;
    if (FinalClass::getConstantNumberOfIterations(forOp).has_value())
      return true;
    // A dynamic trip count has no compile-time N to take the square root of,
    // so the period cannot be defaulted: it has to be stated, and (being the
    // segment count, which the scaffold divides by) it has to be positive.
    if (!FinalClass::supportsDynamicPeriodic())
      return false;
    auto period = FinalClass::getCheckpointBudget(forOp);
    return period.has_value() && *period > 0;
  }

  static bool needsBinomialCheckpointing(OpName forOp) {
    return FinalClass::checkpointingEnabled(forOp) &&
           FinalClass::hasBinomialAttr(forOp);
  }

  static std::optional<int64_t> getCheckpointBudget(OpName forOp) {
    if (auto a = forOp->template getAttrOfType<IntegerAttr>(
            FinalClass::checkpointPeriodAttrName()))
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

  static Value getInductionVar(OpName forOp) { return forOp.getInductionVar(); }

  // The op's body block. Spelling differs across dialects (scf.for and
  // affine.for hand out the block, stablehlo.while a whole region), and every
  // scaffold loop the periodic hooks build is itself an OpName, so this is the
  // one accessor the shared code uses.
  static Block *getBodyBlock(OpName forOp) { return forOp.getBody(); }

  // Move `builder` to where the next op of `block`'s body belongs. scf.for and
  // affine.for hand out a body whose terminator the scaffold has yet to create,
  // so that is simply the end of the block; an op whose builder pre-creates the
  // terminator (stablehlo.while) has to insert before it instead.
  static void setInsertionPointToBodyEnd(OpBuilder &builder, Block *block) {
    builder.setInsertionPointToEnd(block);
  }

  static size_t getNumRegionIterArgs(OpName forOp) {
    return forOp.getNumRegionIterArgs();
  }

  // The loop-carried subset of a body terminator's operands, in iter-arg
  // order. An op whose terminator also yields the next induction variable
  // (stablehlo.while) drops it here, so that the shared code can zip these
  // against the iter args.
  static OperandRange getCarriedTerminatorOperands(Operation *term) {
    return term->getOperands();
  }

  // The loop-carried subset of a scaffold loop's results, in iter-arg order.
  // Same asymmetry as above: an op that carries its induction variable as a
  // regular operand also returns it as a regular result.
  static ResultRange getCarriedResults(Operation *loop) {
    return loop->getResults();
  }

  //===--------------------------------------------------------------------===//
  // Scalar ("index-like") arithmetic. Everything the scaffold computes for
  // itself -- step counters, segment bounds, checkpoint slot numbers -- is
  // built out of these, so that a dialect which does not count on `index`
  // (stablehlo counts in tensor<i64>) can supply its own primitives.
  //===--------------------------------------------------------------------===//

  static Type getIndexLikeType(OpBuilder &builder) {
    return builder.getIndexType();
  }

  static Value emitConst(OpBuilder &b, Location loc, int64_t v) {
    return arith::ConstantIndexOp::create(b, loc, v);
  }

  static Value emitAdd(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::AddIOp::create(b, loc, l, r);
  }

  static Value emitSub(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::SubIOp::create(b, loc, l, r);
  }

  static Value emitMul(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::MulIOp::create(b, loc, l, r);
  }

  static Value emitDivU(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::DivUIOp::create(b, loc, l, r);
  }

  static Value emitMin(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::MinUIOp::create(b, loc, l, r);
  }

  static Value getNumIterationsValue(OpBuilder &builder, Location loc,
                                     OpName forOp,
                                     MGradientUtilsReverse *gutils) {
    Value lb = FinalClass::materializeLowerBound(builder, loc, forOp, gutils);
    Value ub = FinalClass::materializeUpperBound(builder, loc, forOp, gutils);
    Value step = FinalClass::materializeStep(builder, loc, forOp, gutils);
    Value diff = FinalClass::emitSub(builder, loc, ub, lb);
    return FinalClass::emitDivU(builder, loc, diff, step);
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

  // Whether copyBlockWithoutTerminator also publishes the caller's seed
  // mappings -- the values the copied block reads from above, mapped to popped
  // caches or to the per-segment clone that replays them -- into
  // gutils->originalToNewFn, alongside the values the block itself defines.
  //
  // The memref dialects need them published: a load's reverse rule looking up
  // getNewFromOriginal() of a memref read from above must see the clone this
  // replay writes through, not the caller's. Dialects whose outside refs are
  // immutable values instead keep them private, so that the ops visited after
  // the loop (in reverse order) still see the caller's own value.
  static bool publishesCopiedSeeds() { return true; }

  // True when `v` is defined inside `b` (directly or in a nested region),
  // i.e. it is one of the values the copied block itself brings into being
  // rather than one of the caller's seeds.
  static bool isDefinedInBlock(Block *b, Value v) {
    Block *owner = isa<BlockArgument>(v) ? cast<BlockArgument>(v).getOwner()
                                         : v.getDefiningOp()->getBlock();
    for (; owner; owner = owner->getParentOp()
                              ? owner->getParentOp()->getBlock()
                              : nullptr)
      if (owner == b)
        return true;
    return false;
  }

  static void copyBlockWithoutTerminator(OpBuilder &builder, Block *b,
                                         MGradientUtilsReverse *gutils,
                                         IRMapping &mapping) {
    for (auto &it : b->without_terminator()) {
      OpBuilder::InsertionGuard g(builder);
      FinalClass::cloneOp(builder, it, mapping);
    }

    for (auto [oldVal, newVal] : mapping.getValueMap())
      if (FinalClass::publishesCopiedSeeds() || isDefinedInBlock(b, oldVal))
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
      l.numIterArgs = FinalClass::getNumRegionIterArgs(forOp);
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

  //===--------------------------------------------------------------------===//
  // Checkpoint storage.
  //
  // A store holds `budget` snapshots of one value, indexed by the checkpoint
  // stack pointer. Two shapes of storage are covered. A mutable buffer has one
  // identity that every write goes through, so it can be allocated once and
  // referred to from anywhere (memref). An immutable one has no identity: a
  // write produces a *new* value, which then has to be carried out of every
  // loop it was written inside (a tensor). storesAreLoopCarried() picks between
  // them, and the scheme below threads the store values through its scaffold
  // loops as a trailing group of iteration arguments when it is set.
  //===--------------------------------------------------------------------===//

  static bool storesAreLoopCarried() { return false; }

  static Value createStore(OpBuilder &b, Location loc, int64_t budget,
                           Type valTy) {
    return memref::AllocOp::create(b, loc, checkpointBufferType(budget, valTy));
  }

  // Returns the store to use from here on: the same one for a buffer, the
  // updated value for an immutable store.
  static Value storeSlot(OpBuilder &b, Location loc, Value store, Value slot,
                         Value val) {
    storeCheckpoint(b, loc, store, slot, val);
    return store;
  }

  static Value loadSlot(OpBuilder &b, Location loc, Value store, Value slot,
                        Type valTy) {
    return loadCheckpoint(b, loc, store, slot, valTy);
  }

  static void destroyStore(OpBuilder &b, Location loc, Value store) {
    memref::DeallocOp::create(b, loc, store);
  }

  //===--------------------------------------------------------------------===//
  // Scaffold loops -- the scheme's own bookkeeping loops, as opposed to the
  // replayed copies of the differentiated loop. A counted one has body
  // arguments [induction variable, carried...]; a while loop carries no
  // induction variable of its own.
  //===--------------------------------------------------------------------===//

  struct ScaffoldLoop {
    Operation *op = nullptr;
    Block *body = nullptr;
    // Index of the first carried value among the body's arguments and among
    // the op's results: an op that models its induction variable as a regular
    // iteration argument (stablehlo.while) has one of each in front.
    unsigned firstCarriedArg = 0;
    unsigned firstCarriedResult = 0;

    Value getIV() const { return body->getArgument(0); }
    Block::BlockArgListType args() const {
      return body->getArguments().drop_front(firstCarriedArg);
    }
    ResultRange results() const {
      return op->getResults().drop_front(firstCarriedResult);
    }
  };

  static ScaffoldLoop createScaffoldForLoop(OpBuilder &b, Location loc,
                                            Value lb, Value ub, Value step,
                                            ValueRange inits) {
    auto loop = scf::ForOp::create(b, loc, lb, ub, step, inits);
    // scf.for materializes a terminator of its own when nothing is carried;
    // finalizeScaffoldLoop creates it in either case.
    if (!loop.getBody()->empty())
      loop.getBody()->front().erase();
    return {loop, loop.getBody(), /*firstCarriedArg=*/1,
            /*firstCarriedResult=*/0};
  }

  // `cond` is called with the carried values and returns the i1 deciding
  // whether to run the body again.
  static ScaffoldLoop createScaffoldWhileLoop(
      OpBuilder &b, Location loc, ValueRange inits,
      llvm::function_ref<Value(OpBuilder &, Location, ValueRange)> cond) {
    SmallVector<Type> types = llvm::to_vector(inits.getTypes());
    SmallVector<Location> locs(inits.size(), loc);
    auto loop = scf::WhileOp::create(b, loc, types, inits);
    {
      OpBuilder::InsertionGuard g(b);
      Block *before = b.createBlock(&loop.getBefore(), {}, types, locs);
      b.setInsertionPointToEnd(before);
      scf::ConditionOp::create(b, loc, cond(b, loc, before->getArguments()),
                               before->getArguments());
      b.createBlock(&loop.getAfter(), {}, types, locs);
    }
    return {loop, &loop.getAfter().front(), 0, 0};
  }

  static void finalizeScaffoldLoop(OpBuilder &b, Location loc,
                                   const ScaffoldLoop &loop,
                                   ValueRange yields) {
    scf::YieldOp::create(b, loc, yields);
  }

  static Value emitCmpLT(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, l, r);
  }

  static Value emitCmpEQ(OpBuilder &b, Location loc, Value l, Value r) {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, l, r);
  }

  static Value emitSelect(OpBuilder &b, Location loc, Value c, Value t,
                          Value f) {
    return arith::SelectOp::create(b, loc, c, t, f);
  }

  // The values that stand in for the original loop's results once a scaffold
  // has replaced it. `state` is the scaffold's final carried state; an op that
  // returns its own induction variable has to supply that itself, since the
  // scaffold's counter is not the original's.
  static SmallVector<Value> getPrimalResults(OpBuilder &b, Location loc,
                                             OpName forOp, ValueRange state,
                                             MGradientUtilsReverse *gutils) {
    return llvm::to_vector(state);
  }

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
    Type idxTy = FinalClass::getIndexLikeType(builder);

    Value c0 = FinalClass::emitConst(builder, loc, 0);
    Value c1 = FinalClass::emitConst(builder, loc, 1);

    // Loop trip count / lower bound / step as index values (constant-folded
    // when the bounds are constant).
    Value numItersV, startV, stepV;
    if (isDynamic) {
      startV = FinalClass::materializeLowerBound(builder, loc, forOp, gutils);
      stepV = FinalClass::materializeStep(builder, loc, forOp, gutils);
      numItersV =
          FinalClass::getNumIterationsValue(builder, loc, forOp, gutils);
      numItersV = FinalClass::castToType(builder, loc, numItersV, idxTy);
    } else {
      int64_t numIters =
          FinalClass::getConstantNumberOfIterations(forOp).value();
      numItersV = FinalClass::emitConst(builder, loc, numIters);
      startV = FinalClass::emitConst(builder, loc,
                                     FinalClass::getConstantStart(forOp));
      stepV = FinalClass::emitConst(builder, loc,
                                    FinalClass::getConstantStep(forOp));
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
    Value budgetV = FinalClass::emitMin(
        builder, loc, FinalClass::emitConst(builder, loc, budget), numItersV);

    SmallVector<Value> immutableRefs, mutableRefs;
    FinalClass::splitOutsideRefs(forOp, mutableRefs, immutableRefs);
    auto layout = BinomialCacheLayout::get(forOp, mutableRefs, immutableRefs,
                                           isDynamic, gutils);

    IRMapping mapping;
    SmallVector<Value> caches;

    // One checkpoint store per iter arg, plus one holding the forward step each
    // checkpoint was taken at. Kept in a single list, since they are carried,
    // updated and transported together; the index store is the last entry.
    size_t numIterArgs = FinalClass::getNumRegionIterArgs(forOp);
    SmallVector<Value> stores;
    for (auto arg : FinalClass::getInits(newForOp))
      stores.push_back(
          FinalClass::createStore(builder, loc, budget, arg.getType()));
    stores.push_back(FinalClass::createStore(builder, loc, budget, idxTy));
    const size_t idxStore = stores.size() - 1;
    const bool carriedStores = FinalClass::storesAreLoopCarried();

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
    // (stepCtr, state..., [stores...]).
    SmallVector<Value> outerInit;
    outerInit.push_back(c0);
    auto newForOpInits = FinalClass::getInits(newForOp);
    outerInit.append(newForOpInits.begin(), newForOpInits.end());
    if (carriedStores)
      outerInit.append(stores.begin(), stores.end());
    auto outerFwd = FinalClass::createScaffoldForLoop(builder, loc, c0, budgetV,
                                                      c1, outerInit);
    FinalClass::preserveAttributesButCheckpointing(outerFwd.op, forOp);

    builder.setInsertionPointToStart(outerFwd.body);
    Value k = outerFwd.getIV();
    auto outerArgs = outerFwd.args();
    Value stepCtr = outerArgs[0];
    auto state = outerArgs.slice(1, numIterArgs);

    // Inside the loop the stores are the iteration arguments, not the values
    // they were initialized from.
    SmallVector<Value> liveStores(stores);
    if (carriedStores)
      for (auto &&[i, arg] :
           llvm::enumerate(outerArgs.slice(1 + numIterArgs, stores.size())))
        liveStores[i] = arg;

    for (auto &&[i, val] : llvm::enumerate(state))
      liveStores[i] =
          FinalClass::storeSlot(builder, loc, liveStores[i], k, val);
    liveStores[idxStore] =
        FinalClass::storeSlot(builder, loc, liveStores[idxStore], k, stepCtr);

    Value numStepsRem = FinalClass::emitSub(builder, loc, numItersV, stepCtr);
    Value budgetRem = FinalClass::emitSub(builder, loc, budgetV, k);
    // Never use more checkpoints than remaining steps (binomial_progress is
    // degenerate for budget > steps).
    budgetRem = FinalClass::emitMin(builder, loc, budgetRem, numStepsRem);
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
    auto innerFwd = FinalClass::createScaffoldForLoop(
        builder, loc, c0, split, c1,
        SmallVector<Value>(state.begin(), state.end()));
    FinalClass::preserveAttributesButCheckpointing(innerFwd.op, forOp);

    builder.setInsertionPointToStart(innerFwd.body);
    Value i = innerFwd.getIV();
    Value globalStep = FinalClass::emitAdd(builder, loc, stepCtr, i);
    globalStep =
        FinalClass::castToType(builder, loc, globalStep, stepV.getType());
    Value iv = FinalClass::emitAdd(
        builder, loc, startV,
        FinalClass::emitMul(builder, loc, stepV, globalStep));

    Block *newForOpBody = FinalClass::getBodyBlock(newForOp);
    for (auto &&[oldArg, newArg] : llvm::zip_equal(
             newForOpBody->getArguments().drop_front(), innerFwd.args()))
      mapping.map(oldArg, newArg);
    mapping.map(FinalClass::getInductionVar(newForOp), iv);

    copyBlockWithoutTerminator(builder, newForOpBody, gutils, mapping);

    SmallVector<Value> innerYields;
    for (auto operand : FinalClass::getCarriedTerminatorOperands(
             newForOpBody->getTerminator()))
      innerYields.push_back(mapping.lookupOrDefault(operand));
    FinalClass::finalizeScaffoldLoop(builder, loc, innerFwd, innerYields);

    FinalClass::setInsertionPointToBodyEnd(builder, outerFwd.body);
    SmallVector<Value> outerYields;
    outerYields.push_back(FinalClass::emitAdd(builder, loc, stepCtr, split));
    auto innerResults = innerFwd.results();
    outerYields.append(innerResults.begin(), innerResults.end());
    if (carriedStores)
      outerYields.append(liveStores.begin(), liveStores.end());
    FinalClass::finalizeScaffoldLoop(builder, loc, outerFwd, outerYields);

    builder.setInsertionPointAfter(outerFwd.op);

    // Carried stores leave the loop as its results; a buffer is the same value
    // it always was.
    auto outerResults = outerFwd.results();
    SmallVector<Value> finalStores(stores);
    if (carriedStores)
      for (auto &&[i, res] :
           llvm::enumerate(outerResults.slice(1 + numIterArgs, stores.size())))
        finalStores[i] = res;

    // Cache the stores + outside refs (single push each).
    for (auto store : finalStores)
      caches.push_back(gutils->initAndPushCache(store, builder));

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
    gutils->replaceOrigOpWith(
        forOp, FinalClass::getPrimalResults(builder, loc, forOp,
                                            outerResults.slice(1, numIterArgs),
                                            gutils));
    gutils->erase(newForOp);
    gutils->originalToNewFnOps[forOp] = outerFwd.op;

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
    auto numIterArgs = FinalClass::getNumRegionIterArgs(forOp);

    SmallVector<Value> immutableRefs, mutableRefs;
    FinalClass::splitOutsideRefs(forOp, mutableRefs, immutableRefs);
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

    // Pop the checkpoint stores (indices from the shared layout); the index
    // store is the last of them, same as in cacheBinomial.
    SmallVector<Value> stores;
    for (size_t j = 0; j < numIterArgs; ++j)
      stores.push_back(gutils->popCache(caches[layout.ckptBuf(j)], builder));
    stores.push_back(gutils->popCache(caches[layout.idxBuf()], builder));
    const size_t idxStore = stores.size() - 1;
    const bool carriedStores = FinalClass::storesAreLoopCarried();

    SmallVector<Value> immutableRefsCaches;
    for (auto &&[i, ref] : llvm::enumerate(immutableRefs)) {
      Value cached = gutils->popCache(caches[layout.immutable(i)], builder);
      mapping.map(ref, cached);
      immutableRefsCaches.push_back(cached);
    }

    Value c0 = FinalClass::emitConst(builder, loc, 0);
    Value c1 = FinalClass::emitConst(builder, loc, 1);

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
      numItersV = FinalClass::emitConst(builder, loc, numIters);
      startV = FinalClass::emitConst(builder, loc,
                                     FinalClass::getConstantStart(forOp));
      stepV = FinalClass::emitConst(builder, loc,
                                    FinalClass::getConstantStep(forOp));
    }

    // Effective budget = min(requested budget, trip count); must match
    // cacheBinomial, including being unconditional.
    Value budgetV = FinalClass::emitMin(
        builder, loc, FinalClass::emitConst(builder, loc, budget), numItersV);

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

    // Outer reverse loop over all N steps; carries (sp, adjoints..., [stores]).
    SmallVector<Value> outerInit;
    outerInit.push_back(budgetV); // live checkpoint count
    outerInit.append(incomingGradients.begin(), incomingGradients.end());
    if (carriedStores)
      outerInit.append(stores.begin(), stores.end());

    auto revOuter = FinalClass::createScaffoldForLoop(builder, loc, c0,
                                                      numItersV, c1, outerInit);
    FinalClass::preserveAttributesButCheckpointing(revOuter.op, forOp);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(revOuter.body);

    Value ivO = revOuter.getIV();
    auto revOuterArgs = revOuter.args();
    Value sp = revOuterArgs[0];
    auto adjArgs = revOuterArgs.slice(1, incomingGradients.size());

    SmallVector<Value> liveStores(stores);
    if (carriedStores)
      for (auto &&[i, arg] : llvm::enumerate(
               revOuterArgs.slice(1 + incomingGradients.size(), stores.size())))
        liveStores[i] = arg;

    Value capo = FinalClass::emitSub(builder, loc, sp, c1);
    Value currentRevStep = FinalClass::emitSub(builder, loc, numItersV, ivO);

    // Load the top checkpoint state + its forward step.
    SmallVector<Value> ckptState;
    for (auto &&[store, arg] : llvm::zip_equal(
             ArrayRef<Value>(liveStores).drop_back(),
             FinalClass::getBodyBlock(forOp)->getArguments().drop_front()))
      ckptState.push_back(
          FinalClass::loadSlot(builder, loc, store, capo, arg.getType()));
    Value ckptStep =
        FinalClass::loadSlot(builder, loc, liveStores[idxStore], capo,
                             FinalClass::getIndexLikeType(builder));

    // Re-prime each working clone from the snapshot paired with slot `capo`,
    // so the replay below starts from the mutable memory as it was at
    // ckptStep.
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs))
      cast<ClonableTypeInterface>(ref.getType())
          .copyValue(builder, workClones[r],
                     cloneSlot(builder, loc, mutBufs[r], capo));

    // Inner remat loop: reconstruct state at (currentRevStep - 1), carrying
    // (pos, capo, state..., [stores...]).
    SmallVector<Value> whileInit;
    whileInit.push_back(ckptStep);
    whileInit.push_back(capo);
    whileInit.append(ckptState.begin(), ckptState.end());
    if (carriedStores)
      whileInit.append(liveStores.begin(), liveStores.end());

    auto revWhile = FinalClass::createScaffoldWhileLoop(
        builder, loc, whileInit,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value posPlus1 = FinalClass::emitAdd(b, l, args[0], c1);
          return FinalClass::emitCmpLT(b, l, posPlus1, currentRevStep);
        });
    {
      FinalClass::setInsertionPointToBodyEnd(builder, revWhile.body);
      auto wargs = revWhile.args();
      Value pos = wargs[0];
      Value acapo = wargs[1];
      auto astate = wargs.slice(2, numIterArgs);

      SmallVector<Value> wStores(liveStores);
      if (carriedStores)
        for (auto &&[i, arg] :
             llvm::enumerate(wargs.slice(2 + numIterArgs, stores.size())))
          wStores[i] = arg;

      Value remaining = FinalClass::emitSub(builder, loc, currentRevStep, pos);
      Value budgetRem = FinalClass::emitSub(builder, loc, budgetV, acapo);
      // Never use more checkpoints than remaining steps (binomial_progress
      // is degenerate for budget > steps).
      budgetRem = FinalClass::emitMin(builder, loc, budgetRem, remaining);
      Value split = enzyme::BinomialProgressOp::create(
          builder, loc, FinalClass::getIndexLikeType(builder), remaining,
          budgetRem);

      // Place a checkpoint at slot `acapo`. The mutable-ref snapshot has to
      // go with it: the working clones currently hold the content at step
      // `pos` (innerRemat below advances them only after this store), and a
      // slot whose state and snapshot came from different steps replays
      // wrongly.
      for (auto &&[i, val] : llvm::enumerate(astate))
        wStores[i] =
            FinalClass::storeSlot(builder, loc, wStores[i], acapo, val);
      wStores[idxStore] =
          FinalClass::storeSlot(builder, loc, wStores[idxStore], acapo, pos);
      for (auto &&[r, ref] : llvm::enumerate(mutableRefs))
        cast<ClonableTypeInterface>(ref.getType())
            .copyValue(builder, cloneSlot(builder, loc, mutBufs[r], acapo),
                       workClones[r]);

      Value posPlusSplit = FinalClass::emitAdd(builder, loc, pos, split);
      Value isLast =
          FinalClass::emitCmpEQ(builder, loc, posPlusSplit, currentRevStep);
      Value rematUB = FinalClass::emitSelect(
          builder, loc, isLast,
          FinalClass::emitSub(builder, loc, posPlusSplit, c1), posPlusSplit);

      // Recompute the primal from `pos` to `rematUB`.
      auto innerRemat = FinalClass::createScaffoldForLoop(
          builder, loc, pos, rematUB, c1,
          SmallVector<Value>(astate.begin(), astate.end()));
      FinalClass::preserveAttributesButCheckpointing(innerRemat.op, forOp);

      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(innerRemat.body);
        Value idx = FinalClass::castToType(builder, loc, innerRemat.getIV(),
                                           stepV.getType());
        Value iv =
            FinalClass::emitAdd(builder, loc, startV,
                                FinalClass::emitMul(builder, loc, stepV, idx));

        Block *origBodyBlock = FinalClass::getBodyBlock(forOp);
        for (auto &&[oldArg, newArg] : llvm::zip_equal(
                 origBodyBlock->getArguments().drop_front(), innerRemat.args()))
          mapping.map(oldArg, newArg);
        mapping.map(FinalClass::getInductionVar(forOp), iv);

        copyBlockWithoutTerminator(builder, origBodyBlock, gutils, mapping);

        SmallVector<Value> yields;
        for (auto operand : FinalClass::getCarriedTerminatorOperands(
                 origBodyBlock->getTerminator()))
          yields.push_back(mapping.lookupOrDefault(operand));
        FinalClass::finalizeScaffoldLoop(builder, loc, innerRemat, yields);
      }

      Value newCapo = FinalClass::emitAdd(builder, loc, acapo, c1);
      SmallVector<Value> afterYields;
      afterYields.push_back(posPlusSplit);
      afterYields.push_back(newCapo);
      auto rematResults = innerRemat.results();
      afterYields.append(rematResults.begin(), rematResults.end());
      if (carriedStores)
        afterYields.append(wStores.begin(), wStores.end());
      FinalClass::finalizeScaffoldLoop(builder, loc, revWhile, afterYields);
    }

    FinalClass::setInsertionPointToBodyEnd(builder, revOuter.body);
    auto whileResults = revWhile.results();
    Value newSp = whileResults[1];
    auto reconState = whileResults.slice(2, numIterArgs);
    if (carriedStores)
      for (auto &&[i, res] :
           llvm::enumerate(whileResults.slice(2 + numIterArgs, stores.size())))
        liveStores[i] = res;

    // Adjoint of a single body step at (currentRevStep - 1).
    Value stepAdj = FinalClass::emitSub(builder, loc, currentRevStep, c1);
    Value stepAdjC =
        FinalClass::castToType(builder, loc, stepAdj, stepV.getType());
    Value ivAdj =
        FinalClass::emitAdd(builder, loc, startV,
                            FinalClass::emitMul(builder, loc, stepV, stepAdjC));

    mapping = IRMapping();

    for (auto [ref, cached] :
         llvm::zip_equal(immutableRefs, immutableRefsCaches))
      mapping.map(ref, cached);

    // Re-bind after the mapping reset above; the shadows were bound once,
    // before revOuter, and do not need rebinding per iteration.
    for (auto &&[r, ref] : llvm::enumerate(mutableRefs))
      mapping.map(ref, workClones[r]);

    for (auto &&[oldArg, newArg] : llvm::zip_equal(
             FinalClass::getBodyBlock(forOp)->getArguments().drop_front(),
             reconState))
      mapping.map(oldArg, newArg);
    mapping.map(FinalClass::getInductionVar(forOp), ivAdj);

    // Re-materialize primal ops of this step for the reverse visitor.
    copyBlockWithoutTerminator(builder, FinalClass::getBodyBlock(forOp), gutils,
                               mapping);

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
      FinalClass::getBodyBlock(forOp)->walk([&](Operation *inner) {
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
    auto term = FinalClass::getBodyBlock(forOp)->getTerminator();
    FinalClass::primeStepGradients(builder, gutils,
                                   FinalClass::getBodyBlock(forOp),
                                   revOuter.body, operandsActive);

    // Seed adjoints of the yielded operands from the outer carried
    // gradients.
    unsigned revIdx = 0;
    for (auto &&[active, operand] : llvm::zip_equal(
             operandsActive, FinalClass::getCarriedTerminatorOperands(term))) {
      if (active) {
        gutils->addToDiffe(operand, adjArgs[revIdx], builder);
        revIdx++;
      }
    }

    bool valid = true;
    auto first = FinalClass::getBodyBlock(forOp)->rbegin();
    first++; // skip terminator
    auto last = FinalClass::getBodyBlock(forOp)->rend();
    {
      // Same as in reversePeriodic: a cache a body op's reverse rule asks for
      // is per-step, so it is created outside the remat loop.
      SegmentCacheCreatorGuard cacheGuard(gutils, revWhile.op);
      for (auto it = first; it != last; ++it)
        valid &= gutils->Logic.visitChild(&*it, builder, gutils).succeeded();
    }

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
             operandsActive,
             FinalClass::getBodyBlock(forOp)->getArguments().drop_front())) {
      if (active) {
        newAdjoints.push_back(gutils->diffe(arg, builder));
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, builder);
      }
    }

    SmallVector<Value> outerYields;
    outerYields.push_back(newSp);
    outerYields.append(newAdjoints.begin(), newAdjoints.end());
    if (carriedStores)
      outerYields.append(liveStores.begin(), liveStores.end());
    FinalClass::finalizeScaffoldLoop(builder, loc, revOuter, outerYields);

    builder.setInsertionPointAfter(revOuter.op);

    revIdx = 0;
    auto revOuterResults = revOuter.results();
    auto forOpInits = FinalClass::getInits(forOp);
    for (auto &&[active, arg] : llvm::zip_equal(operandsActive, forOpInits)) {
      if (active) {
        if (!gutils->isConstantValue(arg))
          gutils->addToDiffe(arg, revOuterResults[revIdx + 1], builder);
        revIdx++;
      }
    }

    // Release the checkpoint stores (a no-op for storage that is a value
    // rather than an allocation) and the cloned mutable refs.
    for (auto store : stores)
      FinalClass::destroyStore(builder, loc, store);
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
  // steps in a checkpoint; the reverse pass replays one segment at a time. A
  // stated period replaces the sqrt(N) segment count with one the caller
  // budgets for, and the segment length grows to match.
  //===--------------------------------------------------------------------===//

  // The compile-time half of the decomposition. A stated period is a budget on
  // the *segment* count -- see PeriodicSchedule -- so it is the segment length
  // that is derived from it, as ceil(numIters / period). Without one the
  // classic sqrt(N) split is used, which deliberately keeps nOuter == nInner
  // and lets the remainder spill into a trailing segment that may be as long as
  // nInner itself.
  static PeriodicSchedule getStaticPeriodicSchedule(OpName forOp) {
    PeriodicSchedule sched;
    auto period = FinalClass::getCheckpointBudget(forOp);
    auto numIters = FinalClass::getConstantNumberOfIterations(forOp);
    if (!numIters) {
      // Same reading of the period, with the division left to the runtime:
      // nInnerV ends up holding ceil(numIters / nOuter).
      // needsCheckpointing() only admits a dynamic trip count with a positive
      // period.
      sched.nInner = -1;
      sched.nOuter = *period;
      return sched;
    }
    if (*numIters <= 0) {
      // Nothing to segment. Spelled out because both splits below divide by a
      // quantity derived from the trip count, and because a zero-length segment
      // would give the scaffold loop a zero step.
      sched.nInner = 1;
      sched.nOuter = 0;
      return sched;
    }
    if (period && *period > 0) {
      // Rounding the length up is what holds the segment count to the budget:
      // numSegments() then lands at or below `period`, never above.
      sched.nInner = (*numIters + *period - 1) / *period;
    } else {
      sched.nInner = (int64_t)std::sqrt(*numIters);
    }
    // Whole segments, plus a remainder shorter than one of them. The remainder
    // has to be the shorter part: a dialect whose segment bound is
    // min(nInner, remaining) -- affine.for's boundary-tile map, stablehlo's
    // runtime clamp -- silently truncates a trailing segment longer than
    // nInner, and the tail of the loop is then never replayed at all. Splitting
    // sqrt(N) as nOuter == nInner and letting the remainder run on up to
    // 2*sqrt(N), which is what this did, is what made that reachable.
    sched.nOuter = *numIters / sched.nInner;
    sched.trailingIters = *numIters % sched.nInner;
    return sched;
  }

  // Where a body op's reverse rule materializes the caches it asks for while a
  // segment is being replayed. Returning null keeps the enclosing function's
  // own cache creator, which is what the memref dialects want; a dialect that
  // must place them relative to the replay loop instead (stablehlo emits an
  // enzyme.init just before it) returns a creator anchored on `anchor`.
  using CacheCreator = std::function<std::pair<Value, Value>(Type)>;

  static CacheCreator makeSegmentCacheCreator(MGradientUtilsReverse *gutils,
                                              Operation *anchor) {
    return nullptr;
  }

  // Registers that creator, if there is one, for as long as the replayed body
  // is being differentiated.
  class SegmentCacheCreatorGuard {
  public:
    SegmentCacheCreatorGuard(MGradientUtilsReverse *gutils, Operation *anchor)
        : gutils(gutils),
          hook(FinalClass::makeSegmentCacheCreator(gutils, anchor)) {
      if (hook)
        gutils->registerCacheCreatorHook(hook);
    }
    ~SegmentCacheCreatorGuard() {
      if (hook)
        gutils->deregisterCacheCreatorHook(hook);
    }
    SegmentCacheCreatorGuard(const SegmentCacheCreatorGuard &) = delete;
    SegmentCacheCreatorGuard &
    operator=(const SegmentCacheCreatorGuard &) = delete;

  private:
    MGradientUtilsReverse *gutils;
    CacheCreator hook;
  };

  // Prepare the gradient slots for one replayed segment, before the caller
  // seeds the incoming adjoints from the reverse outer loop's iter_args.
  //
  // Resetting every (non-mutable) intermediate slot to zero and zeroing the
  // diffe of the yielded operands is what keeps a scalar slot -- e.g. the diffe
  // of a value loaded from an enzyme_dup'ed memref -- from leaking across
  // reverse iterations, getting promoted to a loop-carried iter_arg and
  // over-accumulating into the external shadow.
  static void primeSegmentGradients(OpBuilder &builder,
                                    MGradientUtilsReverse *gutils,
                                    Block *origBody, Block *revLoopBody,
                                    ArrayRef<bool> operandsActive) {
    builder.setInsertionPointToStart(revLoopBody);
    mlir::enzyme::localizeGradients(builder, gutils, origBody);

    builder.setInsertionPointToEnd(revLoopBody);
    for (auto &&[active, operand] : llvm::zip_equal(
             operandsActive, FinalClass::getCarriedTerminatorOperands(
                                 origBody->getTerminator()))) {
      if (active)
        gutils->zeroDiffe(operand, builder);
    }
  }

  // The binomial counterpart of primeSegmentGradients: one *step*, rather than
  // a whole segment, is differentiated per iteration of the reverse loop, so
  // the slots are reset at the top of that loop's body.
  static void primeStepGradients(OpBuilder &builder,
                                 MGradientUtilsReverse *gutils, Block *origBody,
                                 Block *revOuterBody,
                                 ArrayRef<bool> operandsActive) {
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(revOuterBody);
      mlir::enzyme::localizeGradients(builder, gutils, origBody);
    }
    for (auto &&[active, operand] : llvm::zip_equal(
             operandsActive, FinalClass::getCarriedTerminatorOperands(
                                 origBody->getTerminator()))) {
      if (active)
        gutils->zeroDiffe(operand, builder);
    }
  }

  // The segment length as a value, which every segment bound and replayed
  // induction variable is derived from. Split out (rather than transported from
  // the forward pass) because the reverse pass recomputes it from the popped
  // trip count, by the very same formula: it is a pure function of the
  // schedule.
  static void materializeSegmentValues(OpBuilder &builder, Location loc,
                                       PeriodicSchedule &sched) {
    if (!sched.isDynamic()) {
      sched.nInnerV = FinalClass::emitConst(builder, loc, sched.nInner);
      return;
    }
    // nInner = ceil(numIters / nOuter). Rounding *up* is what keeps the
    // statically-many segments covering the whole trip count; it is also what
    // lets the last segments start past the end of the loop, so each segment's
    // length has to be clamped against the trip count where it is built.
    sched.nOuterV = FinalClass::emitConst(builder, loc, sched.nOuter);
    Value roundUp = FinalClass::emitAdd(
        builder, loc, sched.numItersV,
        FinalClass::emitConst(builder, loc, sched.nOuter - 1));
    sched.nInnerV = FinalClass::emitDivU(builder, loc, roundUp, sched.nOuterV);
  }

  // How many iterations the segment based at `base` covers: nInner, except for
  // the trailing one. Which segment that is, and how short, is the one thing
  // the two trip counts disagree on, so it is decided here rather than in each
  // direction's hook -- both directions ask this the same question, about a
  // base they derived the same way.
  static Value segmentLength(OpBuilder &builder, Location loc,
                             const PeriodicSchedule &sched, Value base) {
    if (!sched.isDynamic()) {
      if (!sched.hasTrailing())
        return sched.nInnerV;
      // The trailing segment is the one based just past the last full one.
      Value isTrailing = FinalClass::emitCmpEQ(
          builder, loc, base,
          FinalClass::emitConst(builder, loc, sched.nInner * sched.nOuter));
      return FinalClass::emitSelect(
          builder, loc, isTrailing,
          FinalClass::emitConst(builder, loc, sched.trailingIters),
          sched.nInnerV);
    }

    // A runtime trip count cannot say which segment is short, so every one of
    // them is clamped: min(nInner, numIters - base). The subtraction saturates
    // rather than wrapping, which is the other half of rounding the segment
    // length up -- the last segments can then start at or past the end of the
    // loop, and an unsigned wrap there would turn an empty segment into a full
    // one running off the end.
    Value inBounds = FinalClass::emitMin(builder, loc, base, sched.numItersV);
    return FinalClass::emitMin(
        builder, loc, sched.nInnerV,
        FinalClass::emitSub(builder, loc, sched.numItersV, inBounds));
  }

  // The base iteration of segment `index`, counted from the start of the loop.
  static Value segmentBase(OpBuilder &builder, Location loc,
                           const PeriodicSchedule &sched, Value index) {
    return FinalClass::emitMul(builder, loc, index, sched.nInnerV);
  }

  // Materialize trip count / lower bound / step for a dynamic trip count, at
  // the builder's insertion point -- which must dominate the whole scaffold,
  // since every segment bound and induction variable is derived from these.
  static void materializeDynamicSchedule(OpBuilder &builder, Location loc,
                                         OpName forOp,
                                         MGradientUtilsReverse *gutils,
                                         PeriodicSchedule &sched) {
    sched.startV =
        FinalClass::materializeLowerBound(builder, loc, forOp, gutils);
    sched.stepV = FinalClass::materializeStep(builder, loc, forOp, gutils);
    sched.numItersV = FinalClass::castToType(
        builder, loc,
        FinalClass::getNumIterationsValue(builder, loc, forOp, gutils),
        FinalClass::getIndexLikeType(builder));
    FinalClass::materializeSegmentValues(builder, loc, sched);
  }

  static SmallVector<Value> cachePeriodic(OpName forOp, Operation *op,
                                          MGradientUtilsReverse *gutils) {
    Operation *newOpBase = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOpBase);
    Location loc = forOp.getLoc();

    PeriodicSchedule sched = FinalClass::getStaticPeriodicSchedule(forOp);
    if (!FinalClass::getConstantNumberOfIterations(forOp).has_value())
      materializeDynamicSchedule(cacheBuilder, loc, forOp, gutils, sched);
    else
      FinalClass::materializeSegmentValues(cacheBuilder, loc, sched);

    SmallVector<Value> immutableRefs, mutableRefs;
    FinalClass::splitOutsideRefs(forOp, mutableRefs, immutableRefs);

    SmallVector<Value> caches;

    OpName newForOp = cast<OpName>(newOpBase);

    auto newForOpInits = FinalClass::getInits(newForOp);
    auto outerFwd = FinalClass::createForwardOuterLoop(cacheBuilder, loc, sched,
                                                       newForOpInits);
    FinalClass::preserveAttributesButCheckpointing(outerFwd, forOp);

    Block *outerFwdBody = FinalClass::getBodyBlock(outerFwd);
    cacheBuilder.setInsertionPointToStart(outerFwdBody);

    // A segment index, in every dialect and whether or not the trip count is
    // known: the outer loops run one iteration per checkpoint. Recovering the
    // segment's base iteration from it (nInner * index) is left to the hooks,
    // which is what `fwdHint` carries.
    Value outerFwdIV = FinalClass::getInductionVar(outerFwd);

    IRMapping mapping;

    // The bound computation (for scf.for) must happen before the
    // mutable-ref cloning loop below, and the loop itself must be created
    // after it, matching the original code's exact order: canonicalize does
    // not freely reorder ops, so this keeps scf.for's output byte-identical.
    SmallVector<Value> fwdHint = FinalClass::computeForwardSegmentHint(
        cacheBuilder, loc, outerFwdIV, sched);

    SmallVector<Value> mutableRefsCaches;
    for (auto ref : mutableRefs) {
      auto iface = cast<ClonableTypeInterface>(ref.getType());
      auto clone =
          iface.cloneValue(cacheBuilder, gutils->getNewFromOriginal(ref));
      mutableRefsCaches.push_back(
          gutils->initAndPushCache(clone, cacheBuilder));
    }

    auto innerFwd = FinalClass::createForwardSegmentLoop(
        cacheBuilder, loc, outerFwdIV, fwdHint, sched,
        outerFwdBody->getArguments().drop_front());
    FinalClass::preserveAttributesButCheckpointing(innerFwd, forOp);

    Block *innerFwdBody = FinalClass::getBodyBlock(innerFwd);
    FinalClass::setInsertionPointToBodyEnd(cacheBuilder, innerFwdBody);

    Value currentIV = FinalClass::computeForwardSegmentIV(
        cacheBuilder, loc, forOp, outerFwdIV,
        FinalClass::getInductionVar(innerFwd), sched, fwdHint);

    Block *newForOpBody = FinalClass::getBodyBlock(newForOp);
    for (auto [oldArg, newArg] : llvm::zip_equal(newForOpBody->getArguments(),
                                                 innerFwdBody->getArguments()))
      mapping.map(oldArg, newArg);

    mapping.map(FinalClass::getInductionVar(newForOp), currentIV);

    copyBlockWithoutTerminator(cacheBuilder, newForOpBody, gutils, mapping);

    Operation *fwdTerm = newForOpBody->getTerminator();
    SmallVector<Value> fwdYields;
    for (auto operand : FinalClass::getCarriedTerminatorOperands(fwdTerm))
      fwdYields.push_back(mapping.lookupOrDefault(operand));
    FinalClass::createScaffoldYield(cacheBuilder, fwdTerm->getLoc(), fwdYields);

    FinalClass::setInsertionPointToBodyEnd(cacheBuilder, outerFwdBody);
    for (auto initArg : FinalClass::getInits(innerFwd))
      caches.push_back(gutils->initAndPushCache(initArg, cacheBuilder));

    FinalClass::createScaffoldYield(
        cacheBuilder,
        FinalClass::getBodyBlock(forOp)->getTerminator()->getLoc(),
        FinalClass::getCarriedResults(innerFwd));

    cacheBuilder.setInsertionPointAfter(outerFwd);

    caches.append(mutableRefsCaches);

    for (auto ref : immutableRefs)
      caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(ref),
                                                cacheBuilder));

    // A dynamic trip count cannot be recovered from constants in the reverse
    // pass, so transport it along with the bounds it was derived from.
    if (sched.isDynamic()) {
      caches.push_back(gutils->initAndPushCache(sched.numItersV, cacheBuilder));
      caches.push_back(gutils->initAndPushCache(sched.startV, cacheBuilder));
      caches.push_back(gutils->initAndPushCache(sched.stepV, cacheBuilder));
    }

    gutils->replaceOrigOpWith(op, FinalClass::getPrimalResults(
                                      cacheBuilder, loc, forOp,
                                      FinalClass::getCarriedResults(outerFwd),
                                      gutils));
    hoistPlaceholdersBefore(newForOp, newForOp);
    gutils->erase(newForOp);
    gutils->originalToNewFnOps[op] = outerFwd;

    // caches is composed of:
    // [
    //  <caches of iter args>...,
    //  <caches of mutable values>...,
    //  <caches of immutable values>...,
    //  <numIters, start, step -- only when the trip count is dynamic>
    // ]
    return caches;
  }

  static LogicalResult reversePeriodic(OpName forOp, Operation *op,
                                       OpBuilder &builder,
                                       MGradientUtilsReverse *gutils,
                                       SmallVector<Value> caches,
                                       ArrayRef<bool> operandsActive,
                                       ArrayRef<Value> incomingGradients) {
    auto numIterArgs = FinalClass::getNumRegionIterArgs(forOp);
    Location loc = forOp.getLoc();

    PeriodicSchedule sched = FinalClass::getStaticPeriodicSchedule(forOp);

    SmallVector<Value> immutableRefs, mutableRefs;
    FinalClass::splitOutsideRefs(forOp, mutableRefs, immutableRefs);

    IRMapping mapping;

    assert(caches.size() ==
               numIterArgs + mutableRefs.size() + immutableRefs.size() +
                   (FinalClass::getConstantNumberOfIterations(forOp) ? 0 : 3) &&
           "periodic cache layout mismatch");

    for (auto [i, ref] : llvm::enumerate(immutableRefs)) {
      Value refVal = gutils->popCache(
          caches[numIterArgs + mutableRefs.size() + i], builder);
      mapping.map(ref, refVal);
    }

    // The trailing three caches carry what a dynamic trip count cannot get
    // from constants; the segment count is recomputed from them here rather
    // than transported, since it is a pure function of the trip count.
    if (!FinalClass::getConstantNumberOfIterations(forOp).has_value()) {
      size_t base = numIterArgs + mutableRefs.size() + immutableRefs.size();
      sched.numItersV = gutils->popCache(caches[base], builder);
      sched.startV = gutils->popCache(caches[base + 1], builder);
      sched.stepV = gutils->popCache(caches[base + 2], builder);
    }
    FinalClass::materializeSegmentValues(builder, loc, sched);

    auto revOuter = FinalClass::createReverseOuterLoop(builder, loc, sched,
                                                       incomingGradients);
    FinalClass::preserveAttributesButCheckpointing(revOuter, forOp);

    OpBuilder::InsertionGuard guard(builder);
    Block *revOuterBody = FinalClass::getBodyBlock(revOuter);
    // The same segment index as the forward direction, counted from the end:
    // segment numSegments() - 1 is replayed first. `revHint` carries whatever
    // the hooks derive from it.
    Value revOuterIV = FinalClass::getInductionVar(revOuter);
    FinalClass::setInsertionPointToBodyEnd(builder, revOuterBody);

    SmallVector<Value> cachedOutsideRefs;
    for (auto [i, ref] : llvm::enumerate(mutableRefs)) {
      Value refVal = gutils->popCache(caches[numIterArgs + i], builder);
      cachedOutsideRefs.push_back(refVal);
      mapping.map(ref, refVal);
    }

    // Must happen here, before the initArgs pop loop below, matching the
    // original code's exact order (canonicalize does not freely reorder
    // ops, so this keeps scf.for's output byte-identical). Whatever the dialect
    // returns is handed back to it unchanged by the two hooks below; it is
    // computed once, here, only so that its ops land at this point.
    SmallVector<Value> revHint =
        FinalClass::computeReverseSegmentHint(builder, loc, revOuterIV, sched);

    SmallVector<Value> initArgs(numIterArgs, nullptr);
    for (size_t i = 0; i < numIterArgs; ++i) {
      initArgs[i] = gutils->popCache(caches[i], builder);
    }

    auto revInner = FinalClass::createReverseSegmentLoop(
        builder, loc, revOuterIV, revHint, sched, initArgs);
    FinalClass::preserveAttributesButCheckpointing(revInner, forOp);

    Block *revInnerBody = FinalClass::getBodyBlock(revInner);
    FinalClass::setInsertionPointToBodyEnd(builder, revInnerBody);

    Value currentIV = FinalClass::computeReverseSegmentIV(
        builder, loc, forOp, revOuterIV, FinalClass::getInductionVar(revInner),
        sched, revHint);

    Block *origBodyBlock = FinalClass::getBodyBlock(forOp);
    for (auto [oldArg, newArg] : llvm::zip_equal(origBodyBlock->getArguments(),
                                                 revInnerBody->getArguments()))
      mapping.map(oldArg, newArg);

    mapping.map(FinalClass::getInductionVar(forOp), currentIV);

    copyBlockWithoutTerminator(builder, origBodyBlock, gutils, mapping);
    Operation *segTerm = origBodyBlock->getTerminator();
    SmallVector<Value> segYields;
    for (auto operand : FinalClass::getCarriedTerminatorOperands(segTerm))
      segYields.push_back(mapping.lookupOrDefault(operand));
    FinalClass::createScaffoldYield(builder, segTerm->getLoc(), segYields);

    FinalClass::setInsertionPointToBodyEnd(builder, revOuterBody);

    for (auto outsideRef : cachedOutsideRefs) {
      if (auto cachableT =
              dyn_cast<ClonableTypeInterface>(outsideRef.getType())) {
        cachableT.freeClonedValue(builder, outsideRef);
      }
    }

    auto revLoop = FinalClass::createLoopWithSameBounds(
        builder, loc, revInner, revOuterBody->getArguments().drop_front());
    FinalClass::preserveAttributesButCheckpointing(revLoop, forOp);

    Block *revLoopBody = FinalClass::getBodyBlock(revLoop);
    Block *origBody = origBodyBlock;

    FinalClass::primeSegmentGradients(builder, gutils, origBody, revLoopBody,
                                      operandsActive);

    FinalClass::setInsertionPointToBodyEnd(builder, revLoopBody);
    int revIdx = 1;
    for (auto &&[active, operand] : llvm::zip_equal(
             operandsActive, FinalClass::getCarriedTerminatorOperands(
                                 origBody->getTerminator()))) {
      if (active) {
        gutils->addToDiffe(operand, revLoopBody->getArgument(revIdx), builder);
        revIdx++;
      }
    }

    bool valid = true;

    auto first = origBody->rbegin();
    first++; // skip terminator

    auto last = origBody->rend();

    {
      // Any cache a body op's reverse rule needs is per-segment, so it has to
      // be created outside the replayed segment -- `revInner` is the anchor the
      // dialect gets to place it before.
      SegmentCacheCreatorGuard cacheGuard(gutils, revInner);
      for (auto it = first; it != last; ++it) {
        Operation *o = &*it;
        valid &= gutils->Logic.visitChild(o, builder, gutils).succeeded();
      }
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

    FinalClass::setInsertionPointToBodyEnd(builder, revLoopBody);
    FinalClass::createScaffoldYield(
        builder, origBody->getTerminator()->getLoc(), newResults);

    FinalClass::setInsertionPointToBodyEnd(builder, revOuterBody);
    FinalClass::createScaffoldYield(builder,
                                    origBody->getTerminator()->getLoc(),
                                    FinalClass::getCarriedResults(revLoop));

    builder.setInsertionPointAfter(revOuter);

    revIdx = 0;
    auto revOuterResults = FinalClass::getCarriedResults(revOuter);
    auto forOpInits = FinalClass::getInits(forOp);
    for (auto &&[active, arg] : llvm::zip_equal(operandsActive, forOpInits)) {
      if (active) {
        if (!gutils->isConstantValue(arg)) {
          gutils->addToDiffe(arg, revOuterResults[revIdx], builder);
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
    if (!FinalClass::needsBinomialCheckpointing(forOp) &&
        !FinalClass::needsCheckpointing(forOp))
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

    if (FinalClass::needsBinomialCheckpointing(forOp)) {
      auto budget = FinalClass::getCheckpointBudget(forOp);
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

    if (FinalClass::needsCheckpointing(forOp))
      return cachePeriodic(forOp, op, gutils);

    return std::nullopt;
  }

  static std::optional<LogicalResult> tryCreateReverseModeAdjoint(
      OpName forOp, Operation *op, OpBuilder &builder,
      MGradientUtilsReverse *gutils, SmallVector<Value> caches,
      ArrayRef<bool> operandsActive, ArrayRef<Value> incomingGradients) {
    if (!FinalClass::needsBinomialCheckpointing(forOp) &&
        !FinalClass::needsCheckpointing(forOp))
      return std::nullopt;

    // Mirrors the guard in tryCacheValues; already reported there, but
    // createReverseModeAdjoint must still fail (not fall through to the
    // plain-loop path, whose cache layout wouldn't match what cacheValues
    // actually cached).
    if (failed(FinalClass::requireSingleResultBounds(forOp)))
      return failure();

    if (FinalClass::needsBinomialCheckpointing(forOp)) {
      auto budget = FinalClass::getCheckpointBudget(forOp);
      if (!budget || *budget <= 1) {
        op->emitError() << "binomial checkpointing requires a "
                        << FinalClass::checkpointPeriodAttrName()
                        << " attribute greater than 1";
        return failure();
      }
      return reverseBinomial(forOp, *budget, builder, gutils, caches,
                             operandsActive, incomingGradients);
    }

    if (FinalClass::needsCheckpointing(forOp))
      return reversePeriodic(forOp, op, builder, gutils, caches, operandsActive,
                             incomingGradients);

    return std::nullopt;
  }
};

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_IMPLEMENTATIONS_LOOPCHECKPOINTING_H
