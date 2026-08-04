//===- HoistLoopAllocations.cpp - Hoist per-iteration scratch buffers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that lifts a matched allocation/deallocation
// pair out of a sequential loop, so a scratch buffer is allocated once for the
// whole loop instead of once per iteration.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_HOISTLOOPALLOCATIONSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

/// Returns the buffer allocated by `op`, if `op` does nothing but allocate it.
/// An op with any further effect (a read, a write, a second allocation) is not
/// something we can replay once for the whole loop.
static Value getAllocatedBuffer(Operation *op) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface || op->getNumResults() != 1 ||
      !isa<MemRefType>(op->getResult(0).getType()))
    return {};

  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);

  bool allocates = false;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Allocate>(effect.getEffect()))
      return {};
    if (effect.getValue() != op->getResult(0))
      return {};
    allocates = true;
  }
  return allocates ? op->getResult(0) : Value();
}

/// Walks everything reachable from `buffer` and returns the single op that
/// frees it, or null if the buffer is not a self-contained per-iteration
/// scratch buffer.
///
/// This is the whole legality argument of the transformation. Hoisting makes
/// every iteration share one buffer, so it is only correct when nothing
/// derived from the buffer survives the iteration that allocated it: no yield
/// through `iter_args`, no store of the pointer, no call that might capture.
/// Rather than enumerate the ways a buffer can escape, we accept only the uses
/// we can prove harmless -- plain reads and writes, and views that are
/// themselves only read and written -- and reject the rest.
static Operation *findPairedDealloc(Value buffer, Operation *loop) {
  SmallVector<Value> worklist{buffer};
  SmallPtrSet<Value, 4> seen;
  Operation *dealloc = nullptr;

  while (!worklist.empty()) {
    Value alias = worklist.pop_back_val();
    if (!seen.insert(alias).second)
      continue;

    for (OpOperand &use : alias.getUses()) {
      Operation *user = use.getOwner();

      // Values defined inside the loop cannot be used after it, so this only
      // fires for uses in a sibling region of a multi-region loop.
      if (!loop->isProperAncestor(user))
        return nullptr;

      if (hasSingleEffect<MemoryEffects::Free>(user, alias)) {
        // Only a free of the allocation itself can move after the loop. A free
        // of a view would leave the view op behind in the body, where the
        // dealloc can no longer reach it.
        if (alias != buffer || dealloc)
          return nullptr;
        dealloc = user;
        continue;
      }

      // A view is re-derived from the buffer each iteration, so it stays in
      // the body; we just have to keep following it.
      if (isa<ViewLikeOpInterface>(user)) {
        worklist.append(user->result_begin(), user->result_end());
        continue;
      }

      // A terminator hands the buffer to the next iteration or out of the
      // loop; a region we do not model may do anything with it.
      if (user->hasTrait<OpTrait::IsTerminator>() || user->getNumRegions() != 0)
        return nullptr;

      auto iface = dyn_cast<MemoryEffectOpInterface>(user);
      if (!iface)
        return nullptr;

      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffectsOnValue(alias, effects);
      if (effects.empty())
        return nullptr;
      for (const MemoryEffects::EffectInstance &effect : effects)
        if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
          return nullptr;

      // A result that may alias the buffer without being a recognized view is
      // outside what this analysis can follow.
      if (llvm::any_of(user->getResultTypes(), llvm::IsaPred<MemRefType>))
        return nullptr;
    }
  }

  return dealloc;
}

/// Hoisting turns a sum of per-iteration live ranges into a set of overlapping
/// ones, so a body with several scratch buffers can raise peak memory. The
/// budget lets a caller bound that; a dynamically sized buffer has no size we
/// can check it against.
static bool fitsBudget(Value buffer, uint64_t maxHoistedBytes) {
  if (maxHoistedBytes == 0)
    return true;

  auto type = cast<MemRefType>(buffer.getType());
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return false;

  uint64_t bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8);
  return type.getNumElements() * bytes <= maxHoistedBytes;
}

/// Only sequential loops qualify: hoisting out of a parallel loop gives every
/// concurrent iteration the same buffer.
static bool isSupportedLoop(Operation *op) {
  return isa<scf::ForOp, affine::AffineForOp>(op);
}

} // end anonymous namespace

bool mlir::enzyme::hoistLoopAllocations(LoopLikeOpInterface loop,
                                        uint64_t maxHoistedBytes) {
  if (!isSupportedLoop(loop))
    return false;

  SmallVector<Region *> regions = loop.getLoopRegions();
  if (regions.size() != 1 || !regions.front()->hasOneBlock())
    return false;
  Block *body = &regions.front()->front();

  // Collect first: hoisting rewrites the block we are iterating over.
  SmallVector<std::pair<Operation *, Operation *>> pairs;
  for (Operation &op : *body) {
    Value buffer = getAllocatedBuffer(&op);
    if (!buffer || !fitsBudget(buffer, maxHoistedBytes))
      continue;

    // The size, alignment and memory space have to mean the same thing before
    // the loop as they did inside it.
    if (!llvm::all_of(op.getOperands(), [&](Value operand) {
          return loop.isDefinedOutsideOfLoop(operand);
        }))
      continue;

    Operation *dealloc = findPairedDealloc(buffer, loop);

    // v1 requires both ops directly in the body's entry block, which is the
    // shape bufferization emits. It is stricter than necessary -- the real
    // condition is that the dealloc post-dominates the alloc within the body,
    // which also admits an alloc/dealloc pair nested together under the same
    // `scf.if` -- but it needs no dominance information to check.
    if (!dealloc || dealloc->getBlock() != body)
      continue;

    pairs.emplace_back(&op, dealloc);
  }

  for (auto [alloc, dealloc] : pairs) {
    alloc->moveBefore(loop);
    dealloc->moveAfter(loop);
  }
  return !pairs.empty();
}

namespace {

struct HoistLoopAllocationsPass
    : public enzyme::impl::HoistLoopAllocationsPassBase<
          HoistLoopAllocationsPass> {
  using HoistLoopAllocationsPassBase::HoistLoopAllocationsPassBase;

  void runOnOperation() override {
    // Post-order gives us innermost loops first. Hoisting out of an inner loop
    // leaves the pair in the enclosing body, where the enclosing loop's turn
    // lifts it another level, so a nest is drained bottom-up in one walk.
    SmallVector<LoopLikeOpInterface> loops;
    getOperation()->walk([&](LoopLikeOpInterface loop) {
      if (isSupportedLoop(loop))
        loops.push_back(loop);
    });

    for (LoopLikeOpInterface loop : loops)
      hoistLoopAllocations(loop, maxHoistedBytes);
  }
};

} // end anonymous namespace
