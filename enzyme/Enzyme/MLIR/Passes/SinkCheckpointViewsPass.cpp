//===- SinkCheckpointViewsPass.cpp - Sink views into checkpointed loops ---- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that sinks pure views of mutable values into the
// checkpointed loops that use them, so that a view and the value it views are
// never captured by the same loop.
//
//===----------------------------------------------------------------------===//

#include "Interfaces/AutoDiffTypeInterface.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace enzyme {
using namespace mlir::enzyme;
#define GEN_PASS_DEF_SINKCHECKPOINTVIEWSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
using namespace mlir;
using namespace enzyme;

// Checkpointing snapshots every mutable value a loop body uses from outside it,
// so that the reverse pass can replay a segment of the primal from a restored
// state. When both a value and a pure view of that value are captured, both get
// snapshotted -- but they are the same memory, so the loop ends up recomputing
// from two independent copies of one buffer, and only the writes that happen to
// go through the copy that was restored last are visible. No alias analysis
// makes that correct; the two clones are the bug.
//
// It also fails much more loudly than that on a GPU: the view's type carries no
// memory space (a `memref<?xi8>` view of a device pointer looks like host
// memory), so its snapshot is a host malloc/memcpy of device memory, which
// segfaults before any wrong answer can be observed.
//
// Sinking the view into the loop region leaves only its source captured. The
// view is then re-derived inside the body from whatever the source resolves to,
// which during a replay is the one restored checkpoint.
static bool isCheckpointed(Operation *op) {
  auto enable = op->getAttrOfType<BoolAttr>("enzyme.enable_checkpointing");
  return enable && enable.getValue();
}

/// A view op is safe to re-evaluate anywhere its operands are available, which
/// is what makes sinking it a no-op semantically.
static bool isSinkableView(Operation *op) {
  return op && isa<ViewLikeOpInterface>(op) && isMemoryEffectFree(op) &&
         op->getNumResults() == 1 && op->getNumRegions() == 0;
}

/// Replaces uses of `view`'s result inside `loop` with a copy of `view` placed
/// at the top of each of the loop's region entry blocks. `view`'s operands are
/// defined above the loop, so they dominate those insertion points.
static void sinkViewInto(Operation *loop, Operation *view) {
  Value result = view->getResult(0);

  for (Region &region : loop->getRegions()) {
    if (region.empty())
      continue;

    // Only clone if this region actually reads the view.
    bool used = llvm::any_of(result.getUses(), [&](OpOperand &use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
    if (!used)
      continue;

    OpBuilder builder(&region.front(), region.front().begin());
    Operation *sunk = builder.clone(*view);
    result.replaceUsesWithIf(sunk->getResult(0), [&](OpOperand &use) {
      return use.getOwner() != sunk &&
             region.isAncestor(use.getOwner()->getParentRegion());
    });
  }
}

/// Sinks captured views until none of `loop`'s captures is a view of a mutable
/// value. Iterates because sinking a view exposes its source as a new capture,
/// which may itself be a view.
static void sinkCapturedViews(Operation *loop) {
  bool changed = true;
  while (changed) {
    changed = false;

    SetVector<Value> captures;
    getUsedValuesDefinedAbove(loop->getRegions(), captures);

    for (Value capture : captures) {
      // Only mutable captures get snapshotted, so only they can be double
      // cloned; sinking anything else would just be code motion.
      if (!isa<ClonableTypeInterface>(capture.getType()))
        continue;

      Operation *def = capture.getDefiningOp();
      if (!isSinkableView(def))
        continue;

      sinkViewInto(loop, def);
      changed = true;
      break; // captures is stale now
    }
  }
}

struct SinkCheckpointViewsPass
    : public enzyme::impl::SinkCheckpointViewsPassBase<
          SinkCheckpointViewsPass> {
  using SinkCheckpointViewsPassBase::SinkCheckpointViewsPassBase;

  void runOnOperation() override {
    SmallVector<Operation *> loops;
    getOperation()->walk([&](LoopLikeOpInterface loop) {
      if (isCheckpointed(loop))
        loops.push_back(loop);
    });

    for (Operation *loop : loops)
      sinkCapturedViews(loop);
  }
};

} // end anonymous namespace
