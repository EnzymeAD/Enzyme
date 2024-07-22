//===- RemoveUnusedEnzymeOps.cpp - Remove unnecessary or unused gradient and
// cache ops
//------------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/Dominance.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace enzyme;
namespace {

// Starting at the beginning of blk, is there a path that can execute
// check before end.
bool mayExecuteBefore(Block *blk, Operation *check, Operation *end) {
  auto reg = blk->getParent();
  (void)reg;
  assert(reg->isAncestor(end->getParentRegion()));

  DenseSet<Block *> visitedBlocks;

  SmallVector<Block *> blocksToVisit;
  for (auto succ : blk->getSuccessors()) {
    blocksToVisit.push_back(succ);
  }

  while (!blocksToVisit.empty()) {
    Block *cur = blocksToVisit.pop_back_val();

    if (visitedBlocks.contains(cur))
      continue;

    visitedBlocks.insert(cur);

    bool seenEnd = false;
    for (auto &op : *cur) {

      // If we've seen the thing to check with, it may execute before
      if (op.isAncestor(check)) {
        // The sole exception to this is if they are in the same sub region,
        // which is known to execute only once. TODO this later
        /*
        if (op.isAncestor(end)) {

          for (auto reg2 : op.getRegions()) {

          }
        }
        */

        return true;
      }

      // Otherwise if we've seen the end op, this path is over as the route we
      // found here didn't first find a check.
      if (op.isAncestor(end)) {
        seenEnd = true;
        break;
      }
    }

    if (seenEnd)
      continue;

    // If we didn't find the end, try all successors
    for (auto succ : cur->getSuccessors()) {
      blocksToVisit.push_back(succ);
    }
  }

  return false;
}

bool mayExecuteBetween(Operation *start, Operation *check, Operation *end) {

  for (auto op = start->getNextNode(); op != nullptr; op = op->getNextNode()) {
    // This check op has been found after start in its block
    if (op->isAncestor(check)) {
      return true;
    }
    if (op->isAncestor(end)) {
      return false;
    }
  }

  Block *blk = start->getBlock();

  auto reg = blk->getParent();
  if (reg->isAncestor(end->getParentRegion())) {
    return mayExecuteBefore(blk, check, end);
  }

  // If the check is in the parent op, but the end is not, assume
  // we may execute that parent op part before going to any later ops
  if (reg->isAncestor(check->getParentRegion())) {
    return true;
  }

  return mayExecuteBetween(start->getParentOp(), check, end);
}

// TODO this isn't necessarily correct. This is because there could be a
// non dominating use bewteen the dominating one and the op, causing
// correctness issues when not seen. In interim, be conservative and only
// succeed if these have the same parent block, and no other ops in path
template <class T, class T2 = T>
T findNearestDominatingOpByUse(Operation *op, Value v) {
  DominanceInfo dInfo;
  PostDominanceInfo pdInfo;

  SmallVector<T, 1> options;
  SmallVector<Operation *, 1> conflicts;
  for (Operation *userSet : v.getUsers()) {
    if (auto setOp = dyn_cast<T>(userSet)) {
      options.push_back(setOp);
      conflicts.push_back(setOp);
      continue;
    }
    if (auto setOp = dyn_cast<T2>(userSet)) {
      conflicts.push_back(setOp);
      continue;
    }
  }

  for (auto opt : options) {
    if (!dInfo.dominates(opt, op))
      continue;
    bool conflict = false;
    for (auto opt2 : conflicts) {
      if (opt == opt2)
        continue;
      if (opt2 == op)
        continue;

      if (!mayExecuteBetween(opt, opt2, op)) {
        continue;
      }

      conflict = true;
    }
    if (!conflict) {
      return opt;
    }
  }

  return nullptr;
}

struct PopSimplify : public OpRewritePattern<enzyme::PopOp> {
  using OpRewritePattern<enzyme::PopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::PopOp pop,
                                PatternRewriter &rewriter) const final {

    auto init = pop.getCache().getDefiningOp<enzyme::InitOp>();
    if (!init)
      return failure();

    SmallVector<enzyme::PopOp, 1> pops;
    SmallVector<enzyme::PushOp, 1> pushes;
    for (Operation *userSet : init.getResult().getUsers()) {
      if (auto push = dyn_cast<enzyme::PushOp>(userSet)) {
        pushes.push_back(push);
        continue;
      }
      if (auto pop = dyn_cast<enzyme::PopOp>(userSet)) {
        pops.push_back(pop);
        continue;
      }
      return failure();
    }

    if (auto push = findNearestDominatingOpByUse<enzyme::PushOp, enzyme::PopOp>(
            pop, init)) {
      // Do the block check to conservatively avoid multi execute push/pop
      if (pop->getBlock() == push->getBlock()) {
        rewriter.replaceOp(pop, push.getValue());
        rewriter.eraseOp(push);
        return success();
      }
    }

    return failure();
  }
};

struct GetSimplify : public OpRewritePattern<enzyme::GetOp> {
  using OpRewritePattern<enzyme::GetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::GetOp get,
                                PatternRewriter &rewriter) const final {

    auto init = get.getGradient().getDefiningOp<enzyme::InitOp>();
    if (!init)
      return failure();

    for (Operation *userSet : init.getResult().getUsers()) {
      if (isa<enzyme::GetOp>(userSet))
        continue;
      if (isa<enzyme::SetOp>(userSet))
        continue;
      return failure();
    }

    if (auto set = findNearestDominatingOpByUse<enzyme::SetOp>(get, init)) {
      rewriter.replaceOp(get, set.getValue());
      return success();
    }
    return failure();
  }
};

struct SetSimplify : public OpRewritePattern<enzyme::SetOp> {
  using OpRewritePattern<enzyme::SetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::SetOp get,
                                PatternRewriter &rewriter) const final {

    auto init = get.getGradient().getDefiningOp<enzyme::InitOp>();
    if (!init)
      return failure();

    for (Operation *userSet : init.getResult().getUsers()) {
      if (isa<enzyme::SetOp>(userSet))
        continue;
      return failure();
    }

    rewriter.eraseOp(get);
    return success();
  }
};

struct PushSimplify : public OpRewritePattern<enzyme::PushOp> {
  using OpRewritePattern<enzyme::PushOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::PushOp get,
                                PatternRewriter &rewriter) const final {

    auto init = get.getCache().getDefiningOp<enzyme::InitOp>();
    if (!init)
      return failure();

    for (Operation *userSet : init.getResult().getUsers()) {
      if (isa<enzyme::PushOp>(userSet))
        continue;
      return failure();
    }

    rewriter.eraseOp(get);
    return success();
  }
};

struct InitSimplify : public OpRewritePattern<enzyme::InitOp> {
  using OpRewritePattern<enzyme::InitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::InitOp get,
                                PatternRewriter &rewriter) const final {

    if (get.use_empty()) {
      rewriter.eraseOp(get);
      return success();
    }
    return failure();
  }
};

struct RemoveUnusedEnzymeOpsPass
    : public enzyme::RemoveUnusedEnzymeOpsPassBase<RemoveUnusedEnzymeOpsPass> {
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    patterns.insert<PopSimplify, GetSimplify, PushSimplify, SetSimplify,
                    InitSimplify>(&getContext());

    GreedyRewriteConfig config;
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createRemoveUnusedEnzymeOpsPass() {
  return std::make_unique<RemoveUnusedEnzymeOpsPass>();
}
} // namespace enzyme
} // namespace mlir
