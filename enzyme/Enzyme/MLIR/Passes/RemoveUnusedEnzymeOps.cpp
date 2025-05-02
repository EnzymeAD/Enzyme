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
#include "Interfaces/AutoDiffOpInterface.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "Passes/RemovalUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  Block *blk = start->getBlock();
  auto checkAnc = check;
  while (checkAnc && checkAnc->getBlock() != blk)
    checkAnc = checkAnc->getParentOp();

  auto endAnc = end;
  while (endAnc && endAnc->getBlock() != blk)
    endAnc = endAnc->getParentOp();

  // Both operations are either the same, or in a joint parent (which could
  // rerun)
  if (checkAnc && checkAnc == endAnc)
    return true;

  if (checkAnc) {
    if (checkAnc->isBeforeInBlock(start))
      checkAnc = nullptr;
  }
  if (endAnc) {
    if (endAnc->isBeforeInBlock(start))
      endAnc = nullptr;
  }

  if (checkAnc && endAnc) {
    if (checkAnc->isBeforeInBlock(endAnc))
      return true;
    else
      return false;
  }

  if (checkAnc)
    return true;
  if (endAnc)
    return false;

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

static void applyPatterns(Operation *op) {
  RewritePatternSet patterns(op->getContext());
  patterns.insert<PopSimplify, GetSimplify, PushSimplify, SetSimplify,
                  InitSimplify>(op->getContext());

  GreedyRewriteConfig config;
  config.enableFolding();
  (void)applyPatternsGreedily(op, std::move(patterns), config);
}

// A worklist that supports removing operations
// original implementation is from
// https://github.com/llvm/llvm-project/blob/9d8d538e40ef040cb53e8db7a32f3024865187f3/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L198
class Worklist {
public:
  Worklist() { list.reserve(8); }

  bool empty();
  void push(Operation *op);
  void remove(Operation *op);
  Operation *pop();
  void reverse();

private:
  std::vector<Operation *> list;
  llvm::DenseMap<Operation *, unsigned> map;
};

bool Worklist::empty() {
  // Skip all nullptr.
  return !llvm::any_of(list,
                       [](Operation *op) { return static_cast<bool>(op); });
}

void Worklist::push(Operation *op) {
  assert(op && "cannot push nullptr to worklist");
  // Check to see if the worklist already contains this op.
  if (!map.insert({op, list.size()}).second)
    return;
  list.push_back(op);
}

void Worklist::reverse() {
  std::reverse(list.begin(), list.end());
  for (size_t i = 0, e = list.size(); i < e; ++i)
    map[list[i]] = i;
}

Operation *Worklist::pop() {
  // Skip and remove all trailing nullptr.
  while (!list.back())
    list.pop_back();
  Operation *op = list.back();
  list.pop_back();
  map.erase(op);
  // Cleanup: Remove all trailing nullptr.
  while (!list.empty() && !list.back())
    list.pop_back();
  return op;
}

void Worklist::remove(Operation *op) {
  assert(op && "cannot remove nullptr from worklist");
  auto it = map.find(op);
  if (it != map.end()) {
    assert(list[it->second] == op && "malformed worklist data structure");
    list[it->second] = nullptr;
    map.erase(it);
  }
}

// Drives Enzyme ops removal with the following goals:
//  * Each EnzymeOpsRemoverOpInterface should be processed once.
//  * Inserted ops next in the post order should still be run.
class PostOrderWalkDriver : public RewriterBase::Listener {
public:
  PostOrderWalkDriver(Operation *root_) : root(root_) {}

  void initializeWorklist();
  LogicalResult processWorklist();

protected:
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;
  void notifyOperationErased(Operation *op) override;

  void notifyMatchFailure(Location loc,
                          function_ref<void(Diagnostic &)> reasonCallback);

private:
  void addToWorklist(Operation *op);

  Worklist worklist;

  Operation *current = nullptr;
  Operation *root;
};

void PostOrderWalkDriver::addToWorklist(Operation *op) {
  // This driver only processes EnzymeOpsRemoverOpInterface ops.
  if (!isa<EnzymeOpsRemoverOpInterface>(op))
    return;
  worklist.push(op);
}

void PostOrderWalkDriver::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  if (!isa<EnzymeOpsRemoverOpInterface>(op))
    return;

  if (!current) {
    addToWorklist(op);
    return;
  }

  // Check if the inserted op would be next in the post order or not compared to
  // the current operation.
  bool shouldInsert = false;
  (void)root->walk([&](EnzymeOpsRemoverOpInterface iface) {
    if ((Operation *)iface == current) {
      shouldInsert = true;
      return WalkResult::interrupt();
    }

    if ((Operation *)iface == op) {
      shouldInsert = false;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (shouldInsert)
    addToWorklist(op);
}

void PostOrderWalkDriver::notifyOperationErased(Operation *op) {
  if (op == current) {
    current = nullptr;
  }
  if (!isa<EnzymeOpsRemoverOpInterface>(op))
    return;
  worklist.remove(op);
}

void PostOrderWalkDriver::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  auto diag = mlir::emitError(loc);
  reasonCallback(*diag.getUnderlyingDiagnostic());
}

void PostOrderWalkDriver::initializeWorklist() {
  root->walk(
      [this](EnzymeOpsRemoverOpInterface iface) { addToWorklist(iface); });
  worklist.reverse();
}

LogicalResult PostOrderWalkDriver::processWorklist() {
  PatternRewriter rewriter(root->getContext());
  rewriter.setListener(this);

  bool result = true;
  while (!worklist.empty()) {
    auto op = worklist.pop();
    auto iface = cast<EnzymeOpsRemoverOpInterface>(op);
    current = op;
    rewriter.setInsertionPoint(current);
    result &= iface.removeEnzymeOps(rewriter).succeeded();
    current = nullptr;
  }

  return LogicalResult::success(result);
}

struct RemoveUnusedEnzymeOpsPass
    : public enzyme::RemoveUnusedEnzymeOpsPassBase<RemoveUnusedEnzymeOpsPass> {
  void runOnOperation() override {
    auto op = getOperation();

    applyPatterns(op);

    bool failed = false;
    op->walk([&](FunctionOpInterface func) {
      PostOrderWalkDriver driver(func);
      driver.initializeWorklist();
      failed |= driver.processWorklist().failed();
    });

    if (failed) {
      signalPassFailure();
      return;
    }

    applyPatterns(op);
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
