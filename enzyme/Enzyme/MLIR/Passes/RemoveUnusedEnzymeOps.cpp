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

#include "mlir/IR/Dominance.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace enzyme;
namespace {


// Starting at the beginning of blk, is there a path that can execute
// check before end. 
bool mayExecuteBefore(Block* blk, Operation* check, Operation *end) {
  auto reg = blk->getParent();
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
        // The sole exception to this is if they are in the same sub region, which is 
        // known to execute only once. TODO this later
        /*
        if (op.isAncestor(end)) {

          for (auto reg2 : op.getRegions()) {

          }
        }
        */

        return true;
      }

      // Otherwise if we've seen the end op, this path is over as the route we found here
      // didn't first find a check.
      if (op.isAncestor(end)) {
        seenEnd = true;
        break;
      }
    }

    if (seenEnd) continue;

    // If we didn't find the end, try all successors
    for (auto succ : cur->getSuccessors()) {
      blocksToVisit.push_back(succ);
    }
  }

  return false;
}

bool mayExecuteBetween(Operation *start, Operation* check, Operation *end) {

  for (auto op = start->getNextNode(); op != nullptr; op++) {
    // This check op has been found after start in its block 
    if (op->isAncestor(check)) {
      return true;
    }
    if (op->isAncestor(end)) {
      return false;
    }
  }

  Block* blk = start->getBlock();

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
template <class T>
T findNearestDominatingOpByUse(Operation *op, Value v) {
  DominanceInfo dInfo;
  PostDominanceInfo pdInfo;

  SmallVector<T, 1> options;
  for (Operation *userSet : v.getUsers()) {
    if (auto setOp = dyn_cast<T>(userSet)) {
      options.push_back(setOp);
    }
  }
  if (options.size() == 1 && dInfo.dominates(options[0], op))
    return options[0];

  llvm::errs() << " scope: " << *op->getParentOp() << "\n";
    llvm::errs() << "  want to replace " << *op << "\n";
  for (auto opt : options) {
    if (!dInfo.dominates(opt, op))
      continue;
    bool conflict = false;
    llvm::errs() << " trying: " << *opt << "\n";
    for (auto opt2 : options) {
      if (opt == opt2) continue;

      llvm::errs() << " conflict check: " << *opt2 << "\n";

      if (!mayExecuteBetween(opt, opt2, op)) {
        llvm::errs() << " + known good since occurs before store\n";
        continue;
      }

      conflict = true;
    }
    if (!conflict) {
      llvm::errs() << " - replaced with " << *opt << "\n";
      return opt;
    }
  }

  return nullptr;
}

struct RemoveUnusedEnzymeOpsPass
    : public enzyme::RemoveUnusedEnzymeOpsPassBase<RemoveUnusedEnzymeOpsPass> {
  void runOnOperation() override {

    SmallVector<enzyme::InitOp, 1> inits;
    getOperation()->walk([&](Operation *op) {
      if (auto initOp = dyn_cast<enzyme::InitOp>(op)) {
        inits.push_back(initOp);
      }
    });

    for (auto initOp : inits) {
      DominanceInfo dInfo;
        Value v = initOp;
        if (auto type = dyn_cast<enzyme::GradientType>(initOp.getType())) {
          bool replaceable = true;
          for (Operation *userSet : v.getUsers()) {
            if (isa<enzyme::SetOp>(userSet)) continue;
            if (isa<enzyme::GetOp>(userSet)) continue;
            llvm::errs() << " unknown user of grad: " << *userSet << "\n";
            replaceable = false;
          }
          if (replaceable) {
            // Do replacing
            bool allDelete = true;
            for (Operation *userGet : make_early_inc_range(v.getUsers())) {
              if (auto getOp = dyn_cast<enzyme::GetOp>(userGet)) {
                if (auto setOp =
                    findNearestDominatingOpByUse<enzyme::SetOp>(userGet, v)) {
                  getOp.replaceAllUsesWith(setOp.getValue());
                  getOp->erase();
                  continue;
                }
                allDelete = false;
              }
            }
            if (allDelete) {
              for (Operation *userGet : make_early_inc_range(v.getUsers())) {
                userGet->erase();
              }
              initOp->erase();
            }
            continue;
          }
        } else if (auto type = dyn_cast<enzyme::CacheType>(initOp.getType())) {
          bool replaceable = true;

          SmallVector<enzyme::PopOp, 1> pops;
          for (Operation *userSet : v.getUsers()) {
            if (isa<enzyme::PushOp>(userSet)) continue;
            if (auto pop = dyn_cast<enzyme::PopOp>(userSet)) {
              pops.push_back(pop);
              continue;
            }
            llvm::errs() << " unknown user of cache: " << *userSet << "\n";
            replaceable = false;
          }

          if (replaceable) 
          for (auto pop : pops) {
            if (auto push = findNearestDominatingOpByUse<enzyme::PushOp>(pop, v)) {
              pop.replaceAllUsesWith(push.getValue());
              pop->erase();
              push->erase();
            }
          }
          if (v.use_empty()) {            
            initOp->erase();
          }
          continue;
        }
    }
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
