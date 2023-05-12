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
using llvm::errs;
namespace {

// TODO: Expand to region branches??
bool reachable(Operation *a, Operation *b) {
  Block *aBlock = a->getBlock();
  Block *bBlock = b->getBlock();
  if (aBlock == bBlock) {
    if (a->isBeforeInBlock(b)) {
      return true;
    }
  }
  DenseSet<Block *> visitedBlocks;
  SmallVector<Block *> blocksToVisit;

  blocksToVisit.push_back(aBlock);
  while (!blocksToVisit.empty()) {
    Block *processedBlock = blocksToVisit[blocksToVisit.size() - 1];
    blocksToVisit.pop_back();

    for (Block *successor : processedBlock->getSuccessors()) {
      if (!visitedBlocks.contains(successor)) {
        visitedBlocks.insert(successor);
        blocksToVisit.push_back(successor);

        if (successor == bBlock)
          return true;
      }
    }
  }
  return false;
}

template <class T>
Operation *findNearestDominatingOpByUse(Operation *op, Value v) {
  DominanceInfo dInfo;

  Operation *closestSetOp = nullptr;
  for (Operation *userSet : v.getUsers()) {
    if (auto setOp = dyn_cast<T>(userSet)) {
      if (dInfo.dominates(userSet, op)) {
        if (closestSetOp == nullptr) {
          closestSetOp = userSet;
        } else if (dInfo.dominates(closestSetOp, userSet)) {
          closestSetOp = userSet;
        }
      }
    }
  }
  return closestSetOp;
}

struct RemoveUnusedEnzymeOpsPass
    : public enzyme::RemoveUnusedEnzymeOpsPassBase<RemoveUnusedEnzymeOpsPass> {
  void runOnOperation() override {

    getOperation()->walk([&](Operation *op) {
      DominanceInfo dInfo;
      if (auto initOp = dyn_cast<enzyme::InitOp>(op)) {
        Value v = initOp;
        if (auto type = dyn_cast<enzyme::GradientType>(initOp.getType())) {
          bool replaceable = true;
          for (Operation *userSet : v.getUsers()) {
            if (auto setOp = dyn_cast<enzyme::SetOp>(userSet)) {
              for (Operation *userGet : v.getUsers()) {
                if (auto getOp = dyn_cast<enzyme::GetOp>(userGet)) {
                  // We can safely delete an enzyme.gradient op if each pair of
                  // enzyme.set and enzyme.get ops are either not reachable or
                  // are reachable and do not exist inside a loop
                  bool relatedButNotInLoop =
                      dInfo.dominates(userSet, userGet) &&
                      !reachable(getOp, setOp);
                  bool unrelated = !reachable(setOp, getOp);
                  if (!(relatedButNotInLoop || unrelated)) {
                    replaceable = false;
                  }
                }
              }
            }
          }
          if (replaceable) {
            // Do replacing
            for (Operation *userGet : v.getUsers()) {
              if (auto getOp = dyn_cast<enzyme::GetOp>(userGet)) {
                Operation *closestSetOp =
                    findNearestDominatingOpByUse<enzyme::SetOp>(userGet, v);
                auto setOp = dyn_cast<enzyme::SetOp>(closestSetOp);
                getOp.replaceAllUsesWith(setOp.getValue());
              }
            }
            for (Operation *userGet : v.getUsers()) {
              userGet->erase();
            }
            op->erase();
          }
        } else if (auto type = dyn_cast<enzyme::CacheType>(initOp.getType())) {
          bool replaceable = true;
          for (Operation *userPush : v.getUsers()) {
            if (auto pushOp = dyn_cast<enzyme::PushOp>(userPush)) {
              // There should only be exactly one push per pop
              if (reachable(userPush, userPush)) {
                replaceable = false;
              }
              int numAssociatedPops = 0;
              for (Operation *user : v.getUsers()) {
                if (auto popOp = dyn_cast<enzyme::PopOp>(user)) {
                  if (reachable(userPush, user)) {
                    // Pops always need to be dominated by the push
                    if (dInfo.dominates(userPush, user)) {
                      numAssociatedPops++;
                    } else {
                      replaceable = false;
                    }
                  }
                }
                if (auto getOp = dyn_cast<enzyme::GetOp>(user)) {
                  if (reachable(userPush, user)) {
                    // Gets always need to be dominated by the push
                    if (!dInfo.dominates(userPush, user)) {
                      replaceable = false;
                    }
                  }
                }
              }
              // There should only be one pop per push
              if (numAssociatedPops > 1) {
                replaceable = false;
              }
            }
          }
          if (replaceable) {
            // Do replacing
            for (Operation *user : v.getUsers()) {
              if (auto popOp = dyn_cast<enzyme::PopOp>(user)) {
                Operation *closestPushOp =
                    findNearestDominatingOpByUse<enzyme::PushOp>(user, v);
                auto pushOp = dyn_cast<enzyme::PushOp>(closestPushOp);
                popOp.replaceAllUsesWith(pushOp.getValue());
              }
              if (auto getOp = dyn_cast<enzyme::GetOp>(user)) {
                Operation *closestPushOp =
                    findNearestDominatingOpByUse<enzyme::PushOp>(user, v);
                auto pushOp = dyn_cast<enzyme::PushOp>(closestPushOp);
                getOp.replaceAllUsesWith(pushOp.getValue());
              }
            }
            for (Operation *user : v.getUsers()) {
              user->erase();
            }
            op->erase();
          }
        }
      }
    });
  };
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createRemoveUnusedEnzymeOpsPass() {
  return std::make_unique<RemoveUnusedEnzymeOpsPass>();
}
} // namespace enzyme
} // namespace mlir
