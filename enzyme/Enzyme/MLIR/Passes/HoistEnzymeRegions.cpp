//===- HoistEnzymeRegions.cpp - Invariant code motion ------------===//
//===- within enzyme.autodiff_region                   ----------=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements passes to hoist computations within autodiff_region ops
// to the caller
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/Utils.h"
#include "Passes/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_HOISTENZYMEFROMREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

#define DEBUG_TYPE "enzyme-hoist"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

namespace {

static bool checkRangeDominance(DominanceInfo &dominance,
                                enzyme::AutoDiffRegionOp &rootOp,
                                SetVector<Operation *> &specialOps,
                                ValueRange values) {
  for (auto value : values) {
    if (dominance.properlyDominates(value, rootOp))
      continue;
    // Block arguments within autodiff_region are not supported
    // TODO: add support for enzyme_const block arguments
    if (isa<BlockArgument>(value)) {
      return false;
    }
    if (!llvm::is_contained(specialOps, value.getDefiningOp())) {
      return false;
    }
  }
  return true;
}

struct HoistEnzymeAutoDiff : public OpRewritePattern<enzyme::AutoDiffRegionOp> {
  using OpRewritePattern<enzyme::AutoDiffRegionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(enzyme::AutoDiffRegionOp rootOp,
                                PatternRewriter &rewriter) const override {
    DominanceInfo dominance(rootOp);
    PostDominanceInfo postDominance(rootOp);
    Region &autodiffRegion = rootOp.getBody();
    SmallVector<Value> primalArgs = rootOp.getPrimalInputs();
    SmallVector<Value> regionPrimalArgs(autodiffRegion.getArguments());

    if (primalArgs.size() != regionPrimalArgs.size())
      return failure();

    llvm::SetVector<Value> freeValues;
    getUsedValuesDefinedAbove(autodiffRegion, freeValues);

    for (Value value : freeValues) {
      for (auto [pval, bval] : llvm::zip(primalArgs, regionPrimalArgs)) {
        if (value == pval) {
          for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
            if (rootOp->isProperAncestor(use.getOwner()))
              use.assign(bval);
          }
        }
      }
    }

    llvm::SetVector<Operation *> liftOps;
    llvm::SetVector<Operation *> stationaryOps;
    llvm::SmallVector<MemoryEffects::EffectInstance> stationaryEffects;
    for (Block &blk : autodiffRegion.getBlocks()) {

      // If bodyOp is in a block which does not post-dominate the entry
      // block to the regionOp, then we disable lifting it
      if (postDominance.postDominates(&blk, &autodiffRegion.front())) {
        for (Operation &bodyOp : blk.without_terminator()) {
          bool canLift = true;
          llvm::SmallVector<MemoryEffects::EffectInstance> bodyOpEffects;

          bool couldCollectEffects =
              enzyme::oputils::collectOpEffects(&bodyOp, bodyOpEffects);

          if (!couldCollectEffects)
            canLift = false;

          canLift = checkRangeDominance(dominance, rootOp, liftOps,
                                        bodyOp.getOperands());

          if (bodyOp.getNumRegions()) {
            canLift = false;
            llvm::SetVector<Value> values;
            getUsedValuesDefinedAbove(bodyOp.getRegions(), values);
            canLift = checkRangeDominance(dominance, rootOp, liftOps,
                                          values.getArrayRef());
          }

          // Check for memory conflicts with current set of stationary ops
          for (auto stationaryEffect : stationaryEffects) {
            for (auto bodyOpEffect : bodyOpEffects) {
              if ((isa<MemoryEffects::Write>(stationaryEffect.getEffect()) &&
                   isa<MemoryEffects::Read>(bodyOpEffect.getEffect())) ||
                  (isa<MemoryEffects::Read>(stationaryEffect.getEffect()) &&
                   isa<MemoryEffects::Write>(bodyOpEffect.getEffect())) ||
                  (isa<MemoryEffects::Write>(stationaryEffect.getEffect()) &&
                   isa<MemoryEffects::Write>(bodyOpEffect.getEffect()))) {

                if (enzyme::oputils::mayAlias(bodyOpEffect, stationaryEffect)) {
                  canLift = false;
                  break;
                }
              }
            }
          }

          if (canLift) {
            liftOps.insert(&bodyOp);
          } else {
            stationaryOps.insert(&bodyOp);
            stationaryEffects.append(bodyOpEffects.begin(),
                                     bodyOpEffects.end());
          }
        }
      }
    }

    // Lift operations
    for (Operation *op : llvm::make_early_inc_range(liftOps)) {
      rewriter.moveOpBefore(op, rootOp);
    }

    return success();
  }
};

struct HoistEnzymeFromRegion
    : public enzyme::impl::HoistEnzymeFromRegionPassBase<
          HoistEnzymeFromRegion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<HoistEnzymeAutoDiff>(&getContext());
    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace
