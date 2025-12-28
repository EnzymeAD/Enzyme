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
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace enzyme;
namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_HOISTENZYMEFROMREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

#define DEBUG_TYPE "enzyme-hoist"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

namespace {

static bool checkRangeDominance(IRMapping &btop, DominanceInfo &dom,
                                AutoDiffRegionOp &rootOp,
                                SetVector<Operation *> &specialOps,
                                ValueRange values) {
  SmallVector<Value> blockArgs(rootOp.getBody().getArguments());
  for (auto value : values) {
    if (dom.properlyDominates(value, rootOp))
      continue;

    // Block arguments within autodiff_region are not supported
    //
    if (isa<BlockArgument>(value)) {
      // check if it's a block argument of type enzyme_const
      if (btop.contains(value))
        continue;
      else
        return false;
    }

    if (!llvm::is_contained(specialOps, value.getDefiningOp())) {
      return false;
    }
  }

  // if we reach this point, it means that these current set of values are safe
  // to hoist
  return true;
}

struct HoistEnzymeAutoDiff : public OpRewritePattern<AutoDiffRegionOp> {
  using OpRewritePattern<AutoDiffRegionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AutoDiffRegionOp rootOp,
                                PatternRewriter &rewriter) const override {
    DominanceInfo dom(rootOp);
    PostDominanceInfo pdom(rootOp);
    Region &autodiffRegion = rootOp.getBody();
    SmallVector<Value> primalArgs = rootOp.getPrimalInputs();
    SmallVector<Value> blockArgs(autodiffRegion.getArguments());

    if (primalArgs.size() != blockArgs.size())
      return failure();

    // map for block arg -> primal arg iff activity is enzyme_const
    IRMapping btop;
    for (auto [pval, bval, act] :
         llvm::zip(primalArgs, blockArgs,
                   rootOp.getActivity().getAsRange<ActivityAttr>())) {
      auto act_val = act.getValue();
      if (act_val == Activity::enzyme_const) {
        btop.map(bval, pval);
      }
    }

    // rename all uses of primal
    llvm::SetVector<Value> freeValues;
    getUsedValuesDefinedAbove(autodiffRegion, freeValues);
    for (Value value : freeValues) {
      for (auto [pval, bval] : llvm::zip(primalArgs, blockArgs)) {
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
      if (pdom.postDominates(&blk, &autodiffRegion.front())) {
        for (Operation &bodyOp : blk.without_terminator()) {
          bool canLift = true;
          llvm::SmallVector<MemoryEffects::EffectInstance> bodyOpEffects;

          bool couldCollectEffects =
              enzyme::oputils::collectOpEffects(&bodyOp, bodyOpEffects);

          if (!couldCollectEffects)
            canLift = false;

          canLift = checkRangeDominance(btop, dom, rootOp, liftOps,
                                        bodyOp.getOperands());

          llvm::SetVector<Value> inside_values;
          if (bodyOp.getNumRegions()) {
            canLift = false;
            getUsedValuesDefinedAbove(bodyOp.getRegions(), inside_values);
            canLift = checkRangeDominance(btop, dom, rootOp, liftOps,
                                          inside_values.getArrayRef());
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
            // replace all instances of enzyme_const block args with the
            // equivalent primal args for both inside_values and
            // bodyOp.getOperands()

            for (Value inner : inside_values) {
              if (btop.contains(inner)) {
                auto pval = btop.lookup(inner);
                for (auto &region : bodyOp.getRegions()) {
                  replaceAllUsesInRegionWith(inner, pval, region);
                }
              }
            }

            for (OpOperand &inner : bodyOp.getOpOperands()) {
              inner.assign(btop.lookupOrDefault(inner.get()));
            }

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
