//===- HoistEnzymeRegions.cpp -LICM for  enzyme.autodiff_region ----------=== //
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
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

struct HoistEnzymeAutoDiff : public OpRewritePattern<enzyme::AutoDiffRegionOp> {
  using OpRewritePattern<enzyme::AutoDiffRegionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(enzyme::AutoDiffRegionOp rootOp,
                                PatternRewriter &rewriter) const override {
    DominanceInfo dInfo;
    Region &autodiffRegion = rootOp.getBody();
    SmallVector<Value> primalArgs = rootOp.getPrimalInputs();
    SmallVector<Value> regionPrimalArgs(autodiffRegion.getArguments());

    // int cnt = 0;
    // for(auto &op : autodiffRegion.getOps()){
    //   ENZYME_DBGS << (++cnt) << " : " << op << "\n ---- \n";
    // }
    //
    // ENZYME_DBGS << "======\n";
    // auto barrierOp = mlir::NVVM::Barrier0Op::create(rewriter,
    // rootOp->getLoc()); SmallVector<MemoryEffects::EffectInstance> bEffects;
    // bool couldCollect = enzyme::oputils::collectOpEffects(barrierOp,
    // bEffects); ENZYME_DBGS << "barrier: " << couldCollect <<  " effects" <<
    // "\n"; return failure();
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

    llvm::SmallVector<Operation *> liftOps;
    llvm::SmallVector<Operation *> stationaryOps;
    llvm::SmallVector<MemoryEffects::EffectInstance> stationaryEffects;

    for (Operation &bodyOp : rootOp.getBody().getOps()) {
      bool canLift = true;
      llvm::SmallVector<MemoryEffects::EffectInstance> bodyOpEffects;
      bool couldCollectEffects =
          enzyme::oputils::collectOpEffects(&bodyOp, bodyOpEffects);

      if (!couldCollectEffects)
        canLift = false;
      if (bodyOp.hasTrait<OpTrait::ReturnLike>() ||
          bodyOp.hasTrait<OpTrait::IsTerminator>())
        canLift = false;

      // establish dominance of values
      for (Value op_arg : bodyOp.getOperands()) {
        if (auto op_argBA = dyn_cast<BlockArgument>(op_arg)) {
          if (!dInfo.properlyDominates(op_arg, rootOp)) {
            canLift = false;
            break;
          }
        } else if (auto op_argOR = dyn_cast<OpResult>(op_arg)) {
          if (!dInfo.properlyDominates(op_arg, rootOp)) {
            if (!llvm::is_contained(liftOps, op_arg.getDefiningOp())) {
              canLift = false;
              break;
            }
          }
        }
      }

      // Check for memory conflicts with current set of stationary ops
      for (auto st_eff : stationaryEffects) {
        for (auto op_eff : bodyOpEffects) {
          if ((isa<MemoryEffects::Write>(st_eff.getEffect()) &&
               isa<MemoryEffects::Read>(op_eff.getEffect())) ||
              (isa<MemoryEffects::Read>(st_eff.getEffect()) &&
               isa<MemoryEffects::Write>(op_eff.getEffect())) ||
              (isa<MemoryEffects::Write>(st_eff.getEffect()) &&
               isa<MemoryEffects::Write>(op_eff.getEffect()))) {

            if (enzyme::oputils::mayAlias(op_eff, st_eff)) {
              canLift = false;
              break;
            }
          }
        }
      }

      if (canLift) {
        liftOps.push_back(&bodyOp);
      } else {
        stationaryOps.push_back(&bodyOp);
        stationaryEffects.append(bodyOpEffects.begin(), bodyOpEffects.end());
      }
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
