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
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  LogicalResult matchAndRewrite(enzyme::AutoDiffRegionOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value> primalArgs = op.getPrimalInputs(); 
    Region &body = op.getBody();
    SmallVector<Value> primalBlockArgs(body.getArguments());
    
    if (primalArgs.size() != primalBlockArgs.size())
      return failure();

    SmallVector<Operation *> moveList, dontmoveList;
    ;

    return failure();
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
