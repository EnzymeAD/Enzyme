//===- SplitMultiResults.cpp - Split multi-result region ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that splits region control flow that produces
// multiple results into separate operations for each result.
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SPLITMULTIRESULTSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
namespace {

// Given a block terminator, determine if the terminator's operands are solely
// computed by pure operations within the block.
static bool allComputedByPureOps(Operation *terminator) {
  Block *parent = terminator->getBlock();
  DenseSet<Value> visited;
  visited.insert_range(terminator->getOperands());
  SmallVector<Value> frontier(terminator->getOperands());
  while (!frontier.empty()) {
    Value curr = frontier.pop_back_val();
    Operation *definingOp = curr.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != parent)
      continue;

    if (!mlir::isPure(definingOp))
      return false;

    for (Value operand : definingOp->getOperands()) {
      if (!visited.contains(operand)) {
        visited.insert(operand);
        frontier.push_back(operand);
      }
    }
  }

  return true;
}
struct SplitMultiIf : public mlir::OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getNumResults() < 2)
      return failure();
    if (llvm::all_of(ifOp.getResults(),
                     [](OpResult res) { return res.use_empty(); })) {
      return failure();
    }
    // all operands should be computed by pure operations.
    if (!(allComputedByPureOps(ifOp.thenYield()) &&
          allComputedByPureOps(ifOp.elseYield()))) {
      return failure();
    }

    for (OpResult ifRes : ifOp.getResults()) {
      auto newIf = scf::IfOp::create(rewriter, ifOp.getLoc(), ifRes.getType(),
                                     ifOp.getCondition(),
                                     /*withElseRegion=*/true);
      IRMapping map;
      OpBuilder::InsertionGuard guard(rewriter);
      for (auto [oldReg, newReg] :
           llvm::zip(ifOp.getRegions(), newIf.getRegions())) {
        for (auto &&[oldBlk, newBlk] : llvm::zip(*oldReg, *newReg)) {
          rewriter.setInsertionPointToStart(&newBlk);
          for (auto &op : oldBlk.without_terminator()) {
            if (mlir::isPure(&op))
              rewriter.clone(op, map);
          }
          scf::YieldOp::create(
              rewriter, ifOp.getLoc(),
              map.lookupOrDefault(
                  oldBlk.getTerminator()->getOperand(ifRes.getResultNumber())));
        }
      }
      ifRes.replaceAllUsesWith(newIf.getResult(0));
    }
    return success();
  }
};

struct SplitMultiResultsPass
    : public enzyme::impl::SplitMultiResultsPassBase<SplitMultiResultsPass> {
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    patterns.insert<SplitMultiIf>(&getContext());

    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace
