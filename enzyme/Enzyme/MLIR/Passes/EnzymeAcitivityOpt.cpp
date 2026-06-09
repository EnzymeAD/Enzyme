//===- EnzymeAcitivtyOpt.cpp - Optimize activity for differentiation -------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/DataFlowActivityAnalysis.h"
#include "Analysis/DataFlowAliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "Passes/Utils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "enzyme-activity-opt"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::dataflow;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEACTIVITYOPT
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

class FwdOpt : public OpRewritePattern<enzyme::ForwardDiffOp> {
public:
  using OpRewritePattern<enzyme::ForwardDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForwardDiffOp uop,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    FunctionOpInterface callee = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(uop, uop.getFnAttr()));

    if (!callee)
      return failure();

    SmallVector<Activity> argumentActivity =
        llvm::to_vector(uop.getActivity().getAsValueRange<ActivityAttr>());
    SmallVector<Activity> resultActivity =
        llvm::to_vector(uop.getRetActivity().getAsValueRange<ActivityAttr>());

    // Run activity analysis(both forward and backward) on the primal function.
    // This helps us determine if the current assignments to activity or
    // ret_activity are optimal or conservative.
    DataFlowSolver solver;
    solver.load<enzyme::PointsToPointerAnalysis>();
    solver.load<enzyme::AliasAnalysis>(callee.getContext());
    solver.load<enzyme::SparseForwardActivityAnalysis>();
    solver.load<enzyme::DenseForwardActivityAnalysis>(
        &callee.getFunctionBody().front(), argumentActivity);
    solver.load<enzyme::SparseBackwardActivityAnalysis>(symbolTable);
    solver.load<enzyme::DenseBackwardActivityAnalysis>(symbolTable, callee,
                                                       argumentActivity);
    // Required for the dataflow framework to traverse region-based control flow
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();

    if (uop.getOutputs().size() == 0)
      return failure();

    // auto inActivity = uop.getActivity();
    //
    // auto in_idx = 0;
    // SmallVector<mlir::Value, 2> in_args;
    // SmallVector<ActivityAttr, 2> newInActivityArgs;
    // bool changed = false;
    // for (auto [idx, act] : llvm::enumerate(inActivity)) {
    //   auto iattr = cast<ActivityAttr>(act);
    //   auto val = iattr.getValue();
    //
    //   // Forward mode Input activities can only take values {dup, dupnoneed,
    //   // const }
    //   mlir::Value inp = uop.getInputs()[in_idx];
    //
    //   switch (val) {
    //
    //   case mlir::enzyme::Activity::enzyme_const:
    //     in_args.push_back(inp);
    //     newInActivityArgs.push_back(iattr);
    //     break;
    //
    //   case Activity::enzyme_dupnoneed: {
    //     // always pass in primal
    //     in_args.push_back(inp);
    //     in_idx++;
    //
    //     // selectively push or skip directional derivative
    //     inp = uop.getInputs()[in_idx];
    //     auto ET = inp.getType();
    //     auto ETintf = dyn_cast<AutoDiffTypeInterface>(ET);
    //
    //     if (ETintf && !isMutable(ET) && ETintf.isZero(inp)) {
    //       // skip and promote to const
    //       auto new_const = mlir::enzyme::ActivityAttr::get(
    //           rewriter.getContext(), mlir::enzyme::Activity::enzyme_const);
    //       newInActivityArgs.push_back(new_const);
    //       changed = true;
    //     } else {
    //       // push derivative value
    //       in_args.push_back(inp);
    //       newInActivityArgs.push_back(iattr);
    //     }
    //     break;
    //   }
    //
    //   case Activity::enzyme_dup: {
    //     // always pass in primal
    //     in_args.push_back(inp);
    //     in_idx++;
    //
    //     // selectively push or skip directional derivative
    //     inp = uop.getInputs()[in_idx];
    //     auto ET = inp.getType();
    //     auto ETintf = dyn_cast<AutoDiffTypeInterface>(ET);
    //
    //     if (ETintf && !isMutable(ET) && ETintf.isZero(inp)) {
    //       // skip and promote to const
    //       auto new_const = mlir::enzyme::ActivityAttr::get(
    //           rewriter.getContext(), mlir::enzyme::Activity::enzyme_const);
    //       newInActivityArgs.push_back(new_const);
    //       changed = true;
    //     } else {
    //       // push derivative value
    //       in_args.push_back(inp);
    //       newInActivityArgs.push_back(iattr);
    //     }
    //     break;
    //   }
    //   default:
    //     llvm_unreachable("unexpected input activity arg");
    //   }
    //
    //   in_idx++;
    // }

    // create the new op
    // ArrayAttr newInActivity =
    //     ArrayAttr::get(rewriter.getContext(),
    //                    llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
    //                                              newInActivityArgs.end()));
    //
    // if constexpr (std::is_same_v<SourceOp, ForwardDiffOp>) {
    //
    //   rewriter.replaceOpWithNewOp<ForwardDiffOp>(
    //       uop, uop->getResultTypes(), uop.getFnAttr(), in_args,
    //       newInActivity, uop.getRetActivityAttr(), uop.getWidthAttr(),
    //       uop.getStrongZeroAttr());
    // } else {
    //   rewriter.replaceOpWithNewOp<ForwardDiffRegionOp>(
    //       uop, uop->getResultTypes(), in_args, newInActivity,
    //       uop.getRetActivityAttr(), uop.getWidthAttr(),
    //       uop.getStrongZeroAttr(), uop.getFnAttr());
    // }
    return failure();
  }
};

struct EnzymeActivityOptPass
    : public enzyme::impl::EnzymeActivityOptBase<EnzymeActivityOptPass> {
  using EnzymeActivityOptBase::EnzymeActivityOptBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
} // namespace
