//===- EnzymeAcitivtyOpt.cpp - Optimize activity for differentiation -------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "Passes/Utils.h"

#include "Analysis/DataFlowAliasAnalysis.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "enzyme-activity-opt"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEACTIVITYOPT
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

class FwdInpOpt : public OpRewritePattern<enzyme::ForwardDiffOp> {
public:
  using OpRewritePattern<enzyme::ForwardDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForwardDiffOp uop,
                                PatternRewriter &rewriter) const override {

    SymbolTableCollection symbolTable;
    FunctionOpInterface fn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(uop, uop.getFnAttr()));

    if (!fn)
      return failure();

    DataFlowSolver solver;

    solver.load<enzyme::PointsToPointerAnalysis>();
    solver.load<enzyme::AliasAnalysis>(callee.getContext());
    solver.load<SparseForwardActivityAnalysis>();
    solver.load<DenseForwardActivityAnalysis>(&callee.getFunctionBody().front(),
                                              argumentActivity);
    solver.load<SparseBackwardActivityAnalysis>(symbolTable);
    solver.load<DenseBackwardActivityAnalysis>(symbolTable, callee,
                                               argumentActivity);

    // Required for the dataflow framework to traverse region-based control flow
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();

    if (uop.getOutputs().size() == 0)
      return failure();

    auto inActivity = uop.getActivity();

    auto in_idx = 0;
    SmallVector<mlir::Value, 2> in_args;
    SmallVector<ActivityAttr, 2> newInActivityArgs;
    bool changed = false;
    for (auto [idx, act] : llvm::enumerate(inActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      // Forward mode Input activities can only take values {dup, dupnoneed,
      // const }
      mlir::Value inp = uop.getInputs()[in_idx];

      switch (val) {

      case mlir::enzyme::Activity::enzyme_const:
        in_args.push_back(inp);
        newInActivityArgs.push_back(iattr);
        break;

      case Activity::enzyme_dupnoneed: {
        // always pass in primal
        in_args.push_back(inp);
        in_idx++;

        // selectively push or skip directional derivative
        inp = uop.getInputs()[in_idx];
        auto ET = inp.getType();
        auto ETintf = dyn_cast<AutoDiffTypeInterface>(ET);

        if (ETintf && !isMutable(ET) && ETintf.isZero(inp)) {
          // skip and promote to const
          auto new_const = mlir::enzyme::ActivityAttr::get(
              rewriter.getContext(), mlir::enzyme::Activity::enzyme_const);
          newInActivityArgs.push_back(new_const);
          changed = true;
        } else {
          // push derivative value
          in_args.push_back(inp);
          newInActivityArgs.push_back(iattr);
        }
        break;
      }

      case Activity::enzyme_dup: {
        // always pass in primal
        in_args.push_back(inp);
        in_idx++;

        // selectively push or skip directional derivative
        inp = uop.getInputs()[in_idx];
        auto ET = inp.getType();
        auto ETintf = dyn_cast<AutoDiffTypeInterface>(ET);

        if (ETintf && !isMutable(ET) && ETintf.isZero(inp)) {
          // skip and promote to const
          auto new_const = mlir::enzyme::ActivityAttr::get(
              rewriter.getContext(), mlir::enzyme::Activity::enzyme_const);
          newInActivityArgs.push_back(new_const);
          changed = true;
        } else {
          // push derivative value
          in_args.push_back(inp);
          newInActivityArgs.push_back(iattr);
        }
        break;
      }
      default:
        llvm_unreachable("unexpected input activity arg");
      }

      in_idx++;
    }

    if (!changed)
      return failure();

    // create the new op
    ArrayAttr newInActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                 newInActivityArgs.end()));

    if constexpr (std::is_same_v<SourceOp, ForwardDiffOp>) {

      rewriter.replaceOpWithNewOp<ForwardDiffOp>(
          uop, uop->getResultTypes(), uop.getFnAttr(), in_args, newInActivity,
          uop.getRetActivityAttr(), uop.getWidthAttr(),
          uop.getStrongZeroAttr());
    } else {
      rewriter.replaceOpWithNewOp<ForwardDiffRegionOp>(
          uop, uop->getResultTypes(), in_args, newInActivity,
          uop.getRetActivityAttr(), uop.getWidthAttr(), uop.getStrongZeroAttr(),
          uop.getFnAttr());
    }
    return success();
  }
};

class ReverseRetOpt : public OpRewritePattern<enzyme::AutoDiffOp> {
public:
  using OpRewritePattern<enzyme::AutoDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AutoDiffOp uop,
                                PatternRewriter &rewriter) const override {
    // early return if there are no outputs
    if (uop.getOutputs().size() == 0)
      return failure();

    auto inpActivity = uop.getActivity();
    auto retActivity = uop.getRetActivity();
    auto out_idx = 0;
    SmallVector<mlir::Value, 2> in_args;
    SmallVector<mlir::Value, 2> outs_args;
    SmallVector<Type, 2> in_ty;
    SmallVector<Type, 2> out_ty;
    SmallVector<ActivityAttr, 2> newInActivityArgs;
    SmallVector<ActivityAttr, 2> newRetActivityArgs;

    bool changed = false;
    auto in_idx = 0;

    // go upto dOutput
    for (auto [idx, act] : llvm::enumerate(inpActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();
      mlir::Value res = uop.getInputs()[in_idx];
      in_args.push_back(res);
      in_ty.push_back(res.getType());
      in_idx++;

      if (val == Activity::enzyme_dup || val == Activity::enzyme_dupnoneed) {
        mlir::Value dres = uop.getInputs()[in_idx];
        in_args.push_back(dres);
        in_ty.push_back(dres.getType());
        in_idx++;
      }
    }
    // function isn't differentiable
    if (in_idx == uop.getInputs().size())
      return failure();

    // handle pOutput
    for (auto [idx, act] : llvm::enumerate(retActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      // skip primal return
      if (val == Activity::enzyme_constnoneed ||
          val == Activity::enzyme_dupnoneed) {
        newRetActivityArgs.push_back(iattr);
        continue;
      }

      mlir::Value res = uop.getOutputs()[out_idx];

      switch (val) {
      case Activity::enzyme_active: {
        // active -> activenoneed(if res isn't used)
        // active -> const(if dres == 0)
        // active -> constnoneed(both)

        mlir::Value dres = uop.getInputs()[in_idx];
        in_idx++;

        auto dres_type = dres.getType();
        auto dres_type_intf = dyn_cast<AutoDiffTypeInterface>(dres_type);

        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          ActivityAttr new_act = iattr;
          if (dres_type_intf && !isMutable(dres_type) &&
              dres_type_intf.isZero(dres)) {
            // const
            changed = true;
            new_act = ActivityAttr::get(rewriter.getContext(),
                                        Activity::enzyme_const);
          } else {
            in_args.push_back(dres);
            in_ty.push_back(dres_type);
          }
          newRetActivityArgs.push_back(new_act);
        } else {
          changed = true;
          ActivityAttr new_act = ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_activenoneed);
          if (dres_type_intf && !isMutable(dres_type) &&
              dres_type_intf.isZero(dres)) {
            // constnoneed
            new_act = ActivityAttr::get(rewriter.getContext(),
                                        Activity::enzyme_constnoneed);
          } else {
            // activenoneed
            in_args.push_back(dres);
            in_ty.push_back(dres_type);
          }
          newRetActivityArgs.push_back(new_act);
        }

        ++out_idx;
        break;
      }

      case Activity::enzyme_activenoneed:
        // activenoneed -> constnoneed
        {
          mlir::Value dres = uop.getInputs()[in_idx];
          in_idx++;
          auto new_act = iattr;

          auto dres_type = dres.getType();
          auto dres_type_intf = dyn_cast<AutoDiffTypeInterface>(dres_type);
          if (dres_type_intf && !isMutable(dres_type) &&
              dres_type_intf.isZero(dres)) {
            // constnoneed
            new_act = ActivityAttr::get(rewriter.getContext(),
                                        Activity::enzyme_constnoneed);
          } else {
            in_args.push_back(dres);
            in_ty.push_back(dres_type);
          }
          newRetActivityArgs.push_back(iattr);
          break;
        }
      case Activity::enzyme_const:
        // const -> constnoneed
        {
          auto new_act = iattr;
          if (!res.use_empty()) {
            outs_args.push_back(res);
            out_ty.push_back(res.getType());
            newRetActivityArgs.push_back(new_act);
          } else {
            changed = true;
            new_act = ActivityAttr::get(rewriter.getContext(),
                                        Activity::enzyme_constnoneed);
            newRetActivityArgs.push_back(new_act);
          }
          ++out_idx;
          break;
        }

      case Activity::enzyme_dup:
        // TODO: check if ret_arg == enzyme_dup inserts a derivative as the
        // output and input both
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        ++out_idx;
        break;

      case Activity::enzyme_constnoneed:
      case Activity::enzyme_dupnoneed:
        break;

      default:
        llvm_unreachable("unexpected activity arg");
      }
    }

    // handle dInputs
    for (auto [idx, act] : llvm::enumerate(inpActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();

      if (val == Activity::enzyme_active) {
        mlir::Value res = uop.getOutputs()[out_idx];
        if (!res.use_empty()) {
          out_ty.push_back(res.getType());
          outs_args.push_back(res);
          newInActivityArgs.push_back(iattr);
        } else {
          // TODO: check if we can relax immutability here
          if (!isMutable(res.getType())) {
            changed = true;
            auto new_const = ActivityAttr::get(rewriter.getContext(),
                                               Activity::enzyme_const);
            newInActivityArgs.push_back(new_const);
          } else {
            // noop even if its not used.
            out_ty.push_back(res.getType());
            outs_args.push_back(res);
            newInActivityArgs.push_back(iattr);
          }
        }

        ++out_idx;
      } else if (val == Activity::enzyme_activenoneed) {
        mlir::Value res = uop.getOutputs()[out_idx];
        out_ty.push_back(res.getType());
        outs_args.push_back(res);
        newInActivityArgs.push_back(iattr);
        ++out_idx;
        llvm_unreachable("unsupported arg activenoneed");
      } else {
        newInActivityArgs.push_back(iattr);
      }
    }

    if (!changed)
      return failure();

    ArrayAttr newInActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                 newInActivityArgs.end()));
    ArrayAttr newRetActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                 newRetActivityArgs.end()));

    SourceOp newOp = SourceOpCreator::create(rewriter, uop, out_ty, in_args,
                                             newInActivity, newRetActivity);

    // Map old uses of uop to newOp
    auto oldIdx = 0;
    auto newIdx = 0;
    for (auto [idx, old_act, new_act] :
         llvm::enumerate(retActivity, newRetActivityArgs)) {
      auto iattr = cast<ActivityAttr>(old_act);
      auto old_val = iattr.getValue();
      auto new_val = new_act.getValue();

      if (old_val == new_val) {
        // don't index into op if no primal is returned
        if (old_val == Activity::enzyme_constnoneed ||
            old_val == Activity::enzyme_activenoneed ||
            old_val == Activity::enzyme_dupnoneed) {
          continue;
        }
        // replace current Primal
        uop.getOutputs()[oldIdx++].replaceAllUsesWith(
            newOp.getOutputs()[newIdx++]);
      } else {
        // handle all substitutions
        if (new_val == Activity::enzyme_activenoneed &&
            old_val == Activity::enzyme_active) {
          ++oldIdx; // skip active primal
        } else if (new_val == Activity::enzyme_constnoneed &&
                   old_val == Activity::enzyme_const) {
          ++oldIdx; // skip const primal
        } else if (old_val == Activity::enzyme_active &&
                   new_val == Activity::enzyme_const) {
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else if (old_val == Activity::enzyme_active &&
                   new_val == Activity::enzyme_constnoneed) {
          ++oldIdx;
        } else if (old_val == Activity::enzyme_activenoneed &&
                   new_val == Activity::enzyme_constnoneed) {
          // just skip
        }
      }
    }

    for (auto [idx, old_act, new_act] :
         llvm::enumerate(inpActivity, newInActivityArgs)) {
      auto iattr = cast<ActivityAttr>(old_act);
      auto old_val = iattr.getValue();
      auto new_val = new_act.getValue();

      if (old_val == new_val) {
        if (old_val == Activity::enzyme_active ||
            old_val == Activity::enzyme_activenoneed) {
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else {
          continue;
        }
      } else {
        if (old_val == Activity::enzyme_active &&
            new_val == Activity::enzyme_const) {
          oldIdx++; // skip derivative
        }
      }
    }
    rewriter.eraseOp(uop);
    return success();
  }
};

struct EnzymeActivityOptPass
    : public enzyme::impl::EnzymeActivityOptBase<EnzymeActivityOptPass> {
  using EnzymeActivityOptBase::EnzymeActivityOptBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    patterns.add<FwdInpOpt, ReverseRetOpt>(&getContext());

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};
} // namespace
