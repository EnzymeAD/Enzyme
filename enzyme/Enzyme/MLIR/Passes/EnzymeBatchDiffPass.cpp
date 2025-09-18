//===- EnzymeBatchDiffPass.cpp - Merge autodiff calls into their batched
// versions
//------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Passes/EnzymeBatchDiffPass.h"
#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "enzyme-diff-batch"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_BATCHDIFFPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

struct BatchDiffPass : public enzyme::impl::BatchDiffPassBase<BatchDiffPass> {
  void runOnOperation() override;

  bool isReadOnly(Operation *op) {
    bool hasRecursiveEffects =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      for (Region &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &nestedOp : block)
            if (!isReadOnly(&nestedOp))
              return false;
        }
      }
      return true;
    }

    // If the op has memory effects, try to characterize them to see if the op
    // is trivially dead here.
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Check to see if this op either has no effects, or only allocates/reads
      // memory.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);
      if (!llvm::all_of(effects, [op](const MemoryEffects::EffectInstance &it) {
            return isa<MemoryEffects::Read>(it.getEffect());
          })) {
        return false;
      }
      return true;
    }
    return false;
  }

  // Map tracking batchable subset of fwddiff calls
  void mergeFwddiffCalls(SymbolTableCollection &symbolTable,
                         FunctionOpInterface op) {

    // map tracking batchable AD calls
    std::map<enzyme::batchutils::BatchDiffCacheKey,
             SmallVector<enzyme::ForwardDiffOp>>
        toMerge;

    op->walk([&](enzyme::ForwardDiffOp dop) {
      // lookup function, check if its readOnly
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
      auto fnOp = cast<FunctionOpInterface>(symbolOp);

      // skip if fn isn't readonly
      if (!isReadOnly(fnOp)) {
        return mlir::WalkResult::skip();
      }

      // add to map
      SmallVector<Activity> inActivity;
      SmallVector<Activity> retActivity;
      SmallVector<Value> in_args;

      auto in_idx = 0;
      for (auto [idx, act] : llvm::enumerate(dop.getActivity())) {
        auto iattr = cast<ActivityAttr>(act);
        auto val = iattr.getValue();
        inActivity.push_back(val);

        in_args.push_back(dop.getInputs()[in_idx]);
        in_idx++;
        if (val == Activity::enzyme_dup || val == Activity::enzyme_dupnoneed) {
          in_idx++;
        }
      }

      for (auto [idx, ract] : llvm::enumerate(dop.getRetActivity())) {
        auto iattr = cast<ActivityAttr>(ract);
        auto val = iattr.getValue();
        retActivity.push_back(val);
      }

      batchutils::BatchDiffCacheKey key{fnOp, in_args, inActivity, retActivity};

      auto mergeItr = toMerge.find(key);
      if (mergeItr != toMerge.end()) {
        mergeItr->second.push_back(dop);
      } else {
        SmallVector<enzyme::ForwardDiffOp> v;
        v.push_back(dop);
        toMerge[key] = v;
      }
    });

    OpBuilder builder(op);
    // process map
    for (auto mergeItr = toMerge.begin(); mergeItr != toMerge.end();
         ++mergeItr) {
      auto key = mergeItr->first;
      SmallVector<enzyme::ForwardDiffOp> allOps = mergeItr->second;
      auto width = allOps.size();

      if (width < 2)
        continue;

      auto lastOp = allOps.back();
      auto loc = lastOp->getLoc();
      auto context = builder.getContext();

      SmallVector<mlir::Value> in_args;
      SmallVector<ActivityAttr, 2> inActivity;
      SmallVector<ActivityAttr, 2> retActivity;
      SmallVector<mlir::Value> out_args;
      SmallVector<mlir::Type, 2> out_ty;
      auto in_idx = 0;

      // process inputs
      for (auto [idx, act] : llvm::enumerate(key.inActivity)) {
        ActivityAttr iattr = ActivityAttr::get(context, act);
        inActivity.push_back(iattr);
        in_args.push_back(key.inputs[in_idx]);
        in_idx++;

        // collect derivatives
        SmallVector<mlir::Value> derivToBatch;
        if (act == Activity::enzyme_dup || act == Activity::enzyme_dupnoneed) {
          for (auto uop : allOps) {
            derivToBatch.push_back(uop.getInputs()[in_idx]);
          }

          auto derivTy = derivToBatch[0].getType();
          auto T = dyn_cast<TensorType>(derivTy);
          mlir::Value batchedDeriv;
          if (!T) {
            // use tensor.from_elements
            batchedDeriv =
                builder.create<tensor::FromElementsOp>(loc, derivToBatch);
          } else {
            // use tensor.concat on dim 0
            batchedDeriv =
                builder.create<tensor::ConcatOp>(loc, 0, derivToBatch);
          }

          in_args.push_back(batchedDeriv);
          in_idx++;
        }
      }

      // process output type
      auto out_idx = 0;
      for (auto [idx, ract] : llvm::enumerate(key.retActivity)) {
        ActivityAttr iattr = ActivityAttr::get(context, ract);

        retActivity.push_back(iattr);
        switch (ract) {

        case Activity::enzyme_active: {
          mlir::Value res = lastOp.getOutputs()[out_idx];
          out_ty.push_back(res.getType());
          ++out_idx;
          break;
        }

        case Activity::enzyme_const: {
          mlir::Value res = lastOp.getOutputs()[out_idx];
          out_ty.push_back(res.getType());
          ++out_idx;
          break;
        }

        case Activity::enzyme_dupnoneed: {
          // derivative

          mlir::Value dres = lastOp.getOutputs()[out_idx];
          auto dresTy = dres.getType();
          auto T = dyn_cast<TensorType>(dresTy);
          if (!T) {
            out_ty.push_back(RankedTensorType::get(width, dresTy));
          } else {
            // prepend to shape
            SmallVector<int64_t> shape;
            shape.push_back(width);
            shape.append(T.getShape().begin(), T.getShape().end());
            auto T2 = T.clone(shape);
            out_ty.push_back(T2);
          }
          ++out_idx;
          break;
        }

        case Activity::enzyme_dup: {
          mlir::Value res = lastOp.getOutputs()[out_idx];
          out_ty.push_back(res.getType());

          ++out_idx;

          // derivative
          mlir::Value dres = lastOp.getOutputs()[out_idx];
          auto dresTy = dres.getType();
          auto T = dyn_cast<TensorType>(dresTy);
          if (!T) {
            out_ty.push_back(RankedTensorType::get(width, dresTy));
          } else {
            // prepend to shape
            SmallVector<int64_t> shape;
            shape.push_back(width);
            shape.append(T.getShape().begin(), T.getShape().end());
            auto T2 = T.clone(shape);
            out_ty.push_back(T2);
          }
          ++out_idx;
          break;
        }

        case Activity::enzyme_constnoneed: {
          break;
        }

        case Activity::enzyme_activenoneed: {
          mlir::Value res = lastOp.getOutputs()[out_idx];
          out_ty.push_back(res.getType());
          ++out_idx;
          break;
        }

        default:
          llvm_unreachable(
              "unknown activity value encountered for ret_activity");
        }
      }
      
      // create new FwdDiffOp
      
      IntegerAttr newWidthAttr = IntegerAttr::get(context, llvm::APSInt(width));
      auto newMergedDiffOp = builder.create<ForwardDiffOp>(
          loc, out_ty, lastOp.getFnAttr(), in_args, inActivity, retActivity,
          newWidthAttr, lastOp.getStrongZeroAttr());
      
      // 

      // rename uses from old to new results

    };
  };
};

} // end anonymous namespace

void BatchDiffPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk(
      [&](FunctionOpInterface op) { mergeFwddiffCalls(symbolTable, op); });
}
