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
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

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

  bool isReadOnly2(Operation *op) {
    // If the op has memory effects, try to characterize them to see if the op
    // is trivially dead here.
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      LLVM_DEBUG(ENZYME_DBGS << "querying memory effects of fn op" << "\n");
      // Check to see if this op either has no effects, or only allocates/reads
      // memory.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);
      if (!llvm::all_of(effects, [op](const MemoryEffects::EffectInstance &it) {
            return isa<MemoryEffects::Read>(it.getEffect());
          })) {
        return false;
      }
    }

    bool isRecursiveContainer =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
        isa<FunctionOpInterface>(op);
    if (isRecursiveContainer) {
      LLVM_DEBUG(ENZYME_DBGS << "has rec mem effects" << "\n");
      for (Region &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &nestedOp : block)
            if (!isReadOnly(&nestedOp))
              return false;
        }
      }
    }

    return true;
  }

  bool isReadOnly(Operation *op) {
    LLVM_DEBUG(ENZYME_DBGS << "inside isReadOnly check" << "\n");
    bool hasRecursiveEffects =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      LLVM_DEBUG(ENZYME_DBGS << "has rec mem effects" << "\n");
      for (Region &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &nestedOp : block)
            if (!isReadOnly(&nestedOp))
              return false;
        }
      }
      return true;
    }
    LLVM_DEBUG(ENZYME_DBGS << "has no rec mem effects" << "\n");

    // If the op has memory effects, try to characterize them to see if the op
    // is trivially dead here.
    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      LLVM_DEBUG(ENZYME_DBGS << "querying memory effects of fn op" << "\n");
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

    LLVM_DEBUG(ENZYME_DBGS << "has no mem effects" << "\n");
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
      LLVM_DEBUG(ENZYME_DBGS << "found fwddiff" << "\n");
      // lookup function, check if its readOnly
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
      auto fnOp = cast<FunctionOpInterface>(symbolOp);

      // skip if fn isn't readonly(iterate through toplevel ops)
      LLVM_DEBUG(ENZYME_DBGS << *symbolOp << "\n");
      LLVM_DEBUG(ENZYME_DBGS << *fnOp << "\n");
      LLVM_DEBUG(ENZYME_DBGS << fnOp.getFunctionBody().front().front() << "\n");
      mlir::Region &fnReg = fnOp.getFunctionBody();
      for (mlir::Block &fnBlk : fnReg) {
        LLVM_DEBUG({
          llvm::dbgs() << "Processing block: ";
          fnBlk.printAsOperand(llvm::dbgs());
          llvm::dbgs() << "\n";
        });

        for (mlir::Operation &fnOp : fnBlk.getOperations()) {
          LLVM_DEBUG(llvm::dbgs() << "Processing op " << fnOp << "\n");
        }
      }
      if (!isReadOnly2(fnOp)) {
        LLVM_DEBUG(ENZYME_DBGS << "skipping fn." << "\n");
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
        LLVM_DEBUG(llvm::dbgs() << "adding to map" << "\n");
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

      {
        // insert merged op BEFORE first fwddiff op
        auto firstDiffOp = allOps.front();
        IRRewriter::InsertionGuard insertGuard(builder);
        builder.setInsertionPoint(firstDiffOp);
        auto loc = firstDiffOp->getLoc();
        auto context = builder.getContext();

        SmallVector<mlir::Value> in_args;
        SmallVector<ActivityAttr, 2> inActivityAttrs;
        SmallVector<ActivityAttr, 2> retActivityAttrs;
        SmallVector<mlir::Type, 2> out_ty;
        auto in_idx = 0;

        // process inputs
        for (auto [idx, act] : llvm::enumerate(key.inActivity)) {
          ActivityAttr iattr = ActivityAttr::get(context, act);
          inActivityAttrs.push_back(iattr);
          in_args.push_back(key.inputs[in_idx]);
          in_idx++;

          // collect derivatives
          SmallVector<mlir::Value> derivToBatch;
          if (act == Activity::enzyme_dup ||
              act == Activity::enzyme_dupnoneed) {
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
              batchedDeriv = builder.create<tensor::ConcatOp>(loc, /*dim*/ 0,
                                                              derivToBatch);
            }

            in_args.push_back(batchedDeriv);
            in_idx++;
          }
        }

        // process output type
        auto out_idx = 0;
        for (auto [idx, ract] : llvm::enumerate(key.retActivity)) {
          ActivityAttr iattr = ActivityAttr::get(context, ract);

          retActivityAttrs.push_back(iattr);
          switch (ract) {

          case Activity::enzyme_active: {
            mlir::Value res = firstDiffOp.getOutputs()[out_idx];
            out_ty.push_back(res.getType());
            ++out_idx;
            break;
          }

          case Activity::enzyme_const: {
            mlir::Value res = firstDiffOp.getOutputs()[out_idx];
            out_ty.push_back(res.getType());
            ++out_idx;
            break;
          }

          case Activity::enzyme_dupnoneed: {
            // derivative

            mlir::Value dres = firstDiffOp.getOutputs()[out_idx];
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
            mlir::Value res = firstDiffOp.getOutputs()[out_idx];
            out_ty.push_back(res.getType());

            ++out_idx;

            // derivative
            mlir::Value dres = firstDiffOp.getOutputs()[out_idx];
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
            mlir::Value res = firstDiffOp.getOutputs()[out_idx];
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
        ArrayAttr newInActivity = ArrayAttr::get(
            context, llvm::ArrayRef<Attribute>(inActivityAttrs.begin(),
                                               inActivityAttrs.end()));

        ArrayAttr newRetActivity = ArrayAttr::get(
            context, llvm::ArrayRef<Attribute>(retActivityAttrs.begin(),
                                               retActivityAttrs.end()));

        IntegerAttr newWidthAttr =
            IntegerAttr::get(context, llvm::APSInt(width));

        ForwardDiffOp newDiffOp = builder.create<ForwardDiffOp>(
            loc, out_ty, firstDiffOp.getFnAttr(), in_args, newInActivity,
            newRetActivity, newWidthAttr, firstDiffOp.getStrongZeroAttr());

        // map old uses to new uses
        // reduce primal from multiple to 1
        // preserve derivative uses
        out_idx = 0;
        for (auto [idx, ract] : llvm::enumerate(key.retActivity)) {
          switch (ract) {
          case Activity::enzyme_constnoneed:
            // no-op
            break;
          case Activity::enzyme_const: {
            mlir::Value new_out = newDiffOp.getOutputs()[out_idx];

            for (ForwardDiffOp dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            out_idx++;
            break;
          }

          case Activity::enzyme_dupnoneed: {
            // derivative
            mlir::Value batch_dout = newDiffOp.getOutputs()[out_idx];
            for (auto [dop_idx, dop] : llvm::enumerate(allOps)) {

              mlir::Value old_dres = dop.getOutputs()[out_idx];
              auto old_dresTy = old_dres.getType();
              mlir::Value new_dres;
              auto T = cast<TensorType>(old_dresTy);
              mlir::Value indexOp =
                  builder.create<arith::ConstantIndexOp>(loc, dop_idx);

              if (!T) {
                new_dres =
                    builder.create<tensor::ExtractOp>(loc, batch_dout, indexOp);
              } else {
                auto rankedType = cast<RankedTensorType>(batch_dout.getType());
                SmallVector<OpFoldResult> offsets, sizes, strides;

                // Offsets: [dop_idx, 0, 0, ...]
                offsets.push_back(builder.getI64IntegerAttr(dop_idx));
                for (int i = 1; i < rankedType.getRank(); ++i) {
                  offsets.push_back(builder.getI64IntegerAttr(0));
                }
                // Sizes: [1, original_dim1, original_dim2, ...]
                sizes.push_back(builder.getI64IntegerAttr(1));
                for (auto dim : cast<RankedTensorType>(T).getShape()) {
                  sizes.push_back(builder.getI64IntegerAttr(dim));
                }

                // Strides: [1, 1, 1, ...]
                for (int i = 0; i < rankedType.getRank(); ++i) {
                  strides.push_back(builder.getI64IntegerAttr(1));
                }

                new_dres = builder.create<tensor::ExtractSliceOp>(
                    loc, batch_dout, offsets, sizes, strides);
              }

              old_dres.replaceAllUsesWith(new_dres);
            }
            ++out_idx;
            break;
          }

          case Activity::enzyme_dup: {
            mlir::Value new_out = newDiffOp.getOutputs()[out_idx];

            for (ForwardDiffOp dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            out_idx++;

            // derivative
            mlir::Value batch_dout = newDiffOp.getOutputs()[out_idx];
            for (auto [dop_idx, dop] : llvm::enumerate(allOps)) {

              mlir::Value old_dres = dop.getOutputs()[out_idx];
              auto old_dresTy = old_dres.getType();
              mlir::Value new_dres;
              auto T = cast<TensorType>(old_dresTy);
              mlir::Value indexOp =
                  builder.create<arith::ConstantIndexOp>(loc, dop_idx);

              if (!T) {
                new_dres =
                    builder.create<tensor::ExtractOp>(loc, batch_dout, indexOp);
              } else {
                auto rankedType = cast<RankedTensorType>(batch_dout.getType());
                SmallVector<OpFoldResult> offsets, sizes, strides;

                // Offsets: [dop_idx, 0, 0, ...]
                offsets.push_back(builder.getI64IntegerAttr(dop_idx));
                for (int i = 1; i < rankedType.getRank(); ++i) {
                  offsets.push_back(builder.getI64IntegerAttr(0));
                }
                // Sizes: [1, original_dim1, original_dim2, ...]
                sizes.push_back(builder.getI64IntegerAttr(1));
                for (auto dim : cast<RankedTensorType>(T).getShape()) {
                  sizes.push_back(builder.getI64IntegerAttr(dim));
                }

                // Strides: [1, 1, 1, ...]
                for (int i = 0; i < rankedType.getRank(); ++i) {
                  strides.push_back(builder.getI64IntegerAttr(1));
                }

                new_dres = builder.create<tensor::ExtractSliceOp>(
                    loc, batch_dout, offsets, sizes, strides);
              }

              old_dres.replaceAllUsesWith(new_dres);
            }

            ++out_idx;
            break;
          }
          case Activity::enzyme_active: {
            // TODO: check later
            mlir::Value new_out = newDiffOp.getOutputs()[out_idx];

            for (ForwardDiffOp dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            out_idx++;
            break;
          }

          case Activity::enzyme_activenoneed: {
            // TODO: check later
            mlir::Value new_out = newDiffOp.getOutputs()[out_idx];

            for (ForwardDiffOp dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            out_idx++;
            break;
          }
          }
        }

        // erase old ops
        for (auto dop : allOps) {
          op->erase();
        }
      }
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
