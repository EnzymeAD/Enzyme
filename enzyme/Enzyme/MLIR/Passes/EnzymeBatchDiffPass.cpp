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

namespace mlir {
namespace enzyme {
namespace batchutils {

Value tensorizeArg(OpBuilder &builder, Location &loc,
                   SmallVector<Value> &argList) {
  auto argTy = argList.front().getType();
  auto T = dyn_cast<TensorType>(argTy);
  mlir::Value out;
  if (!T) {
    // use tensor.from_elements
    out = builder.create<tensor::FromElementsOp>(loc, argList);
  } else {
    // use tensor.concat on dim 0
    out = builder.create<tensor::ConcatOp>(loc, /*dim*/ 0, argList);
  }
  return out;
}

Value extractArg(OpBuilder &builder, Location &loc, Type &argTy, Value &val,
                 int64_t index) {
  // Extract the original output from the tensorized output at the given index.
  auto T = dyn_cast<TensorType>(argTy);
  Value out;
  if (!T) {
    Value indexOp = builder.create<arith::ConstantIndexOp>(loc, index);
    out = builder.create<tensor::ExtractOp>(loc, val, indexOp);
  } else {
    auto valRTy = cast<RankedTensorType>(val.getType());
    SmallVector<OpFoldResult> offsets, sizes, strides;

    // Offsets: [index, 0, 0, ...]
    offsets.push_back(builder.getI64IntegerAttr(index));
    for (auto i = 1; i < valRTy.getRank(); ++i) {
      offsets.push_back(builder.getI64IntegerAttr(0));
    }

    // Sizes: [1, original_dim1, original_dim2, ...]
    sizes.push_back(builder.getI64IntegerAttr(1));
    for (auto dim : cast<RankedTensorType>(T).getShape()) {
      sizes.push_back(builder.getI64IntegerAttr(dim));
    }

    // Strides: [1, 1, 1, ...]
    for (int i = 0; i < valRTy.getRank(); ++i) {
      strides.push_back(builder.getI64IntegerAttr(1));
    }

    // reduce rank
    auto out_ty = RankedTensorType::get(valRTy.getShape().drop_front(),
                                        valRTy.getElementType());
    out = builder.create<tensor::ExtractSliceOp>(loc, out_ty, val, offsets,
                                                 sizes, strides);
  }

  return out;
}

bool isReadOnly(Operation *op) {
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
  }

  bool isRecursiveContainer =
      op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
      isa<FunctionOpInterface>(op);
  if (isRecursiveContainer) {
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

} // namespace batchutils
} // namespace enzyme
} // namespace mlir
namespace {

struct BatchDiffPass : public enzyme::impl::BatchDiffPassBase<BatchDiffPass> {
  void runOnOperation() override;

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

      // skip if fn isn't readonly(iterate through toplevel ops)
      if (!batchutils::isReadOnly(fnOp)) {
        return mlir::WalkResult::skip();
      }

      batchutils::BatchDiffCacheKey key =
          batchutils::createDiffCacheKey(dop, fnOp);

      auto mergeItr = toMerge.find(key);
      if (mergeItr != toMerge.end()) {
        mergeItr->second.push_back(dop);
      } else {
        toMerge[key] = {dop};
      }
      return mlir::WalkResult::advance();
    });

    OpBuilder builder(op);

    // Merge call subsets
    for (auto mergeItr = toMerge.begin(); mergeItr != toMerge.end();
         ++mergeItr) {
      auto key = mergeItr->first;
      SmallVector<enzyme::ForwardDiffOp> allOps = mergeItr->second;
      int64_t width = allOps.size();

      if (width < 2)
        continue;

      {
        // We will insert the merged op before the first fwddiff call
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

        // process input, d<input>
        for (auto [idx, act] : llvm::enumerate(key.inActivity)) {
          ActivityAttr iattr = ActivityAttr::get(context, act);
          inActivityAttrs.push_back(iattr);
          in_args.push_back(key.inputs[in_idx]);
          in_idx++;

          SmallVector<mlir::Value> derivList;
          if (act == Activity::enzyme_dup ||
              act == Activity::enzyme_dupnoneed) {
            for (auto uop : allOps) {
              derivList.push_back(uop.getInputs()[in_idx]);
            }

            mlir::Value batchedDeriv =
                batchutils::tensorizeArg(builder, loc, derivList);
            in_args.push_back(batchedDeriv);
            in_idx++;
          }
        }

        // process out, d<out> (only need types)
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
              SmallVector<int64_t> shape = {width};
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
              SmallVector<int64_t> shape = {width};
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
            IntegerAttr::get(firstDiffOp.getWidthAttr().getType(), width);

        auto newDiffOp = builder.create<ForwardDiffOp>(
            loc, out_ty, firstDiffOp.getFnAttr(), in_args, newInActivity,
            newRetActivity, newWidthAttr, firstDiffOp.getStrongZeroAttr());

        // Rename old users of out,d<out> to new users
        out_idx = 0;
        for (auto [idx, ract] : llvm::enumerate(key.retActivity)) {
          switch (ract) {
          case Activity::enzyme_constnoneed:
            // no-op
            break;
          case Activity::enzyme_const: {
            auto new_out = newDiffOp.getOutputs()[out_idx];

            for (auto dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }

            out_idx++;
            break;
          }

          case Activity::enzyme_dupnoneed: {
            // derivative
            auto batch_dout = newDiffOp.getOutputs()[out_idx];
            for (auto [dop_idx, dop] : llvm::enumerate(allOps)) {
              auto old_dout = dop.getOutputs()[out_idx];
              auto doutTy = old_dout.getType();
              auto new_dout = batchutils::extractArg(builder, loc, doutTy,
                                                     batch_dout, dop_idx);

              old_dout.replaceAllUsesWith(new_dout);
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
            auto batch_dout = newDiffOp.getOutputs()[out_idx];
            for (auto [dop_idx, dop] : llvm::enumerate(allOps)) {

              auto old_dout = dop.getOutputs()[out_idx];
              auto doutTy = old_dout.getType();
              auto new_dout = batchutils::extractArg(builder, loc, doutTy,
                                                     batch_dout, dop_idx);

              old_dout.replaceAllUsesWith(new_dout);
            }

            ++out_idx;
            break;
          }
          case Activity::enzyme_active: {
            // TODO: check later
            auto new_out = newDiffOp.getOutputs()[out_idx];

            for (ForwardDiffOp dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            out_idx++;
            break;
          }

          case Activity::enzyme_activenoneed: {
            // TODO: check later
            auto new_out = newDiffOp.getOutputs()[out_idx];

            for (ForwardDiffOp dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            out_idx++;
            break;
          }
          }
        }

        // erase all old ops
        for (auto dop : allOps) {
          dop->erase();
        }
      }
    }
  }

  void mergeRevdiffCalls(SymbolTableCollection &symbolTable,
                         FunctionOpInterface op) {

    // map tracking batchable AD calls
    std::map<enzyme::batchutils::BatchDiffCacheKey,
             SmallVector<enzyme::AutoDiffOp>>
        toMerge;

    op->walk([&](enzyme::AutoDiffOp dop) {
      // lookup function, check if its readOnly
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
      auto fnOp = cast<FunctionOpInterface>(symbolOp);

      // skip if fn isn't readonly(iterate through toplevel ops)
      if (!batchutils::isReadOnly(fnOp)) {
        return mlir::WalkResult::skip();
      }

      // add to map
      batchutils::BatchDiffCacheKey key =
          batchutils::createDiffCacheKey(dop, fnOp);

      auto mergeItr = toMerge.find(key);
      if (mergeItr != toMerge.end()) {
        mergeItr->second.push_back(dop);
      } else {
        toMerge[key] = {dop};
      }
      return mlir::WalkResult::advance();
    });

    OpBuilder builder(op);

    // process map
    for (auto mergeItr = toMerge.begin(); mergeItr != toMerge.end();
         ++mergeItr) {
      auto key = mergeItr->first;
      SmallVector<enzyme::AutoDiffOp> allOps = mergeItr->second;
      int64_t width = allOps.size();

      if (width < 2)
        continue;

      {
        auto firstDiffOp = allOps.front();
        IRRewriter::InsertionGuard insertGuard(builder);
        builder.setInsertionPoint(firstDiffOp);
        auto loc = firstDiffOp->getLoc();
        auto context = builder.getContext();

        // Prepare args for merged operation

        SmallVector<mlir::Value> in_args;
        SmallVector<ActivityAttr, 2> inActivityAttrs;
        SmallVector<ActivityAttr, 2> retActivityAttrs;
        SmallVector<mlir::Type, 2> out_ty;

        // fill in_args using inputs
        auto call_idx = 0;
        for (auto [idx, act] : llvm::enumerate(key.inActivity)) {
          auto iattr = ActivityAttr::get(context, act);
          inActivityAttrs.push_back(iattr);
          in_args.push_back(key.inputs[call_idx]);
          call_idx++;

          if (act == Activity::enzyme_dup ||
              act == Activity::enzyme_dupnoneed) {

            SmallVector<mlir::Value> derivList;
            for (auto uop : allOps) {
              derivList.push_back(uop.getInputs()[call_idx]);
            }

            mlir::Value b_din =
                batchutils::tensorizeArg(builder, loc, derivList);

            in_args.push_back(b_din);
            call_idx++;
          }
        }

        // Skip if function is non-differentiable
        if (call_idx == firstDiffOp.getInputs().size()) {
          continue;
        }

        // fill in_args using d<out>, fill out_ty using out
        auto out_idx = 0;
        for (auto ract : key.retActivity) {
          auto iattr = ActivityAttr::get(context, ract);
          retActivityAttrs.push_back(iattr);

          // no effect on out or d<out>
          if (ract == Activity::enzyme_constnoneed ||
              ract == Activity::enzyme_dupnoneed) {
            continue;
          }

          // handle d<out>
          if (ract == Activity::enzyme_active ||
              ract == Activity::enzyme_activenoneed) {
            SmallVector<mlir::Value> derivList;
            for (auto uop : allOps) {
              derivList.push_back(uop.getInputs()[call_idx]);
            }

            Value batch_dout =
                batchutils::tensorizeArg(builder, loc, derivList);
            in_args.push_back(batch_dout);
            call_idx++;
          }

          // handle out
          if (ract == Activity::enzyme_active ||
              ract == Activity::enzyme_const || ract == Activity::enzyme_dup) {
            Value out = firstDiffOp.getOutputs()[out_idx];
            out_ty.push_back(out.getType());
            ++out_idx;
          }
        }

        // fill out_ty using d<in>
        for (auto act : key.inActivity) {
          if (act == Activity::enzyme_active) {
            Value din = firstDiffOp.getOutputs()[out_idx];
            auto dinTy = din.getType();
            auto T = dyn_cast<TensorType>(dinTy);
            if (!T) {
              out_ty.push_back(RankedTensorType::get(width, dinTy));
            } else {
              SmallVector<int64_t> shape = {width};
              shape.append(T.getShape().begin(), T.getShape().end());
              out_ty.push_back(T.clone(shape));
            }
            ++out_idx;
          }
        }

        ArrayAttr newInActivity = ArrayAttr::get(
            context, llvm::ArrayRef<Attribute>(inActivityAttrs.begin(),
                                               inActivityAttrs.end()));

        ArrayAttr newRetActivity = ArrayAttr::get(
            context, llvm::ArrayRef<Attribute>(retActivityAttrs.begin(),
                                               retActivityAttrs.end()));

        IntegerAttr newWidthAttr =
            IntegerAttr::get(firstDiffOp.getWidthAttr().getType(), width);

        auto newDiffOp = builder.create<AutoDiffOp>(
            loc, out_ty, firstDiffOp.getFnAttr(), in_args, newInActivity,
            newRetActivity, newWidthAttr, firstDiffOp.getStrongZeroAttr());

        // Map old uses to new uses

        out_idx = 0;
        for (auto ract : key.retActivity) {
          if (ract == Activity::enzyme_active ||
              ract == Activity::enzyme_const || ract == Activity::enzyme_dup) {
            Value new_out = newDiffOp.getOutputs()[out_idx];
            for (auto dop : allOps) {
              dop.getOutputs()[out_idx].replaceAllUsesWith(new_out);
            }
            ++out_idx;
          }
        }

        for (auto act : key.inActivity) {
          if (act == Activity::enzyme_active) {
            Value batch_din = newDiffOp.getOutputs()[out_idx];
            for (auto [dop_idx, dop] : llvm::enumerate(allOps)) {
              Value old_din = dop.getOutputs()[out_idx];
              auto dinTy = old_din.getType();
              auto new_din = batchutils::extractArg(builder, loc, dinTy,
                                                    batch_din, dop_idx);

              old_din.replaceAllUsesWith(new_din);
            }
          }
        }
        // erase all old ops
        for (auto dop : allOps) {
          dop->erase();
        }
      }
    };
  };
};

} // end anonymous namespace

void BatchDiffPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk([&](FunctionOpInterface op) {
    mergeFwddiffCalls(symbolTable, op);
    mergeRevdiffCalls(symbolTable, op);
  });
}
