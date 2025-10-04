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
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"

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

Type getTensorizedType(Value val, int64_t width) {
  auto valTy = val.getType();
  auto valTensorTy = dyn_cast<TensorType>(valTy);
  if (!valTensorTy) {
    // val is not a tensor
    return RankedTensorType::get(width, valTy);
  } else {
    // val is a tensor, prepend batch width to shape
    SmallVector<int64_t> out_shape = {width};
    out_shape.append(valTensorTy.getShape().begin(),
                     valTensorTy.getShape().end());
    auto outTy = valTensorTy.clone(out_shape);
    return outTy;
  }
}

Value getTensorizedValue(OpBuilder &builder, Location &loc,
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

Value extractValueAtIdx(OpBuilder &builder, Location &loc, Type &argTy,
                        Value &val, int64_t index) {
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

} // namespace batchutils
} // namespace enzyme
} // namespace mlir
namespace {

struct BatchDiffPass : public enzyme::impl::BatchDiffPassBase<BatchDiffPass> {
  void runOnOperation() override;

  void mergeFwddiffCalls(SymbolTableCollection &symbolTable,
                         FunctionOpInterface op) {
  // TODO: Use a modified version of inter-procedural DataFlowAliasAnalysis
    // for mapping primal effects
    std::map<FunctionOpInterface, SmallVector<MemoryEffects::EffectInstance>>
        innerEffectCache;

    OpBuilder builder(op);

    op->walk([&](Block *blk) {
      // map tracking batchable AD calls
      std::map<enzyme::batchutils::BatchDiffCacheKey,
               SmallVector<enzyme::ForwardDiffOp>>
          toMerge;

      for (auto fwdOp : blk->getOps<enzyme::ForwardDiffOp>()) {
        auto fnOp = dyn_cast_or_null<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(fwdOp, fwdOp.getFnAttr()));
        if (!fnOp)
          continue;

        batchutils::BatchDiffCacheKey key =
            batchutils::createDiffCacheKey(fwdOp, fnOp);

        toMerge[key].push_back(fwdOp);
      }

      for (auto &pair : toMerge) {
        auto key = pair.first;
        auto allDiffs = pair.second;
        if (allDiffs.size() < 2)
          continue;

        // Collect inner effects of function
        if (innerEffectCache.find(key.function) == innerEffectCache.end()) {
          innerEffectCache[key.function] =
              oputils::collectFnEffects(key.function);
        }
        SmallVector<MemoryEffects::EffectInstance> &calleeEffects =
            innerEffectCache[key.function];

        // TODO skip if known readnone from existing analyses

        bool skipMergeEntry = false;
        // Map callee(primal function) memory effects to the calling
        // function's(autodiff op's) memory effects. This allows us to also
        // reason about memory effects on the derivative argument potentially
        // being passed in(if the primal argument has activity enzyme_dup)

        // ForwardDiff read(primal) ?-> read(deriv);
        // write(primal) ?-> write(deriv)
        //
        // ReverseDiff read(primal) ?-> read or write(deriv);
        // write(primal) ?-> read or write(deriv);
        //
        // Unknown effects are retained in the caller, and will always alias to
        // true with any other effect

        SmallVector<MemoryEffects::EffectInstance, 4> callerEffects;

        for (auto &eff : calleeEffects) {
          if (isa<MemoryEffects::Read>(eff.getEffect()) ||
              isa<MemoryEffects::Write>(eff.getEffect())) {
            if (Value resource = eff.getValue()) {

              auto argnum = 0;
              bool found = false;
              for (auto fnArg : key.function.getArguments()) {
                if (fnArg == resource) {
                  argnum = fnArg.getArgNumber();
                  found = true;
                  break;
                }
              }
              auto e = eff.getEffect();
              if (found) {

                // Primal value as effect
                Value primalVal = key.inputs[argnum];
                if (auto primalOpResult = dyn_cast<OpResult>(primalVal))
                  callerEffects.emplace_back(MemoryEffects::EffectInstance(
                      eff.getEffect(), primalOpResult, eff.getResource()));
                else if (auto primalBlockArg =
                             dyn_cast<BlockArgument>(primalVal)) {
                  callerEffects.emplace_back(MemoryEffects::EffectInstance(
                      eff.getEffect(), primalBlockArg, eff.getResource()));
                } else {
                  llvm_unreachable("Value is neither an argument nor a result "
                                   "of an op. This is not allowed by SSA");
                }

                // Derivative effects(remain the same for fwddiff)
                for (auto dop : allDiffs) {
                  auto act_val =
                      cast<ActivityAttr>(dop.getActivity()[argnum]).getValue();
                  if (act_val == Activity::enzyme_dup ||
                      act_val == Activity::enzyme_dupnoneed) {
                    // derivative index is always argnum + 1
                    Value dVal = dop.getInputs()[argnum + 1];
                    if (auto dOpResult = dyn_cast<OpResult>(dVal)) {
                      callerEffects.emplace_back(MemoryEffects::EffectInstance(
                          eff.getEffect(), dOpResult, eff.getResource()));
                    } else if (auto dBlockArg = dyn_cast<BlockArgument>(dVal)) {
                      callerEffects.emplace_back(MemoryEffects::EffectInstance(
                          eff.getEffect(), dBlockArg, eff.getResource()));
                    } else {
                      llvm_unreachable(
                          "Value is neither an argument nor a result "
                          "of an op. This is not allowed by SSA");
                    }
                  }
                }
              } else {
                // effect on some unknown mlir::Value which is not a primal
                // function argument
                // TODO: Handle this either as a global value, or a value which
                // is inside of the MLIR function.
                callerEffects.emplace_back(eff);
              }

            } else {
              // unknown effect which isnt a value
              callerEffects.emplace_back(eff);
            }
          } else {
            // encountered an allocate / load effect. Skip to next map entry
            skipMergeEntry = true;
            break;
          }
        }

        if (skipMergeEntry)
          continue;

        SmallVector<enzyme::ForwardDiffOp> legalMerge;
        if (callerEffects.size() == 0) {
          // legal to merge since there is no effect overwrite
          legalMerge = allDiffs;
        } else {

          legalMerge.emplace_back(allDiffs[0]);
          SmallVector<MemoryEffects::EffectInstance, 4> betweenEffects;

          for (auto *cur = allDiffs[0]->getNextNode(); cur != allDiffs.back();
               cur = cur->getNextNode()) {
            auto curOpEffects = mlir::getEffectsRecursively(cur);
            if (curOpEffects.has_value()) {
              betweenEffects.clear();
              betweenEffects.append(curOpEffects->begin(), curOpEffects->end());
            }

            bool stillOk = true;

            for (auto eff : betweenEffects) {
              // read before write
              if (isa<MemoryEffects::Read>(eff.getEffect())) {
                for (auto fneff : callerEffects) {
                  if (isa<MemoryEffects::Write>(fneff.getEffect())) {
                    if (oputils::mayAlias(eff, fneff)) {
                      stillOk = false;
                      break;
                    }
                  }
                }
              }
              // write before read
              if (isa<MemoryEffects::Write>(eff.getEffect())) {
                for (auto fneff : callerEffects) {
                  if (isa<MemoryEffects::Read>(fneff.getEffect())) {
                    if (oputils::mayAlias(eff, fneff)) {
                      stillOk = false;
                      break;
                    }
                  }
                }
              }
            }
            if (!stillOk) {
              break;
            }

            auto curFnOp = dyn_cast<enzyme::ForwardDiffOp>(cur);
            if (curFnOp && (std::find(pair.second.begin(), pair.second.end(),
                                      curFnOp) != pair.second.end())) {
              legalMerge.push_back(cast<enzyme::ForwardDiffOp>(cur));
            }
          }
        }

        // go ahead and actually do the merge now
        {
          SmallVector<enzyme::ForwardDiffOp> allOps = legalMerge;
          int64_t width = allOps.size();

          if (width < 2)
            continue;

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
                  batchutils::getTensorizedValue(builder, loc, derivList);
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
              out_ty.push_back(batchutils::getTensorizedType(dres, width));
              ++out_idx;
              break;
            }

            case Activity::enzyme_dup: {
              mlir::Value res = firstDiffOp.getOutputs()[out_idx];
              out_ty.push_back(res.getType());

              ++out_idx;

              // derivative
              mlir::Value dres = firstDiffOp.getOutputs()[out_idx];
              out_ty.push_back(batchutils::getTensorizedType(dres, width));
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
                auto new_dout = batchutils::extractValueAtIdx(
                    builder, loc, doutTy, batch_dout, dop_idx);

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
                auto new_dout = batchutils::extractValueAtIdx(
                    builder, loc, doutTy, batch_dout, dop_idx);

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
    }); // block walker
  }

  void mergeRevdiffCalls(SymbolTableCollection &symbolTable,
                         FunctionOpInterface op) {

    // TODO: Use a modified version of inter-procedural DataFlowAliasAnalysis
    // for mapping primal effects

    // list of values read/written to inside fn
    std::map<FunctionOpInterface, SmallVector<MemoryEffects::EffectInstance>>
        innerEffectCache;

    OpBuilder builder(op);

    op->walk([&](Block *blk) {
      // map tracking batchable AD calls
      std::map<enzyme::batchutils::BatchDiffCacheKey,
               SmallVector<enzyme::AutoDiffOp>>
          toMerge;

      for (auto revOp : blk->getOps<enzyme::AutoDiffOp>()) {
        auto fnOp = dyn_cast_or_null<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(revOp, revOp.getFnAttr()));
        if (!fnOp)
          continue;

        batchutils::BatchDiffCacheKey key =
            batchutils::createDiffCacheKey(revOp, fnOp);

        toMerge[key].push_back(revOp);
      }

      for (auto &pair : toMerge) {
        auto key = pair.first;
        auto allDiffs = pair.second;
        if (allDiffs.size() < 2)
          continue;

        // Collect inner effects of function
        if (innerEffectCache.find(key.function) == innerEffectCache.end()) {
          innerEffectCache[key.function] =
              oputils::collectFnEffects(key.function);
        }
        SmallVector<MemoryEffects::EffectInstance> &calleeEffects =
            innerEffectCache[key.function];

        // TODO: skip if known readonly from existing analyses

        bool skipMergeEntry = false;
        // Map callee(primal function) memory effects to the calling
        // function's(autodiff op's) memory effects. This allows us to also
        // reason about memory effects on the derivative argument potentially
        // being passed in(if the primal argument has activity enzyme_dup)

        // ForwardDiff read(primal) ?-> read(deriv);
        // write(primal) ?-> write(deriv)
        //
        // ReverseDiff read(primal) ?-> read or write(deriv);
        // write(primal) ?-> read or write(deriv);
        //
        // Unknown effects are retained in the caller, and will always alias to
        // true with any other effect

        SmallVector<MemoryEffects::EffectInstance, 4> callerEffects;

        for (auto &eff : calleeEffects) {
          if (isa<MemoryEffects::Read>(eff.getEffect()) ||
              isa<MemoryEffects::Write>(eff.getEffect())) {
            if (Value resource = eff.getValue()) {

              auto argnum = 0;
              bool found = false;
              for (auto fnArg : key.function.getArguments()) {
                if (fnArg == resource) {
                  argnum = fnArg.getArgNumber();
                  found = true;
                  break;
                }
              }

              if (found) {

                // Primal value as effect
                Value primalVal = key.inputs[argnum];
                if (auto primalOpResult = dyn_cast<OpResult>(primalVal))
                  callerEffects.emplace_back(MemoryEffects::EffectInstance(
                      eff.getEffect(), primalOpResult, eff.getResource()));
                else if (auto primalBlockArg =
                             dyn_cast<BlockArgument>(primalVal)) {
                  callerEffects.emplace_back(MemoryEffects::EffectInstance(
                      eff.getEffect(), primalBlockArg, eff.getResource()));
                } else {
                  llvm_unreachable("Value is neither an argument nor a result "
                                   "of an op. This is not allowed by SSA");
                }

                auto deff_read =
                    isa<MemoryEffects::Read>(eff.getEffect()) ? true : false;

                // Derivative effects(flipped for revdiff)
                for (auto dop : allDiffs) {
                  auto act_val =
                      cast<ActivityAttr>(dop.getActivity()[argnum]).getValue();
                  if (act_val == Activity::enzyme_dup ||
                      act_val == Activity::enzyme_dupnoneed) {
                    // derivative index is always argnum + 1
                    Value dVal = dop.getInputs()[argnum + 1];
                    if (auto dOpResult = dyn_cast<OpResult>(dVal)) {
                      callerEffects.emplace_back(MemoryEffects::EffectInstance(
                          eff.getEffect(), dOpResult, eff.getResource()));
                      if (deff_read) {
                        callerEffects.emplace_back(
                            MemoryEffects::EffectInstance(
                                MemoryEffects::Write::get(), dOpResult,
                                eff.getResource()));
                      } else {
                        callerEffects.emplace_back(
                            MemoryEffects::EffectInstance(
                                MemoryEffects::Read::get(), dOpResult,
                                eff.getResource()));
                      }
                    } else if (auto dBlockArg = dyn_cast<BlockArgument>(dVal)) {
                      callerEffects.emplace_back(MemoryEffects::EffectInstance(
                          eff.getEffect(), dBlockArg, eff.getResource()));
                      if (deff_read) {
                        callerEffects.emplace_back(
                            MemoryEffects::EffectInstance(
                                MemoryEffects::Write::get(), dBlockArg,
                                eff.getResource()));
                      } else {
                        callerEffects.emplace_back(
                            MemoryEffects::EffectInstance(
                                MemoryEffects::Read::get(), dBlockArg,
                                eff.getResource()));
                      }
                    } else {
                      llvm_unreachable(
                          "Value is neither an argument nor a result "
                          "of an op. This is not allowed by SSA");
                    }
                  }
                }
              } else {
                // effect on some unknown mlir::Value which is not a primal
                // function argument
                // TODO: Handle this either as a global value, or a value which
                // is inside of the MLIR function.
                callerEffects.emplace_back(eff);
              }

            } else {
              // unknown effect which isnt a value
              callerEffects.emplace_back(eff);
            }
          } else {
            // encountered an allocate / load effect. Skip to next map entry
            skipMergeEntry = true;
            break;
          }
        }

        if (skipMergeEntry)
          continue;

        SmallVector<enzyme::AutoDiffOp> legalMerge;
        if (callerEffects.size() == 0) {
          // legal to merge since there is no effect overwrite
          legalMerge = allDiffs;
        } else {

          legalMerge.emplace_back(allDiffs[0]);
          SmallVector<MemoryEffects::EffectInstance, 4> betweenEffects;

          for (auto *cur = allDiffs[0]->getNextNode(); cur != allDiffs.back();
               cur = cur->getNextNode()) {
            auto curOpEffects = mlir::getEffectsRecursively(cur);
            if (curOpEffects.has_value()) {
              betweenEffects.clear();
              betweenEffects.append(curOpEffects->begin(), curOpEffects->end());
            }

            bool stillOk = true;

            for (auto eff : betweenEffects) {
              // read before write
              if (isa<MemoryEffects::Read>(eff.getEffect())) {
                for (auto fneff : callerEffects) {
                  if (isa<MemoryEffects::Write>(fneff.getEffect())) {
                    if (oputils::mayAlias(eff, fneff)) {
                      stillOk = false;
                      break;
                    }
                  }
                }
              }
              // write before read
              if (isa<MemoryEffects::Write>(eff.getEffect())) {
                for (auto fneff : callerEffects) {
                  if (isa<MemoryEffects::Read>(fneff.getEffect())) {
                    if (oputils::mayAlias(eff, fneff)) {
                      stillOk = false;
                      break;
                    }
                  }
                }
              }
            }
            if (!stillOk) {
              break;
            }

            auto curFnOp = dyn_cast<enzyme::AutoDiffOp>(cur);
            if (curFnOp && (std::find(pair.second.begin(), pair.second.end(),
                                      curFnOp) != pair.second.end())) {
              legalMerge.push_back(cast<enzyme::AutoDiffOp>(cur));
            }
          }
        }

        // go ahead and actually do the merge now
        {

          SmallVector<enzyme::AutoDiffOp> &allOps = legalMerge;
          int64_t width = allOps.size();

          if (width < 2)
            continue;

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
                  batchutils::getTensorizedValue(builder, loc, derivList);

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
                  batchutils::getTensorizedValue(builder, loc, derivList);
              in_args.push_back(batch_dout);
              call_idx++;
            }

            // handle out
            if (ract == Activity::enzyme_active ||
                ract == Activity::enzyme_const ||
                ract == Activity::enzyme_dup) {
              Value out = firstDiffOp.getOutputs()[out_idx];
              out_ty.push_back(out.getType());
              ++out_idx;
            }
          }

          // fill out_ty using d<in>
          for (auto act : key.inActivity) {
            if (act == Activity::enzyme_active) {
              Value din = firstDiffOp.getOutputs()[out_idx];
              out_ty.push_back(batchutils::getTensorizedType(din, width));
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
                ract == Activity::enzyme_const ||
                ract == Activity::enzyme_dup) {
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
                auto new_din = batchutils::extractValueAtIdx(
                    builder, loc, dinTy, batch_din, dop_idx);

                old_din.replaceAllUsesWith(new_din);
              }
            }
          }
          // erase all old ops
          for (auto dop : allOps) {
            dop->erase();
          }
        }
      }
    }); // block walker
  }
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
