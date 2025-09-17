//===- EnzymeBatchDiffPass.cpp - Merge autodiff calls into their batched
// versions
//------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
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

      auto mergeIt = toMerge.find(key);
      if (mergeIt != toMerge.end()) {
        mergeIt->second.push_back(dop);
      } else {
        SmallVector<enzyme::ForwardDiffOp> v;
        v.push_back(dop);
        toMerge[key] = v;
      }
    });

    // process map
    ;
  };
};

} // end anonymous namespace

void BatchDiffPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk(
      [&](FunctionOpInterface op) { mergeFwddiffCalls(symbolTable, op); });
}
