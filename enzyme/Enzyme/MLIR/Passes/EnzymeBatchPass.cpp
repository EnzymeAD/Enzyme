//===- EnzymeBatchPass.cpp - Replace calls with their batched versions
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

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "enzyme-batch"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct BatchPass : public BatchPassBase<BatchPass> {
  MEnzymeLogic Logic;

  void runOnOperation() override;

  template <typename T>
  LogicalResult HandleBatch(SymbolTableCollection &symbolTable, T CI) {
    SmallVector<mlir::Value, 2> args;

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    SmallVector<DIFFE_TYPE> RetActivity(CI.getResults().size(),
                                        DIFFE_TYPE::CONSTANT);
    SmallVector<DIFFE_TYPE> ArgActivity(CI.getInputs().size(),
                                        DIFFE_TYPE::CONSTANT);
    std::vector<bool> returnPrimals(CI.getResults().size(), true);
    std::vector<bool> returnShadows(CI.getResults().size(), false);

    IRMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    IRMapping invertedPointers;

    FunctionOpInterface newFunc = CloneFunctionWithReturns(
        /*mode*/ DerivativeMode::ForwardMode, /*width*/ 1, fn, invertedPointers,
        ArgActivity, constant_values, nonconstant_values, returnvals,
        returnPrimals, returnShadows, RetActivity, "batched_" + fn.getName(),
        originalToNew, originalToNewOps,
        /*additionalArg*/ nullptr, CI.getBatchShape());

    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    auto dCI =
        builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                     newFunc.getResultTypes(), CI.getInputs());
    CI.replaceAllUsesWith(dCI);
    CI->erase();
    return success();
  }

  void lowerEnzymeBatchCalls(SymbolTableCollection &symbolTable,
                             FunctionOpInterface op) {
    {
      SmallVector<Operation *> toLower;
      op->walk([&](enzyme::BatchOp dop) {
        auto *symbolOp =
            symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
        auto callableOp = cast<FunctionOpInterface>(symbolOp);

        lowerEnzymeBatchCalls(symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::BatchOp>(T)) {
          auto res = HandleBatch(symbolTable, F);
          if (!res.succeeded()) {
            signalPassFailure();
            return;
          }
        } else {
          llvm_unreachable("Illegal type");
        }
      }
    };
  };
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createBatchPass() {
  return std::make_unique<BatchPass>();
}
} // namespace enzyme
} // namespace mlir

void BatchPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeBatchCalls(symbolTable, op); });
}
