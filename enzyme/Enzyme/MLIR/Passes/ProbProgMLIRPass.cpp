//===- ProbProgMLIRPass.cpp - Replace calls with ProbProg operations
//------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to handle probabilistic programming operations
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/ProbProgUtils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "probprog"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct ProbProgPass : public ProbProgPassBase<ProbProgPass> {
  MEnzymeLogic Logic;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::OpPassManager pm;
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (!mlir::failed(result)) {
      pm.getDependentDialects(registry);
    }

    registry.insert<mlir::arith::ArithDialect, mlir::complex::ComplexDialect,
                    mlir::cf::ControlFlowDialect, mlir::tensor::TensorDialect,
                    mlir::enzyme::EnzymeDialect>();
  }

  LogicalResult HandleGenerate(SymbolTableCollection &symbolTable,
                               enzyme::GenerateOp CI) {
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

    auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Generate);
    FunctionOpInterface NewF = putils->newFunc;

    // Replace SampleOps with distribution function calls
    SmallVector<Operation *, 4> toErase;
    NewF.walk([&](enzyme::SampleOp sampleOp) {
      OpBuilder b(sampleOp);
      auto distFn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
      auto distCall = b.create<func::CallOp>(
          sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
          sampleOp.getOperands());
      sampleOp.replaceAllUsesWith(distCall);

      toErase.push_back(sampleOp);
    });

    for (Operation *op : toErase) {
      op->erase();
    }

    OpBuilder b(CI);
    auto newCallOp = b.create<func::CallOp>(
        CI.getLoc(), NewF.getName(), NewF.getResultTypes(), CI.getOperands());

    CI->replaceAllUsesWith(newCallOp);
    CI->erase();

    return success();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    op->walk([&](enzyme::GenerateOp cop) {
      auto res = HandleGenerate(symbolTable, cop);
      if (!res.succeeded()) {
        signalPassFailure();
        return;
      }
    });
  };
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createProbProgPass() {
  return std::make_unique<ProbProgPass>();
}
} // namespace enzyme
} // namespace mlir

void ProbProgPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeCalls(symbolTable, op); });
}
