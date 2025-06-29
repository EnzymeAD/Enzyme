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

  LogicalResult HandleUntracedCall(SymbolTableCollection &symbolTable,
                                   enzyme::UntracedCallOp CI) {
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

    auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Call);
    FunctionOpInterface NewF = putils->newFunc;

    {
      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](enzyme::SampleOp sampleOp) {
        OpBuilder b(sampleOp);
        auto distFn =
            cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                sampleOp, sampleOp.getFnAttr()));
        auto distCall = b.create<func::CallOp>(
            sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
            sampleOp.getInputs());
        sampleOp.replaceAllUsesWith(distCall);

        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase) {
        op->erase();
      }
    }

    OpBuilder b(CI);
    auto newCI = b.create<func::CallOp>(
        CI.getLoc(), NewF.getName(), NewF.getResultTypes(), CI.getOperands());

    CI->replaceAllUsesWith(newCI);
    CI->erase();

    return success();
  }

  LogicalResult HandleSimulate(SymbolTableCollection &symbolTable,
                               enzyme::SimulateOp CI) {
    auto symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    if (fn.getFunctionBody().empty()) {
      llvm::errs() << fn << "\n";
      llvm_unreachable("Simulating empty function");
    }

    auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Simulate);
    FunctionOpInterface NewF = putils->newFunc;

    {
      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](enzyme::SampleOp sampleOp) {
        OpBuilder b(sampleOp);
        auto distFn =
            cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                sampleOp, sampleOp.getFnAttr()));
        auto distCall = b.create<func::CallOp>(
            sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
            sampleOp.getInputs());

        auto tracedOutputIndices = sampleOp.getTracedOutputIndicesAttr();
        if (tracedOutputIndices) {
          for (auto idx : tracedOutputIndices.asArrayRef()) {
            b.create<enzyme::AddSampleToTraceOp>(
                sampleOp.getLoc(),
                /*trace*/ putils->getTrace(),
                /*symbol*/ sampleOp.getSymbolAttr(),
                /*sample*/ ValueRange{distCall.getResult(idx)});
          }
        }

        sampleOp.replaceAllUsesWith(distCall);
        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase) {
        op->erase();
      }
    }

    // Return the trace as the first result
    NewF.walk([&](func::ReturnOp retOp) {
      OpBuilder b(retOp);
      SmallVector<Value> newRetVals;
      newRetVals.push_back(putils->getTrace());
      newRetVals.append(retOp.getOperands().begin(), retOp.getOperands().end());

      b.create<func::ReturnOp>(retOp.getLoc(), newRetVals);
      retOp.erase();
    });

    if (!NewF)
      return failure();

    delete putils;

    OpBuilder b(CI);
    auto newCI = b.create<func::CallOp>(
        CI.getLoc(), NewF.getName(), NewF.getResultTypes(), CI.getOperands());

    CI->replaceAllUsesWith(newCI);
    CI->erase();

    return success();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    op->walk([&](enzyme::UntracedCallOp ucop) {
      auto res = HandleUntracedCall(symbolTable, ucop);
      if (!res.succeeded()) {
        signalPassFailure();
        return;
      }
    });
    op->walk([&](enzyme::SimulateOp sop) {
      auto res = HandleSimulate(symbolTable, sop);
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

  if (!postpasses.empty()) {
    mlir::PassManager pm(getOperation()->getContext());

    if (mlir::failed(mlir::parsePassPipeline(postpasses, pm))) {
      getOperation()->emitError()
          << "Failed to parse probprog post-passes pipeline: " << postpasses;
      signalPassFailure();
      return;
    }

    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
      return;
    }
  }
}
