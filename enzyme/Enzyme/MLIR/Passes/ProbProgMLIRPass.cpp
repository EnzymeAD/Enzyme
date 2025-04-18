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

  LogicalResult HandleSimulate(SymbolTableCollection &symbolTable,
                               enzyme::SimulateOp CI) {
    auto symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    if (fn.getFunctionBody().empty()) {
      llvm::errs() << fn << "\n";
      llvm_unreachable("Tracing empty function");
    }

    auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Simulate);
    FunctionOpInterface NewF = putils->newFunc;

    // Initialize execution trace object
    putils->initTrace();

    // Process SampleOps to produce the trace
    SmallVector<Operation *, 4> toErase;
    for (auto &block : NewF.getFunctionBody()) {
      for (auto &op : block) {
        if (auto sampleOp = dyn_cast<enzyme::SampleOp>(op)) {
          OpBuilder b(sampleOp);
          putils->processSampleOp(sampleOp, b, symbolTable);
          toErase.push_back(sampleOp);
        }
      }
    }

    for (Operation *op : toErase) {
      op->erase();
    }

    for (auto &block : NewF.getFunctionBody()) {
      OpBuilder b(&block, block.end());
      auto term = block.getTerminator();

      auto retloc = block.getTerminator()->getLoc();
      if (auto retOp = dyn_cast<func::ReturnOp>(term)) {
        retOp->replaceAllUsesWith(
            b.create<func::ReturnOp>(retloc, putils->getTrace()));
        retOp->erase();
      }
    }

    if (!NewF)
      return failure();

    llvm::errs() << "Creating new function\n";
    NewF.dump();

    delete putils;

    OpBuilder b(CI);
    auto tCI = b.create<func::CallOp>(CI.getLoc(), NewF.getName(),
                                      NewF.getResultTypes(), CI.getInputs());
    // if (tCI.getNumResults() != CI.getNumResults()) {
    //   CI.emitError() << "Incorrect number of results for enzyme operation: "
    //                  << *CI << " expected " << *tCI;
    //   return failure();
    // }

    CI->replaceAllUsesWith(tCI);
    CI->erase();

    return success();
  }

  LogicalResult HandleTrace(SymbolTableCollection &symbolTable,
                            enzyme::TraceOp CI) {
    auto symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);
    (void)fn;

    FunctionOpInterface newFunc; // TODO
    if (!newFunc)
      return failure();

    llvm::errs() << "Creating new function\n";
    newFunc.dump();

    OpBuilder builder(CI);
    auto tCI =
        builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                     newFunc.getResultTypes(), CI.getInputs());
    if (tCI.getNumResults() != CI.getNumResults()) {
      CI.emitError() << "Incorrect number of results for enzyme operation: "
                     << *CI << " expected " << *tCI;
      return failure();
    }

    CI->replaceAllUsesWith(tCI);
    CI->erase();

    return success();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    op->walk([&](enzyme::SimulateOp sop) {
      auto res = HandleSimulate(symbolTable, sop);
      if (!res.succeeded()) {
        signalPassFailure();
        return;
      }
    });

    op->walk([&](enzyme::TraceOp top) {
      auto res = HandleTrace(symbolTable, top);
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
