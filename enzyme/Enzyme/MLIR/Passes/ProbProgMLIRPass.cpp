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
#include "Interfaces/GradientUtilsReverse.h"
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

  LogicalResult HandleTrace(SymbolTableCollection &symbolTable,
                            enzyme::TraceOp CI) {
    auto symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    bool freeMemory = true;
    bool omp = false;

    FunctionOpInterface newFunc = Logic.CreateTrace(
        fn, {}, {}, TA, {}, {}, freeMemory, 0, /*addedType*/ nullptr, type_args,
        {}, /*augmented*/ nullptr, omp, "");
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

    // if (isa<enzyme::SampleOp>(CI)) {
    //   auto symbolOp = symbolTable.lookupNearestSymbolFrom(CI,
    //   CI.getFnAttr()); auto fn = cast<FunctionOpInterface>(symbolOp);

    //   if (fn.getNumArguments() != CI.getNumOperands()) {
    //     CI.emitError() << "Incorrect number of arguments for enzyme
    //     operation: "
    //                    << *CI << " expected " << fn.getNumArguments()
    //                    << " but got " << CI.getNumOperands();
    //     return failure();
    //   }

    //   OpBuilder builder(CI);
    //   auto res = builder.create<func::CallOp>(
    //       CI.getLoc(), fn.getName(), fn.getResultTypes(), CI.getOperands());

    //   if (res.getNumResults() != CI.getNumResults()) {
    //     CI.emitError() << "Incorrect number of results for enzyme operation:
    //     "
    //                    << res.getNumResults() << " expected "
    //                    << CI.getNumResults();
    //     return failure();
    //   }

    //   // res->setAttr("name", CI.getNameAttr());

    //   CI.replaceAllUsesWith(res);
    //   CI->erase();
    // } else {
    //   CI.emitError() << "Unsupported ProbProg operation: " << *CI;
    //   return failure();
    // }
    return success();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
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
