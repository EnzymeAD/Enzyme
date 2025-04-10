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

  LogicalResult HandleSimulate(SymbolTableCollection &symbolTable,
                               enzyme::SimulateOp CI) {
    auto symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    if (fn.getFunctionBody().empty()) {
      llvm::errs() << fn << "\n";
      llvm_unreachable("Tracing empty function");
    }

    // Assume the same trace object base type as the traced function return
    // type.
    auto traceType = enzyme::TraceType::get(
        fn.getContext(),
        fn.getFunctionType().cast<mlir::FunctionType>().getResult(0));

    auto originalInputs =
        fn.getFunctionType().cast<mlir::FunctionType>().getInputs();
    SmallVector<mlir::Type, 4> ArgTypes(originalInputs.begin(),
                                        originalInputs.end());

    OpBuilder builder(fn.getContext());
    auto FTy = builder.getFunctionType(ArgTypes, {traceType});

    auto NewF = cast<FunctionOpInterface>(fn->cloneWithoutRegions());
    SymbolTable::setSymbolName(NewF, fn.getName().str() + ".simulate");
    NewF.setType(FTy);

    Operation *parent = fn->getParentWithTrait<OpTrait::SymbolTable>();
    SymbolTable table(parent);
    table.insert(NewF);

    IRMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;
    cloneInto(&fn.getFunctionBody(), &NewF.getFunctionBody(), originalToNew,
              originalToNewOps);

    DenseMap<Block *, Value> blockTrace;

    // Ensure the execution trace is passed through.
    for (auto &block : NewF.getFunctionBody()) {
      if (&block == &NewF.getFunctionBody().front()) {
        OpBuilder b(&block, block.begin());
        auto initOp = b.create<enzyme::InitOp>(block.getTerminator()->getLoc(),
                                               traceType);
        blockTrace[&block] = initOp.getResult();
      } else {
        block.insertArgument(block.args_begin(), traceType,
                             block.getTerminator()->getLoc());
        blockTrace[&block] = block.getArgument(0);
      }
    }

    // Process SampleOps to produce the trace
    SmallVector<Operation *, 4> toErase;
    for (auto &block : NewF.getFunctionBody()) {
      for (auto &op : block) {
        if (auto sampleOp = dyn_cast<enzyme::SampleOp>(op)) {
          OpBuilder b(sampleOp);
          auto distFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));
          auto logpdfFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getLogpdfAttr()));
          auto inputs = sampleOp.getInputs();
          auto nameAttr = sampleOp.getNameAttr();

          // 1. Insert distribution function call
          auto distCall =
              b.create<func::CallOp>(sampleOp.getLoc(), distFn.getName(),
                                     distFn.getResultTypes(), inputs);
          Value sampleVal = distCall.getResult(0);

          // 2. Insert logpdf function call
          SmallVector<Value, 4> logpdfInputs;
          logpdfInputs.push_back(sampleVal);
          for (auto input : inputs)
            logpdfInputs.push_back(input);
          auto logpdfCall =
              b.create<func::CallOp>(sampleOp.getLoc(), logpdfFn.getName(),
                                     logpdfFn.getResultTypes(), logpdfInputs);
          Value logpdfVal = logpdfCall.getResult(0);

          // 3. Record the computed sample and logpdf in the trace
          Value trace = blockTrace[&block];
          auto traceCall = b.create<enzyme::insertSampleToTraceOp>(
              sampleOp.getLoc(), traceType, trace, sampleVal, logpdfVal,
              nameAttr);
          blockTrace[&block] = traceCall.getResult();

          sampleOp.replaceAllUsesWith(distCall);
          toErase.push_back(sampleOp);
        }
      }
    }

    for (Operation *op : toErase) {
      op->erase();
    }

    // Update terminators to make sure the trace is propagated. Return only the
    // trace.
    for (auto &block : NewF.getFunctionBody()) {
      OpBuilder b(&block, block.end());
      auto term = block.getTerminator();

      auto retloc = block.getTerminator()->getLoc();
      if (auto brOp = dyn_cast<cf::BranchOp>(term)) {
        SmallVector<Value, 4> newOperands(term->getOperands().begin(),
                                          term->getOperands().end());
        newOperands.insert(newOperands.begin(), blockTrace[&block]);
        brOp->replaceAllUsesWith(
            b.create<cf::BranchOp>(retloc, brOp.getDest(), newOperands));
      } else if (auto retOp = dyn_cast<func::ReturnOp>(term)) {
        retOp->replaceAllUsesWith(
            b.create<func::ReturnOp>(retloc, blockTrace[&block]));
      } else {
        fn.emitError() << "Unsupported terminator found in traced function: "
                       << *term;
        return failure();
      }

      term->erase();
    }

    if (!NewF)
      return failure();

    llvm::errs() << "Creating new function\n";
    NewF.dump();

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
