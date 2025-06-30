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

    OpBuilder entryBuilder(putils->initializationBlock,
                           putils->initializationBlock->begin());
    auto tensorType = RankedTensorType::get({}, entryBuilder.getF64Type());
    auto zeroWeight = entryBuilder.create<arith::ConstantOp>(
        putils->initializationBlock->begin()->getLoc(), tensorType,
        DenseElementsAttr::get(tensorType, 0.0));
    Value weightAccumulator = zeroWeight;

    {
      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](enzyme::SampleOp sampleOp) {
        OpBuilder b(sampleOp);

        // 1. Generate sampled function call and replace uses.
        auto fn = cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
            sampleOp, sampleOp.getFnAttr()));
        auto fnCall =
            b.create<func::CallOp>(sampleOp.getLoc(), fn.getName(),
                                   fn.getResultTypes(), sampleOp.getInputs());
        sampleOp.replaceAllUsesWith(fnCall);

        // 2. Add sampled values to trace.
        if (auto tracedOutputIndices = sampleOp.getTracedOutputIndicesAttr()) {
          SmallVector<Value> tracedOutputs;
          for (auto idx : tracedOutputIndices.asArrayRef()) {
            tracedOutputs.push_back(fnCall.getResult(idx));
          }
          b.create<enzyme::AddSampleToTraceOp>(
              sampleOp.getLoc(),
              /*trace*/ putils->getTrace(),
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*sample*/ tracedOutputs);
        }

        // 3. If there is a logpdf attribute, consider `fn` a distribution
        // function. Call logpdf on the sampled values and accumulate the
        // weight. If there is no logpdf attribute, consider `fn` a generative
        // function. Generate a simulate op to produce a subtrace and accumulate
        // the returned weight.
        if (sampleOp.getLogpdfAttr()) {
          auto logpdfFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getLogpdfAttr()));
          SmallVector<Value> logpdfOperands;
          if (auto tracedOutputIndices =
                  sampleOp.getTracedOutputIndicesAttr()) {
            for (auto idx : tracedOutputIndices.asArrayRef()) {
              logpdfOperands.push_back(fnCall.getResult(idx));
            }
          }
          if (auto tracedInputIndices = sampleOp.getTracedInputIndicesAttr()) {
            for (auto idx : tracedInputIndices.asArrayRef()) {
              logpdfOperands.push_back(fnCall.getOperand(idx));
            }
          }
          assert(logpdfOperands.size() == logpdfFn.getNumArguments());
          auto logpdf =
              b.create<func::CallOp>(sampleOp.getLoc(), logpdfFn.getName(),
                                     logpdfFn.getResultTypes(), logpdfOperands);

          weightAccumulator = b.create<arith::AddFOp>(
              sampleOp.getLoc(), weightAccumulator, logpdf.getResult(0));
        } else {
          auto simulateOp = b.create<enzyme::SimulateOp>(
              sampleOp.getLoc(),
              /*trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*weight*/ RankedTensorType::get({}, b.getF64Type()),
              /*outputs*/ sampleOp.getResultTypes(),
              /*fn*/ sampleOp.getFnAttr(),
              /*inputs*/ sampleOp.getInputs(),
              /*name*/ sampleOp.getNameAttr());
          b.create<enzyme::AddSubtraceOp>(sampleOp.getLoc(),
                                          /*subtrace*/ simulateOp->getResult(0),
                                          /*trace*/ putils->getTrace());
          weightAccumulator = b.create<arith::AddFOp>(
              sampleOp.getLoc(), weightAccumulator, simulateOp->getResult(1));
        }

        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase) {
        op->erase();
      }
    }

    // Return the trace and weight as the first and second results.
    NewF.walk([&](func::ReturnOp retOp) {
      OpBuilder b(retOp);
      SmallVector<Value> newRetVals;
      newRetVals.push_back(putils->getTrace());
      newRetVals.push_back(weightAccumulator);
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

  getOperation()->dump();
}
