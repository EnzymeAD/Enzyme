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

    OpBuilder entryBuilder(&NewF.getFunctionBody().front().front());
    auto loc = NewF.getLoc();
    auto tensorType = RankedTensorType::get({}, entryBuilder.getF64Type());
    auto zeroAttr = DenseElementsAttr::get(
        tensorType, APFloat(entryBuilder.getF64Type().getFloatSemantics(), 0));
    auto zeroWeight =
        entryBuilder.create<arith::ConstantOp>(loc, tensorType, zeroAttr);
    Value weightAccumulator = zeroWeight;

    {
      auto constraintsAttr = CI.getConstraintsAttr();
      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](enzyme::SampleOp sampleOp) {
        assert(sampleOp.getSymbolAttr() && "SampleOp requires symbol");
        assert(sampleOp.getLogpdfAttr() &&
               "GenerateOp requires logpdf in SampleOps");

        OpBuilder b(sampleOp);
        auto distFn =
            cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                sampleOp, sampleOp.getFnAttr()));

        SmallVector<Value> finalValues;

        auto tracedOutputIndices = sampleOp.getTracedOutputIndicesAttr();
        auto symbolAttr = sampleOp.getSymbolAttr();
        bool hasConstraint = false;
        SmallVector<Attribute> opConstraintValues;

        if (constraintsAttr && symbolAttr && tracedOutputIndices &&
            !constraintsAttr.empty() && !tracedOutputIndices.empty()) {
          uint64_t symPtr = symbolAttr.getValue().getZExtValue();

          for (auto constraint : constraintsAttr) {
            auto c = cast<ConstraintAttr>(constraint);
            if (c.getSymbol() != symPtr)
              continue;

            assert(c.getValues().size() == tracedOutputIndices.size() &&
                   "Constraint entry value count must match number of traced "
                   "output indices");
            for (auto v : c.getValues())
              opConstraintValues.push_back(v);

            hasConstraint = true;
            break;
          }
        }

        if (!hasConstraint) {
          auto distCall = b.create<func::CallOp>(
              sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
              sampleOp.getInputs());

          finalValues.append(distCall.getResults().begin(),
                             distCall.getResults().end());
        } else {
          // Constraint for current SampleOp is found. Replace traced outputs
          // with constraint values and try to forward other results from
          // SampleOp inputs when possible (e.g., RNG state).
          unsigned numResults = sampleOp->getNumResults();
          finalValues.resize(numResults, Value());

          // Map constraint values to traced output indices in order.
          if (tracedOutputIndices) {
            unsigned i = 0;
            for (auto outIdx : tracedOutputIndices.asArrayRef()) {
              auto elemAttr = cast<ElementsAttr>(opConstraintValues[i]);
              auto constOp = b.create<arith::ConstantOp>(
                  sampleOp.getLoc(), elemAttr.getType(), elemAttr);
              finalValues[outIdx] = constOp.getResult();
              i++;
            }
          }

          // Try forwarding an operand with identical type (e.g., RNG state).
          for (unsigned i = 0; i < numResults; ++i) {
            if (finalValues[i])
              continue;

            for (Value inVal : sampleOp.getInputs()) {
              if (inVal.getType() == sampleOp.getResult(i).getType()) {
                finalValues[i] = inVal;
                break;
              }
            }

            assert(finalValues[i] && "Cannot retrieve or forward result");
          }
        }

        // For traced outputs: weight using logpdf and accumulate and add to
        // trace.
        if (tracedOutputIndices && !tracedOutputIndices.empty()) {
          auto logpdfFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getLogpdfAttr()));

          // 1. Construct args for logpdf call: sample + specified distFn args
          SmallVector<Value> logpdfOperands;
          // NOTE: This assumes that the traced output indices come in the same
          // order as the first few logpdf args.
          for (auto idx : tracedOutputIndices.asArrayRef()) {
            logpdfOperands.push_back(finalValues[idx]);
          }
          auto tracedInputIndices = sampleOp.getTracedInputIndicesAttr();
          if (tracedInputIndices) {
            for (auto idx : tracedInputIndices.asArrayRef()) {
              logpdfOperands.push_back(sampleOp.getOperand(idx));
            }
          }
          assert(logpdfOperands.size() == logpdfFn.getNumArguments() &&
                 "Logpdf call arguments must match number of logpdf function "
                 "arguments; double check traced input indices attribute");
          auto logpdfCall =
              b.create<func::CallOp>(sampleOp.getLoc(), logpdfFn.getName(),
                                     logpdfFn.getResultTypes(), logpdfOperands);

          // 2. Accumulate weights
          weightAccumulator = b.create<arith::AddFOp>(
              sampleOp.getLoc(), weightAccumulator, logpdfCall.getResult(0));

          // 3. Add traced outputs to trace
          for (auto idx : tracedOutputIndices.asArrayRef()) {
            b.create<enzyme::AddSampleToTraceOp>(
                sampleOp.getLoc(), finalValues[idx], sampleOp.getSymbolAttr(),
                CI.getTraceAttr(), sampleOp.getNameAttr());
          }
        }

        sampleOp.replaceAllUsesWith(finalValues);
        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase) {
        op->erase();
      }
    }

    // Return the weight (0th) and original results
    NewF.walk([&](func::ReturnOp returnOp) {
      OpBuilder b(returnOp);
      SmallVector<Value> newReturnValues;
      newReturnValues.push_back(weightAccumulator);
      newReturnValues.append(returnOp.getOperands().begin(),
                             returnOp.getOperands().end());

      auto fnType = cast<FunctionType>(NewF.getFunctionType());
      SmallVector<Type> newResultTypes;
      newResultTypes.push_back(RankedTensorType::get({}, b.getF64Type()));
      newResultTypes.append(fnType.getResults().begin(),
                            fnType.getResults().end());
      auto newFnType = b.getFunctionType(fnType.getInputs(), newResultTypes);
      NewF.setType(newFnType);

      b.create<func::ReturnOp>(returnOp.getLoc(), newReturnValues);
      returnOp.erase();
    });

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
                sampleOp.getLoc(), distCall.getResult(idx),
                sampleOp.getSymbolAttr(), CI.getTraceAttr(),
                sampleOp.getNameAttr());
          }
        }

        sampleOp.replaceAllUsesWith(distCall);
        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase) {
        op->erase();
      }
    }

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
    op->walk([&](enzyme::GenerateOp cop) {
      auto res = HandleGenerate(symbolTable, cop);
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
}
