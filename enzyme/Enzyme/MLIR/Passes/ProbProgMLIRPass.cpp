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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

  struct LowerUntracedCallPattern
      : public mlir::OpRewritePattern<enzyme::UntracedCallOp> {
    using mlir::OpRewritePattern<enzyme::UntracedCallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(enzyme::UntracedCallOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError("ProbProg: trying to call an empty function");
        return failure();
      }

      auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Call);
      FunctionOpInterface NewF = putils->newFunc;

      SmallVector<Operation *, 4> toErase;
      NewF.walk([&](enzyme::SampleOp sampleOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        auto distFn =
            cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                sampleOp, sampleOp.getFnAttr()));
        auto distCall = rewriter.create<func::CallOp>(
            sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
            sampleOp.getInputs());
        sampleOp.replaceAllUsesWith(distCall);

        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      rewriter.setInsertionPoint(CI);
      auto newCI = rewriter.create<func::CallOp>(
          CI.getLoc(), NewF.getName(), NewF.getResultTypes(), CI.getOperands());

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  struct LowerSimulatePattern
      : public mlir::OpRewritePattern<enzyme::SimulateOp> {
    using mlir::OpRewritePattern<enzyme::SimulateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(enzyme::SimulateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "ProbProg: calling `simulate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "logpdf attribute to avoid recursive `simulate` calls which is "
            "intended for generative functions");
        return failure();
      }

      auto putils =
          MProbProgUtils::CreateFromClone(fn, MProbProgMode::Simulate);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      auto tensorType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight = entryBuilder.create<arith::ConstantOp>(
          putils->initializationBlock->begin()->getLoc(), tensorType,
          DenseElementsAttr::get(tensorType, 0.0));
      Value weightAccumulator = zeroWeight;
      Value currTrace = putils->getTrace();

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues; // Values to replace uses of sample op
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: replace sample op uses with call to the
          // distribution function.
          auto distFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          auto distCall = rewriter.create<func::CallOp>(
              sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
              sampleOp.getInputs());

          sampledValues.append(distCall.getResults().begin(),
                               distCall.getResults().end());

          auto logpdfFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getLogpdfAttr()));

          // logpdf operands: (<non-RNG outputs>..., <non-RNG inputs>...)
          SmallVector<Value> logpdfOperands;
          for (unsigned i = 1; i < sampledValues.size(); ++i) {
            logpdfOperands.push_back(sampledValues[i]);
          }
          for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
            logpdfOperands.push_back(sampleOp.getOperand(i));
          }

          if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
            sampleOp.emitError("ProbProg: failed to construct logpdf call; "
                               "logpdf function has wrong number of arguments");
            return WalkResult::interrupt();
          }

          // A2. Compute and accumulate weight.
          auto logpdf = rewriter.create<func::CallOp>(
              sampleOp.getLoc(), logpdfFn.getName(), logpdfFn.getResultTypes(),
              logpdfOperands);
          weightAccumulator = rewriter.create<arith::AddFOp>(
              sampleOp.getLoc(), weightAccumulator, logpdf.getResult(0));
        } else {
          // B1. Generative functions: generate a simulate op that will itself
          // be lowered in a subsequent rewrite. No direct call to the
          // generative function should be emitted here.
          auto simulateOp = rewriter.create<enzyme::SimulateOp>(
              sampleOp.getLoc(),
              /*trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*weight*/ RankedTensorType::get({}, rewriter.getF64Type()),
              /*outputs*/ sampleOp.getResultTypes(),
              /*fn*/ sampleOp.getFnAttr(),
              /*inputs*/ sampleOp.getInputs(),
              /*name*/ sampleOp.getNameAttr());

          // The first two results of simulateOp are the subtrace and weight.
          // The remaining results correspond 1-to-1 with the original sample
          // op's results. We will replace uses of the sample op with these
          // values in the next step.
          for (unsigned i = 0; i < sampleOp.getNumResults(); ++i)
            sampledValues.push_back(simulateOp->getResult(i + 2));

          // B2. Add subtrace to trace.
          auto addSubtraceOp = rewriter.create<enzyme::AddSubtraceOp>(
              sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*subtrace*/ simulateOp->getResult(0),
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*trace*/ currTrace);
          currTrace = addSubtraceOp.getUpdatedTrace();

          // B3. Accumulate weight returned by simulateOp.
          weightAccumulator = rewriter.create<arith::AddFOp>(
              sampleOp.getLoc(), weightAccumulator, simulateOp->getResult(1));
        }

        // C. Add non-RNG sampled values to trace (common for both cases).
        SmallVector<Value> valuesToTrace;
        for (unsigned i = 1; i < sampledValues.size(); ++i) {
          valuesToTrace.push_back(sampledValues[i]);
        }

        if (!valuesToTrace.empty()) {
          auto addSampleToTraceOp = rewriter.create<enzyme::AddSampleToTraceOp>(
              sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*trace*/ currTrace,
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*sample*/ valuesToTrace);
          currTrace = addSampleToTraceOp.getUpdatedTrace();
        }

        // D. Replace uses of the original sample op with the new values.
        sampleOp.replaceAllUsesWith(sampledValues);

        toErase.push_back(sampleOp);
        return WalkResult::advance();
      });

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (result.wasInterrupted()) {
        CI.emitError("ProbProg: failed to walk sample ops");
        return failure();
      }

      // E. Before returning, record the aggregated weight and the function
      // return value(s) in the trace, then rewrite the return to return the
      // updated trace and aggregated weight.
      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        // E1. Add the accumulated weight to the trace.
        auto addWeightOp = rewriter.create<enzyme::AddWeightToTraceOp>(
            retOp.getLoc(),
            /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
            /*trace*/ currTrace, /*weight*/ weightAccumulator);
        currTrace = addWeightOp.getUpdatedTrace();

        // E2. Add non-RNG return values to the trace.
        SmallVector<Value> retvals;
        for (unsigned i = 1; i < retOp.getNumOperands(); ++i) {
          retvals.push_back(retOp.getOperand(i));
        }

        if (!retvals.empty()) {
          auto addRetvalOp = rewriter.create<enzyme::AddRetvalToTraceOp>(
              retOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
              /*trace*/ currTrace,
              /*retval*/ retvals);
          currTrace = addRetvalOp.getUpdatedTrace();
        }

        // E3. Construct new return values: (trace, weight, <original return
        // values>...)
        SmallVector<Value> newRetVals;
        newRetVals.push_back(currTrace);
        newRetVals.push_back(weightAccumulator);
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        rewriter.create<func::ReturnOp>(retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      auto newCI = rewriter.create<func::CallOp>(
          CI.getLoc(), NewF.getName(), NewF.getResultTypes(), CI.getInputs());

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  struct LowerGeneratePattern
      : public mlir::OpRewritePattern<enzyme::GenerateOp> {
    using mlir::OpRewritePattern<enzyme::GenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(enzyme::GenerateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "ProbProg: calling `generate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "logpdf attribute to avoid recursive `generate` calls which is "
            "intended for generative functions");
        return failure();
      }

      auto putils =
          MProbProgUtils::CreateFromClone(fn, MProbProgMode::Generate);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      auto tensorType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight = entryBuilder.create<arith::ConstantOp>(
          putils->initializationBlock->begin()->getLoc(), tensorType,
          DenseElementsAttr::get(tensorType, 0.0));
      Value weightAccumulator = zeroWeight;
      Value currTrace = putils->getTrace();
      Value constraint = NewF.getArgument(0);

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: replace sample op uses with call to the
          // distribution function.
          bool isConstrained = false;
          for (auto addr : CI.getConstrainedAddressesAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError(
                    "ProbProg: distribution function cannot have composite "
                    "constrained address");
                return WalkResult::interrupt();
              }
              isConstrained = true;
              break;
            }
          }

          if (isConstrained) {
            // Get sampled values from the constraint instead of sampling
            // from the distribution.
            sampledValues.resize(sampleOp.getNumResults());

            SmallVector<Type> constraintOutputTypes;
            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              constraintOutputTypes.push_back(sampleOp.getResult(i).getType());
            }

            auto gsfcOp = rewriter.create<enzyme::GetSampleFromConstraintOp>(
                sampleOp.getLoc(), constraintOutputTypes, constraint,
                sampleOp.getSymbolAttr());

            // Pass RNG state from input to output.
            sampledValues[0] = sampleOp.getOperand(0);
            for (unsigned i = 0; i < gsfcOp->getNumResults(); ++i) {
              sampledValues[i + 1] = gsfcOp->getResult(i);
            }

            // Compute weight via logpdf using constrained values.
            auto logpdfFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getLogpdfAttr()));

            // logpdf operands: (<non-RNG outputs>..., <non-RNG inputs>...)
            SmallVector<Value> logpdfOperands;
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              logpdfOperands.push_back(sampledValues[i]);
            }
            for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
              logpdfOperands.push_back(sampleOp.getOperand(i));
            }

            if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
              sampleOp.emitError(
                  "ProbProg: failed to construct logpdf call for constrained "
                  "sample; logpdf function has wrong number of arguments");
              return WalkResult::interrupt();
            }

            auto logpdf = rewriter.create<func::CallOp>(
                sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator = rewriter.create<arith::AddFOp>(
                sampleOp.getLoc(), weightAccumulator, logpdf.getResult(0));
          } else {
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));

            auto distCall = rewriter.create<func::CallOp>(
                sampleOp.getLoc(), distFn.getName(), distFn.getResultTypes(),
                sampleOp.getInputs());

            sampledValues.append(distCall.getResults().begin(),
                                 distCall.getResults().end());

            auto logpdfFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getLogpdfAttr()));

            // logpdf operands: (<non-RNG outputs>..., <non-RNG inputs>...)
            SmallVector<Value> logpdfOperands;
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              logpdfOperands.push_back(sampledValues[i]);
            }
            for (unsigned i = 1; i < sampleOp.getNumOperands(); ++i) {
              logpdfOperands.push_back(sampleOp.getOperand(i));
            }

            if (logpdfOperands.size() != logpdfFn.getNumArguments()) {
              sampleOp.emitError(
                  "ProbProg: failed to construct logpdf call; "
                  "logpdf function has wrong number of arguments");
              return WalkResult::interrupt();
            }

            // A2. Compute and accumulate weight.
            auto logpdf = rewriter.create<func::CallOp>(
                sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator = rewriter.create<arith::AddFOp>(
                sampleOp.getLoc(), weightAccumulator, logpdf.getResult(0));
          }
        } else {
          // B1. Generative functions: generate a recursive op (simulate when
          // unconstrained, generate when constrained) that will itself be
          // lowered in a subsequent rewrite.
          SmallVector<Attribute> subContrainedAddresses;
          bool isConstrained = false;
          for (auto addr : CI.getConstrainedAddressesAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              isConstrained = true;
              if (address.size() == 1) {
                sampleOp.emitError(
                    "ProbProg: generative function cannot be constrained with "
                    "singleton address - composite addresses are required to "
                    "ensure proper weight calculation");
                return WalkResult::interrupt();
              }

              SmallVector<Attribute> tailAddresses;
              for (size_t i = 1; i < address.size(); ++i) {
                tailAddresses.push_back(address[i]);
              }
              subContrainedAddresses.push_back(
                  rewriter.getArrayAttr(tailAddresses));
              break;
            }
          }

          // B2. Generate a recursive op (simulate when unconstrained, generate
          // when constrained).
          Operation *recursiveOp = nullptr;

          if (isConstrained) {
            auto getSubconstraintOp =
                rewriter.create<enzyme::GetSubconstraintOp>(
                    sampleOp.getLoc(),
                    /*subconstraint*/
                    enzyme::ConstraintType::get(sampleOp.getContext()),
                    /*constraint*/ constraint,
                    /*symbol*/ sampleOp.getSymbolAttr());
            Value subConstraint = getSubconstraintOp.getSubconstraint();

            recursiveOp = rewriter.create<enzyme::GenerateOp>(
                sampleOp.getLoc(),
                /*trace*/ enzyme::TraceType::get(sampleOp.getContext()),
                /*weight*/ RankedTensorType::get({}, rewriter.getF64Type()),
                /*outputs*/ sampleOp.getResultTypes(),
                /*fn*/ sampleOp.getFnAttr(),
                /*inputs*/ sampleOp.getInputs(),
                /*constrained_addresses*/
                rewriter.getArrayAttr(subContrainedAddresses),
                /*constraint*/
                subConstraint,
                /*name*/ sampleOp.getNameAttr());
          } else {
            recursiveOp = rewriter.create<enzyme::SimulateOp>(
                sampleOp.getLoc(),
                /*trace*/ enzyme::TraceType::get(sampleOp.getContext()),
                /*weight*/ RankedTensorType::get({}, rewriter.getF64Type()),
                /*outputs*/ sampleOp.getResultTypes(),
                /*fn*/ sampleOp.getFnAttr(),
                /*inputs*/ sampleOp.getInputs(),
                /*name*/ sampleOp.getNameAttr());
          }

          // The first two results of the recursive op are the subtrace and
          // weight. The remaining results correspond 1-to-1 with the original
          // sample op's results.
          for (unsigned i = 0; i < sampleOp.getNumResults(); ++i)
            sampledValues.push_back(recursiveOp->getResult(i + 2));

          // B2. Add subtrace to trace.
          auto addSubtraceOp = rewriter.create<enzyme::AddSubtraceOp>(
              sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*subtrace*/ recursiveOp->getResult(0),
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*trace*/ currTrace);
          currTrace = addSubtraceOp.getUpdatedTrace();

          // B3. Accumulate weight returned by recursive op.
          weightAccumulator = rewriter.create<arith::AddFOp>(
              sampleOp.getLoc(), weightAccumulator, recursiveOp->getResult(1));
        }

        // C. Add non-RNG sampled values to trace (common for both cases).
        SmallVector<Value> valuesToTrace;
        for (unsigned i = 1; i < sampledValues.size(); ++i) {
          valuesToTrace.push_back(sampledValues[i]);
        }

        if (!valuesToTrace.empty()) {
          auto addSampleToTraceOp = rewriter.create<enzyme::AddSampleToTraceOp>(
              sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*trace*/ currTrace,
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*sample*/ valuesToTrace);
          currTrace = addSampleToTraceOp.getUpdatedTrace();
        }

        // D. Replace uses of the original sample op with the new values.
        sampleOp.replaceAllUsesWith(sampledValues);

        toErase.push_back(sampleOp);
        return WalkResult::advance();
      });

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      if (result.wasInterrupted()) {
        CI.emitError("ProbProg: failed to walk sample ops");
        return failure();
      }

      // E. Before returning, record the aggregated weight and the function
      // return value(s) in the trace, then rewrite the return to return the
      // updated trace and aggregated weight.
      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        // E1. Add the accumulated weight to the trace.
        auto addWeightOp = rewriter.create<enzyme::AddWeightToTraceOp>(
            retOp.getLoc(),
            /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
            /*trace*/ currTrace, /*weight*/ weightAccumulator);
        currTrace = addWeightOp.getUpdatedTrace();

        // E2. Add non-RNG return values to the trace.
        SmallVector<Value> retvals;
        for (unsigned i = 1; i < retOp.getNumOperands(); ++i) {
          retvals.push_back(retOp.getOperand(i));
        }

        if (!retvals.empty()) {
          auto addRetvalOp = rewriter.create<enzyme::AddRetvalToTraceOp>(
              retOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
              /*trace*/ currTrace,
              /*retval*/ retvals);
          currTrace = addRetvalOp.getUpdatedTrace();
        }

        // E3. Construct new return values: (trace, weight, <original return
        // values>...)
        SmallVector<Value> newRetVals;
        newRetVals.push_back(currTrace);
        newRetVals.push_back(weightAccumulator);
        newRetVals.append(retOp.getOperands().begin(),
                          retOp.getOperands().end());

        rewriter.create<func::ReturnOp>(retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getConstraint());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = rewriter.create<func::CallOp>(
          CI.getLoc(), NewF.getName(), NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
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
  // Old direct lowering disabled; pattern-based lowering is now used.

  RewritePatternSet patterns(&getContext());
  patterns.add<LowerUntracedCallPattern, LowerSimulatePattern,
               LowerGeneratePattern>(&getContext());

  mlir::GreedyRewriteConfig config;

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                          config))) {
    signalPassFailure();
    return;
  }

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
