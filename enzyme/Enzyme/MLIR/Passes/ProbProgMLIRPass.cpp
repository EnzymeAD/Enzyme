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
#include "Interfaces/HMCUtils.h"
#include "Interfaces/ProbProgUtils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "probprog"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;
using namespace enzyme::MCMC;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PROBPROGPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

static bool computePositionSizeForAddress(Operation *op,
                                          FunctionOpInterface func,
                                          ArrayRef<Attribute> address,
                                          SymbolTableCollection &symbolTable,
                                          int64_t &positionSize) {
  if (address.empty())
    return false;

  auto targetSymbol = address[0];
  bool found = false;

  func.walk([&](enzyme::SampleOp sampleOp) {
    if (found)
      return WalkResult::interrupt();

    auto sampleSymbol = sampleOp.getSymbolAttr();
    if (!sampleSymbol || sampleSymbol != targetSymbol)
      return WalkResult::advance();

    found = true;

    if (address.size() > 1) {
      if (sampleOp.getLogpdfAttr()) {
        op->emitError("Cannot select nested address in distribution function");
        return WalkResult::interrupt();
      }

      auto genFn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
      if (!genFn || genFn.getFunctionBody().empty()) {
        op->emitError("Cannot find generative function for nested address");
        return WalkResult::interrupt();
      }

      if (!computePositionSizeForAddress(op, genFn, address.drop_front(),
                                         symbolTable, positionSize))
        return WalkResult::interrupt();

      return WalkResult::interrupt();
    }

    for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
      auto resultType = sampleOp.getResult(i).getType();
      if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
        int64_t elemCount = 1;
        for (auto dim : tensorType.getShape()) {
          if (dim == ShapedType::kDynamic) {
            op->emitError("Dynamic tensor dimensions not supported");
            return WalkResult::interrupt();
          }
          elemCount *= dim;
        }
        positionSize += elemCount;
      } else {
        op->emitError("Expected ranked tensor type for sample result");
        return WalkResult::interrupt();
      }
    }

    return WalkResult::interrupt();
  });

  return found;
}

static int64_t
computePositionSizeForSelection(Operation *op, FunctionOpInterface fn,
                                ArrayAttr selection,
                                SymbolTableCollection &symbolTable) {
  int64_t positionSize = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty()) {
      op->emitError("Empty address in selection");
      return -1;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, fn, tailAddresses, symbolTable,
                                       positionSize)) {
      op->emitError("Could not find sample with symbol in address chain");
      return -1;
    }
  }

  return positionSize;
}

struct ProbProgPass : public enzyme::impl::ProbProgPassBase<ProbProgPass> {
  using ProbProgPassBase::ProbProgPassBase;

  MEnzymeLogic Logic;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::OpPassManager pm;
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (!mlir::failed(result)) {
      pm.getDependentDialects(registry);
    }

    registry.insert<mlir::arith::ArithDialect, mlir::math::MathDialect,
                    mlir::complex::ComplexDialect, mlir::cf::ControlFlowDialect,
                    mlir::tensor::TensorDialect, mlir::enzyme::EnzymeDialect>();
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
        auto distCall =
            func::CallOp::create(rewriter, sampleOp.getLoc(), distFn.getName(),
                                 distFn.getResultTypes(), sampleOp.getInputs());
        sampleOp.replaceAllUsesWith(distCall);

        toErase.push_back(sampleOp);
      });

      for (Operation *op : toErase)
        rewriter.eraseOp(op);

      rewriter.setInsertionPoint(CI);
      auto newCI =
          func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                               NewF.getResultTypes(), CI.getOperands());

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
      auto zeroWeight = arith::ConstantOp::create(
          entryBuilder, putils->initializationBlock->begin()->getLoc(),
          tensorType, DenseElementsAttr::get(tensorType, 0.0));
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

          auto distCall = func::CallOp::create(
              rewriter, sampleOp.getLoc(), distFn.getName(),
              distFn.getResultTypes(), sampleOp.getInputs());

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
          auto logpdf = func::CallOp::create(
              rewriter, sampleOp.getLoc(), logpdfFn.getName(),
              logpdfFn.getResultTypes(), logpdfOperands);
          weightAccumulator =
              arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                    weightAccumulator, logpdf.getResult(0));
        } else {
          // B1. Generative functions: generate a simulate op that will itself
          // be lowered in a subsequent rewrite. No direct call to the
          // generative function should be emitted here.
          auto simulateOp = enzyme::SimulateOp::create(
              rewriter, sampleOp.getLoc(),
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
          auto addSubtraceOp = enzyme::AddSubtraceOp::create(
              rewriter, sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*subtrace*/ simulateOp->getResult(0),
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*trace*/ currTrace);
          currTrace = addSubtraceOp.getUpdatedTrace();

          // B3. Accumulate weight returned by simulateOp.
          weightAccumulator = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                    weightAccumulator,
                                                    simulateOp->getResult(1));
        }

        // C. Add non-RNG sampled values to trace (common for both cases).
        SmallVector<Value> valuesToTrace;
        for (unsigned i = 1; i < sampledValues.size(); ++i) {
          valuesToTrace.push_back(sampledValues[i]);
        }

        if (!valuesToTrace.empty()) {
          auto addSampleToTraceOp = enzyme::AddSampleToTraceOp::create(
              rewriter, sampleOp.getLoc(),
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
        auto addWeightOp = enzyme::AddWeightToTraceOp::create(
            rewriter, retOp.getLoc(),
            /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
            /*trace*/ currTrace, /*weight*/ weightAccumulator);
        currTrace = addWeightOp.getUpdatedTrace();

        // E2. Add non-RNG return values to the trace.
        SmallVector<Value> retvals;
        for (unsigned i = 1; i < retOp.getNumOperands(); ++i) {
          retvals.push_back(retOp.getOperand(i));
        }

        if (!retvals.empty()) {
          auto addRetvalOp = enzyme::AddRetvalToTraceOp::create(
              rewriter, retOp.getLoc(),
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

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), CI.getInputs());

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  struct LowerMCMCPattern : public mlir::OpRewritePattern<enzyme::MCMCOp> {
    bool debugDump;

    LowerMCMCPattern(MLIRContext *context, bool debugDump,
                     PatternBenefit benefit = 1)
        : OpRewritePattern(context, benefit), debugDump(debugDump) {}

    LogicalResult matchAndRewrite(enzyme::MCMCOp mcmcOp,
                                  PatternRewriter &rewriter) const override {
      if (mcmcOp.getHmcConfig().has_value()) {
        return lowerHMC(mcmcOp, rewriter);
      } else if (mcmcOp.getNutsConfig().has_value()) {
        return lowerNUTS(mcmcOp, rewriter);
      } else {
        mcmcOp.emitError("ProbProg: Unknown MCMC algorithm");
        return failure();
      }
    }

  private:
    LogicalResult lowerHMC(enzyme::MCMCOp mcmcOp,
                           PatternRewriter &rewriter) const {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        mcmcOp.emitError(
            "ProbProg: calling `mcmc` with HMC on an empty function");
        return failure();
      }

      if (!mcmcOp.getStepSize()) {
        mcmcOp.emitError("ProbProg: HMC requires step_size parameter");
        return failure();
      }

      auto loc = mcmcOp.getLoc();
      auto invMass = mcmcOp.getInverseMassMatrix();
      auto stepSize = mcmcOp.getStepSize();

      auto hmcConfig = mcmcOp.getHmcConfig().value();
      int64_t numLeapfrogSteps = hmcConfig.getNumSteps();

      auto inputs = mcmcOp.getInputs();
      if (inputs.empty()) {
        mcmcOp.emitError("ProbProg: HMC requires at least rng_state input");
        return failure();
      }

      auto rngInput = inputs[0];
      SmallVector<Value> fnInputs(inputs.begin() + 1, inputs.end());

      auto originalTrace = mcmcOp.getOriginalTrace();
      auto selection = mcmcOp.getSelectionAttr();

      int64_t positionSize =
          computePositionSizeForSelection(mcmcOp, fn, selection, symbolTable);
      if (positionSize <= 0)
        return failure();

      // 1. Setup context
      HMCContext ctx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                     invMass, stepSize, positionSize);

      // 2. Initialize HMC state from trace
      auto initState = InitHMC(rewriter, loc, rngInput, ctx, debugDump);

      // 3. Single kernel sample step
      auto sample =
          SampleHMC(rewriter, loc, initState.q0, initState.grad0, initState.U0,
                    initState.rng, numLeapfrogSteps, ctx, debugDump);

      // 4. PostProcess
      auto result = PostProcessHMC(rewriter, loc, sample.q, sample.accepted,
                                   sample.rng, ctx);

      rewriter.replaceOp(mcmcOp, {result.trace, result.accepted, result.rng});

      return success();
    }

    LogicalResult lowerNUTS(enzyme::MCMCOp mcmcOp,
                            PatternRewriter &rewriter) const {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        mcmcOp.emitError(
            "ProbProg: calling `mcmc` with NUTS on an empty function");
        return failure();
      }

      if (!mcmcOp.getStepSize()) {
        mcmcOp.emitError("ProbProg: NUTS requires step_size parameter");
        return failure();
      }

      auto loc = mcmcOp.getLoc();
      auto invMass = mcmcOp.getInverseMassMatrix();
      auto stepSize = mcmcOp.getStepSize();

      auto inputs = mcmcOp.getInputs();
      if (inputs.empty()) {
        mcmcOp.emitError("ProbProg: NUTS requires at least rng_state input");
        return failure();
      }

      auto rngInput = inputs[0];
      SmallVector<Value> fnInputs(inputs.begin() + 1, inputs.end());

      auto originalTrace = mcmcOp.getOriginalTrace();
      auto selection = mcmcOp.getSelectionAttr();

      int64_t positionSize =
          computePositionSizeForSelection(mcmcOp, fn, selection, symbolTable);
      if (positionSize <= 0)
        return failure();

      // 1. Setup base context for InitHMC
      HMCContext baseCtx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                         invMass, stepSize, positionSize);

      // 2. Initialize HMC state from trace
      auto initState = InitHMC(rewriter, loc, rngInput, baseCtx, debugDump);

      // 3. Get NUTS-specific configuration
      auto nutsConfig = mcmcOp.getNutsConfig().value();
      int64_t maxTreeDepthVal = nutsConfig.getMaxTreeDepth();
      double maxDeltaEnergyVal =
          nutsConfig.getMaxDeltaEnergy()
              ? nutsConfig.getMaxDeltaEnergy().getValueAsDouble()
              : 1000.0;

      auto F64TensorType = RankedTensorType::get({}, rewriter.getF64Type());
      auto maxDeltaEnergy = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType,
                                 rewriter.getF64FloatAttr(maxDeltaEnergyVal)));

      // Create NUTSContext (energyCurrent will be recomputed by SampleNUTS)
      NUTSContext ctx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                      invMass, stepSize, positionSize, initState.U0,
                      maxDeltaEnergy, maxTreeDepthVal);

      // 4. Single kernel sample step
      auto sample = SampleNUTS(rewriter, loc, initState.q0, initState.grad0,
                               initState.U0, initState.rng, ctx, debugDump);

      // 5. PostProcess
      auto result = PostProcessNUTS(rewriter, loc, sample.q, sample.rng, ctx);

      rewriter.replaceOp(mcmcOp, {result.trace, result.accepted, result.rng});

      return success();
    }
  };

  struct LowerMHPattern : public mlir::OpRewritePattern<enzyme::MHOp> {
    using mlir::OpRewritePattern<enzyme::MHOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(enzyme::MHOp mhOp,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mhOp, mhOp.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        mhOp.emitError(
            "ProbProg: calling `mh` on an empty function; if this is a "
            "distribution function, its sample op should have a logpdf "
            "attribute to avoid recursive `mh` calls which is intended for "
            "generative functions");
        return failure();
      }

      auto tensorType = RankedTensorType::get({}, rewriter.getF64Type());
      auto traceType = enzyme::TraceType::get(mhOp.getContext());
      auto rngStateType = mhOp.getInputs()[0].getType();

      // 1. Create regenerate op with the same function and selection
      // `enzyme.regenerate` returns the same results as `enzyme.generate`
      // (trace, weight, outputs...<rng_state, original_outputs...>), but
      // takes addresses to regenerate instead of to specify constraints.
      auto regenerateOp = enzyme::RegenerateOp::create(
          rewriter, mhOp.getLoc(),
          /*trace*/ traceType,
          /*weight*/ tensorType,
          /*output_rng_state*/ rngStateType,
          /*fn*/ mhOp.getFnAttr(),
          /*inputs*/ mhOp.getInputs(),
          /*original_trace*/ mhOp.getOriginalTrace(),
          /*selection*/ mhOp.getSelectionAttr(),
          /*name*/ mhOp.getNameAttr());

      // 2. Metropolis-Hastings accept/reject step
      auto getOriginalWeightOp = enzyme::GetWeightFromTraceOp::create(
          rewriter, mhOp.getLoc(), tensorType, mhOp.getOriginalTrace());
      auto logAlpha = arith::SubFOp::create(rewriter, mhOp.getLoc(),
                                            regenerateOp.getWeight(),
                                            getOriginalWeightOp.getWeight());

      auto zeroConst =
          arith::ConstantOp::create(rewriter, mhOp.getLoc(), tensorType,
                                    DenseElementsAttr::get(tensorType, 0.0));
      auto oneConst =
          arith::ConstantOp::create(rewriter, mhOp.getLoc(), tensorType,
                                    DenseElementsAttr::get(tensorType, 1.0));

      auto randomOp = enzyme::RandomOp::create(
          rewriter, mhOp.getLoc(), TypeRange{rngStateType, tensorType},
          regenerateOp.getOutputRngState(), zeroConst, oneConst,
          enzyme::RngDistributionAttr::get(rewriter.getContext(),
                                           enzyme::RngDistribution::UNIFORM));
      auto logRand =
          math::LogOp::create(rewriter, mhOp.getLoc(), randomOp.getResult());

      // 3. Check if proposal is accepted: log(rand()) < log_alpha
      auto accepted =
          arith::CmpFOp::create(rewriter, mhOp.getLoc(),
                                arith::CmpFPredicate::OLT, logRand, logAlpha);

      // 4. Select between new and original trace based on acceptance
      auto selectedTrace = enzyme::SelectTraceOp::create(
          rewriter, mhOp.getLoc(), traceType, accepted, regenerateOp.getTrace(),
          mhOp.getOriginalTrace());

      rewriter.replaceOp(
          mhOp, {selectedTrace, accepted, randomOp.getOutputRngState()});
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
      auto zeroWeight = arith::ConstantOp::create(
          entryBuilder, putils->initializationBlock->begin()->getLoc(),
          tensorType, DenseElementsAttr::get(tensorType, 0.0));
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

            auto gsfcOp = enzyme::GetSampleFromConstraintOp::create(
                rewriter, sampleOp.getLoc(), constraintOutputTypes, constraint,
                sampleOp.getSymbolAttr());

            // Pass along the RNG state.
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

            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
          } else {
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));

            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());

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
            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
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
            auto getSubconstraintOp = enzyme::GetSubconstraintOp::create(
                rewriter, sampleOp.getLoc(),
                /*subconstraint*/
                enzyme::ConstraintType::get(sampleOp.getContext()),
                /*constraint*/ constraint,
                /*symbol*/ sampleOp.getSymbolAttr());
            Value subConstraint = getSubconstraintOp.getSubconstraint();

            recursiveOp = enzyme::GenerateOp::create(
                rewriter, sampleOp.getLoc(),
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
            recursiveOp = enzyme::SimulateOp::create(
                rewriter, sampleOp.getLoc(),
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
          auto addSubtraceOp = enzyme::AddSubtraceOp::create(
              rewriter, sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*subtrace*/ recursiveOp->getResult(0),
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*trace*/ currTrace);
          currTrace = addSubtraceOp.getUpdatedTrace();

          // B3. Accumulate weight returned by recursive op.
          weightAccumulator = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                    weightAccumulator,
                                                    recursiveOp->getResult(1));
        }

        // C. Add non-RNG sampled values to trace (common for both cases).
        SmallVector<Value> valuesToTrace;
        for (unsigned i = 1; i < sampledValues.size(); ++i) {
          valuesToTrace.push_back(sampledValues[i]);
        }

        if (!valuesToTrace.empty()) {
          auto addSampleToTraceOp = enzyme::AddSampleToTraceOp::create(
              rewriter, sampleOp.getLoc(),
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
        auto addWeightOp = enzyme::AddWeightToTraceOp::create(
            rewriter, retOp.getLoc(),
            /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
            /*trace*/ currTrace, /*weight*/ weightAccumulator);
        currTrace = addWeightOp.getUpdatedTrace();

        // E2. Add non-RNG return values to the trace.
        SmallVector<Value> retvals;
        for (unsigned i = 1; i < retOp.getNumOperands(); ++i) {
          retvals.push_back(retOp.getOperand(i));
        }

        if (!retvals.empty()) {
          auto addRetvalOp = enzyme::AddRetvalToTraceOp::create(
              rewriter, retOp.getLoc(),
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

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getConstraint());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  struct LowerRegeneratePattern
      : public mlir::OpRewritePattern<enzyme::RegenerateOp> {
    using mlir::OpRewritePattern<enzyme::RegenerateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(enzyme::RegenerateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError(
            "ProbProg: calling `regenerate` on an empty function; if this "
            "is a distribution function, its sample op should have a "
            "logpdf attribute to avoid recursive `regenerate` calls which is "
            "intended for generative functions");
        return failure();
      }

      auto putils =
          MProbProgUtils::CreateFromClone(fn, MProbProgMode::Regenerate);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      auto traceType = enzyme::TraceType::get(CI.getContext());
      auto tensorType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight = arith::ConstantOp::create(
          entryBuilder, putils->initializationBlock->begin()->getLoc(),
          tensorType, DenseElementsAttr::get(tensorType, 0.0));
      Value weightAccumulator = zeroWeight;
      Value currTrace = putils->getTrace();

      Value prevTrace = NewF.getArgument(0);

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: replace sample op uses with call to the
          // distribution function.
          bool isSelected = false;
          for (auto addr : CI.getSelectionAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError(
                    "ProbProg: distribution function cannot have composite "
                    "selected address");
                return WalkResult::interrupt();
              }
              isSelected = true;
              break;
            }
          }

          if (isSelected) {
            // Regenerate selected addresses.
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));

            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());

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
            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
          } else {
            // Use sampled values from the original trace.
            sampledValues.resize(sampleOp.getNumResults());

            SmallVector<Type> sampledValueTypes;
            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              sampledValueTypes.push_back(sampleOp->getResultTypes()[i]);
            }

            auto gsftOp = enzyme::GetSampleFromTraceOp::create(
                rewriter, sampleOp.getLoc(), sampledValueTypes, prevTrace,
                sampleOp.getSymbolAttr());

            // Pass along the RNG state.
            sampledValues[0] = sampleOp.getOperand(0);
            for (unsigned i = 0; i < gsftOp->getNumResults(); ++i) {
              sampledValues[i + 1] = gsftOp->getResult(i);
            }

            // Compute weight via logpdf using originally traced values.
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
            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
          }
        } else {
          // B1. Generative functions: Get the subselection (potentially empty).
          SmallVector<Attribute> subselection;
          for (auto addr : CI.getSelectionAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() == 1) {
                sampleOp.emitError(
                    "ProbProg: generative function cannot be selected with "
                    "singleton address - composite addresses are required to "
                    "ensure proper weight calculation");
                return WalkResult::interrupt();
              }

              SmallVector<Attribute> tailAddresses;
              for (size_t i = 1; i < address.size(); ++i) {
                tailAddresses.push_back(address[i]);
              }
              subselection.push_back(rewriter.getArrayAttr(tailAddresses));
              break;
            }
          }

          // B2. Generate a recursive regenerate op.
          auto getSubtraceOp = enzyme::GetSubtraceOp::create(
              rewriter, sampleOp.getLoc(),
              /*subtrace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*trace*/ prevTrace,
              /*symbol*/ sampleOp.getSymbolAttr());
          auto recursiveOp = enzyme::RegenerateOp::create(
              rewriter, sampleOp.getLoc(),
              /*trace*/ traceType,
              /*weight*/ tensorType,
              /*output_rng_state*/ sampleOp.getOperand(0).getType(),
              /*fn*/ sampleOp.getFnAttr(),
              /*inputs*/ sampleOp.getInputs(),
              /*original_trace*/ getSubtraceOp.getSubtrace(),
              /*selection*/
              rewriter.getArrayAttr(subselection),
              /*name*/ sampleOp.getNameAttr());

          for (unsigned i = 0; i < sampleOp.getNumResults(); ++i)
            sampledValues.push_back(recursiveOp->getResult(i + 2));

          // B2. Add subtrace to trace.
          auto addSubtraceOp = enzyme::AddSubtraceOp::create(
              rewriter, sampleOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(sampleOp.getContext()),
              /*subtrace*/ recursiveOp->getResult(0),
              /*symbol*/ sampleOp.getSymbolAttr(),
              /*trace*/ currTrace);
          currTrace = addSubtraceOp.getUpdatedTrace();

          // B3. Accumulate weight returned by recursive op.
          weightAccumulator = arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                                    weightAccumulator,
                                                    recursiveOp->getResult(1));
        }

        // C. Add non-RNG sampled values to trace (common for both cases).
        SmallVector<Value> valuesToTrace;
        for (unsigned i = 1; i < sampledValues.size(); ++i) {
          valuesToTrace.push_back(sampledValues[i]);
        }

        if (!valuesToTrace.empty()) {
          auto addSampleToTraceOp = enzyme::AddSampleToTraceOp::create(
              rewriter, sampleOp.getLoc(),
              /*updated_trace*/ traceType,
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
        auto addWeightOp = enzyme::AddWeightToTraceOp::create(
            rewriter, retOp.getLoc(),
            /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
            /*trace*/ currTrace, /*weight*/ weightAccumulator);
        currTrace = addWeightOp.getUpdatedTrace();

        // E2. Add non-RNG return values to the trace.
        SmallVector<Value> retvals;
        for (unsigned i = 1; i < retOp.getNumOperands(); ++i) {
          retvals.push_back(retOp.getOperand(i));
        }

        if (!retvals.empty()) {
          auto addRetvalOp = enzyme::AddRetvalToTraceOp::create(
              rewriter, retOp.getLoc(),
              /*updated_trace*/ enzyme::TraceType::get(retOp.getContext()),
              /*trace*/ currTrace,
              /*retval*/ retvals);
          currTrace = addRetvalOp.getUpdatedTrace();
        }

        // E3. Construct new return values: (trace, weight, output_rng_state)
        SmallVector<Value> newRetVals;
        newRetVals.push_back(currTrace);
        newRetVals.push_back(weightAccumulator);
        newRetVals.push_back(retOp.getOperand(0));

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getOriginalTrace());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };

  struct LowerUpdatePattern : public mlir::OpRewritePattern<enzyme::UpdateOp> {
    using mlir::OpRewritePattern<enzyme::UpdateOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(enzyme::UpdateOp CI,
                                  PatternRewriter &rewriter) const override {
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        CI.emitError("ProbProg: calling `update` on an empty function");
        return failure();
      }

      int64_t positionSize = computePositionSizeForSelection(
          CI, fn, CI.getSelectionAttr(), symbolTable);
      if (positionSize <= 0) {
        CI.emitError("ProbProg: failed to compute position size for update");
        return failure();
      }

      auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Update,
                                                    positionSize);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      auto tensorType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight = arith::ConstantOp::create(
          entryBuilder, putils->initializationBlock->begin()->getLoc(),
          tensorType, DenseElementsAttr::get(tensorType, 0.0));
      Value weightAccumulator = zeroWeight;
      Value currTrace = putils->getTrace();

      Value originalTrace = NewF.getArgument(0);
      Value position = NewF.getArgument(1);

      size_t positionOffset = 0;

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          bool isSelected = false;
          for (auto addr : CI.getSelectionAttr()) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              if (address.size() != 1) {
                sampleOp.emitError(
                    "ProbProg: distribution function cannot have composite "
                    "selected address");
                return WalkResult::interrupt();
              }
              isSelected = true;
              break;
            }
          }

          if (isSelected) {
            sampledValues.resize(sampleOp.getNumResults());
            sampledValues[0] = sampleOp.getOperand(0); // RNG state

            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              auto shape = resultType.getShape();

              int64_t numElements = 1;
              for (auto dim : shape) {
                if (dim == ShapedType::kDynamic) {
                  sampleOp.emitError(
                      "ProbProg: dynamic tensor dimensions not supported in "
                      "update");
                  return WalkResult::interrupt();
                }
                numElements *= dim;
              }

              // Reconstruct multi-dimensional tensor from position vector
              auto unflattenOp = enzyme::UnflattenSliceOp::create(
                  rewriter, sampleOp.getLoc(), resultType, position,
                  rewriter.getI64IntegerAttr(positionOffset));

              sampledValues[i] = unflattenOp.getResult();
              positionOffset += numElements;
            }

            auto logpdfFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getLogpdfAttr()));

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

            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
          } else {
            sampledValues.resize(sampleOp.getNumResults());

            SmallVector<Type> sampledValueTypes;
            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              sampledValueTypes.push_back(sampleOp->getResultTypes()[i]);
            }

            auto gsftOp = enzyme::GetSampleFromTraceOp::create(
                rewriter, sampleOp.getLoc(), sampledValueTypes, originalTrace,
                sampleOp.getSymbolAttr());

            sampledValues[0] = sampleOp.getOperand(0); // RNG state
            for (unsigned i = 0; i < gsftOp->getNumResults(); ++i) {
              sampledValues[i + 1] = gsftOp->getResult(i);
            }

            auto logpdfFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getLogpdfAttr()));

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

            auto logpdf = func::CallOp::create(
                rewriter, sampleOp.getLoc(), logpdfFn.getName(),
                logpdfFn.getResultTypes(), logpdfOperands);
            weightAccumulator =
                arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                      weightAccumulator, logpdf.getResult(0));
          }
        } else {
          // TODO
          sampleOp.emitError(
              "ProbProg: update on generative functions not implemented");
          return WalkResult::interrupt();
        }

        SmallVector<Value> valuesToTrace;
        for (unsigned i = 1; i < sampledValues.size(); ++i) {
          valuesToTrace.push_back(sampledValues[i]);
        }

        if (!valuesToTrace.empty()) {
          auto addSampleToTraceOp = enzyme::AddSampleToTraceOp::create(
              rewriter, sampleOp.getLoc(),
              enzyme::TraceType::get(sampleOp.getContext()), currTrace,
              sampleOp.getSymbolAttr(), valuesToTrace);
          currTrace = addSampleToTraceOp.getUpdatedTrace();
        }

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

      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

        auto addWeightOp = enzyme::AddWeightToTraceOp::create(
            rewriter, retOp.getLoc(),
            enzyme::TraceType::get(retOp.getContext()), currTrace,
            weightAccumulator);
        currTrace = addWeightOp.getUpdatedTrace();

        SmallVector<Value> retvals;
        for (unsigned i = 1; i < retOp.getNumOperands(); ++i) {
          retvals.push_back(retOp.getOperand(i));
        }

        if (!retvals.empty()) {
          auto addRetvalOp = enzyme::AddRetvalToTraceOp::create(
              rewriter, retOp.getLoc(),
              enzyme::TraceType::get(retOp.getContext()), currTrace, retvals);
          currTrace = addRetvalOp.getUpdatedTrace();
        }

        SmallVector<Value> newRetVals;
        newRetVals.push_back(currTrace);
        newRetVals.push_back(weightAccumulator);
        newRetVals.push_back(retOp.getOperand(0));

        func::ReturnOp::create(rewriter, retOp.getLoc(), newRetVals);
        rewriter.eraseOp(retOp);
      });

      rewriter.setInsertionPoint(CI);
      SmallVector<Value> operands;
      operands.push_back(CI.getOriginalTrace());
      operands.push_back(CI.getPosition());
      operands.append(CI.getInputs().begin(), CI.getInputs().end());
      auto newCI = func::CallOp::create(rewriter, CI.getLoc(), NewF.getName(),
                                        NewF.getResultTypes(), operands);

      rewriter.replaceOp(CI, newCI.getResults());

      delete putils;

      return success();
    }
  };
};

} // end anonymous namespace

void ProbProgPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns
      .add<LowerUpdatePattern, LowerUntracedCallPattern, LowerSimulatePattern,
           LowerGeneratePattern, LowerMHPattern, LowerRegeneratePattern>(
          &getContext());
  patterns.add<LowerMCMCPattern>(&getContext(), debugDump);

  mlir::GreedyRewriteConfig config;

  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
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
