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

#include "llvm/ADT/APFloat.h"

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

static int64_t computeTensorElementCount(RankedTensorType tensorType) {
  int64_t elemCount = 1;
  for (auto dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return -1;
    elemCount *= dim;
  }
  return elemCount;
}

static enzyme::SampleOp findSampleBySymbol(FunctionOpInterface fn,
                                           Attribute targetSymbol) {
  enzyme::SampleOp result = nullptr;
  fn.walk([&](enzyme::SampleOp sampleOp) {
    auto sampleSymbol = sampleOp.getSymbolAttr();
    if (sampleSymbol && sampleSymbol == targetSymbol) {
      result = sampleOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

static int64_t computeSampleElementCount(Operation *op,
                                         enzyme::SampleOp sampleOp) {
  int64_t totalCount = 0;
  for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
    auto resultType = sampleOp.getResult(i).getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType) {
      op->emitError("Expected ranked tensor type for sample result");
      return -1;
    }
    int64_t elemCount = computeTensorElementCount(tensorType);
    if (elemCount < 0) {
      op->emitError("Dynamic tensor dimensions not supported");
      return -1;
    }
    totalCount += elemCount;
  }
  return totalCount;
}

static bool computePositionSizeForAddress(Operation *op,
                                          FunctionOpInterface func,
                                          ArrayRef<Attribute> address,
                                          SymbolTableCollection &symbolTable,
                                          int64_t &positionSize) {
  if (address.empty())
    return false;

  auto sampleOp = findSampleBySymbol(func, address[0]);
  if (!sampleOp)
    return false;

  if (address.size() > 1) {
    if (sampleOp.getLogpdfAttr()) {
      op->emitError("Cannot select nested address in distribution function");
      return false;
    }

    auto genFn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
    if (!genFn || genFn.getFunctionBody().empty()) {
      op->emitError("Cannot find generative function for nested address");
      return false;
    }

    return computePositionSizeForAddress(op, genFn, address.drop_front(),
                                         symbolTable, positionSize);
  }

  int64_t elemCount = computeSampleElementCount(op, sampleOp);
  if (elemCount < 0)
    return false;

  positionSize += elemCount;
  return true;
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

static int64_t
computeOffsetForSampleInSelection(Operation *op, FunctionOpInterface fn,
                                  ArrayAttr selection, Attribute targetSymbol,
                                  SymbolTableCollection &symbolTable) {
  int64_t offset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    auto firstSymbol = address[0];

    if (firstSymbol == targetSymbol) {
      return offset;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, fn, tailAddresses, symbolTable,
                                       offset)) {
      return -1;
    }
  }

  return -1;
}

static SmallVector<MCMC::SupportInfo>
collectSupportInfoForSelection(Operation *op, FunctionOpInterface fn,
                               ArrayAttr selection,
                               SymbolTableCollection &symbolTable) {
  SmallVector<MCMC::SupportInfo> supports;
  int64_t currentOffset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    // TODO: Handle nested cases
    if (address.size() != 1)
      continue;

    auto targetSymbol = address[0];
    auto sampleOp = findSampleBySymbol(fn, targetSymbol);
    if (!sampleOp)
      continue;

    auto supportAttr = sampleOp.getSupportAttr();

    int64_t sampleSize = computeSampleElementCount(op, sampleOp);
    if (sampleSize < 0)
      continue;

    supports.emplace_back(currentOffset, sampleSize, supportAttr);
    currentOffset += sampleSize;
  }

  return supports;
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
      SymbolTableCollection symbolTable;

      auto fn = cast<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));

      if (fn.getFunctionBody().empty()) {
        mcmcOp.emitError("ProbProg: calling `mcmc` on an empty function");
        return failure();
      }

      if (!mcmcOp.getStepSize()) {
        mcmcOp.emitError("ProbProg: MCMC requires step_size parameter");
        return failure();
      }

      bool isHMC = mcmcOp.getHmcConfig().has_value();
      bool isNUTS = mcmcOp.getNutsConfig().has_value();
      if (!isHMC && !isNUTS) {
        mcmcOp.emitError("ProbProg: Unknown MCMC algorithm");
        return failure();
      }

      auto loc = mcmcOp.getLoc();
      auto invMass = mcmcOp.getInverseMassMatrix();
      Value adaptedInvMass = invMass;
      auto stepSize = mcmcOp.getStepSize();

      auto inputs = mcmcOp.getInputs();
      if (inputs.empty()) {
        mcmcOp.emitError("ProbProg: MCMC requires at least rng_state input");
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

      auto supports =
          collectSupportInfoForSelection(mcmcOp, fn, selection, symbolTable);

      int64_t numSamples = mcmcOp.getNumSamples();
      int64_t thinning = mcmcOp.getThinning();
      int64_t numWarmup = mcmcOp.getNumWarmup();

      auto elemType =
          cast<RankedTensorType>(stepSize.getType()).getElementType();
      auto positionType = RankedTensorType::get({positionSize}, elemType);
      auto scalarType = RankedTensorType::get({}, elemType);
      auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());
      auto i1TensorType = RankedTensorType::get({}, rewriter.getI1Type());

      // Algorithm-specific configuration
      Value trajectoryLength;
      Value maxDeltaEnergy;
      int64_t maxTreeDepth = 0;

      bool adaptStepSize = false;
      bool adaptMassMatrix = false;
      auto F64TensorType = RankedTensorType::get({}, rewriter.getF64Type());
      if (isHMC) {
        auto hmcConfig = mcmcOp.getHmcConfig().value();
        double length = hmcConfig.getTrajectoryLength().getValueAsDouble();
        trajectoryLength = arith::ConstantOp::create(
            rewriter, loc, F64TensorType,
            DenseElementsAttr::get(F64TensorType,
                                   rewriter.getF64FloatAttr(length)));
        adaptStepSize = hmcConfig.getAdaptStepSize();
        adaptMassMatrix = hmcConfig.getAdaptMassMatrix();
      } else {
        auto nutsConfig = mcmcOp.getNutsConfig().value();
        maxTreeDepth = nutsConfig.getMaxTreeDepth();
        adaptStepSize = nutsConfig.getAdaptStepSize();
        adaptMassMatrix = nutsConfig.getAdaptMassMatrix();
        double maxDeltaEnergyVal =
            nutsConfig.getMaxDeltaEnergy()
                ? nutsConfig.getMaxDeltaEnergy().getValueAsDouble()
                : 1000.0;
        maxDeltaEnergy = arith::ConstantOp::create(
            rewriter, loc, F64TensorType,
            DenseElementsAttr::get(
                F64TensorType, rewriter.getF64FloatAttr(maxDeltaEnergyVal)));
      }

      bool diagonal = true;
      if (invMass) {
        auto invMassType = cast<RankedTensorType>(invMass.getType());
        diagonal = (invMassType.getRank() == 1);
      }

      auto adaptedMassMatrixSqrt =
          computeMassMatrixSqrt(rewriter, loc, adaptedInvMass, positionType);

      HMCContext baseCtx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                         adaptedInvMass, adaptedMassMatrixSqrt, stepSize,
                         trajectoryLength, positionSize, supports);
      auto initState = InitHMC(rewriter, loc, rngInput, baseCtx, debugDump);

      auto runSampleStepWithStepSize =
          [&](OpBuilder &builder, Location loc, Value q, Value grad, Value U,
              Value rng, Value currentStepSize) -> MCMCKernelResult {
        if (isHMC) {
          HMCContext ctx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                         adaptedInvMass, adaptedMassMatrixSqrt, currentStepSize,
                         trajectoryLength, positionSize, supports);
          return SampleHMC(builder, loc, q, grad, U, rng, ctx, debugDump);
        } else {
          NUTSContext nutsCtx(mcmcOp.getFnAttr(), fnInputs, originalTrace,
                              selection, adaptedInvMass, adaptedMassMatrixSqrt,
                              currentStepSize, positionSize, supports, U,
                              maxDeltaEnergy, maxTreeDepth);
          return SampleNUTS(builder, loc, q, grad, U, rng, nutsCtx, debugDump);
        }
      };

      auto runPostProcess = [&](OpBuilder &builder, Location loc, Value q,
                                Value accepted, Value rng,
                                Value currentStepSize) -> HMCResult {
        if (isHMC) {
          HMCContext ctx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                         adaptedInvMass, adaptedMassMatrixSqrt, currentStepSize,
                         trajectoryLength, positionSize, supports);
          return PostProcessHMC(builder, loc, q, accepted, rng, ctx);
        } else {
          NUTSContext nutsCtx(mcmcOp.getFnAttr(), fnInputs, originalTrace,
                              selection, adaptedInvMass, adaptedMassMatrixSqrt,
                              currentStepSize, positionSize, supports,
                              initState.U0, maxDeltaEnergy, maxTreeDepth);
          return PostProcessNUTS(builder, loc, q, rng, nutsCtx);
        }
      };

      Value currentQ = initState.q0;
      Value currentGrad = initState.grad0;
      Value currentU = initState.U0;
      Value currentRng = initState.rng;
      Value adaptedStepSize = stepSize;

      auto runSampleStepWithInvMass =
          [&](OpBuilder &builder, Location loc, Value q, Value grad, Value U,
              Value rng, Value currentStepSize, Value currentInvMass,
              Value currentMassMatrixSqrt) -> MCMCKernelResult {
        if (isHMC) {
          HMCContext ctx(mcmcOp.getFnAttr(), fnInputs, originalTrace, selection,
                         currentInvMass, currentMassMatrixSqrt, currentStepSize,
                         trajectoryLength, positionSize, supports);
          return SampleHMC(builder, loc, q, grad, U, rng, ctx, debugDump);
        } else {
          NUTSContext nutsCtx(mcmcOp.getFnAttr(), fnInputs, originalTrace,
                              selection, currentInvMass, currentMassMatrixSqrt,
                              currentStepSize, positionSize, supports, U,
                              maxDeltaEnergy, maxTreeDepth);
          return SampleNUTS(builder, loc, q, grad, U, rng, nutsCtx, debugDump);
        }
      };

      if (numWarmup > 0) {
        auto c0 = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(0)));
        auto c1 = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(1)));
        auto numWarmupConst = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(numWarmup)));

        auto schedule = buildAdaptationSchedule(numWarmup);
        int64_t numWindows = static_cast<int64_t>(schedule.size());

        SmallVector<Value> windowEndConstants;
        for (const auto &window : schedule) {
          windowEndConstants.push_back(arith::ConstantOp::create(
              rewriter, loc, i64TensorType,
              DenseElementsAttr::get(i64TensorType,
                                     rewriter.getI64IntegerAttr(window.end))));
        }

        auto numWindowsMinusOne = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(numWindows - 1)));
        auto lastIterConst = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(numWarmup - 1)));

        if (!adaptedInvMass) {
          adaptedInvMass = arith::ConstantOp::create(
              rewriter, loc, positionType,
              DenseElementsAttr::get(positionType,
                                     rewriter.getFloatAttr(elemType, 1.0)));
          adaptedMassMatrixSqrt = arith::ConstantOp::create(
              rewriter, loc, positionType,
              DenseElementsAttr::get(positionType,
                                     rewriter.getFloatAttr(elemType, 1.0)));
        }

        Value initialStepSize = stepSize;
        initialStepSize =
            conditionalDump(rewriter, loc, initialStepSize,
                            "MCMC: initial step size before warmup", debugDump);
        DualAveragingState daState =
            initDualAveraging(rewriter, loc, initialStepSize);

        WelfordState welfordState;
        WelfordConfig welfordConfig;
        if (adaptMassMatrix) {
          welfordState = initWelford(rewriter, loc, positionSize, diagonal);
          welfordConfig.diagonal = diagonal;
          welfordConfig.regularize = true;
        }

        Value windowIdx = arith::ConstantOp::create(
            rewriter, loc, i64TensorType,
            DenseElementsAttr::get(i64TensorType,
                                   rewriter.getI64IntegerAttr(0)));

        // Warmup loop carries by default:
        // [q, grad, U, rng, stepSize, invMass, massMatrixSqrt, daState(5),
        // welfordState(3)?, windowIdx]
        SmallVector<Type> warmupLoopTypes = {positionType,
                                             positionType,
                                             scalarType,
                                             currentRng.getType(),
                                             scalarType, // stepSize
                                             adaptedInvMass.getType(),
                                             adaptedMassMatrixSqrt.getType()};
        for (Type t : daState.getTypes())
          warmupLoopTypes.push_back(t);
        if (adaptMassMatrix) {
          for (Type t : welfordState.getTypes())
            warmupLoopTypes.push_back(t);
        }
        warmupLoopTypes.push_back(i64TensorType); // windowIdx

        SmallVector<Value> warmupInitArgs = {currentQ,
                                             currentGrad,
                                             currentU,
                                             currentRng,
                                             initialStepSize,
                                             adaptedInvMass,
                                             adaptedMassMatrixSqrt};
        for (Value v : daState.toValues())
          warmupInitArgs.push_back(v);
        if (adaptMassMatrix) {
          for (Value v : welfordState.toValues())
            warmupInitArgs.push_back(v);
        }
        warmupInitArgs.push_back(windowIdx);

        auto warmupLoop =
            enzyme::ForLoopOp::create(rewriter, loc, warmupLoopTypes, c0,
                                      numWarmupConst, c1, warmupInitArgs);

        Block *warmupBody = rewriter.createBlock(&warmupLoop.getRegion());
        warmupBody->addArgument(i64TensorType, loc); // iteration index t
        for (Type t : warmupLoopTypes)
          warmupBody->addArgument(t, loc);

        rewriter.setInsertionPointToStart(warmupBody);

        Value iterT = warmupBody->getArgument(0);
        Value qLoop = warmupBody->getArgument(1);
        Value gradLoop = warmupBody->getArgument(2);
        Value ULoop = warmupBody->getArgument(3);
        Value rngLoop = warmupBody->getArgument(4);
        Value stepSizeLoop = warmupBody->getArgument(5);
        Value invMassLoop = warmupBody->getArgument(6);
        Value massMatrixSqrtLoop = warmupBody->getArgument(7);

        SmallVector<Value> daStateLoopValues;
        for (int i = 0; i < 5; ++i)
          daStateLoopValues.push_back(warmupBody->getArgument(8 + i));
        auto daStateLoop = DualAveragingState::fromValues(daStateLoopValues);

        WelfordState welfordStateLoop;
        Value windowIdxLoop;
        if (adaptMassMatrix) {
          SmallVector<Value> welfordStateLoopValues;
          for (int i = 0; i < 3; ++i)
            welfordStateLoopValues.push_back(warmupBody->getArgument(13 + i));
          welfordStateLoop = WelfordState::fromValues(welfordStateLoopValues);
          windowIdxLoop = warmupBody->getArgument(16);
        } else {
          windowIdxLoop = warmupBody->getArgument(13);
        }

        auto sample = runSampleStepWithInvMass(rewriter, loc, qLoop, gradLoop,
                                               ULoop, rngLoop, stepSizeLoop,
                                               invMassLoop, massMatrixSqrtLoop);

        // Update dual averaging state
        DualAveragingConfig daConfig;
        DualAveragingState updatedDaState;
        Value currentStepSizeFromDA;
        Value finalStepSizeFromDA;

        if (adaptStepSize) {
          updatedDaState = updateDualAveraging(rewriter, loc, daStateLoop,
                                               sample.accept_prob, daConfig);
          currentStepSizeFromDA =
              getStepSizeFromDualAveraging(rewriter, loc, updatedDaState);
          finalStepSizeFromDA =
              getStepSizeFromDualAveraging(rewriter, loc, updatedDaState, true);
        } else {
          updatedDaState = daStateLoop;
          currentStepSizeFromDA = stepSizeLoop;
          finalStepSizeFromDA = stepSizeLoop;
        }

        // Use log_step_size_avg at last iteration
        auto isLastIter = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::eq, iterT, lastIterConst);
        Value adaptedStepSizeInLoop = enzyme::SelectOp::create(
            rewriter, loc, scalarType, isLastIter, finalStepSizeFromDA,
            currentStepSizeFromDA);

        const auto &floatSemantics =
            cast<FloatType>(elemType).getFloatSemantics();
        auto tinyConst = arith::ConstantOp::create(
            rewriter, loc, scalarType,
            DenseElementsAttr::get(
                scalarType, FloatAttr::get(elemType, llvm::APFloat::getSmallest(
                                                         floatSemantics))));
        auto maxConst = arith::ConstantOp::create(
            rewriter, loc, scalarType,
            DenseElementsAttr::get(
                scalarType, FloatAttr::get(elemType, llvm::APFloat::getLargest(
                                                         floatSemantics))));
        adaptedStepSizeInLoop = arith::MaximumFOp::create(
            rewriter, loc, adaptedStepSizeInLoop, tinyConst);
        adaptedStepSizeInLoop = arith::MinimumFOp::create(
            rewriter, loc, adaptedStepSizeInLoop, maxConst);

        auto windowIdxGtZero = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::sgt, windowIdxLoop, c0);
        auto windowIdxLtLast =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                  windowIdxLoop, numWindowsMinusOne);
        auto isMiddleWindow = arith::AndIOp::create(
            rewriter, loc, windowIdxGtZero, windowIdxLtLast);

        // Conditionally update Welford
        WelfordState conditionalWelford;
        if (adaptMassMatrix) {
          WelfordState updatedWelfordAfterSample = updateWelford(
              rewriter, loc, welfordStateLoop, sample.q, welfordConfig);

          conditionalWelford.mean = enzyme::SelectOp::create(
              rewriter, loc, welfordStateLoop.mean.getType(), isMiddleWindow,
              updatedWelfordAfterSample.mean, welfordStateLoop.mean);
          conditionalWelford.m2 = enzyme::SelectOp::create(
              rewriter, loc, welfordStateLoop.m2.getType(), isMiddleWindow,
              updatedWelfordAfterSample.m2, welfordStateLoop.m2);
          conditionalWelford.n = enzyme::SelectOp::create(
              rewriter, loc, welfordStateLoop.n.getType(), isMiddleWindow,
              updatedWelfordAfterSample.n, welfordStateLoop.n);
        }

        Value atWindowEnd = arith::ConstantOp::create(
            rewriter, loc, i1TensorType,
            DenseElementsAttr::get(i1TensorType, rewriter.getBoolAttr(false)));

        for (int64_t w = 0; w < numWindows; ++w) {
          auto windowIdxIsW = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, windowIdxLoop,
              arith::ConstantOp::create(
                  rewriter, loc, i64TensorType,
                  DenseElementsAttr::get(i64TensorType,
                                         rewriter.getI64IntegerAttr(w))));
          auto tEqualsWindowEnd =
              arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                    iterT, windowEndConstants[w]);
          auto matchesThisWindow = arith::AndIOp::create(
              rewriter, loc, windowIdxIsW, tEqualsWindowEnd);
          atWindowEnd = arith::OrIOp::create(rewriter, loc, atWindowEnd,
                                             matchesThisWindow);
        }

        Value newWindowIdx =
            arith::AddIOp::create(rewriter, loc, windowIdxLoop, c1);
        Value windowIdxAfterIncrement =
            enzyme::SelectOp::create(rewriter, loc, i64TensorType, atWindowEnd,
                                     newWindowIdx, windowIdxLoop);

        auto atMiddleWindowEnd =
            arith::AndIOp::create(rewriter, loc, atWindowEnd, isMiddleWindow);

        Value finalInvMass;
        Value finalMassMatrixSqrt;
        WelfordState finalWelfordState;
        Value finalStepSizeValue;
        DualAveragingState finalDaState;

        SmallVector<Type> ifResultTypes;
        ifResultTypes.push_back(invMassLoop.getType());
        ifResultTypes.push_back(massMatrixSqrtLoop.getType());
        if (adaptMassMatrix) {
          ifResultTypes.push_back(conditionalWelford.mean.getType());
          ifResultTypes.push_back(conditionalWelford.m2.getType());
          ifResultTypes.push_back(conditionalWelford.n.getType());
        }
        for (Type t : updatedDaState.getTypes())
          ifResultTypes.push_back(t);

        auto ifOp = enzyme::IfOp::create(rewriter, loc, ifResultTypes,
                                         atMiddleWindowEnd);

        {
          Block *trueBranch = rewriter.createBlock(&ifOp.getTrueBranch());
          rewriter.setInsertionPointToStart(trueBranch);

          SmallVector<Value> trueYieldValues;

          if (adaptMassMatrix) {
            auto newInvMass = finalizeWelford(rewriter, loc, conditionalWelford,
                                              welfordConfig);
            auto newMassMatrixSqrt =
                computeMassMatrixSqrt(rewriter, loc, newInvMass, positionType);
            auto reinitWelford =
                initWelford(rewriter, loc, positionSize, diagonal);

            trueYieldValues.push_back(newInvMass);
            trueYieldValues.push_back(newMassMatrixSqrt);
            trueYieldValues.push_back(reinitWelford.mean);
            trueYieldValues.push_back(reinitWelford.m2);
            trueYieldValues.push_back(reinitWelford.n);
          } else {
            trueYieldValues.push_back(invMassLoop);
            trueYieldValues.push_back(massMatrixSqrtLoop);
          }

          if (adaptStepSize) {
            auto reinitDaState =
                initDualAveraging(rewriter, loc, adaptedStepSizeInLoop);
            for (auto v : reinitDaState.toValues())
              trueYieldValues.push_back(v);
          } else {
            for (auto v : updatedDaState.toValues())
              trueYieldValues.push_back(v);
          }

          enzyme::YieldOp::create(rewriter, loc, trueYieldValues);
        }

        {
          Block *falseBranch = rewriter.createBlock(&ifOp.getFalseBranch());
          rewriter.setInsertionPointToStart(falseBranch);

          SmallVector<Value> falseYieldValues;
          falseYieldValues.push_back(invMassLoop);
          falseYieldValues.push_back(massMatrixSqrtLoop);
          if (adaptMassMatrix) {
            falseYieldValues.push_back(conditionalWelford.mean);
            falseYieldValues.push_back(conditionalWelford.m2);
            falseYieldValues.push_back(conditionalWelford.n);
          }
          for (auto v : updatedDaState.toValues())
            falseYieldValues.push_back(v);

          enzyme::YieldOp::create(rewriter, loc, falseYieldValues);
        }

        rewriter.setInsertionPointAfter(ifOp);

        size_t resultIdx = 0;
        finalInvMass = ifOp.getResult(resultIdx++);
        finalMassMatrixSqrt = ifOp.getResult(resultIdx++);
        if (adaptMassMatrix) {
          finalWelfordState.mean = ifOp.getResult(resultIdx++);
          finalWelfordState.m2 = ifOp.getResult(resultIdx++);
          finalWelfordState.n = ifOp.getResult(resultIdx++);
        }
        finalDaState.log_step_size = ifOp.getResult(resultIdx++);
        finalDaState.log_step_size_avg = ifOp.getResult(resultIdx++);
        finalDaState.gradient_avg = ifOp.getResult(resultIdx++);
        finalDaState.step_count = ifOp.getResult(resultIdx++);
        finalDaState.prox_center = ifOp.getResult(resultIdx++);

        finalStepSizeValue = adaptedStepSizeInLoop;

        SmallVector<Value> warmupYieldValues = {
            sample.q,           sample.grad,  sample.U,           sample.rng,
            finalStepSizeValue, finalInvMass, finalMassMatrixSqrt};
        for (Value v : finalDaState.toValues())
          warmupYieldValues.push_back(v);
        if (adaptMassMatrix) {
          for (Value v : finalWelfordState.toValues())
            warmupYieldValues.push_back(v);
        }
        warmupYieldValues.push_back(windowIdxAfterIncrement);

        enzyme::YieldOp::create(rewriter, loc, warmupYieldValues);

        rewriter.setInsertionPointAfter(warmupLoop);

        currentQ = warmupLoop.getResult(0);
        currentGrad = warmupLoop.getResult(1);
        currentU = warmupLoop.getResult(2);
        currentRng = warmupLoop.getResult(3);
        adaptedStepSize = warmupLoop.getResult(4);
        adaptedInvMass = warmupLoop.getResult(5);
        adaptedMassMatrixSqrt = warmupLoop.getResult(6);

        adaptedStepSize =
            conditionalDump(rewriter, loc, adaptedStepSize,
                            "MCMC: adapted step size after warmup", debugDump);
        if (adaptMassMatrix) {
          adaptedInvMass = conditionalDump(
              rewriter, loc, adaptedInvMass,
              "MCMC: adapted inverse mass matrix after warmup", debugDump);
        }
      }

      // TODO: Remove?
      if (numSamples == 1) {
        auto sample =
            runSampleStepWithStepSize(rewriter, loc, currentQ, currentGrad,
                                      currentU, currentRng, adaptedStepSize);
        auto result = runPostProcess(rewriter, loc, sample.q, sample.accepted,
                                     sample.rng, adaptedStepSize);
        rewriter.replaceOp(mcmcOp, {result.trace, result.accepted, result.rng});
        return success();
      }

      int64_t collectionSize = numSamples / thinning;
      int64_t startIdx = numSamples % thinning;

      auto samplesBufferType =
          RankedTensorType::get({collectionSize, positionSize}, elemType);
      auto acceptedBufferType =
          RankedTensorType::get({collectionSize}, rewriter.getI1Type());

      auto samplesBuffer = arith::ConstantOp::create(
          rewriter, loc, samplesBufferType,
          DenseElementsAttr::get(samplesBufferType,
                                 rewriter.getFloatAttr(elemType, 0.0)));
      auto acceptedBuffer = arith::ConstantOp::create(
          rewriter, loc, acceptedBufferType,
          DenseElementsAttr::get(acceptedBufferType,
                                 rewriter.getBoolAttr(isNUTS)));

      auto c0 = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(0)));
      auto c1 = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(1)));
      auto numSamplesConst = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType,
                                 rewriter.getI64IntegerAttr(numSamples)));
      auto startIdxConst = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType,
                                 rewriter.getI64IntegerAttr(startIdx)));
      auto thinningConst = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType,
                                 rewriter.getI64IntegerAttr(thinning)));

      // Loop carries: [q, grad, U, rng, samplesBuffer, acceptedBuffer]
      SmallVector<Type> loopResultTypes = {
          positionType,         positionType,      scalarType,
          currentRng.getType(), samplesBufferType, acceptedBufferType};
      auto forLoopOp = enzyme::ForLoopOp::create(
          rewriter, loc, loopResultTypes, c0, numSamplesConst, c1,
          ValueRange{currentQ, currentGrad, currentU, currentRng, samplesBuffer,
                     acceptedBuffer});

      Block *loopBody = rewriter.createBlock(&forLoopOp.getRegion());
      loopBody->addArgument(i64TensorType, loc);        // i (iteration index)
      loopBody->addArgument(positionType, loc);         // q
      loopBody->addArgument(positionType, loc);         // grad
      loopBody->addArgument(scalarType, loc);           // U
      loopBody->addArgument(currentRng.getType(), loc); // rng
      loopBody->addArgument(samplesBufferType, loc);    // samplesBuffer
      loopBody->addArgument(acceptedBufferType, loc);   // acceptedBuffer

      rewriter.setInsertionPointToStart(loopBody);
      Value iterIdx = loopBody->getArgument(0);
      Value qLoop = loopBody->getArgument(1);
      Value gradLoop = loopBody->getArgument(2);
      Value ULoop = loopBody->getArgument(3);
      Value rngLoop = loopBody->getArgument(4);
      Value samplesBufferLoop = loopBody->getArgument(5);
      Value acceptedBufferLoop = loopBody->getArgument(6);

      auto sample = runSampleStepWithStepSize(rewriter, loc, qLoop, gradLoop,
                                              ULoop, rngLoop, adaptedStepSize);
      auto q_constrained =
          MCMC::constrainPosition(rewriter, loc, sample.q, supports);

      // Storage index: idx = (i - start_idx) / thinning
      auto iMinusStart =
          arith::SubIOp::create(rewriter, loc, iterIdx, startIdxConst);
      auto storageIdx =
          arith::DivSIOp::create(rewriter, loc, iMinusStart, thinningConst);

      // Store condition:
      // (i >= start_idx) && ((i - start_idx) % thinning == 0)
      auto geStartIdx = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sge, iterIdx, startIdxConst);
      auto modThinning =
          arith::RemSIOp::create(rewriter, loc, iMinusStart, thinningConst);
      auto modIsZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, modThinning, c0);
      auto shouldStore =
          arith::AndIOp::create(rewriter, loc, geStartIdx, modIsZero);

      auto updatedSamplesBuffer = enzyme::DynamicUpdateOp::create(
          rewriter, loc, samplesBufferType, samplesBufferLoop, storageIdx,
          q_constrained);
      auto selectedSamplesBuffer = enzyme::SelectOp::create(
          rewriter, loc, samplesBufferType, shouldStore, updatedSamplesBuffer,
          samplesBufferLoop);

      auto updatedAcceptedBuffer = enzyme::DynamicUpdateOp::create(
          rewriter, loc, acceptedBufferType, acceptedBufferLoop, storageIdx,
          sample.accepted);
      auto selectedAcceptedBuffer = enzyme::SelectOp::create(
          rewriter, loc, acceptedBufferType, shouldStore, updatedAcceptedBuffer,
          acceptedBufferLoop);

      enzyme::YieldOp::create(rewriter, loc,
                              ValueRange{sample.q, sample.grad, sample.U,
                                         sample.rng, selectedSamplesBuffer,
                                         selectedAcceptedBuffer});

      rewriter.setInsertionPointAfter(forLoopOp);
      Value finalSamplesBuffer = forLoopOp.getResult(4);
      Value finalAcceptedBuffer = forLoopOp.getResult(5);
      Value finalRng = forLoopOp.getResult(3);

      finalSamplesBuffer =
          conditionalDump(rewriter, loc, finalSamplesBuffer,
                          "MCMC: collected samples", debugDump);

      auto traceType = enzyme::TraceType::get(rewriter.getContext());
      auto initTraceOp = enzyme::InitTraceOp::create(rewriter, loc, traceType);
      Value currTrace = initTraceOp.getTrace();

      size_t positionOffset = 0;

      for (auto addr : selection) {
        auto address = cast<ArrayAttr>(addr);
        if (address.empty())
          continue;

        auto targetSymbol = cast<enzyme::SymbolAttr>(address[0]);
        auto sampleOp = findSampleBySymbol(fn, targetSymbol);
        if (!sampleOp) {
          mcmcOp.emitError("Could not find sample for address in selection");
          return failure();
        }

        SmallVector<Value> batchedSamples;
        for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
          auto resultType =
              cast<RankedTensorType>(sampleOp.getResult(i).getType());
          auto shape = resultType.getShape();

          int64_t numElements = computeTensorElementCount(resultType);
          if (numElements < 0) {
            mcmcOp.emitError("Dynamic tensor dimensions not supported");
            return failure();
          }

          // Result type: [collectionSize, originalShape...]
          SmallVector<int64_t> batchedShape;
          batchedShape.push_back(collectionSize);
          batchedShape.append(shape.begin(), shape.end());
          auto batchedResultType =
              RankedTensorType::get(batchedShape, resultType.getElementType());

          auto unflattenOp = enzyme::RecoverSampleOp::create(
              rewriter, loc, batchedResultType, finalSamplesBuffer,
              rewriter.getI64IntegerAttr(positionOffset));

          batchedSamples.push_back(unflattenOp.getResult());
          positionOffset += numElements;
        }

        if (!batchedSamples.empty()) {
          auto addSampleOp = enzyme::AddSampleToTraceOp::create(
              rewriter, loc, traceType, currTrace, targetSymbol,
              batchedSamples);
          currTrace = addSampleOp.getUpdatedTrace();
        }
      }

      // TODO
      auto getWeightOp = enzyme::GetWeightFromTraceOp::create(
          rewriter, loc, RankedTensorType::get({}, elemType), originalTrace);
      auto addWeightOp = enzyme::AddWeightToTraceOp::create(
          rewriter, loc, traceType, currTrace, getWeightOp.getWeight());
      currTrace = addWeightOp.getUpdatedTrace();

      rewriter.replaceOp(mcmcOp, {currTrace, finalAcceptedBuffer, finalRng});

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
      auto selectedTrace = enzyme::SelectOp::create(
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
            int64_t sampleOffset = computeOffsetForSampleInSelection(
                CI, fn, CI.getSelectionAttr(), sampleOp.getSymbolAttr(),
                symbolTable);
            if (sampleOffset < 0) {
              sampleOp.emitError(
                  "ProbProg: could not compute offset for sample in selection");
              return WalkResult::interrupt();
            }

            sampledValues.resize(sampleOp.getNumResults());
            sampledValues[0] = sampleOp.getOperand(0); // RNG state

            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());

              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "ProbProg: dynamic tensor dimensions not supported in "
                    "update");
                return WalkResult::interrupt();
              }

              // Reconstruct multi-dimensional tensor from position vector
              auto unflattenOp = enzyme::RecoverSampleOp::create(
                  rewriter, sampleOp.getLoc(), resultType, position,
                  rewriter.getI64IntegerAttr(sampleOffset));

              sampledValues[i] = unflattenOp.getResult();
              sampleOffset += numElements;
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
