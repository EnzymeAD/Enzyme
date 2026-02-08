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

using SampleOpMap = DenseMap<Attribute, enzyme::SampleOp>;

static SampleOpMap buildSampleOpMap(FunctionOpInterface fn) {
  SampleOpMap map;
  fn.walk([&](enzyme::SampleOp sampleOp) {
    if (auto symbol = sampleOp.getSymbolAttr())
      map[symbol] = sampleOp;
  });
  return map;
}

static enzyme::SampleOp findSampleBySymbol(const SampleOpMap &map,
                                           Attribute targetSymbol) {
  auto it = map.find(targetSymbol);
  return it != map.end() ? it->second : nullptr;
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
                                          const SampleOpMap &sampleMap,
                                          ArrayRef<Attribute> address,
                                          SymbolTableCollection &symbolTable,
                                          int64_t &positionSize) {
  if (address.empty())
    return false;

  auto sampleOp = findSampleBySymbol(sampleMap, address[0]);
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

    auto nestedMap = buildSampleOpMap(genFn);
    return computePositionSizeForAddress(op, nestedMap, address.drop_front(),
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
  auto sampleMap = buildSampleOpMap(fn);
  int64_t positionSize = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty()) {
      op->emitError("Empty address in selection");
      return -1;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, positionSize)) {
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
  auto sampleMap = buildSampleOpMap(fn);
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
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, offset)) {
      return -1;
    }
  }

  return -1;
}

static SmallVector<MCMC::SupportInfo>
collectSupportInfoForSelection(Operation *op, FunctionOpInterface fn,
                               ArrayAttr selection, ArrayAttr allAddresses,
                               SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  SmallVector<MCMC::SupportInfo> supports;
  int64_t currentPositionOffset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    // TODO: Handle nested cases
    if (address.size() != 1)
      continue;

    auto targetSymbol = address[0];
    auto sampleOp = findSampleBySymbol(sampleMap, targetSymbol);
    if (!sampleOp)
      continue;

    auto supportAttr = sampleOp.getSupportAttr();

    int64_t sampleSize = computeSampleElementCount(op, sampleOp);
    if (sampleSize < 0)
      continue;

    int64_t traceOffset = computeOffsetForSampleInSelection(
        op, fn, allAddresses, targetSymbol, symbolTable);
    if (traceOffset < 0) {
      op->emitError("Symbol in selection not found in all_addresses - cannot "
                    "determine trace offset for scattered selection");
      return {};
    }

    supports.emplace_back(currentPositionOffset, traceOffset, sampleSize,
                          supportAttr);
    currentPositionOffset += sampleSize;
  }

  return supports;
}

static ArrayAttr buildSubSelection(OpBuilder &builder, ArrayAttr selection,
                                   Attribute targetSymbol) {
  SmallVector<Attribute> subAddresses;
  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;
    if (address[0] == targetSymbol && address.size() > 1) {
      SmallVector<Attribute> tail(address.begin() + 1, address.end());
      subAddresses.push_back(builder.getArrayAttr(tail));
    }
  }
  return builder.getArrayAttr(subAddresses);
}

static int64_t
computeOffsetForNestedSample(Operation *op, FunctionOpInterface fn,
                             ArrayAttr selection, Attribute targetSymbol,
                             SymbolTableCollection &symbolTable) {
  auto sampleMap = buildSampleOpMap(fn);
  int64_t offset = 0;

  for (auto addr : selection) {
    auto address = cast<ArrayAttr>(addr);
    if (address.empty())
      continue;

    if (address[0] == targetSymbol) {
      return offset;
    }

    SmallVector<Attribute> tailAddresses(address.begin(), address.end());
    if (!computePositionSizeForAddress(op, sampleMap, tailAddresses,
                                       symbolTable, offset)) {
      return -1;
    }
  }

  return -1;
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

      ArrayAttr selection = CI.getSelectionAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, selection, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("ProbProg: failed to compute position size for simulate");
        return failure();
      }

      auto putils = MProbProgUtils::CreateFromClone(fn, MProbProgMode::Simulate,
                                                    positionSize);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();
      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight =
          arith::ConstantOp::create(entryBuilder, initLoc, scalarType,
                                    DenseElementsAttr::get(scalarType, 0.0));
      Value weightAccumulator = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      auto zeroTrace =
          arith::ConstantOp::create(entryBuilder, initLoc, traceType,
                                    DenseElementsAttr::get(traceType, 0.0));
      Value currTrace = zeroTrace;
      int64_t currentOffset = 0;

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues; // Values to replace uses of sample op
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: call the distribution function.
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

          // A3. Check if this sample is in the selection and insert into trace
          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }

          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleType = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "ProbProg: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleType.getElementType());
              auto flatSample = enzyme::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentOffset)));
              currTrace = enzyme::DynamicUpdateSliceOp::create(
                              rewriter, sampleOp.getLoc(), traceType, currTrace,
                              flatSample, ValueRange{row0, colOff})
                              .getResult();
              currentOffset += numElements;
            }
          }
        } else {
          // B. Generative function: recursively simulate the nested function
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "ProbProg: generative function body is empty; "
                "if this is a distribution, add a logpdf attribute");
            return WalkResult::interrupt();
          }

          ArrayAttr subSelection =
              buildSubSelection(rewriter, selection, sampleOp.getSymbolAttr());
          if (subSelection.empty()) {
            // No samples from this generative function are in the selection
            // Just call the function directly
            auto genCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), genFn.getName(),
                genFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(genCall.getResults().begin(),
                                 genCall.getResults().end());
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subSelection, symbolTable);
            if (subPositionSize <= 0) {
              sampleOp.emitError(
                  "ProbProg: failed to compute sub-position size");
              return WalkResult::interrupt();
            }

            // Build result types: (trace, weight, original_returns...)
            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> simResultTypes;
            simResultTypes.push_back(subTraceType);
            simResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              simResultTypes.push_back(t);

            auto nestedSimulate = enzyme::SimulateOp::create(
                rewriter, sampleOp.getLoc(), simResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subSelection);
            auto subTrace = nestedSimulate.getTrace();
            auto subWeight = nestedSimulate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            int64_t mergeOffset = computeOffsetForNestedSample(
                sampleOp, fn, selection, sampleOp.getSymbolAttr(), symbolTable);
            if (mergeOffset < 0) {
              sampleOp.emitError("ProbProg: failed to compute merge offset");
              return WalkResult::interrupt();
            }

            auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
            auto row0 = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
            auto colOff = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(
                    i64S, rewriter.getI64IntegerAttr(mergeOffset)));
            currTrace = enzyme::DynamicUpdateSliceOp::create(
                            rewriter, sampleOp.getLoc(), traceType, currTrace,
                            subTrace, ValueRange{row0, colOff})
                            .getResult();
            currentOffset =
                std::max(currentOffset, mergeOffset + subPositionSize);

            for (auto output : nestedSimulate.getOutputs())
              sampledValues.push_back(output);
          }
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

      // E. Rewrite the return to return (trace, weight, <original returns>...)
      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);
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

      bool hasLogpdfFn = static_cast<bool>(mcmcOp.getLogpdfFnAttr());

      if (!hasLogpdfFn) {
        auto fnAttr = mcmcOp.getFnAttr();
        if (!fnAttr) {
          mcmcOp.emitError("ProbProg: either fn or logpdf_fn must be provided");
          return failure();
        }
        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(mcmcOp, fnAttr));
        if (fn.getFunctionBody().empty()) {
          mcmcOp.emitError("ProbProg: calling `mcmc` on an empty function");
          return failure();
        }
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

      int64_t positionSize;
      SmallVector<Value> fnInputs;
      SmallVector<Type> fnResultTypes;
      Value originalTrace;
      ArrayAttr selection, allAddresses;
      SmallVector<SupportInfo> supports;
      FlatSymbolRefAttr logpdfFnAttr;

      if (hasLogpdfFn) {
        logpdfFnAttr = mcmcOp.getLogpdfFnAttr();
        auto initialPos = mcmcOp.getInitialPosition();
        positionSize =
            cast<RankedTensorType>(initialPos.getType()).getShape()[1];
        selection = mcmcOp.getSelectionAttr();
        allAddresses = mcmcOp.getAllAddressesAttr();
      } else {
        fnInputs.assign(inputs.begin() + 1, inputs.end());
        originalTrace = mcmcOp.getOriginalTrace();
        selection = mcmcOp.getSelectionAttr();
        allAddresses = mcmcOp.getAllAddressesAttr();

        auto fn = cast<FunctionOpInterface>(
            symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));
        positionSize =
            computePositionSizeForSelection(mcmcOp, fn, selection, symbolTable);
        if (positionSize <= 0)
          return failure();

        supports = collectSupportInfoForSelection(mcmcOp, fn, selection,
                                                  allAddresses, symbolTable);

        auto fnType = cast<FunctionType>(fn.getFunctionType());
        fnResultTypes.assign(fnType.getResults().begin(),
                             fnType.getResults().end());
      }

      int64_t numSamples = mcmcOp.getNumSamples();
      int64_t thinning = mcmcOp.getThinning();
      int64_t numWarmup = mcmcOp.getNumWarmup();

      auto elemType =
          cast<RankedTensorType>(stepSize.getType()).getElementType();
      auto positionType = RankedTensorType::get({1, positionSize}, elemType);
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

      auto makeHMCContext = [&](Value currentInvMass,
                                Value currentMassMatrixSqrt,
                                Value currentStepSize) -> HMCContext {
        if (hasLogpdfFn) {
          return HMCContext(logpdfFnAttr, currentInvMass, currentMassMatrixSqrt,
                            currentStepSize, trajectoryLength, positionSize);
        } else {
          return HMCContext(
              mcmcOp.getFnAttr(), fnInputs, fnResultTypes, originalTrace,
              selection, allAddresses, currentInvMass, currentMassMatrixSqrt,
              currentStepSize, trajectoryLength, positionSize, supports);
        }
      };

      auto makeNUTSContext =
          [&](Value currentInvMass, Value currentMassMatrixSqrt,
              Value currentStepSize, Value U) -> NUTSContext {
        if (hasLogpdfFn) {
          return NUTSContext(logpdfFnAttr, currentInvMass,
                             currentMassMatrixSqrt, currentStepSize,
                             positionSize, U, maxDeltaEnergy, maxTreeDepth);
        } else {
          return NUTSContext(mcmcOp.getFnAttr(), fnInputs, fnResultTypes,
                             originalTrace, selection, allAddresses,
                             currentInvMass, currentMassMatrixSqrt,
                             currentStepSize, positionSize, supports, U,
                             maxDeltaEnergy, maxTreeDepth);
        }
      };

      auto baseCtx =
          makeHMCContext(adaptedInvMass, adaptedMassMatrixSqrt, stepSize);
      auto initState = InitHMC(
          rewriter, loc, rngInput, baseCtx,
          hasLogpdfFn ? mcmcOp.getInitialPosition() : Value(), debugDump);

      auto runSampleStepWithStepSize =
          [&](OpBuilder &builder, Location loc, Value q, Value grad, Value U,
              Value rng, Value currentStepSize) -> MCMCKernelResult {
        if (isHMC) {
          auto ctx = makeHMCContext(adaptedInvMass, adaptedMassMatrixSqrt,
                                    currentStepSize);
          return SampleHMC(builder, loc, q, grad, U, rng, ctx, debugDump);
        } else {
          auto nutsCtx = makeNUTSContext(adaptedInvMass, adaptedMassMatrixSqrt,
                                         currentStepSize, U);
          return SampleNUTS(builder, loc, q, grad, U, rng, nutsCtx, debugDump);
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
          auto ctx = makeHMCContext(currentInvMass, currentMassMatrixSqrt,
                                    currentStepSize);
          return SampleHMC(builder, loc, q, grad, U, rng, ctx, debugDump);
        } else {
          auto nutsCtx = makeNUTSContext(currentInvMass, currentMassMatrixSqrt,
                                         currentStepSize, U);
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
          auto sampleType1D = RankedTensorType::get({positionSize}, elemType);
          Value sample1D =
              enzyme::ReshapeOp::create(rewriter, loc, sampleType1D, sample.q);
          WelfordState updatedWelfordAfterSample = updateWelford(
              rewriter, loc, welfordStateLoop, sample1D, welfordConfig);

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

      auto zeroCol = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(0)));
      auto updatedSamplesBuffer = enzyme::DynamicUpdateSliceOp::create(
          rewriter, loc, samplesBufferType, samplesBufferLoop, q_constrained,
          ValueRange{storageIdx, zeroCol});
      auto selectedSamplesBuffer = enzyme::SelectOp::create(
          rewriter, loc, samplesBufferType, shouldStore, updatedSamplesBuffer,
          samplesBufferLoop);

      auto accepted1D = enzyme::ReshapeOp::create(
          rewriter, loc, RankedTensorType::get({1}, rewriter.getI1Type()),
          sample.accepted);
      auto updatedAcceptedBuffer = enzyme::DynamicUpdateSliceOp::create(
          rewriter, loc, acceptedBufferType, acceptedBufferLoop, accepted1D,
          ValueRange{storageIdx});
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

      rewriter.replaceOp(mcmcOp,
                         {finalSamplesBuffer, finalAcceptedBuffer, finalRng});

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

      auto loc = mhOp.getLoc();

      Value oldTrace = mhOp.getOperand(0);
      Value oldWeight = mhOp.getOperand(1);
      SmallVector<Value> inputs;
      for (unsigned i = 2; i < mhOp.getNumOperands(); ++i)
        inputs.push_back(mhOp.getOperand(i));
      auto selection = mhOp.getSelectionAttr();

      auto traceType = oldTrace.getType();
      auto weightType = cast<RankedTensorType>(oldWeight.getType());
      auto rngStateType = inputs[0].getType();

      // 1. Create regenerate op with the same function and selection
      auto nameAttr = mhOp.getNameAttr();
      if (!nameAttr)
        nameAttr = rewriter.getStringAttr("");

      auto regenerateAddresses = mhOp.getRegenerateAddressesAttr();

      SmallVector<Type> regenResultTypes;
      regenResultTypes.push_back(traceType);
      regenResultTypes.push_back(weightType);
      for (auto t : fn.getResultTypes())
        regenResultTypes.push_back(t);

      auto regenerateOp = rewriter.create<enzyme::RegenerateOp>(
          loc,
          /*resultTypes*/ regenResultTypes,
          /*fn*/ mhOp.getFnAttr(),
          /*inputs*/ inputs,
          /*original_trace*/ oldTrace,
          /*selection*/ selection,
          /*regenerate_addresses*/ regenerateAddresses,
          /*name*/ nameAttr);

      Value newTrace = regenerateOp.getNewTrace();
      Value newWeight = regenerateOp.getWeight();
      Value newRng = regenerateOp.getOutputs()[0];

      // 2. Compute log_alpha = new_weight - old_weight
      auto logAlpha =
          arith::SubFOp::create(rewriter, loc, newWeight, oldWeight);

      // 3. Sample uniform random in (0, 1) and compute log
      auto zeroConst = arith::ConstantOp::create(
          rewriter, loc, weightType, DenseElementsAttr::get(weightType, 0.0));
      auto oneConst = arith::ConstantOp::create(
          rewriter, loc, weightType, DenseElementsAttr::get(weightType, 1.0));

      auto randomOp = enzyme::RandomOp::create(
          rewriter, loc, TypeRange{rngStateType, weightType}, newRng, zeroConst,
          oneConst,
          enzyme::RngDistributionAttr::get(rewriter.getContext(),
                                           enzyme::RngDistribution::UNIFORM));
      auto logRand = math::LogOp::create(rewriter, loc, randomOp.getResult());
      Value finalRng = randomOp.getOutputRngState();

      // 4. Check if proposal is accepted: log(rand()) < log_alpha
      auto accepted = arith::CmpFOp::create(
          rewriter, loc, arith::CmpFPredicate::OLT, logRand, logAlpha);

      // 5. Select trace and weight based on acceptance
      auto selectedTrace = enzyme::SelectOp::create(
          rewriter, loc, traceType, accepted, newTrace, oldTrace);
      auto selectedWeight = arith::SelectOp::create(rewriter, loc, accepted,
                                                    newWeight, oldWeight);

      rewriter.replaceOp(mhOp,
                         {selectedTrace, selectedWeight, accepted, finalRng});
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

      ArrayAttr selection = CI.getSelectionAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, selection, symbolTable);
      if (positionSize <= 0) {
        CI.emitError("ProbProg: failed to compute position size for generate");
        return failure();
      }

      int64_t constraintSize = computePositionSizeForSelection(
          CI, fn, CI.getConstrainedAddressesAttr(), symbolTable);
      if (constraintSize < 0) {
        CI.emitError(
            "ProbProg: failed to compute constraint size for generate");
        return failure();
      }

      auto putils = MProbProgUtils::CreateFromClone(
          fn, MProbProgMode::Generate, positionSize, constraintSize);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();

      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight =
          arith::ConstantOp::create(entryBuilder, initLoc, scalarType,
                                    DenseElementsAttr::get(scalarType, 0.0));
      Value weightAccumulator = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      auto zeroTrace =
          arith::ConstantOp::create(entryBuilder, initLoc, traceType,
                                    DenseElementsAttr::get(traceType, 0.0));
      Value currTrace = zeroTrace;
      Value constraint = NewF.getArgument(0);
      int64_t currentTraceOffset = 0;

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: call the distribution function.
          bool isConstrained = false;
          int64_t constrainedOffset = -1;
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
              constrainedOffset = computeOffsetForSampleInSelection(
                  CI, fn, CI.getConstrainedAddressesAttr(),
                  sampleOp.getSymbolAttr(), symbolTable);
              break;
            }
          }

          if (isConstrained) {
            // Extract sampled values from constraint tensor
            sampledValues.resize(sampleOp.getNumResults());
            sampledValues[0] = sampleOp.getOperand(0); // RNG state

            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "ProbProg: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto sliceType = RankedTensorType::get(
                  {1, numElements}, resultType.getElementType());
              auto sliced = enzyme::SliceOp::create(
                  rewriter, sampleOp.getLoc(), sliceType, constraint,
                  rewriter.getDenseI64ArrayAttr({0, constrainedOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, constrainedOffset + numElements}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
              auto extracted = enzyme::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), resultType, sliced);
              sampledValues[i] = extracted.getResult();
              constrainedOffset += numElements;
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
            // Unconstrained: call the distribution function
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

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }

          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleType = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "ProbProg: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleType.getElementType());
              auto flatSample = enzyme::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentTraceOffset)));
              currTrace = enzyme::DynamicUpdateSliceOp::create(
                              rewriter, sampleOp.getLoc(), traceType, currTrace,
                              flatSample, ValueRange{row0, colOff})
                              .getResult();
              currentTraceOffset += numElements;
            }
          }
        } else {
          // B. Generative function: recursively generate the nested function
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "ProbProg: generative function body is empty; "
                "if this is a distribution, add a logpdf attribute");
            return WalkResult::interrupt();
          }

          ArrayAttr subSelection =
              buildSubSelection(rewriter, selection, sampleOp.getSymbolAttr());
          ArrayAttr subConstrainedAddrs =
              buildSubSelection(rewriter, CI.getConstrainedAddressesAttr(),
                                sampleOp.getSymbolAttr());

          if (subSelection.empty()) {
            // No samples from this generative function are in the selection
            // Just call the function directly
            auto genCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), genFn.getName(),
                genFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(genCall.getResults().begin(),
                                 genCall.getResults().end());
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subSelection, symbolTable);
            int64_t subConstraintSize = computePositionSizeForSelection(
                sampleOp, genFn, subConstrainedAddrs, symbolTable);
            if (subPositionSize <= 0 || subConstraintSize < 0) {
              sampleOp.emitError("ProbProg: failed to compute sub-position or "
                                 "sub-constraint size");
              return WalkResult::interrupt();
            }

            Value subConstraint;
            auto subConstraintType = RankedTensorType::get(
                {1, subConstraintSize}, rewriter.getF64Type());

            if (subConstraintSize > 0) {
              int64_t subConstraintOffset = computeOffsetForNestedSample(
                  sampleOp, fn, CI.getConstrainedAddressesAttr(),
                  sampleOp.getSymbolAttr(), symbolTable);

              subConstraint = enzyme::SliceOp::create(
                  rewriter, sampleOp.getLoc(), subConstraintType, constraint,
                  rewriter.getDenseI64ArrayAttr({0, subConstraintOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, subConstraintOffset + subConstraintSize}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
            } else {
              subConstraint = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), subConstraintType,
                  DenseElementsAttr::get(subConstraintType, {0.0}));
            }

            // Build result types: (trace, weight, original_returns...)
            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> genResultTypes;
            genResultTypes.push_back(subTraceType);
            genResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              genResultTypes.push_back(t);

            auto nestedGenerate = enzyme::GenerateOp::create(
                rewriter, sampleOp.getLoc(), genResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subConstraint,
                subSelection, subConstrainedAddrs);

            Value subTrace = nestedGenerate.getTrace();
            Value subWeight = nestedGenerate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            int64_t mergeOffset = computeOffsetForNestedSample(
                sampleOp, fn, selection, sampleOp.getSymbolAttr(), symbolTable);

            auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
            auto row0 = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
            auto colOff = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(
                    i64S, rewriter.getI64IntegerAttr(mergeOffset)));
            currTrace = enzyme::DynamicUpdateSliceOp::create(
                            rewriter, sampleOp.getLoc(), traceType, currTrace,
                            subTrace, ValueRange{row0, colOff})
                            .getResult();
            currentTraceOffset =
                std::max(currentTraceOffset, mergeOffset + subPositionSize);

            for (auto output : nestedGenerate.getOutputs())
              sampledValues.push_back(output);
          }
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

      // Rewrite the return to return (trace, weight, <original returns>...)
      NewF.walk([&](func::ReturnOp retOp) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(retOp);

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

      ArrayAttr selection = CI.getSelectionAttr();
      int64_t positionSize =
          computePositionSizeForSelection(CI, fn, selection, symbolTable);
      if (positionSize <= 0) {
        CI.emitError(
            "ProbProg: failed to compute position size for regenerate");
        return failure();
      }

      auto putils = MProbProgUtils::CreateFromClone(
          fn, MProbProgMode::Regenerate, positionSize);
      FunctionOpInterface NewF = putils->newFunc;

      OpBuilder entryBuilder(putils->initializationBlock,
                             putils->initializationBlock->begin());
      Location initLoc = putils->initializationBlock->begin()->getLoc();

      auto scalarType = RankedTensorType::get({}, entryBuilder.getF64Type());
      auto zeroWeight =
          arith::ConstantOp::create(entryBuilder, initLoc, scalarType,
                                    DenseElementsAttr::get(scalarType, 0.0));
      Value weightAccumulator = zeroWeight;

      auto traceType =
          RankedTensorType::get({1, positionSize}, entryBuilder.getF64Type());
      auto zeroTrace =
          arith::ConstantOp::create(entryBuilder, initLoc, traceType,
                                    DenseElementsAttr::get(traceType, 0.0));
      Value currTrace = zeroTrace;

      Value prevTrace = NewF.getArgument(0);
      int64_t currentTraceOffset = 0;

      SmallVector<Operation *> toErase;
      auto result = NewF.walk([&](enzyme::SampleOp sampleOp) -> WalkResult {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sampleOp);

        SmallVector<Value> sampledValues;
        bool isDistribution = static_cast<bool>(sampleOp.getLogpdfAttr());

        if (isDistribution) {
          // A1. Distribution function: call the distribution function.
          bool isSelected = false;
          for (auto addr : CI.getRegenerateAddressesAttr()) {
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

          int64_t sampleOffset = computeOffsetForSampleInSelection(
              CI, fn, selection, sampleOp.getSymbolAttr(), symbolTable);

          if (isSelected) {
            // A2. Regenerate: call the distribution function.
            auto distFn =
                cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                    sampleOp, sampleOp.getFnAttr()));

            auto distCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), distFn.getName(),
                distFn.getResultTypes(), sampleOp.getInputs());

            sampledValues.append(distCall.getResults().begin(),
                                 distCall.getResults().end());
          } else {
            // B. Generative function: extract from original trace.
            sampledValues.resize(sampleOp.getNumResults());
            sampledValues[0] = sampleOp.getOperand(0); // RNG state

            int64_t extractOffset = sampleOffset;
            for (unsigned i = 1; i < sampleOp.getNumResults(); ++i) {
              auto resultType =
                  cast<RankedTensorType>(sampleOp.getResult(i).getType());
              int64_t numElements = computeTensorElementCount(resultType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "ProbProg: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto sliceType = RankedTensorType::get(
                  {1, numElements}, resultType.getElementType());
              auto sliced = enzyme::SliceOp::create(
                  rewriter, sampleOp.getLoc(), sliceType, prevTrace,
                  rewriter.getDenseI64ArrayAttr({0, extractOffset}),
                  rewriter.getDenseI64ArrayAttr(
                      {1, extractOffset + numElements}),
                  rewriter.getDenseI64ArrayAttr({1, 1}));
              auto extracted = enzyme::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), resultType, sliced);
              sampledValues[i] = extracted.getResult();
              extractOffset += numElements;
            }
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
            sampleOp.emitError("ProbProg: failed to construct logpdf call; "
                               "logpdf function has wrong number of arguments");
            return WalkResult::interrupt();
          }

          auto logpdf = func::CallOp::create(
              rewriter, sampleOp.getLoc(), logpdfFn.getName(),
              logpdfFn.getResultTypes(), logpdfOperands);
          weightAccumulator =
              arith::AddFOp::create(rewriter, sampleOp.getLoc(),
                                    weightAccumulator, logpdf.getResult(0));

          bool inSelection = false;
          for (auto addr : selection) {
            auto address = cast<ArrayAttr>(addr);
            if (!address.empty() && address[0] == sampleOp.getSymbolAttr()) {
              inSelection = true;
              break;
            }
          }

          if (inSelection) {
            for (unsigned i = 1; i < sampledValues.size(); ++i) {
              auto sampleValue = sampledValues[i];
              auto sampleType = cast<RankedTensorType>(sampleValue.getType());
              int64_t numElements = computeTensorElementCount(sampleType);
              if (numElements < 0) {
                sampleOp.emitError(
                    "ProbProg: dynamic tensor dimensions not supported");
                return WalkResult::interrupt();
              }

              auto flatSampleType = RankedTensorType::get(
                  {1, numElements}, sampleType.getElementType());
              auto flatSample = enzyme::ReshapeOp::create(
                  rewriter, sampleOp.getLoc(), flatSampleType, sampleValue);
              auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
              auto row0 = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
              auto colOff = arith::ConstantOp::create(
                  rewriter, sampleOp.getLoc(), i64S,
                  DenseElementsAttr::get(
                      i64S, rewriter.getI64IntegerAttr(currentTraceOffset)));
              currTrace = enzyme::DynamicUpdateSliceOp::create(
                              rewriter, sampleOp.getLoc(), traceType, currTrace,
                              flatSample, ValueRange{row0, colOff})
                              .getResult();
              currentTraceOffset += numElements;
            }
          }
        } else {
          // B. Generative function: recursively regenerate the nested function
          auto genFn =
              cast<FunctionOpInterface>(symbolTable.lookupNearestSymbolFrom(
                  sampleOp, sampleOp.getFnAttr()));

          if (genFn.getFunctionBody().empty()) {
            sampleOp.emitError(
                "ProbProg: generative function body is empty; "
                "if this is a distribution, add a logpdf attribute");
            return WalkResult::interrupt();
          }

          ArrayAttr subSelection =
              buildSubSelection(rewriter, selection, sampleOp.getSymbolAttr());
          ArrayAttr subRegenerateAddrs =
              buildSubSelection(rewriter, CI.getRegenerateAddressesAttr(),
                                sampleOp.getSymbolAttr());

          if (subSelection.empty()) {
            auto genCall = func::CallOp::create(
                rewriter, sampleOp.getLoc(), genFn.getName(),
                genFn.getResultTypes(), sampleOp.getInputs());
            sampledValues.append(genCall.getResults().begin(),
                                 genCall.getResults().end());
          } else {
            int64_t subPositionSize = computePositionSizeForSelection(
                sampleOp, genFn, subSelection, symbolTable);
            if (subPositionSize <= 0) {
              sampleOp.emitError(
                  "ProbProg: failed to compute sub-position size");
              return WalkResult::interrupt();
            }

            int64_t mergeOffset = computeOffsetForNestedSample(
                sampleOp, fn, selection, sampleOp.getSymbolAttr(), symbolTable);
            if (mergeOffset < 0) {
              sampleOp.emitError("ProbProg: failed to compute merge offset");
              return WalkResult::interrupt();
            }

            auto subTraceType = RankedTensorType::get({1, subPositionSize},
                                                      rewriter.getF64Type());
            Value subPrevTrace = enzyme::SliceOp::create(
                rewriter, sampleOp.getLoc(), subTraceType, prevTrace,
                rewriter.getDenseI64ArrayAttr({0, mergeOffset}),
                rewriter.getDenseI64ArrayAttr(
                    {1, mergeOffset + subPositionSize}),
                rewriter.getDenseI64ArrayAttr({1, 1}));

            // Build result types: (new_trace, weight, original_returns...)
            auto scalarTy = RankedTensorType::get({}, rewriter.getF64Type());
            SmallVector<Type> regenResultTypes;
            regenResultTypes.push_back(subTraceType);
            regenResultTypes.push_back(scalarTy);
            for (auto t : genFn.getResultTypes())
              regenResultTypes.push_back(t);

            auto nestedRegenerate = enzyme::RegenerateOp::create(
                rewriter, sampleOp.getLoc(), regenResultTypes,
                sampleOp.getFnAttr(), sampleOp.getInputs(), subPrevTrace,
                subSelection, subRegenerateAddrs);

            Value subTrace = nestedRegenerate.getNewTrace();
            Value subWeight = nestedRegenerate.getWeight();

            weightAccumulator = arith::AddFOp::create(
                rewriter, sampleOp.getLoc(), weightAccumulator, subWeight);

            auto i64S = RankedTensorType::get({}, rewriter.getI64Type());
            auto row0 = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(i64S, rewriter.getI64IntegerAttr(0)));
            auto colOff = arith::ConstantOp::create(
                rewriter, sampleOp.getLoc(), i64S,
                DenseElementsAttr::get(
                    i64S, rewriter.getI64IntegerAttr(mergeOffset)));
            currTrace = enzyme::DynamicUpdateSliceOp::create(
                            rewriter, sampleOp.getLoc(), traceType, currTrace,
                            subTrace, ValueRange{row0, colOff})
                            .getResult();
            currentTraceOffset =
                std::max(currentTraceOffset, mergeOffset + subPositionSize);

            for (auto output : nestedRegenerate.getOutputs())
              sampledValues.push_back(output);
          }
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
      operands.push_back(CI.getOriginalTrace());
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
  patterns.add<LowerUntracedCallPattern, LowerSimulatePattern,
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
