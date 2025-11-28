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

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PROBPROGPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

static Value createIdentityMatrix(OpBuilder &builder, Location loc,
                                  RankedTensorType matrixType) {
  auto shape = matrixType.getShape();
  assert(shape.size() == 2 && shape[0] == shape[1] &&
         "Identity matrix must be square");
  int64_t n = shape[0];

  SmallVector<double> identityData(n * n, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    identityData[i * n + i] = 1.0;
  }

  return builder.create<arith::ConstantOp>(
      loc, matrixType,
      DenseElementsAttr::get(matrixType, ArrayRef<double>(identityData)));
}

static Value createSigmoid(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  auto oneConst = builder.create<arith::ConstantOp>(
      loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 1.0)));
  auto negX = builder.create<arith::NegFOp>(loc, x);
  auto expNegX = builder.create<math::ExpOp>(loc, negX);
  auto onePlusExp = builder.create<arith::AddFOp>(loc, oneConst, expNegX);
  auto result = builder.create<arith::DivFOp>(loc, oneConst, onePlusExp);
  return result;
}

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
      auto alg = mcmcOp.getAlg();

      switch (alg) {
      case enzyme::MCMCAlgorithm::HMC:
        return lowerHMC(mcmcOp, rewriter);
      case enzyme::MCMCAlgorithm::NUTS:
        return lowerNUTS(mcmcOp, rewriter);
      default:
        mcmcOp.emitError("ProbProg: Unknown MCMC algorithm");
        return failure();
      }
    }

  private:
    // Reference:
    // https://github.com/pyro-ppl/numpyro/blob/d49f71825691b554fb8188f8779dc3a5d13e7b96/numpyro/infer/hmc_util.py#L36
    struct NUTSTree {
      Value q_left, p_left, grad_left;
      Value q_right, p_right, grad_right;
      Value q_proposal, grad_proposal, U_proposal, H_proposal;
      Value depth, weight, turning, diverging;
      Value sum_accept_probs, num_proposals, p_sum;

      static constexpr size_t NUM_FIELDS = 17;

      SmallVector<Value> toValues() const {
        return {q_left,        p_left,        grad_left,
                q_right,       p_right,       grad_right,
                q_proposal,    grad_proposal, U_proposal,
                H_proposal,    depth,         weight,
                turning,       diverging,     sum_accept_probs,
                num_proposals, p_sum};
      }

      static NUTSTree fromValues(ArrayRef<Value> values) {
        assert(values.size() == NUM_FIELDS);
        NUTSTree tree;
        tree.q_left = values[0];
        tree.p_left = values[1];
        tree.grad_left = values[2];
        tree.q_right = values[3];
        tree.p_right = values[4];
        tree.grad_right = values[5];
        tree.q_proposal = values[6];
        tree.grad_proposal = values[7];
        tree.U_proposal = values[8];
        tree.H_proposal = values[9];
        tree.depth = values[10];
        tree.weight = values[11];
        tree.turning = values[12];
        tree.diverging = values[13];
        tree.sum_accept_probs = values[14];
        tree.num_proposals = values[15];
        tree.p_sum = values[16];
        return tree;
      }

      SmallVector<Type> getTypes() const {
        SmallVector<Type> types;
        for (auto val : toValues())
          types.push_back(val.getType());
        return types;
      }
    };

    Value conditionalDump(OpBuilder &builder, Location loc, Value value,
                          StringRef label) const {
      if (debugDump) {
        return enzyme::DumpOp::create(builder, loc, value.getType(), value,
                                      builder.getStringAttr(label))
            .getOutput();
      }
      return value;
    }

    /// Computes v = M^-1 @ p
    Value applyInverseMassMatrix(OpBuilder &builder, Location loc,
                                 Value invMass, Value momentum,
                                 RankedTensorType positionType) const {
      if (!invMass) {
        return momentum;
      }

      auto invMassType = cast<RankedTensorType>(invMass.getType());

      if (invMassType.getRank() == 1) {
        // Diagonal: element-wise
        return arith::MulFOp::create(builder, loc, invMass, momentum);
      } else if (invMassType.getRank() == 2) {
        // Dense: v = invMass @ p
        return enzyme::DotOp::create(builder, loc, positionType, invMass,
                                     momentum, builder.getDenseI64ArrayAttr({}),
                                     builder.getDenseI64ArrayAttr({}),
                                     builder.getDenseI64ArrayAttr({1}),
                                     builder.getDenseI64ArrayAttr({0}));
      }

      emitError(loc,
                "ProbProg: Provided invMass must have rank 1 or 2, got rank " +
                    std::to_string(invMassType.getRank()));
      return nullptr;
    }

    /// Computes K = 0.5 * p^T @ M^-1 @ p
    Value computeKineticEnergy(OpBuilder &builder, Location loc, Value momentum,
                               Value invMass, Value halfConst,
                               RankedTensorType scalarType,
                               RankedTensorType positionType) const {
      // v = M^-1 @ p
      Value v =
          applyInverseMassMatrix(builder, loc, invMass, momentum, positionType);

      // K = 0.5 * p^T @ v
      auto pDotV = enzyme::DotOp::create(
          builder, loc, scalarType, momentum, v,
          builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
          builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));

      return arith::MulFOp::create(builder, loc, halfConst, pDotV);
    }

    /// Samples momentum from N(0, M) where M = invMass^-1
    /// Returns (momentum, updated RNG state)
    std::pair<Value, Value>
    sampleMomentum(OpBuilder &builder, Location loc, Value rngState,
                   Value invMass, Value zeroConst, Value oneConst,
                   RankedTensorType positionType) const {

      // Sample eps ~ N(0, I)
      auto randomOp = enzyme::RandomOp::create(
          builder, loc, TypeRange{rngState.getType(), positionType}, rngState,
          zeroConst, oneConst,
          enzyme::RngDistributionAttr::get(builder.getContext(),
                                           enzyme::RngDistribution::NORMAL));

      Value rngOut = randomOp.getOutputRngState();
      Value eps = randomOp.getResult();

      if (!invMass) {
        return {eps, rngOut};
      }

      auto invMassType = cast<RankedTensorType>(invMass.getType());

      if (invMassType.getRank() == 1) {
        // Diagonal: p = (1/sqrt(invMass)) * eps = sqrt(M) * eps
        auto sqrtInvMass = math::SqrtOp::create(builder, loc, invMass);
        auto onesVector = arith::ConstantOp::create(
            builder, loc, invMassType,
            DenseElementsAttr::get(invMassType, builder.getF64FloatAttr(1.0)));
        auto massMatrixSqrt =
            arith::DivFOp::create(builder, loc, onesVector, sqrtInvMass);
        Value p = arith::MulFOp::create(builder, loc, massMatrixSqrt, eps);
        return {p, rngOut};
      } else {
        // Dense: p = chol(M) @ eps where M = inv(invMass)
        auto identityMatrix = createIdentityMatrix(builder, loc, invMassType);
        auto massMatrixSqrt = enzyme::CholeskySolveOp::create(
            builder, loc, invMassType, invMass, identityMatrix);
        Value p = enzyme::DotOp::create(
            builder, loc, positionType, massMatrixSqrt, eps,
            builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
            builder.getDenseI64ArrayAttr({1}),
            builder.getDenseI64ArrayAttr({0}));
        return {p, rngOut};
      }
    }

    struct GradientResult {
      Value potential; // U(q) = -log p(q)
      Value gradient;  // dU/dq
      Value rngOut;    // Updated RNG state
    };

    /// Computes potential energy U(q) and its gradient dU/dq
    GradientResult computePotentialAndGradient(
        OpBuilder &builder, Location loc, Value position, Value rng,
        FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs, Value originalTrace,
        ArrayAttr selection, enzyme::TraceType traceType,
        RankedTensorType scalarType, RankedTensorType positionType) const {

      auto gradSeed = arith::ConstantOp::create(
          builder, loc, scalarType,
          DenseElementsAttr::get(scalarType, builder.getF64FloatAttr(1.0)));

      auto autodiffOp = enzyme::AutoDiffRegionOp::create(
          builder, loc, TypeRange{scalarType, rng.getType(), positionType},
          ValueRange{position, gradSeed},
          builder.getArrayAttr({enzyme::ActivityAttr::get(
              builder.getContext(), enzyme::Activity::enzyme_active)}),
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_const)}),
          builder.getI64IntegerAttr(1), builder.getBoolAttr(false), nullptr);

      Block *autodiffBlock = builder.createBlock(&autodiffOp.getBody());
      autodiffBlock->addArgument(positionType, loc);

      builder.setInsertionPointToStart(autodiffBlock);
      Value qArg = autodiffBlock->getArgument(0);

      SmallVector<Value> updateInputs;
      updateInputs.push_back(rng);
      updateInputs.append(fnInputs.begin(), fnInputs.end());

      auto updateOp = enzyme::UpdateOp::create(
          builder, loc, TypeRange{traceType, scalarType, rng.getType()}, fn,
          updateInputs, originalTrace, qArg, selection,
          builder.getStringAttr(""));

      Value U = arith::NegFOp::create(builder, loc, updateOp.getWeight());

      enzyme::YieldOp::create(builder, loc,
                              ValueRange{U, updateOp.getOutputRngState()});

      builder.setInsertionPointAfter(autodiffOp);

      return {
          autodiffOp.getResult(0), // potential
          autodiffOp.getResult(2), // gradient
          autodiffOp.getResult(1)  // rngOut
      };
    }

    struct LeapfrogResult {
      Value q_new;
      Value p_new;
      Value grad_new;
      Value U_new;
      Value rng_out;
    };

    /// One leapfrog integration step
    ///   p_half = p - (eps/2) * grad
    ///   q_new = q + eps * M^-1 * p_half
    ///   grad_new = dU/dq(q_new)
    ///   p_new = p_half - (eps/2) * grad_new
    LeapfrogResult leapfrogStep(OpBuilder &builder, Location loc, Value q,
                                Value p, Value grad, Value rng, Value stepSize,
                                Value invMass, FlatSymbolRefAttr fn,
                                ArrayRef<Value> fnInputs, Value originalTrace,
                                ArrayAttr selection,
                                enzyme::TraceType traceType,
                                RankedTensorType scalarType,
                                RankedTensorType positionType) const {

      auto halfConst = arith::ConstantOp::create(
          builder, loc, scalarType,
          DenseElementsAttr::get(scalarType, builder.getF64FloatAttr(0.5)));

      ArrayRef<int64_t> shape = positionType.getShape();
      auto stepSizeBroadcast =
          enzyme::BroadcastOp::create(builder, loc, positionType, stepSize,
                                      builder.getDenseI64ArrayAttr(shape));
      auto halfStep = arith::MulFOp::create(builder, loc, halfConst, stepSize);
      auto halfStepBroadcast =
          enzyme::BroadcastOp::create(builder, loc, positionType, halfStep,
                                      builder.getDenseI64ArrayAttr(shape));

      // 1. Half step momentum: p_half = p - 0.5 * eps * grad
      auto deltaP1 =
          arith::MulFOp::create(builder, loc, halfStepBroadcast, grad);
      Value pHalf = arith::SubFOp::create(builder, loc, p, deltaP1);

      // 2. Full step position: q_new = q + eps * M^-1 * p_half
      Value v =
          applyInverseMassMatrix(builder, loc, invMass, pHalf, positionType);
      auto deltaQ = arith::MulFOp::create(builder, loc, stepSizeBroadcast, v);
      Value qNew = arith::AddFOp::create(builder, loc, q, deltaQ);

      // 3. Compute gradient at new position
      auto gradResult = computePotentialAndGradient(
          builder, loc, qNew, rng, fn, fnInputs, originalTrace, selection,
          traceType, scalarType, positionType);

      // 4. Final half step momentum: p_new = p_half - 0.5 * eps * grad_new
      auto deltaP2 = arith::MulFOp::create(builder, loc, halfStepBroadcast,
                                           gradResult.gradient);
      Value pNew = arith::SubFOp::create(builder, loc, pHalf, deltaP2);

      return {qNew, pNew, gradResult.gradient, gradResult.potential,
              gradResult.rngOut};
    }

    /// NUTS termination check (U-turn)
    /// https://github.com/pyro-ppl/numpyro/blob/47335b83dd02726fb142e61d79193eb761009ba7/numpyro/infer/hmc_util.py#L710-L746
    Value checkTurning(OpBuilder &builder, Location loc, Value invMass,
                       Value pLeft, Value pRight, Value pSum, Value zeroConst,
                       RankedTensorType scalarType,
                       RankedTensorType positionType) const {

      Value vLeft =
          applyInverseMassMatrix(builder, loc, invMass, pLeft, positionType);
      Value vRight =
          applyInverseMassMatrix(builder, loc, invMass, pRight, positionType);

      // p_sum_centered = p_sum - (p_left + p_right) / 2
      auto halfConst = arith::ConstantOp::create(
          builder, loc, scalarType,
          DenseElementsAttr::get(scalarType, builder.getF64FloatAttr(0.5)));
      auto halfBroadcast = enzyme::BroadcastOp::create(
          builder, loc, positionType, halfConst,
          builder.getDenseI64ArrayAttr(positionType.getShape()));

      auto pLeftPlusPRight = arith::AddFOp::create(builder, loc, pLeft, pRight);
      auto halfSum =
          arith::MulFOp::create(builder, loc, halfBroadcast, pLeftPlusPRight);
      Value pSumCentered = arith::SubFOp::create(builder, loc, pSum, halfSum);

      auto leftAngle = enzyme::DotOp::create(
          builder, loc, scalarType, vLeft, pSumCentered,
          builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
          builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
      auto rightAngle = enzyme::DotOp::create(
          builder, loc, scalarType, vRight, pSumCentered,
          builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
          builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));

      // turning = (left_angle <= 0) OR (right_angle <= 0)
      auto leftNeg = arith::CmpFOp::create(
          builder, loc, arith::CmpFPredicate::OLE, leftAngle, zeroConst);
      auto rightNeg = arith::CmpFOp::create(
          builder, loc, arith::CmpFPredicate::OLE, rightAngle, zeroConst);

      return arith::OrIOp::create(builder, loc, leftNeg, rightNeg);
    }

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

      auto tensorType = RankedTensorType::get({}, rewriter.getF64Type());
      auto traceType = enzyme::TraceType::get(mcmcOp.getContext());

      // Extract static HMC parameters
      if (!mcmcOp.getStepSize() || !mcmcOp.getNumSteps()) {
        mcmcOp.emitError(
            "ProbProg: HMC requires step_size and num_steps parameters");
        return failure();
      }

      Value invMass = mcmcOp.getInverseMassMatrix();
      Value stepSize = mcmcOp.getStepSize();
      Value numSteps = mcmcOp.getNumSteps();

      auto inputs = mcmcOp.getInputs();
      if (inputs.empty()) {
        mcmcOp.emitError("ProbProg: HMC requires at least rng_state input");
        return failure();
      }

      Value rngState = inputs[0];
      SmallVector<Value> fnInputs(inputs.begin() + 1, inputs.end());

      auto loc = mcmcOp.getLoc();
      auto originalTrace = mcmcOp.getOriginalTrace();
      auto selection = mcmcOp.getSelectionAttr();

      int64_t positionSize =
          computePositionSizeForSelection(mcmcOp, fn, selection, symbolTable);
      if (positionSize <= 0)
        return failure();

      auto positionType =
          RankedTensorType::get({positionSize}, rewriter.getF64Type());

      // 1. Extract initial position vector q0
      auto q0 = enzyme::GetFlattenedSamplesFromTraceOp::create(
          rewriter, loc, positionType, originalTrace, selection);

      // 2. Compute initial potential energy U0 = -weight
      auto weight0 = enzyme::GetWeightFromTraceOp::create(
          rewriter, loc, tensorType, originalTrace);
      Value U0 = conditionalDump(rewriter, loc,
                                 arith::NegFOp::create(rewriter, loc, weight0),
                                 "HMC: initial potential energy U0");

      auto zeroConst = arith::ConstantOp::create(
          rewriter, loc, tensorType,
          DenseElementsAttr::get(tensorType, rewriter.getF64FloatAttr(0.0)));
      auto oneConst = arith::ConstantOp::create(
          rewriter, loc, tensorType,
          DenseElementsAttr::get(tensorType, rewriter.getF64FloatAttr(1.0)));

      Value rng1;
      Value p0;

      // 3. Sample initial momentum p0 ~ N(0, M) if M is provided,
      //    otherwise p0 ~ N(0, I)
      Value initialMomentum = mcmcOp.getInitialMomentum();
      if (initialMomentum) {
        p0 = initialMomentum;
        rng1 = rngState;
      } else {
        std::tie(p0, rng1) = sampleMomentum(rewriter, loc, rngState, invMass,
                                            zeroConst, oneConst, positionType);
      }

      auto halfConst = arith::ConstantOp::create(
          rewriter, loc, tensorType,
          DenseElementsAttr::get(tensorType, rewriter.getF64FloatAttr(0.5)));

      // 4. Compute initial kinetic energy K0 = 0.5 * p^T * M^-1 * p
      Value K0 = conditionalDump(rewriter, loc,
                                 computeKineticEnergy(rewriter, loc, p0,
                                                      invMass, halfConst,
                                                      tensorType, positionType),
                                 "HMC: initial kinetic energy K0");

      Value H0 = conditionalDump(rewriter, loc,
                                 arith::AddFOp::create(rewriter, loc, U0, K0),
                                 "HMC: initial Hamiltonian H0");

      // 5. Compute initial gradient at q0
      auto gradSeedInit = arith::ConstantOp::create(
          rewriter, loc, tensorType,
          DenseElementsAttr::get(tensorType, rewriter.getF64FloatAttr(1.0)));
      auto autodiffInit = enzyme::AutoDiffRegionOp::create(
          rewriter, loc, TypeRange{rng1.getType(), positionType},
          ValueRange{q0, gradSeedInit},
          rewriter.getArrayAttr({enzyme::ActivityAttr::get(
              rewriter.getContext(), enzyme::Activity::enzyme_active)}),
          rewriter.getArrayAttr(
              {enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_activenoneed),
               enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_const)}),
          rewriter.getI64IntegerAttr(1), rewriter.getBoolAttr(false), nullptr);

      Block *autodiffInitBlock = rewriter.createBlock(&autodiffInit.getBody());
      autodiffInitBlock->addArgument(positionType, loc);

      rewriter.setInsertionPointToStart(autodiffInitBlock);
      Value q0Arg = autodiffInitBlock->getArgument(0);

      SmallVector<Value> updateInputsInit;
      updateInputsInit.push_back(rng1);
      updateInputsInit.append(fnInputs.begin(), fnInputs.end());

      auto updateOpInit = enzyme::UpdateOp::create(
          rewriter, loc, TypeRange{traceType, tensorType, rng1.getType()},
          mcmcOp.getFnAttr(), updateInputsInit, originalTrace, q0Arg, selection,
          rewriter.getStringAttr(""));
      Value w0 = updateOpInit.getWeight();
      Value rng0_out = updateOpInit.getOutputRngState();
      Value U0_init = arith::NegFOp::create(rewriter, loc, w0);

      enzyme::YieldOp::create(rewriter, loc, ValueRange{U0_init, rng0_out});

      rewriter.setInsertionPointAfter(autodiffInit);
      Value rng0_final = autodiffInit.getResult(0);
      Value grad0 = autodiffInit.getResult(1);

      // 6. Leapfrog integration
      auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());
      auto c0 = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(0)));
      auto c1 = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(1)));

      ArrayRef<int64_t> positionShape = positionType.getShape();

      stepSize =
          conditionalDump(rewriter, loc, stepSize, "HMC: step_size (eps)");

      auto stepSizeBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc, positionType, stepSize,
          rewriter.getDenseI64ArrayAttr(positionShape));
      auto halfStepSize =
          arith::MulFOp::create(rewriter, loc, halfConst, stepSize);
      auto halfStepSizeBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc, positionType, halfStepSize,
          rewriter.getDenseI64ArrayAttr(positionShape));

      SmallVector<Type> loopResultTypes = {positionType, positionType,
                                           positionType, rng0_final.getType()};
      auto forLoopOp = enzyme::ForLoopOp::create(
          rewriter, loc, loopResultTypes, c0, numSteps, c1,
          ValueRange{q0, p0, grad0, rng0_final});

      Block *loopBody = rewriter.createBlock(&forLoopOp.getRegion());
      loopBody->addArgument(i64TensorType, loc);        // iv
      loopBody->addArgument(positionType, loc);         // q
      loopBody->addArgument(positionType, loc);         // p
      loopBody->addArgument(positionType, loc);         // gradient
      loopBody->addArgument(rng0_final.getType(), loc); // rng

      rewriter.setInsertionPointToStart(loopBody);
      Value q = conditionalDump(rewriter, loc, loopBody->getArgument(1),
                                "Leapfrog: position q(t)");
      Value p = conditionalDump(rewriter, loc, loopBody->getArgument(2),
                                "Leapfrog: momentum p(t)");
      Value gradient = conditionalDump(rewriter, loc, loopBody->getArgument(3),
                                       "Leapfrog: gradient dU/dq(t)");
      Value loopRng = loopBody->getArgument(4);

      // 6.1 Half step on momentum: p -= (eps/2) * gradient
      auto deltaP1 =
          arith::MulFOp::create(rewriter, loc, halfStepSizeBroadcast, gradient);
      Value p1 = conditionalDump(
          rewriter, loc, arith::SubFOp::create(rewriter, loc, p, deltaP1),
          "Leapfrog: momentum p(t + eps/2)");

      // 6.2 Full step on position: q += eps * M^-1 * p1
      Value v1 =
          applyInverseMassMatrix(rewriter, loc, invMass, p1, positionType);

      auto deltaQ = arith::MulFOp::create(rewriter, loc, stepSizeBroadcast, v1);
      Value q1 = conditionalDump(
          rewriter, loc, arith::AddFOp::create(rewriter, loc, q, deltaQ),
          "Leapfrog: position q(t + eps)");

      // Compute new gradient at q1
      auto gradSeedLoop = arith::ConstantOp::create(
          rewriter, loc, tensorType,
          DenseElementsAttr::get(tensorType, rewriter.getF64FloatAttr(1.0)));
      auto autodiffOp = enzyme::AutoDiffRegionOp::create(
          rewriter, loc, TypeRange{loopRng.getType(), positionType},
          ValueRange{q1, gradSeedLoop},
          rewriter.getArrayAttr({enzyme::ActivityAttr::get(
              rewriter.getContext(), enzyme::Activity::enzyme_active)}),
          rewriter.getArrayAttr(
              {enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_activenoneed),
               enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_const)}),
          rewriter.getI64IntegerAttr(1), rewriter.getBoolAttr(false), nullptr);

      Block *autodiffBlock = rewriter.createBlock(&autodiffOp.getBody());
      autodiffBlock->addArgument(positionType, loc);

      rewriter.setInsertionPointToStart(autodiffBlock);
      Value q1Arg = autodiffBlock->getArgument(0);

      SmallVector<Value> updateInputs;
      updateInputs.push_back(loopRng);
      updateInputs.append(fnInputs.begin(), fnInputs.end());

      auto updateOp = enzyme::UpdateOp::create(
          rewriter, loc, TypeRange{traceType, tensorType, loopRng.getType()},
          mcmcOp.getFnAttr(), updateInputs, originalTrace, q1Arg, selection,
          rewriter.getStringAttr(""));
      Value w1 = updateOp.getWeight();
      Value rng1_inner = updateOp.getOutputRngState();
      Value U1 = arith::NegFOp::create(rewriter, loc, w1);

      enzyme::YieldOp::create(rewriter, loc, ValueRange{U1, rng1_inner});

      rewriter.setInsertionPointAfter(autodiffOp);

      Value newRng = autodiffOp.getResult(0);
      Value newGradient =
          conditionalDump(rewriter, loc, autodiffOp.getResult(1),
                          "Leapfrog: gradient dU/dq(t + eps)");

      // 6.3 Another half step on momentum: p -= (eps/2) * gradient (new)
      auto deltaP2 = arith::MulFOp::create(rewriter, loc, halfStepSizeBroadcast,
                                           newGradient);
      Value p2 = conditionalDump(
          rewriter, loc, arith::SubFOp::create(rewriter, loc, p1, deltaP2),
          "Leapfrog: momentum p(t + eps)");

      // Yield [position, momentum, gradient (new), RNG]
      enzyme::YieldOp::create(rewriter, loc,
                              ValueRange{q1, p2, newGradient, newRng});

      rewriter.setInsertionPointAfter(forLoopOp);
      Value qL = forLoopOp.getResult(0);
      Value pL = forLoopOp.getResult(1);
      Value rngAfterLeapfrog = forLoopOp.getResult(3);

      // 7. Generate final trace with final position qL
      SmallVector<Value> finalUpdateInputs;
      finalUpdateInputs.push_back(rngAfterLeapfrog);
      finalUpdateInputs.append(fnInputs.begin(), fnInputs.end());

      auto finalUpdateOp = enzyme::UpdateOp::create(
          rewriter, loc,
          TypeRange{traceType, tensorType, rngAfterLeapfrog.getType()},
          mcmcOp.getFnAttr(), finalUpdateInputs, originalTrace, qL, selection,
          rewriter.getStringAttr(""));
      Value finalTrace = finalUpdateOp.getUpdatedTrace();
      Value weight1 = finalUpdateOp.getWeight();
      Value rngAfterUpdate = finalUpdateOp.getOutputRngState();

      Value U1_final = conditionalDump(
          rewriter, loc, arith::NegFOp::create(rewriter, loc, weight1),
          "HMC: final potential energy U1");

      // K1 = 0.5 * pL^T * M^-1 * pL
      Value K1 = conditionalDump(rewriter, loc,
                                 computeKineticEnergy(rewriter, loc, pL,
                                                      invMass, halfConst,
                                                      tensorType, positionType),
                                 "HMC: final kinetic energy K1");

      Value H1 = conditionalDump(
          rewriter, loc, arith::AddFOp::create(rewriter, loc, U1_final, K1),
          "HMC: final Hamiltonian H1");

      // 8. Metropolis-Hastings accept/reject step
      // with acceptance probability: α = min(1, exp(H0 - H1))
      auto dH = arith::SubFOp::create(rewriter, loc, H0, H1);
      auto expDH = math::ExpOp::create(rewriter, loc, dH);
      Value accProb = conditionalDump(
          rewriter, loc,
          arith::MinimumFOp::create(rewriter, loc, oneConst, expDH),
          "HMC: acceptance probability α");

      auto randomOp2 = enzyme::RandomOp::create(
          rewriter, loc, TypeRange{rngAfterUpdate.getType(), tensorType},
          rngAfterUpdate, zeroConst, oneConst,
          enzyme::RngDistributionAttr::get(rewriter.getContext(),
                                           enzyme::RngDistribution::UNIFORM));
      Value rngFinal = randomOp2.getOutputRngState();
      Value randUniform = randomOp2.getResult();

      // Accept if U(0,1) < α
      auto acceptedTensor = arith::CmpFOp::create(
          rewriter, loc, arith::CmpFPredicate::OLT, randUniform, accProb);

      // 9. Select trace based on acceptance
      auto selectedTrace = enzyme::SelectTraceOp::create(
          rewriter, loc, traceType, acceptedTensor, finalTrace, originalTrace);

      rewriter.replaceOp(mcmcOp, {selectedTrace, acceptedTensor, rngFinal});

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

      auto F64TensorType = RankedTensorType::get({}, rewriter.getF64Type());
      auto traceType = enzyme::TraceType::get(mcmcOp.getContext());

      Value invMass = mcmcOp.getInverseMassMatrix();
      Value stepSize = mcmcOp.getStepSize();

      if (!stepSize) {
        mcmcOp.emitError("ProbProg: NUTS requires step_size parameter");
        return failure();
      }

      auto inputs = mcmcOp.getInputs();
      if (inputs.empty()) {
        mcmcOp.emitError("ProbProg: initial RNG state is required as the first "
                         "function input by convention");
        return failure();
      }

      Value rngState = inputs[0];
      SmallVector<Value> fnInputs(inputs.begin() + 1, inputs.end());

      auto loc = mcmcOp.getLoc();
      auto originalTrace = mcmcOp.getOriginalTrace();
      auto selection = mcmcOp.getSelectionAttr();

      // 1. Extract initial position vector q0
      int64_t positionSize =
          computePositionSizeForSelection(mcmcOp, fn, selection, symbolTable);
      if (positionSize <= 0)
        return failure();

      auto positionType =
          RankedTensorType::get({positionSize}, rewriter.getF64Type());

      auto q0 = enzyme::GetFlattenedSamplesFromTraceOp::create(
          rewriter, loc, positionType, originalTrace, selection);

      // 2. Compute initial potential energy U0 = -weight
      auto weight0 = enzyme::GetWeightFromTraceOp::create(
          rewriter, loc, F64TensorType, originalTrace);
      Value U0 = conditionalDump(rewriter, loc,
                                 arith::NegFOp::create(rewriter, loc, weight0),
                                 "NUTS: initial potential energy U0");

      auto zeroConst = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType, rewriter.getF64FloatAttr(0.0)));
      auto oneConst = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType, rewriter.getF64FloatAttr(1.0)));

      Value rng1;
      Value pInit;

      // 3. Sample initial momentum p0 ~ N(0, M) if M is provided,
      //    otherwise p0 ~ N(0, I)
      Value initialMomentum = mcmcOp.getInitialMomentum();
      if (initialMomentum) {
        pInit = initialMomentum;
        rng1 = rngState;
      } else {
        std::tie(pInit, rng1) =
            sampleMomentum(rewriter, loc, rngState, invMass, zeroConst,
                           oneConst, positionType);
      }

      auto halfConst = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType, rewriter.getF64FloatAttr(0.5)));

      // 4. Compute initial kinetic energy K0 = 0.5 * p^T * M^-1 * p
      Value K0 = conditionalDump(
          rewriter, loc,
          computeKineticEnergy(rewriter, loc, pInit, invMass, halfConst,
                               F64TensorType, positionType),
          "NUTS: initial kinetic energy K0");

      Value H0 = conditionalDump(rewriter, loc,
                                 arith::AddFOp::create(rewriter, loc, U0, K0),
                                 "NUTS: initial Hamiltonian H0");

      // 5. Compute initial gradient at q0
      auto gradSeedInit = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType, rewriter.getF64FloatAttr(1.0)));
      auto autodiffInit = enzyme::AutoDiffRegionOp::create(
          rewriter, loc, TypeRange{rng1.getType(), positionType},
          ValueRange{q0, gradSeedInit},
          rewriter.getArrayAttr({enzyme::ActivityAttr::get(
              rewriter.getContext(), enzyme::Activity::enzyme_active)}),
          rewriter.getArrayAttr(
              {enzyme::ActivityAttr::get(
                   rewriter.getContext(),
                   enzyme::Activity::enzyme_activenoneed), // U0 not needed here
               enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_const)}),
          rewriter.getI64IntegerAttr(1), rewriter.getBoolAttr(false), nullptr);

      Block *autodiffInitBlock = rewriter.createBlock(&autodiffInit.getBody());
      autodiffInitBlock->addArgument(positionType, loc);

      rewriter.setInsertionPointToStart(autodiffInitBlock);
      Value q0Arg = autodiffInitBlock->getArgument(0);

      SmallVector<Value> updateInputsInit;
      updateInputsInit.push_back(rng1);
      updateInputsInit.append(fnInputs.begin(), fnInputs.end());

      auto updateOpInit = enzyme::UpdateOp::create(
          rewriter, loc, TypeRange{traceType, F64TensorType, rng1.getType()},
          mcmcOp.getFnAttr(), updateInputsInit, originalTrace, q0Arg, selection,
          rewriter.getStringAttr(""));
      Value w0 = updateOpInit.getWeight();
      Value rng0_out = updateOpInit.getOutputRngState();
      Value U0_init = arith::NegFOp::create(rewriter, loc, w0);

      enzyme::YieldOp::create(rewriter, loc, ValueRange{U0_init, rng0_out});

      rewriter.setInsertionPointAfter(autodiffInit);
      Value rng0_final = autodiffInit.getResult(0);
      Value grad0 = autodiffInit.getResult(1);

      // 6. Set up NUTS doubling loop (outer)
      auto i1TensorType = RankedTensorType::get({}, rewriter.getI1Type());
      auto i64TensorType = RankedTensorType::get({}, rewriter.getI64Type());

      auto zeroI64 = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(0)));
      auto oneI64 = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(i64TensorType, rewriter.getI64IntegerAttr(1)));
      auto falseConst = arith::ConstantOp::create(
          rewriter, loc, i1TensorType,
          DenseElementsAttr::get(i1TensorType, rewriter.getBoolAttr(false)));
      auto trueConst = arith::ConstantOp::create(
          rewriter, loc, i1TensorType,
          DenseElementsAttr::get(i1TensorType, rewriter.getBoolAttr(true)));
      auto zeroWeight = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType, rewriter.getF64FloatAttr(0.0)));

      NUTSTree initialTree = {.q_left = q0,
                              .p_left = pInit,
                              .grad_left = grad0,
                              .q_right = q0,
                              .p_right = pInit,
                              .grad_right = grad0,
                              .q_proposal = q0,
                              .grad_proposal = grad0,
                              .U_proposal = U0,
                              .H_proposal = H0,
                              .depth = zeroI64,
                              .weight = zeroWeight,
                              .turning = falseConst,
                              .diverging = falseConst,
                              .sum_accept_probs = oneConst,
                              .num_proposals = oneI64,
                              .p_sum = pInit};

      auto maxTreeDepth = arith::ConstantOp::create(
          rewriter, loc, i64TensorType,
          DenseElementsAttr::get(
              i64TensorType,
              rewriter.getI64IntegerAttr(10))); // TODO: Make adjustable

      auto maxDeltaEnergy = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(
              F64TensorType,
              rewriter.getF64FloatAttr(1000.0))); // TODO: Make adjustable

      SmallVector<Type> whileLoopTypes = initialTree.getTypes();
      whileLoopTypes.push_back(rng0_final.getType());
      SmallVector<Value> whileLoopInitVals = initialTree.toValues();
      whileLoopInitVals.push_back(rng0_final);

      auto outerWhileOp = enzyme::WhileLoopOp::create(
          rewriter, loc, whileLoopTypes, whileLoopInitVals);

      Block *outerCondBlock =
          rewriter.createBlock(&outerWhileOp.getConditionRegion());
      for (auto type : whileLoopTypes)
        outerCondBlock->addArgument(type, loc);

      rewriter.setInsertionPointToStart(outerCondBlock);

      SmallVector<Value> treeArgs(outerCondBlock->getArguments().begin(),
                                  outerCondBlock->getArguments().begin() +
                                      NUTSTree::NUM_FIELDS);
      NUTSTree treeCond = NUTSTree::fromValues(treeArgs);
      Value rngCond = outerCondBlock->getArgument(NUTSTree::NUM_FIELDS);

      // Condition 6a: depth < maxTreeDepth
      auto notMaxDepth =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                treeCond.depth, maxTreeDepth);

      // Condition 6b: NOT turning
      auto notTurning =
          arith::XOrIOp::create(rewriter, loc, treeCond.turning, trueConst);

      // Condition 6c: NOT diverging
      auto notDiverging =
          arith::XOrIOp::create(rewriter, loc, treeCond.diverging, trueConst);

      auto continueDoublingCond = arith::AndIOp::create(
          rewriter, loc,
          arith::AndIOp::create(rewriter, loc, notMaxDepth.getResult(),
                                notTurning.getResult()),
          notDiverging.getResult());

      enzyme::YieldOp::create(rewriter, loc,
                              ValueRange{continueDoublingCond.getResult()});

      Block *outerBodyBlock =
          rewriter.createBlock(&outerWhileOp.getBodyRegion());
      for (auto type : whileLoopTypes)
        outerBodyBlock->addArgument(type, loc);

      rewriter.setInsertionPointToStart(outerBodyBlock);

      SmallVector<Value> treeBodyArgs(outerBodyBlock->getArguments().begin(),
                                      outerBodyBlock->getArguments().begin() +
                                          NUTSTree::NUM_FIELDS);
      NUTSTree treeBody = NUTSTree::fromValues(treeBodyArgs);
      Value rngBody = outerBodyBlock->getArgument(NUTSTree::NUM_FIELDS);

      // Body 6a: Sample direction (left or right)
      auto rngSplitOp = enzyme::RandomSplitOp::create(
          rewriter, loc, TypeRange{rngBody.getType(), rngBody.getType()},
          rngBody);
      Value rngDir = rngSplitOp.getResult(0);
      Value rngDoubling = rngSplitOp.getResult(1);

      auto randomDir = enzyme::RandomOp::create(
          rewriter, loc, TypeRange{rngDir.getType(), F64TensorType}, rngDir,
          zeroConst, oneConst,
          enzyme::RngDistributionAttr::get(rewriter.getContext(),
                                           enzyme::RngDistribution::UNIFORM));
      Value rngDir_out = randomDir.getOutputRngState();
      auto goingRight =
          arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                randomDir.getResult(), halfConst);

      // Body 6b: Build subtree with 2^(currentDepth + 1) proposals.
      Value currentDepth = treeBody.depth;
      auto depthForSubtree =
          arith::AddIOp::create(rewriter, loc, currentDepth, oneI64);

      // 7. Subtree building.
      SmallVector<Type> innerWhileTypes = treeBody.getTypes();
      innerWhileTypes.push_back(rngDoubling.getType());

      SmallVector<Value> innerWhileInitVals = treeBody.toValues();
      innerWhileInitVals.push_back(rngDoubling);

      auto innerWhileOp = enzyme::WhileLoopOp::create(
          rewriter, loc, innerWhileTypes, innerWhileInitVals);

      Block *innerCondBlock =
          rewriter.createBlock(&innerWhileOp.getConditionRegion());
      for (auto type : innerWhileTypes)
        innerCondBlock->addArgument(type, loc);

      rewriter.setInsertionPointToStart(innerCondBlock);

      SmallVector<Value> subtreeCondArgs(
          innerCondBlock->getArguments().begin(),
          innerCondBlock->getArguments().begin() + NUTSTree::NUM_FIELDS);
      NUTSTree subtreeCond = NUTSTree::fromValues(subtreeCondArgs);
      Value rngInnerCond = innerCondBlock->getArgument(NUTSTree::NUM_FIELDS);

      // Condition 7a: depth < depthForSubtree
      auto subtreeDepthOk =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                subtreeCond.depth, depthForSubtree);

      // Condition 7b: NOT turning
      auto subtreeNotTurning =
          arith::XOrIOp::create(rewriter, loc, subtreeCond.turning, trueConst);

      // Condition 7c: NOT diverging
      auto subtreeNotDiverging = arith::XOrIOp::create(
          rewriter, loc, subtreeCond.diverging, trueConst);

      auto continueSubtreeCond = arith::AndIOp::create(
          rewriter, loc,
          arith::AndIOp::create(rewriter, loc, subtreeDepthOk.getResult(),
                                subtreeNotTurning.getResult()),
          subtreeNotDiverging.getResult());

      enzyme::YieldOp::create(rewriter, loc,
                              ValueRange{continueSubtreeCond.getResult()});

      // Body 7a: Set up subtree building loop.
      Block *innerBodyBlock =
          rewriter.createBlock(&innerWhileOp.getBodyRegion());
      for (auto type : innerWhileTypes)
        innerBodyBlock->addArgument(type, loc);

      rewriter.setInsertionPointToStart(innerBodyBlock);

      SmallVector<Value> subtreeIterArgs(
          innerBodyBlock->getArguments().begin(),
          innerBodyBlock->getArguments().begin() + NUTSTree::NUM_FIELDS);
      NUTSTree subtreeIter = NUTSTree::fromValues(subtreeIterArgs);
      Value rngIter = innerBodyBlock->getArgument(NUTSTree::NUM_FIELDS);

      // Body 7a: Extract boundary from subtree based on direction
      auto goingRightBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc,
          RankedTensorType::get(positionType.getShape(), rewriter.getI1Type()),
          goingRight, rewriter.getDenseI64ArrayAttr(positionType.getShape()));

      Value leafQ = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast, subtreeIter.q_right,
          subtreeIter.q_left);
      Value leafP = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast, subtreeIter.p_right,
          subtreeIter.p_left);
      Value leafGrad = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast,
          subtreeIter.grad_right, subtreeIter.grad_left);

      // Body 7b: Prepare RNG states and adjust step size based on direction.
      auto rngSplit3 = enzyme::RandomSplitOp::create(
          rewriter, loc,
          TypeRange{rngIter.getType(), rngIter.getType(), rngIter.getType()},
          rngIter);
      Value rngLeaf = rngSplit3.getResult(0);
      Value rngCombine = rngSplit3.getResult(1);
      Value rngForNext = rngSplit3.getResult(2);

      auto negStepSize = arith::NegFOp::create(rewriter, loc, stepSize);
      Value eps = arith::SelectOp::create(rewriter, loc, F64TensorType,
                                          goingRight, stepSize, negStepSize);

      ArrayRef<int64_t> positionShape = positionType.getShape();
      auto stepSizeBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc, positionType, eps,
          rewriter.getDenseI64ArrayAttr(positionShape));
      auto halfStep = arith::MulFOp::create(rewriter, loc, halfConst, eps);
      auto halfStepBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc, positionType, halfStep,
          rewriter.getDenseI64ArrayAttr(positionShape));

      // Body 7c: Leapfrog integration.

      // Half step momentum: p_half = p - 0.5 * eps * gradient
      auto deltaP1 =
          arith::MulFOp::create(rewriter, loc, halfStepBroadcast, leafGrad);
      Value pHalf = arith::SubFOp::create(rewriter, loc, leafP, deltaP1);

      // Full step position: q_new = q + eps * M^-1 * p_half
      Value v =
          applyInverseMassMatrix(rewriter, loc, invMass, pHalf, positionType);

      auto deltaQ = arith::MulFOp::create(rewriter, loc, stepSizeBroadcast, v);
      Value qNew = arith::AddFOp::create(rewriter, loc, leafQ, deltaQ);

      // Compute potential energy and gradient at new position `qNew`.
      auto gradSeed = arith::ConstantOp::create(
          rewriter, loc, F64TensorType,
          DenseElementsAttr::get(F64TensorType, rewriter.getF64FloatAttr(1.0)));
      // We do need the NLL (a.k.a. potential energy) here, so `enzyme_active`
      // on the NLL.
      auto autodiffOp = enzyme::AutoDiffRegionOp::create(
          rewriter, loc,
          TypeRange{F64TensorType, rngLeaf.getType(), positionType},
          ValueRange{qNew, gradSeed},
          rewriter.getArrayAttr({enzyme::ActivityAttr::get(
              rewriter.getContext(), enzyme::Activity::enzyme_active)}),
          rewriter.getArrayAttr(
              {enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(rewriter.getContext(),
                                         enzyme::Activity::enzyme_const)}),
          rewriter.getI64IntegerAttr(1), rewriter.getBoolAttr(false), nullptr);

      Block *autodiffBlock = rewriter.createBlock(&autodiffOp.getBody());
      autodiffBlock->addArgument(positionType, loc);

      rewriter.setInsertionPointToStart(autodiffBlock);
      Value qNewArg = autodiffBlock->getArgument(0);

      SmallVector<Value> updateInputs;
      updateInputs.push_back(rngLeaf);
      updateInputs.append(fnInputs.begin(), fnInputs.end());

      auto updateResult = enzyme::UpdateOp::create(
          rewriter, loc, TypeRange{traceType, F64TensorType, rngLeaf.getType()},
          mcmcOp.getFnAttr(), updateInputs, originalTrace, qNewArg, selection,
          rewriter.getStringAttr(""));

      Value todiff =
          arith::NegFOp::create(rewriter, loc, updateResult.getWeight());

      enzyme::YieldOp::create(
          rewriter, loc, ValueRange{todiff, updateResult.getOutputRngState()});

      rewriter.setInsertionPointAfter(autodiffOp);

      // AutodiffRegionOp returns: (UNew, RNG, dUNew/dqNew)
      Value UNew = autodiffOp.getResult(0);
      Value gradNew = autodiffOp.getResult(2);

      // Half step momentum: p_new = p_half - 0.5 * eps * grad_new
      auto deltaP2 =
          arith::MulFOp::create(rewriter, loc, halfStepBroadcast, gradNew);
      Value pNew = arith::SubFOp::create(rewriter, loc, pHalf, deltaP2);

      // Body 7d: Compute kinetic energy.
      Value KNew = computeKineticEnergy(rewriter, loc, pNew, invMass, halfConst,
                                        F64TensorType, positionType);
      Value ENew = arith::AddFOp::create(rewriter, loc, UNew, KNew);

      // Body 7e: Various checks.
      auto deltaE = arith::SubFOp::create(rewriter, loc, ENew, H0);

      Value leafDiverging = arith::CmpFOp::create(
          rewriter, loc, arith::CmpFPredicate::OGT, deltaE, maxDeltaEnergy);
      auto treeWeight = arith::NegFOp::create(rewriter, loc, deltaE);

      // Body 7f: Compute acceptance probability.
      auto acceptProbRaw = math::ExpOp::create(rewriter, loc, treeWeight);
      auto acceptProb =
          arith::MinimumFOp::create(rewriter, loc, acceptProbRaw, oneConst);

      // Body 7g: Create leaf tree state.
      NUTSTree newLeaf = {.q_left = qNew,
                          .p_left = pNew,
                          .grad_left = gradNew,
                          .q_right = qNew,
                          .p_right = pNew,
                          .grad_right = gradNew,
                          .q_proposal = qNew,
                          .grad_proposal = gradNew,
                          .U_proposal = UNew,
                          .H_proposal = ENew,
                          .depth = zeroI64,
                          .weight = treeWeight,
                          .turning = falseConst,
                          .diverging = leafDiverging,
                          .sum_accept_probs = acceptProb,
                          .num_proposals = oneI64,
                          .p_sum = pNew};

      // Body 7h: Combine new leaf with the current subtree.
      // 7h.1: Update boundaries based on direction.
      Value qLeft = arith::SelectOp::create(rewriter, loc, positionType,
                                            goingRightBroadcast,
                                            subtreeIter.q_left, newLeaf.q_left);
      Value pLeft = arith::SelectOp::create(rewriter, loc, positionType,
                                            goingRightBroadcast,
                                            subtreeIter.p_left, newLeaf.p_left);
      Value gradLeft = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast,
          subtreeIter.grad_left, newLeaf.grad_left);

      Value qRight = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast, newLeaf.q_right,
          subtreeIter.q_right);
      Value pRight = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast, newLeaf.p_right,
          subtreeIter.p_right);
      Value gradRight = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightBroadcast, newLeaf.grad_right,
          subtreeIter.grad_right);

      // 7h.2: Combine weights using log_add_exp.
      Value combinedWeight = enzyme::LogAddExpOp::create(
          rewriter, loc, F64TensorType, subtreeIter.weight, newLeaf.weight);

      Value weightDiffCombine = arith::SubFOp::create(
          rewriter, loc, newLeaf.weight, subtreeIter.weight);
      Value acceptProbCombine = createSigmoid(rewriter, loc, weightDiffCombine);

      // 7h.3: Select proposal with multinomial sampling.
      auto randomOpCombine = enzyme::RandomOp::create(
          rewriter, loc, TypeRange{rngCombine.getType(), F64TensorType},
          rngCombine, zeroConst, oneConst,
          enzyme::RngDistributionAttr::get(rewriter.getContext(),
                                           enzyme::RngDistribution::UNIFORM));
      Value rngAfterCombine = randomOpCombine.getOutputRngState();
      Value uniformSampleCombine = randomOpCombine.getResult();

      auto acceptNew =
          arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT,
                                uniformSampleCombine, acceptProbCombine);

      auto acceptNewBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc,
          RankedTensorType::get(positionType.getShape(), rewriter.getI1Type()),
          acceptNew, rewriter.getDenseI64ArrayAttr(positionType.getShape()));

      Value qProposal = arith::SelectOp::create(
          rewriter, loc, positionType, acceptNewBroadcast, newLeaf.q_proposal,
          subtreeIter.q_proposal);
      Value gradProposal = arith::SelectOp::create(
          rewriter, loc, positionType, acceptNewBroadcast,
          newLeaf.grad_proposal, subtreeIter.grad_proposal);

      Value UProposal =
          arith::SelectOp::create(rewriter, loc, F64TensorType, acceptNew,
                                  newLeaf.U_proposal, subtreeIter.U_proposal);
      Value EProposal =
          arith::SelectOp::create(rewriter, loc, F64TensorType, acceptNew,
                                  newLeaf.H_proposal, subtreeIter.H_proposal);

      // 7h.4: Update metadata.
      Value combinedDepth =
          arith::AddIOp::create(rewriter, loc, subtreeIter.depth, oneI64);
      Value combinedTurning = arith::OrIOp::create(
          rewriter, loc, subtreeIter.turning, newLeaf.turning);
      Value combinedDiverging = arith::OrIOp::create(
          rewriter, loc, subtreeIter.diverging, newLeaf.diverging);
      Value sumAcceptProbs =
          arith::AddFOp::create(rewriter, loc, subtreeIter.sum_accept_probs,
                                newLeaf.sum_accept_probs);
      Value numProposals = arith::AddIOp::create(
          rewriter, loc, subtreeIter.num_proposals, newLeaf.num_proposals);
      Value pSum = arith::AddFOp::create(rewriter, loc, subtreeIter.p_sum,
                                         newLeaf.p_sum);
      NUTSTree updatedSubtree = {.q_left = qLeft,
                                 .p_left = pLeft,
                                 .grad_left = gradLeft,
                                 .q_right = qRight,
                                 .p_right = pRight,
                                 .grad_right = gradRight,
                                 .q_proposal = qProposal,
                                 .grad_proposal = gradProposal,
                                 .U_proposal = UProposal,
                                 .H_proposal = EProposal,
                                 .depth = combinedDepth,
                                 .weight = combinedWeight,
                                 .turning = combinedTurning,
                                 .diverging = combinedDiverging,
                                 .sum_accept_probs = sumAcceptProbs,
                                 .num_proposals = numProposals,
                                 .p_sum = pSum};

      // Body 7i: Check and update turning flag.
      updatedSubtree.turning = checkTurning(
          rewriter, loc, invMass, updatedSubtree.p_left, updatedSubtree.p_right,
          updatedSubtree.p_sum, zeroConst, F64TensorType, positionType);

      SmallVector<Value> yieldVals = updatedSubtree.toValues();
      yieldVals.push_back(rngIter);
      enzyme::YieldOp::create(rewriter, loc, yieldVals);

      // 8. Combine subtree with main tree.
      rewriter.setInsertionPointAfter(innerWhileOp);
      SmallVector<Value> subtreeValues(innerWhileOp.getResults().begin(),
                                       innerWhileOp.getResults().begin() +
                                           NUTSTree::NUM_FIELDS);
      NUTSTree subtree = NUTSTree::fromValues(subtreeValues);
      Value rngAfterBuild = innerWhileOp.getResult(NUTSTree::NUM_FIELDS);

      auto rngSplitAfterBuild = enzyme::RandomSplitOp::create(
          rewriter, loc,
          TypeRange{rngAfterBuild.getType(), rngAfterBuild.getType()},
          rngAfterBuild);
      Value rngTrans = rngSplitAfterBuild.getResult(0);
      Value rngNext = rngSplitAfterBuild.getResult(1);

      // 8a. Update boundaries based on direction.
      auto goingRightMainBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc,
          RankedTensorType::get(positionType.getShape(), rewriter.getI1Type()),
          goingRight, rewriter.getDenseI64ArrayAttr(positionType.getShape()));

      Value qLeftMain = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightMainBroadcast, treeBody.q_left,
          subtree.q_left);
      Value pLeftMain = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightMainBroadcast, treeBody.p_left,
          subtree.p_left);
      Value gradLeftMain = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightMainBroadcast,
          treeBody.grad_left, subtree.grad_left);
      Value qRightMain = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightMainBroadcast, subtree.q_right,
          treeBody.q_right);
      Value pRightMain = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightMainBroadcast, subtree.p_right,
          treeBody.p_right);
      Value gradRightMain = arith::SelectOp::create(
          rewriter, loc, positionType, goingRightMainBroadcast,
          subtree.grad_right, treeBody.grad_right);

      // 8b. Combine weights using log_add_exp.
      Value combinedWeightMain = enzyme::LogAddExpOp::create(
          rewriter, loc, F64TensorType, treeBody.weight, subtree.weight);

      // 8c. Proposal selection via multinomial sampling.
      Value weightDiffMain =
          arith::SubFOp::create(rewriter, loc, subtree.weight, treeBody.weight);
      Value acceptProbMainRaw = createSigmoid(rewriter, loc, weightDiffMain);

      // 8d. Zero accept probability to 0 if new tree is turning or diverging.
      Value acceptProbMain = arith::SelectOp::create(
          rewriter, loc, F64TensorType,
          arith::OrIOp::create(rewriter, loc, subtree.turning,
                               subtree.diverging),
          zeroConst, acceptProbMainRaw);

      // 8e. Compute acceptance probability on new proposal.
      auto randomOpMain = enzyme::RandomOp::create(
          rewriter, loc, TypeRange{rngTrans.getType(), F64TensorType}, rngTrans,
          zeroConst, oneConst,
          enzyme::RngDistributionAttr::get(rewriter.getContext(),
                                           enzyme::RngDistribution::UNIFORM));
      Value rngAfterCombineFinal = randomOpMain.getOutputRngState();
      Value uniformSampleMain = randomOpMain.getResult();

      auto acceptNewMain =
          arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT,
                                uniformSampleMain, acceptProbMain);

      // 8f. Select proposal components.
      auto acceptNewMainBroadcast = enzyme::BroadcastOp::create(
          rewriter, loc,
          RankedTensorType::get(positionType.getShape(), rewriter.getI1Type()),
          acceptNewMain,
          rewriter.getDenseI64ArrayAttr(positionType.getShape()));

      Value qProposalMain = arith::SelectOp::create(
          rewriter, loc, positionType, acceptNewMainBroadcast,
          subtree.q_proposal, treeBody.q_proposal);
      Value gradProposalMain = arith::SelectOp::create(
          rewriter, loc, positionType, acceptNewMainBroadcast,
          subtree.grad_proposal, treeBody.grad_proposal);

      Value UProposalMain =
          arith::SelectOp::create(rewriter, loc, F64TensorType, acceptNewMain,
                                  subtree.U_proposal, treeBody.U_proposal);
      Value EProposalMain =
          arith::SelectOp::create(rewriter, loc, F64TensorType, acceptNewMain,
                                  subtree.H_proposal, treeBody.H_proposal);
      Value combinedDepthMain =
          arith::AddIOp::create(rewriter, loc, treeBody.depth, oneI64);
      Value combinedTurningMain = arith::OrIOp::create(
          rewriter, loc, treeBody.turning, subtree.turning);
      Value combinedDivergingMain = arith::OrIOp::create(
          rewriter, loc, treeBody.diverging, subtree.diverging);
      Value sumAcceptProbsMain = arith::AddFOp::create(
          rewriter, loc, treeBody.sum_accept_probs, subtree.sum_accept_probs);
      Value numProposalsMain = arith::AddIOp::create(
          rewriter, loc, treeBody.num_proposals, subtree.num_proposals);
      Value pSumMain =
          arith::AddFOp::create(rewriter, loc, treeBody.p_sum, subtree.p_sum);

      NUTSTree combinedTree = {.q_left = qLeftMain,
                               .p_left = pLeftMain,
                               .grad_left = gradLeftMain,
                               .q_right = qRightMain,
                               .p_right = pRightMain,
                               .grad_right = gradRightMain,
                               .q_proposal = qProposalMain,
                               .grad_proposal = gradProposalMain,
                               .U_proposal = UProposalMain,
                               .H_proposal = EProposalMain,
                               .depth = combinedDepthMain,
                               .weight = combinedWeightMain,
                               .turning = combinedTurningMain,
                               .diverging = combinedDivergingMain,
                               .sum_accept_probs = sumAcceptProbsMain,
                               .num_proposals = numProposalsMain,
                               .p_sum = pSumMain};

      // 8g. Yield combined tree.
      SmallVector<Value> outerYieldVals = combinedTree.toValues();
      outerYieldVals.push_back(rngNext);
      enzyme::YieldOp::create(rewriter, loc, outerYieldVals);

      rewriter.setInsertionPointAfter(outerWhileOp);

      // 9. Extract final proposal from combined tree.
      SmallVector<Value> finalTreeValues(outerWhileOp.getResults().begin(),
                                         outerWhileOp.getResults().begin() +
                                             NUTSTree::NUM_FIELDS);
      NUTSTree finalTree = NUTSTree::fromValues(finalTreeValues);
      Value rngFinal = outerWhileOp.getResult(NUTSTree::NUM_FIELDS);

      Value qFinal = finalTree.q_proposal;

      // 10. Generate final trace at proposed position.
      SmallVector<Value> finalUpdateInputs;
      finalUpdateInputs.push_back(rngFinal);
      finalUpdateInputs.append(fnInputs.begin(), fnInputs.end());

      auto finalUpdateOp = enzyme::UpdateOp::create(
          rewriter, loc,
          TypeRange{traceType, F64TensorType, rngFinal.getType()},
          mcmcOp.getFnAttr(), finalUpdateInputs, originalTrace, qFinal,
          selection, rewriter.getStringAttr(""));
      Value finalTrace = finalUpdateOp.getUpdatedTrace();
      Value rngAfterUpdate = finalUpdateOp.getOutputRngState();

      auto acceptedTensor = arith::ConstantOp::create(
          rewriter, loc, i1TensorType,
          DenseElementsAttr::get(i1TensorType, rewriter.getBoolAttr(true)));

      rewriter.replaceOp(mcmcOp, {finalTrace, acceptedTensor, rngAfterUpdate});

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
