
#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "enzyme-legalize-batch"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LEGALIZEBATCHINGPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

struct ExtractOpConversion : public OpConversionPattern<enzyme::ExtractOp> {
  using OpConversionPattern<enzyme::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // filter based on out type
    auto outTy = op.getOutput().getType();
    if (auto outTensorTy = dyn_cast<TensorType>(outTy)) {
      auto outRankTy = dyn_cast<RankedTensorType>(outTy);
      auto rank = outRankTy.getRank();

      // Offsets : [index, 0, 0 ...]
      // Sizes : [1, out_dim1, out_dim2 ...]
      // Strides : [1,1,1,....]
      SmallVector<OpFoldResult> offset = {op.getIndex()},
                                sizes = {rewriter.getI64IntegerAttr(1)},
                                strides(rank + 1,
                                        rewriter.getI64IntegerAttr(1));
      offset.append(
          SmallVector<OpFoldResult>(rank, rewriter.getI64IntegerAttr(0)));

      for (auto dim : outRankTy.getShape()) {
        sizes.push_back(rewriter.getI64IntegerAttr(dim));
      }

      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          op, outRankTy, op.getInput(), offset, sizes, strides);

      return success();
    } else if (outTy.isIntOrIndexOrFloat()) {

      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
          op, op->getResultTypes(), op.getInput(), op.getIndex());
      return success();
    } else {
      // unsupported type
      // TODO: handle memrefs

      return failure();
    }
  }
};

struct ConcatOpConversion : public OpConversionPattern<enzyme::ConcatOp> {
  using OpConversionPattern<enzyme::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // filter based on out type
    SmallVector<Value> in_args = op.getInputs();
    auto inTy = in_args.front().getType();
    if (auto valTensorTy = dyn_cast<TensorType>(inTy)) {
      rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, op->getResultTypes(), 0,
                                                    in_args);
      return success();
    } else if (inTy.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
          op, op->getResultTypes(), in_args);
      return success();
    } else {
      // unsupported type
      // TODO: handle memrefs
      return failure();
    }
  }
};

struct LegalizeBatchingPass
    : public enzyme::impl::LegalizeBatchingPassBase<LegalizeBatchingPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // NOTE: May need a typeConverter here when lowering batched memrefs.

    patterns.add<ConcatOpConversion, ExtractOpConversion>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<enzyme::EnzymeDialect>();
    target.addIllegalOp<enzyme::ConcatOp, enzyme::ExtractOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace
