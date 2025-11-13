#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "enzyme-batch-to-tensor"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEBATCHTOTENSORPASS
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
      SmallVector<OpFoldResult> offset = {op.getIndexAttr()},
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
      // ExtractOp expects only index type arg
      Value indexOp =
          arith::ConstantIndexOp::create(rewriter, op->getLoc(), op.getIndex());
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, op->getResultTypes(),
                                                     op.getInput(), indexOp);
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
    SmallVector<Value> inputs = op.getInputs();
    if (inputs.empty())
      return failure();

    auto firstInTy = inputs.front().getType();

    if (auto firstRankTy = dyn_cast<RankedTensorType>(firstInTy)) {

      // rank has to be the same for all inputs
      auto rank = firstRankTy.getRank();

      // Build the reassociation map attribute for expand_shape
      SmallVector<Attribute> reassociationMap;

      if (rank > 0) {
        reassociationMap.push_back(rewriter.getI64ArrayAttr({0, 1}));
      }

      for (auto i = 1; i < rank; ++i) {
        // src dim 'i' goes to dest dim 'i+1'
        reassociationMap.push_back(rewriter.getI64ArrayAttr({i + 1}));
      }

      ArrayAttr reassociationAttr =
          ArrayAttr::get(rewriter.getContext(), reassociationMap);

      // tensor.expand_shape for every input argument
      SmallVector<Value> expandedInputs;

      for (Value in : inputs) {
        auto inRankTy = cast<RankedTensorType>(in.getType());
        auto inShape = inRankTy.getShape();
        SmallVector<Value> outDynamicDims;

        SmallVector<int64_t> newInShape = {1};
        newInShape.append(inShape.begin(), inShape.end());

        for (auto i = 0; i < rank; ++i) {
          if (inRankTy.isDynamicDim(i)) {
            // extract dynamic dim
            Value dynIdx =
                arith::ConstantIndexOp::create(rewriter, op->getLoc(), i);
            Value dynVal =
                tensor::DimOp::create(rewriter, op->getLoc(), in, dynIdx);
            outDynamicDims.push_back(dynVal);
          }
        }

        auto newInTy = inRankTy.clone(newInShape);
        auto outStaticDimAttr =
            rewriter.getDenseI64ArrayAttr(newInTy.getShape());

        Value newInput = tensor::ExpandShapeOp::create(
            rewriter, op->getLoc(), newInTy, in, reassociationAttr,
            outDynamicDims, outStaticDimAttr);

        expandedInputs.push_back(newInput);
      }

      rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, op->getResultTypes(),
                                                    /*dim*/ 0, expandedInputs);
      return success();
    } else if (firstInTy.isIntOrIndexOrFloat()) {
      rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
          op, op->getResultTypes(), inputs);
      return success();
    } else {
      // unsupported type
      // TODO: handle memrefs
      return failure();
    }
  }
};

struct EnzymeBatchToTensorPass
    : public enzyme::impl::EnzymeBatchToTensorPassBase<
          EnzymeBatchToTensorPass> {
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
