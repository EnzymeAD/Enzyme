
#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

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
    return failure();
  }
};

struct ConcatOpConversion : public OpConversionPattern<enzyme::ConcatOp> {
  using OpConversionPattern<enzyme::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
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
