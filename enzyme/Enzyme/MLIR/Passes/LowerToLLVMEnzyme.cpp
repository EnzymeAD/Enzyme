//===- EnzymeToMemRef.cpp - Lower custom Enzyme operations ------------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower enzyme.diff calls to LLVM-Enzyme calls
// in the LLVM dialect.
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;

namespace {
struct DiffOpLowering : public OpConversionPattern<enzyme::DiffOp> {
  using OpConversionPattern<enzyme::DiffOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::DiffOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr autodiffDecl =
        getOrInsertAutodiffDecl(rewriter, moduleOp);
    return failure();
  }

private:
  static FlatSymbolRefAttr getOrInsertEnzymeConstDecl(OpBuilder &builder,
                                                      ModuleOp moduleOp) {
    MLIRContext *context = builder.getContext();
    if (moduleOp.lookupSymbol<LLVM::GlobalOp>("enzyme_const"))
      return SymbolRefAttr::get(context, "enzyme_const");

    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    IntegerType shortTy = builder.getI8Type();
    builder.create<LLVM::GlobalOp>(moduleOp.getLoc(), shortTy,
                                   /*isConstant=*/true, LLVM::Linkage::Linkonce,
                                   "enzyme_const",
                                   IntegerAttr::get(shortTy, 0));
    return SymbolRefAttr::get(context, "enzyme_const");
  }

  static FlatSymbolRefAttr getOrInsertAutodiffDecl(OpBuilder &builder,
                                                   ModuleOp moduleOp) {
    MLIRContext *context = builder.getContext();
    if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__enzyme_autodiff"))
      return SymbolRefAttr::get(context, "__enzyme_autodiff");

    auto voidPtrType = LLVM::LLVMPointerType::get(builder.getI8Type());

    auto llvmFnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), voidPtrType, /*isVarArg=*/true);
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    builder.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), "__enzyme_autodiff",
                                     llvmFnType);
    return SymbolRefAttr::get(context, "__enzyme_autodiff");
  }
};

struct LowerToLLVMEnzymePass
    : public enzyme::LowerToLLVMEnzymePassBase<LowerToLLVMEnzymePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    LLVMTypeConverter typeConverter(context);
    patterns.add<DiffOpLowering>(typeConverter, context);

    ConversionTarget target(*context);
    target.addIllegalOp<enzyme::DiffOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createLowerToLLVMEnzymePass() {
  return std::make_unique<LowerToLLVMEnzymePass>();
}
} // namespace enzyme
} // namespace mlir
