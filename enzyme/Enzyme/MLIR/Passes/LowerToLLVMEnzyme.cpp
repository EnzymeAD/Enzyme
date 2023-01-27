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
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
void updatePrimalFunc(FunctionOpInterface fn,
                      ConversionPatternRewriter &rewriter) {
  // Update function signature, converting results to destination-passing-style
  // pointers.
  Block &primalEntryBlock = fn.front();
  PatternRewriter::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPointToStart(&primalEntryBlock);
  SmallVector<Type> newArgumentTypes;
  newArgumentTypes.reserve(fn.getNumArguments() + fn.getNumResults());
  for (BlockArgument arg : primalEntryBlock.getArguments()) {
    if (arg.getType().isa<FloatType>()) {
      auto newType = LLVM::LLVMPointerType::get(arg.getType());
      newArgumentTypes.push_back(newType);
      arg.setType(newType);
      auto loaded = rewriter.create<LLVM::LoadOp>(arg.getLoc(), arg);
      rewriter.replaceUsesOfBlockArgument(arg, loaded);
    } else {
      newArgumentTypes.push_back(arg.getType());
    }
  }

  SmallVector<Value> dpsResults;
  dpsResults.reserve(fn.getNumResults());
  for (Type resType : fn.getResultTypes()) {
    if (resType.isa<FloatType>()) {
      auto newType = LLVM::LLVMPointerType::get(resType);
      newArgumentTypes.push_back(newType);
      dpsResults.push_back(primalEntryBlock.addArgument(newType, fn.getLoc()));
    }
  }

  auto newFuncType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(fn.getContext()), newArgumentTypes);
  fn.setType(newFuncType);

  // Replace return operands with DPS stores
  fn.walk([&](LLVM::ReturnOp returnOp) {
    PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.setInsertionPoint(returnOp);
    for (const auto &[returnOperand, dpsResult] :
         llvm::zip(returnOp.getOperands(), dpsResults)) {
      rewriter.create<LLVM::StoreOp>(returnOp.getLoc(), returnOperand,
                                     dpsResult);
    }
    rewriter.create<LLVM::ReturnOp>(returnOp.getLoc(),
                                    /*operands=*/ValueRange{});
    rewriter.eraseOp(returnOp);
  });
}

struct DiffOpLowering : public OpConversionPattern<enzyme::DiffOp> {
  using OpConversionPattern<enzyme::DiffOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::DiffOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    FlatSymbolRefAttr autodiffDecl =
        getOrInsertAutodiffDecl(moduleOp, rewriter);
    auto fn = cast<FunctionOpInterface>(moduleOp.lookupSymbol(op.getFn()));
    if (fn.getNumResults() > 1) {
      op.emitError() << "Expected primal function to have at most one result";
      return failure();
    }

    Value one =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
    SmallVector<Value> operands, shadows;

    // We convert scalars to pointers to support LLVM Enzyme's API with
    // multiple inputs, so we need to convert the pointers back at the end.
    SmallVector<bool> convertMask;
    // The first operand is a function pointer. We use a null placeholder value
    // because we will later mutate the function.
    operands.push_back(Value());

    size_t operandIdx = 0;
    for (Type argType : fn.getArgumentTypes()) {
      Value argument = op.getInputs()[operandIdx++];
      bool isActiveScalar = argType.isa<FloatType>();
      convertMask.push_back(isActiveScalar);
      if (isActiveScalar) {
        auto argPtrType = LLVM::LLVMPointerType::get(argType);
        Value shadow = op.getInputs()[operandIdx++];
        Value argSpace = rewriter.create<LLVM::AllocaOp>(loc, argPtrType, one);
        Value argShadow = rewriter.create<LLVM::AllocaOp>(loc, argPtrType, one);
        rewriter.create<LLVM::StoreOp>(loc, argument, argSpace);
        rewriter.create<LLVM::StoreOp>(loc, shadow, argShadow);

        operands.push_back(argSpace);
        operands.push_back(argShadow);
        shadows.push_back(argShadow);
      } else if (argType.isa<MemRefType>()) {
        llvm_unreachable("memref type not yet supported");
      } else {
        // Generic pointer type
        Value shadow = op.getInputs()[operandIdx++];
        operands.push_back(argument);
        operands.push_back(shadow);
        shadows.push_back(shadow);
      }
    }

    // TODO: Remove assertion in favour of an enzyme.diff verifier when the API
    // is more stable.
    assert(operandIdx == op.getInputs().size() - 1 &&
           "Mismatched # of inputs to enzyme.diff");
    Value seedGradient = op.getInputs().back();
    for (Type outType : fn.getResultTypes()) {
      if (outType.isa<FloatType>()) {
        auto outPtrType = LLVM::LLVMPointerType::get(outType);
        Value outSpace = rewriter.create<LLVM::AllocaOp>(loc, outPtrType, one);
        Value outShadow = rewriter.create<LLVM::AllocaOp>(loc, outPtrType, one);
        rewriter.create<LLVM::StoreOp>(loc, seedGradient, outShadow);
        operands.push_back(outSpace);
        operands.push_back(outShadow);
      }
    }

    updatePrimalFunc(fn, rewriter);
    // Fill in our placeholder now that the function is updated.
    operands[0] = getAddressOfPrimal(loc, fn, rewriter);
    rewriter.create<LLVM::CallOp>(loc, /*resultTypes=*/TypeRange{},
                                  autodiffDecl, operands);

    SmallVector<Value> newResults;
    for (const auto &[shadow, converted] : llvm::zip(shadows, convertMask)) {
      if (converted) {
        newResults.push_back(rewriter.create<LLVM::LoadOp>(loc, shadow));
      } else {
        newResults.push_back(shadow);
      }
    }
    rewriter.replaceOp(op, newResults);
    return success();
  }

private:
  Value getAddressOfPrimal(Location loc, FunctionOpInterface fn,
                           OpBuilder &b) const {
    auto voidPtrType = LLVM::LLVMPointerType::get(b.getI8Type());
    auto primalFuncPtrType = LLVM::LLVMPointerType::get(
        getTypeConverter()->convertType(fn.getFunctionType()));
    Value primalAddr =
        b.create<LLVM::AddressOfOp>(loc, primalFuncPtrType, fn.getName());
    return b.create<LLVM::BitcastOp>(loc, voidPtrType, primalAddr);
  }

  static FlatSymbolRefAttr getOrInsertAutodiffDecl(ModuleOp moduleOp,
                                                   OpBuilder &b) {
    MLIRContext *context = b.getContext();
    if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("__enzyme_autodiff"))
      return SymbolRefAttr::get(context, "__enzyme_autodiff");

    auto voidPtrType = LLVM::LLVMPointerType::get(b.getI8Type());

    auto llvmFnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), voidPtrType, /*isVarArg=*/true);
    OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(moduleOp.getBody());
    b.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), "__enzyme_autodiff",
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
    populateFuncToLLVMFuncOpConversionPattern(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addIllegalOp<enzyme::DiffOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();

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
