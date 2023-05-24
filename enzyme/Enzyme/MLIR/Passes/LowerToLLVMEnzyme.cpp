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
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#if LLVM_VERSION_MAJOR >= 16
#include <optional>
#endif
using namespace mlir;
using llvm::errs;

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
      arg.replaceAllUsesExcept(loaded, loaded);
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

void convertMemRefArgument(Location loc, Value primal,
#if LLVM_VERSION_MAJOR >= 16
                           std::optional<Value> shadow,
#else
                           llvm::Optional<Value> shadow,
#endif
                           Value enzyme_const_addr, int64_t rank, OpBuilder &b,
                           SmallVectorImpl<Value> &operands) {
  MemRefDescriptor memrefPrimal(primal);
  // Mark the allocated pointer as constant
  operands.push_back(enzyme_const_addr);
  operands.push_back(memrefPrimal.allocatedPtr(b, loc));

  if (shadow.has_value()) {
    MemRefDescriptor memrefShadow(shadow.value());
    // Shadow aligned pointer follows the primal aligned pointer
    operands.push_back(memrefPrimal.alignedPtr(b, loc));
    operands.push_back(memrefShadow.alignedPtr(b, loc));
  } else {
    operands.push_back(enzyme_const_addr);
    operands.push_back(memrefPrimal.alignedPtr(b, loc));
  }

  operands.push_back(memrefPrimal.offset(b, loc));
  for (int64_t pos = 0; pos < rank; ++pos)
    operands.push_back(memrefPrimal.size(b, loc, pos));
  for (int64_t pos = 0; pos < rank; ++pos)
    operands.push_back(memrefPrimal.stride(b, loc, pos));
}

struct DiffOpLowering : public OpConversionPattern<enzyme::AutoDiffOp> {
  using OpConversionPattern<enzyme::AutoDiffOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(enzyme::AutoDiffOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    FlatSymbolRefAttr autodiffDecl =
        getOrInsertAutodiffDecl(moduleOp, rewriter);
    FlatSymbolRefAttr const_global =
        getOrInsertEnzymeConstDecl(moduleOp, rewriter);
    auto voidPtrType = LLVM::LLVMPointerType::get(rewriter.getI8Type());
    auto enzyme_const_addr =
        rewriter.create<LLVM::AddressOfOp>(loc, voidPtrType, const_global);

    auto fn = cast<FunctionOpInterface>(moduleOp.lookupSymbol(op.getFn()));
    Value one =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
    SmallVector<Value> operands, shadows;

    // We convert scalars to pointers to support LLVM Enzyme's API with
    // multiple inputs, so we need to convert the pointers back at the end.
    SmallVector<bool> convertMask;
    // The first operand is a function pointer. We use a null placeholder value
    // because we might mutate the function type.
    operands.push_back(Value());

    size_t operandIdx = 0;
    for (auto activity :
         op.getActivity()
             .getAsValueRange<enzyme::ActivityAttr, enzyme::Activity>()) {
      Value arg = op.getInputs()[operandIdx++];
      bool isActiveScalar = activity != enzyme::Activity::enzyme_const &&
                            arg.getType().isa<FloatType>();
      convertMask.push_back(isActiveScalar);
      switch (activity) {
      case enzyme::Activity::enzyme_dup:
      case enzyme::Activity::enzyme_dupnoneed:
        if (arg.getType().isa<FloatType>()) {
          auto argPtrType = LLVM::LLVMPointerType::get(arg.getType());
          Value shadow = op.getInputs()[operandIdx++];
          Value argSpace =
              rewriter.create<LLVM::AllocaOp>(loc, argPtrType, one);
          Value argShadow =
              rewriter.create<LLVM::AllocaOp>(loc, argPtrType, one);
          rewriter.create<LLVM::StoreOp>(loc, arg, argSpace);
          rewriter.create<LLVM::StoreOp>(loc, shadow, argShadow);

          operands.push_back(argSpace);
          operands.push_back(argShadow);
          shadows.push_back(argShadow);
        } else if (auto memrefType = arg.getType().dyn_cast<MemRefType>()) {
          Value casted = adaptor.getInputs()[operandIdx - 1];
          Value shadowCasted = adaptor.getInputs()[operandIdx];
          shadows.push_back(shadowCasted);
          operandIdx++;
          convertMemRefArgument(arg.getLoc(), casted, shadowCasted,
                                enzyme_const_addr, memrefType.getRank(),
                                rewriter, operands);
        }
        break;
      case enzyme::Activity::enzyme_out:
        // TODO: There are API differences between MLIR Enzyme and LLVM Enzyme
        // that make enzyme_out not well-defined.
        op.emitError() << "'enzyme_out' activity not supported";
        return failure();
      case enzyme::Activity::enzyme_const:
        operands.push_back(arg);
        break;
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

  static FlatSymbolRefAttr getOrInsertEnzymeConstDecl(ModuleOp moduleOp,
                                                      OpBuilder &b) {
    MLIRContext *context = b.getContext();
    if (moduleOp.lookupSymbol<LLVM::GlobalOp>("enzyme_const")) {
      return SymbolRefAttr::get(context, "enzyme_const");
    }
    PatternRewriter::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(moduleOp.getBody());
    auto shortTy = b.getI8Type();
    b.create<LLVM::GlobalOp>(moduleOp.getLoc(), shortTy,
                             /*isConstant=*/true, LLVM::Linkage::Linkonce,
                             "enzyme_const", IntegerAttr::get(shortTy, 0));
    return SymbolRefAttr::get(context, "enzyme_const");
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
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMFuncOpConversionPattern(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addIllegalOp<enzyme::AutoDiffOp>();
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
