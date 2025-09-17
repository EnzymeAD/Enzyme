#ifndef ENZYME_BATCH_PASS_H
#define ENZYME_BATCH_PASS_H

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace enzyme {
namespace batchutils {

struct BatchCacheKey {
  FunctionOpInterface function;
  SmallVector<int64_t> batchSizes;

  // for use in std::map:
  bool operator<(const BatchCacheKey &other) const {
    if (const_cast<FunctionOpInterface &>(function).getName() !=
        const_cast<FunctionOpInterface &>(other.function).getName())
      return const_cast<FunctionOpInterface &>(function).getName() <
             const_cast<FunctionOpInterface &>(other.function).getName();
    return batchSizes < other.batchSizes;
  }
};

mlir::TensorType applyBatchSizes(mlir::Type Ty,
                                 llvm::ArrayRef<int64_t> batchSizes);

FunctionOpInterface batchCloneFunction(
    OpBuilder &builder, FunctionOpInterface F, Twine name,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

void batchCloneRegion(
    OpBuilder &builder, Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

LogicalResult handleCallOp(
    func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

template <typename T>
FunctionOpInterface batchOperationWithoutInsertingCallOp(
    OpBuilder &builder, T CI, FunctionOpInterface fn,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

template <typename T>
LogicalResult batchOperation(
    SymbolTableCollection &symbolTable, OpBuilder &builder, T CI,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {

  auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
  return batchOperation(builder, CI, cast<FunctionOpInterface>(symbolOp),
                        batchedFunctionCache);
}

template <typename T>
LogicalResult batchOperation(
    SymbolTableCollection &symbolTable, PatternRewriter &rewriter, T CI,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
  return batchOperation(rewriter, CI, cast<FunctionOpInterface>(symbolOp),
                        batchedFunctionCache);
}

template <typename T>
LogicalResult batchOperation(
    OpBuilder &builder, T CI, FunctionOpInterface fn,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  auto newFunc = batchOperationWithoutInsertingCallOp(builder, CI, fn,
                                                      batchedFunctionCache);

  if (!newFunc)
    return failure();

  {
    IRRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPoint(CI);
    auto dCI =
        builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                     newFunc.getResultTypes(), CI.getInputs());
    CI.replaceAllUsesWith(dCI);
    CI->erase();
  }
  return success();
}

template <typename T>
LogicalResult batchOperation(
    PatternRewriter &rewriter, T CI, FunctionOpInterface fn,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  auto newFunc = batchOperationWithoutInsertingCallOp(rewriter, CI, fn,
                                                      batchedFunctionCache);

  if (!newFunc)
    return failure();

  {
    IRRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPoint(CI);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        CI, newFunc.getName(), newFunc.getResultTypes(), CI.getInputs());
  }
  return success();
}

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_PASS_H
