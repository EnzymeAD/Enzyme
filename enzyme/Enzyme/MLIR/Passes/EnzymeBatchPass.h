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
  SmallVector<bool> batchedArgs;
  SmallVector<bool> batchedResults;

  bool operator<(const BatchCacheKey &other) const {
    Operation *thisSymbolTable = SymbolTable::getNearestSymbolTable(function),
              *otherSymbolTable =
                  SymbolTable::getNearestSymbolTable(other.function);
    if (thisSymbolTable != otherSymbolTable)
      return std::less<Operation *>()(thisSymbolTable, otherSymbolTable);

    auto thisName = const_cast<FunctionOpInterface &>(function).getName(),
         otherName =
             const_cast<FunctionOpInterface &>(other.function).getName();
    if (thisName != otherName)
      return thisName < otherName;

    if (batchSizes != other.batchSizes)
      return std::lexicographical_compare(batchSizes.begin(), batchSizes.end(),
                                          other.batchSizes.begin(),
                                          other.batchSizes.end());

    if (batchedArgs != other.batchedArgs)
      return std::lexicographical_compare(
          batchedArgs.begin(), batchedArgs.end(), other.batchedArgs.begin(),
          other.batchedArgs.end());

    return std::lexicographical_compare(
        batchedResults.begin(), batchedResults.end(),
        other.batchedResults.begin(), other.batchedResults.end());
  }
};

mlir::TensorType applyBatchSizes(mlir::Type Ty,
                                 llvm::ArrayRef<int64_t> batchSizes);

/// Compute the set of values inside \p F that depend on the batched entry
/// arguments specified by \p batchedArgs.
void computeBatchedValues(FunctionOpInterface F,
                          llvm::ArrayRef<bool> batchedArgs,
                          llvm::DenseSet<Value> &batchedValues);

/// Return a batched version of \p origVal, inserting and caching a broadcast or
/// batched constant when \p origVal is unbatched.
Value getBatchedValue(Value origVal, OpBuilder &builder, IRMapping &mapper,
                      llvm::ArrayRef<int64_t> batchSizes,
                      const llvm::DenseSet<Value> &batchedValues,
                      llvm::DenseMap<Value, Value> &broadcastCache,
                      bool insertAtCurrentPoint = false);

FunctionOpInterface batchCloneFunction(
    OpBuilder &builder, FunctionOpInterface F, Twine name,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

FunctionOpInterface batchCloneFunction(
    OpBuilder &builder, FunctionOpInterface F, Twine name,
    llvm::ArrayRef<int64_t> batchSizes, llvm::ArrayRef<bool> batchedArgs,
    llvm::ArrayRef<bool> batchedResults,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

void batchCloneBlock(
    OpBuilder &builder, Block *blk, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    bool withoutTerminator);

void batchCloneBlock(
    OpBuilder &builder, Block *blk, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    bool withoutTerminator, const llvm::DenseSet<Value> &batchedValues,
    llvm::ArrayRef<bool> batchedResults,
    llvm::DenseMap<Value, Value> &broadcastCache, Region *funcBody = nullptr,
    bool insertBroadcastsLocally = false);

void batchCloneRegion(
    OpBuilder &builder, Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

void batchCloneRegion(
    OpBuilder &builder, Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    const llvm::DenseSet<Value> &batchedValues,
    llvm::ArrayRef<bool> batchedResults,
    llvm::DenseMap<Value, Value> &broadcastCache, Region *funcBody = nullptr);

LogicalResult handleCallOp(
    func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

LogicalResult
handleCallOp(func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
             llvm::ArrayRef<int64_t> batchSizes,
             std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
             const llvm::DenseSet<Value> &batchedValues,
             llvm::DenseMap<Value, Value> &broadcastCache);

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
    auto dCI = func::CallOp::create(builder, CI.getLoc(), newFunc.getName(),
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

/// Inline each operation directly into the caller instead of inserting a call.
inline void batchOperationInline(PatternRewriter &rewriter,
                                 enzyme::BatchOp batchOp,
                                 FunctionOpInterface func) {
  auto &origRegion = func.getFunctionBody();
  auto &origBlock = origRegion.front();

  SmallVector<bool> batchedArgs;
  auto origArgTypes = func.getArgumentTypes();
  batchedArgs.reserve(batchOp->getNumOperands());
  for (auto [actualVal, origTy] :
       llvm::zip_equal(batchOp.getInputs(), origArgTypes))
    batchedArgs.push_back(actualVal.getType() != origTy);

  SmallVector<bool> batchedResults;
  auto origResTypes = func.getResultTypes();
  batchedResults.reserve(batchOp->getNumResults());
  for (auto [actualRes, origTy] :
       llvm::zip_equal(batchOp.getOutputs(), origResTypes))
    batchedResults.push_back(actualRes.getType() != origTy);

  IRMapping mapper;
  for (unsigned i = 0; i < batchOp->getNumOperands(); i++) {
    mapper.map(origBlock.getArguments()[i], batchOp->getOperand(i));
  }

  rewriter.setInsertionPoint(batchOp);
  std::map<BatchCacheKey, FunctionOpInterface> batchedFunctionCache;
  DenseSet<Value> batchedValues;
  computeBatchedValues(func, batchedArgs, batchedValues);
  DenseMap<Value, Value> broadcastCache;
  batchCloneBlock(rewriter, &origBlock, mapper, batchOp.getBatchShape(),
                  batchedFunctionCache, true, batchedValues, batchedResults,
                  broadcastCache, &origRegion,
                  /*insertBroadcastsLocally=*/true);

  auto origTerm = origBlock.getTerminator();
  for (auto [i, operand] : llvm::enumerate(origTerm->getOperands())) {
    Value mappedOperand =
        batchedResults[i]
            ? getBatchedValue(operand, rewriter, mapper,
                              batchOp.getBatchShape(), batchedValues,
                              broadcastCache, /*insertAtCurrentPoint=*/true)
            : mapper.lookup(operand);
    rewriter.replaceAllUsesWith(batchOp->getResult(i), mappedOperand);
  }
  rewriter.eraseOp(batchOp);
  rewriter.eraseOp(func);
}

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_PASS_H
