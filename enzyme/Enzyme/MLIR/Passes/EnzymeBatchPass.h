#ifndef ENZYME_BATCH_PASS_H
#define ENZYME_BATCH_PASS_H

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

static mlir::TensorType applyBatchSizes(mlir::Type Ty,
                                        llvm::ArrayRef<int64_t> batchSizes);

static FunctionOpInterface batchCloneFunction(
    FunctionOpInterface F, Twine name, llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

static void batchCloneRegion(
    Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

static LogicalResult handleCallOp(
    func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

template <typename T>
LogicalResult batchOperation(
    SymbolTableCollection &symbolTable, T CI,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache);

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_BATCH_PASS_H
