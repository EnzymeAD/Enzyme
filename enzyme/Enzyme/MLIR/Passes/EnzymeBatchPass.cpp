//===- EnzymeBatchPass.cpp - Replace calls with their batched versions
//------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

#include "Passes/EnzymeBatchPass.h"
#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "enzyme-batch"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
namespace batchutils {

mlir::TensorType applyBatchSizes(mlir::Type Ty,
                                 llvm::ArrayRef<int64_t> batchSizes) {
  auto T = dyn_cast<TensorType>(Ty);
  if (!T) {
    return RankedTensorType::get(batchSizes, Ty);
  }

  SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
  shape.append(T.getShape().begin(), T.getShape().end());
  auto T2 = T.clone(shape);
  return T2;
}

LogicalResult handleCallOp(
    func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  // Get the called function
  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  auto calledFunc =
      dyn_cast<FunctionOpInterface>(moduleOp.lookupSymbol(callOp.getCallee()));
  if (!calledFunc)
    return failure();

  // Create cache key for this function and batch size combination
  BatchCacheKey key{calledFunc,
                    SmallVector<int64_t>(batchSizes.begin(), batchSizes.end())};

  // Look up or create batched version of the called function
  FunctionOpInterface batchedFunc;
  auto it = batchedFunctionCache.find(key);
  if (it != batchedFunctionCache.end()) {
    batchedFunc = it->second;
  } else {
    batchedFunc =
        batchCloneFunction(calledFunc, "batched_" + calledFunc.getName(),
                           batchSizes, batchedFunctionCache);
    if (!batchedFunc)
      return failure();
    batchedFunctionCache[key] = batchedFunc;
  }

  // Create new call operation to the batched function
  SmallVector<Value> newOperands;
  for (auto operand : callOp->getOperands())
    newOperands.push_back(mapper.lookup(operand));

  auto newCall =
      builder.create<func::CallOp>(callOp.getLoc(), batchedFunc.getName(),
                                   batchedFunc.getResultTypes(), newOperands);

  // Map the results
  for (auto [oldResult, newResult] :
       llvm::zip(callOp.getResults(), newCall.getResults()))
    mapper.map(oldResult, newResult);

  return success();
}

void batchCloneRegion(
    Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  // For each block in src, generate a corresponding block in the dest region.
  for (auto &blk : *src) {
    auto newBlk = new Block();
    dest->push_back(newBlk);

    mapper.map(&blk, newBlk);

    for (auto arg : blk.getArguments()) {
      Value newArg = newBlk->addArgument(
          applyBatchSizes(arg.getType(), batchSizes), arg.getLoc());
      mapper.map(arg, newArg);
    }
  }

  for (auto &&[blk, newBlk] : llvm::zip(*src, *dest)) {
    OpBuilder builder(&newBlk, newBlk.end());
    for (auto &src : blk) {

      if (auto callOp = dyn_cast<func::CallOp>(&src)) {
        if (succeeded(handleCallOp(callOp, builder, mapper, batchSizes,
                                   batchedFunctionCache)))
          continue;
      }

      if (auto ifaceOp = dyn_cast<BatchOpInterface>(&src)) {
        auto res = ifaceOp.createBatch(builder, mapper, batchSizes);
        if (res.succeeded())
          continue;
      }

      SmallVector<Value, 8> operands;
      SmallVector<Block *, 2> successors;

      // Remap the operands.
      operands.reserve(src.getNumOperands());
      for (auto opValue : src.getOperands())
        operands.push_back(mapper.lookup(opValue));

      // Remap the successors.
      successors.reserve(src.getNumSuccessors());
      for (Block *successor : src.getSuccessors())
        successors.push_back(mapper.lookup(successor));

      SmallVector<Type> resultTypes(src.getResultTypes().begin(),
                                    src.getResultTypes().end());
      for (auto &Ty : resultTypes) {
        Ty = applyBatchSizes(Ty, batchSizes);
      }

      Operation *newOp = Operation::create(
          src.getLoc(), src.getName(), resultTypes, operands, src.getAttrs(),
          OpaqueProperties(nullptr), successors, src.getNumRegions());

      // Clone the regions.
      for (auto &&[oldReg, newReg] :
           llvm::zip(src.getRegions(), newOp->getRegions())) {
        batchCloneRegion(&oldReg, &newReg, mapper, batchSizes,
                         batchedFunctionCache);
      }

      // Remember the mapping of any results.
      for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
        mapper.map(src.getResult(i), newOp->getResult(i));

      builder.insert(newOp);
    }
  }
}

FunctionOpInterface batchCloneFunction(
    FunctionOpInterface F, Twine name, llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  assert(!F.getFunctionBody().empty());

  auto FTy = cast<FunctionType>(F.getFunctionType());

  llvm::SmallVector<mlir::Type> RetTypes;
  RetTypes.reserve(FTy.getNumResults());

  for (auto Ty : FTy.getResults()) {
    RetTypes.push_back(applyBatchSizes(Ty, batchSizes));
  }

  SmallVector<mlir::Type, 4> ArgTypes;
  ArgTypes.reserve(FTy.getNumInputs());

  for (auto Ty : FTy.getInputs()) {
    ArgTypes.push_back(applyBatchSizes(Ty, batchSizes));
  }

  OpBuilder builder(FTy.getContext());
  FunctionType newFTy = builder.getFunctionType(ArgTypes, RetTypes);

  auto NewF = cast<FunctionOpInterface>(F->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, name.str());
  NewF.setType(newFTy);

  Operation *parent = F->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  // Add the function to the cache BEFORE processing its body to support
  // recursion.
  BatchCacheKey key{F,
                    SmallVector<int64_t>(batchSizes.begin(), batchSizes.end())};
  batchedFunctionCache[key] = NewF;

  auto &origReg = F.getFunctionBody();
  auto &newReg = NewF.getFunctionBody();

  IRMapping mapper;
  batchCloneRegion(&origReg, &newReg, mapper, batchSizes, batchedFunctionCache);

  return NewF;
}

template <typename T>
absl::StatusOr<func::CallOp> batchOperation(
    SymbolTableCollection &symbolTable, T CI,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
  return batchOperation(CI, cast<FunctionOpInterface>(symbolOp),
                        batchedFunctionCache);
}

template <typename T>
absl::StatusOr<func::CallOp> batchOperation(
    T CI, FunctionOpInterface fn,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  enzyme::batchutils::BatchCacheKey key{
      fn, SmallVector<int64_t>(CI.getBatchShape().begin(),
                               CI.getBatchShape().end())};

  // Check if we already have a batched version
  auto it = batchedFunctionCache.find(key);
  FunctionOpInterface newFunc;

  if (it != batchedFunctionCache.end()) {
    newFunc = it->second;
  } else {
    // Create new batched function and store in cache
    newFunc = batchCloneFunction(fn, "batched_" + fn.getName(),
                                 CI.getBatchShape(), batchedFunctionCache);
    if (!newFunc) {
      return absl::InternalError("Failed to batch function.");
    }
  }

  OpBuilder builder(CI);
  auto dCI = builder.create<func::CallOp>(
      CI.getLoc(), newFunc.getName(), newFunc.getResultTypes(), CI.getInputs());
  CI.replaceAllUsesWith(dCI);
  CI->erase();
  return dCI;
}

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

namespace {

struct BatchPass : public BatchPassBase<BatchPass> {
  void runOnOperation() override;

  // Cache mapping original function and batch sizes to batched function
  std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
      batchedFunctionCache;

  void lowerEnzymeBatchCalls(SymbolTableCollection &symbolTable,
                             FunctionOpInterface op) {
    {
      SmallVector<Operation *> toLower;
      op->walk([&](enzyme::BatchOp dop) {
        auto *symbolOp =
            symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
        auto callableOp = cast<FunctionOpInterface>(symbolOp);

        lowerEnzymeBatchCalls(symbolTable, callableOp);
        toLower.push_back(dop);
      });

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::BatchOp>(T)) {
          auto res = enzyme::batchutils::batchOperation(symbolTable, F,
                                                        batchedFunctionCache);
          if (!res.ok()) {
            signalPassFailure();
            return;
          }
        } else {
          llvm_unreachable("Illegal type");
        }
      }
    };
  };
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createBatchPass() {
  return std::make_unique<BatchPass>();
}
} // namespace enzyme
} // namespace mlir

void BatchPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeBatchCalls(symbolTable, op); });
}
