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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "enzyme-batch"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_BATCHPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

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

FailureOr<Value> broadcastFreeValue(OpBuilder &builder, Operation *op,
                                    Value value,
                                    llvm::ArrayRef<int64_t> batchSizes) {
  if (batchSizes.empty())
    return value;
  Type ty = value.getType();
  if (!isa<RankedTensorType>(ty) && !TensorType::isValidElementType(ty)) {
    return op->emitError()
           << "cannot broadcast captured value of unsupported type " << ty;
  }
  TensorType batchedTy = applyBatchSizes(ty, batchSizes);
  return enzyme::BroadcastOp::create(builder, op->getLoc(), batchedTy, value,
                                     batchSizes)
      .getResult();
}

LogicalResult batchOperation(
    OpBuilder &builder, enzyme::BatchRegionOp batchRegionOp,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  Region &body = batchRegionOp.getBody();
  if (body.empty())
    return batchRegionOp.emitError()
           << "enzyme.batch_region body must not be empty";

  if (body.getNumArguments() != batchRegionOp.getInputs().size())
    return batchRegionOp.emitError()
           << "expected " << batchRegionOp.getInputs().size()
           << " block arguments, got " << body.getNumArguments();

  ArrayRef<int64_t> batchSizes = batchRegionOp.getBatchShape();
  llvm::SetVector<Value> freeValues;
  getUsedValuesDefinedAbove(body, freeValues);

  OpBuilder::InsertionGuard insertGuard(builder);
  if (body.hasOneBlock()) {
    IRMapping mapper;
    for (auto [arg, input] :
         llvm::zip_equal(body.getArguments(), batchRegionOp.getInputs()))
      mapper.map(arg, input);

    builder.setInsertionPoint(batchRegionOp);
    for (Value v : freeValues) {
      auto bcast = broadcastFreeValue(builder, batchRegionOp, v, batchSizes);
      if (failed(bcast))
        return failure();
      mapper.map(v, *bcast);
    }

    Block &origBlock = body.front();
    auto yieldOp = dyn_cast<enzyme::YieldOp>(origBlock.getTerminator());
    if (!yieldOp)
      return batchRegionOp.emitError()
             << "expected enzyme.yield terminator in enzyme.batch_region";
    if (yieldOp.getNumOperands() != batchRegionOp.getNumResults())
      return batchRegionOp.emitError()
             << "enzyme.yield operand count does not match "
                "enzyme.batch_region result count";

    batchCloneBlock(builder, &origBlock, mapper, batchSizes,
                    batchedFunctionCache, /*withoutTerminator=*/true);

    SmallVector<Value> mappedResults;
    mappedResults.reserve(yieldOp.getNumOperands());
    for (Value operand : yieldOp.getOperands())
      mappedResults.push_back(mapper.lookup(operand));
    batchRegionOp.replaceAllUsesWith(mappedResults);
    batchRegionOp.erase();
    return success();
  }

  Operation *symbolTableOp =
      batchRegionOp->getParentWithTrait<OpTrait::SymbolTable>();
  if (!symbolTableOp)
    return batchRegionOp.emitError()
           << "multi-block enzyme.batch_region requires an enclosing symbol "
              "table";

  SmallVector<Value> callOperands(batchRegionOp.getInputs().begin(),
                                  batchRegionOp.getInputs().end());
  builder.setInsertionPoint(batchRegionOp);
  for (Value v : freeValues) {
    auto bcast = broadcastFreeValue(builder, batchRegionOp, v, batchSizes);
    if (failed(bcast))
      return failure();
    callOperands.push_back(*bcast);
  }

  SmallVector<Type> argTypes;
  argTypes.reserve(callOperands.size());
  for (Value v : callOperands)
    argTypes.push_back(v.getType());

  std::string fnName = "batched_region";
  if (auto parentSym = batchRegionOp->getParentOfType<SymbolOpInterface>())
    fnName = ("batched_" + parentSym.getName() + "_region").str();
  if (auto origFn = batchRegionOp.getFn())
    fnName = ("batched_" + *origFn).str();

  FunctionType fnType =
      builder.getFunctionType(argTypes, batchRegionOp.getResultTypes());
  builder.setInsertionPointToEnd(&symbolTableOp->getRegion(0).front());
  auto newFunc =
      func::FuncOp::create(builder, batchRegionOp.getLoc(), fnName, fnType);
  SymbolTable table(symbolTableOp);
  table.insert(newFunc);
  SymbolTable::setSymbolVisibility(newFunc, SymbolTable::Visibility::Private);

  Region &newRegion = newFunc.getBody();
  IRMapping mapper;
  for (auto &blk : body) {
    auto *newBlk = new Block();
    newRegion.push_back(newBlk);
    mapper.map(&blk, newBlk);
    bool isEntry = (&blk == &body.front());
    for (auto arg : blk.getArguments()) {
      Type argTy = isEntry
                       ? batchRegionOp.getInputs()[arg.getArgNumber()].getType()
                       : Type(applyBatchSizes(arg.getType(), batchSizes));
      Value newArg = newBlk->addArgument(argTy, arg.getLoc());
      mapper.map(arg, newArg);
    }
  }
  Block &newEntry = newRegion.front();
  for (auto [idx, v] : llvm::enumerate(freeValues)) {
    Type freeTy =
        callOperands[batchRegionOp.getInputs().size() + idx].getType();
    Value newArg = newEntry.addArgument(freeTy, v.getLoc());
    mapper.map(v, newArg);
  }

  for (auto &&[blk, newBlk] : llvm::zip(body, newRegion)) {
    IRRewriter::InsertionGuard blockGuard(builder);
    builder.setInsertionPointToEnd(&newBlk);
    batchCloneBlock(builder, &blk, mapper, batchSizes, batchedFunctionCache,
                    false);
    if (!newBlk.empty()) {
      if (auto yieldOp = dyn_cast<enzyme::YieldOp>(newBlk.getTerminator())) {
        builder.setInsertionPoint(yieldOp);
        func::ReturnOp::create(builder, yieldOp.getLoc(),
                               yieldOp.getOperands());
        yieldOp.erase();
      }
    }
  }

  builder.setInsertionPoint(batchRegionOp);
  auto callOp =
      func::CallOp::create(builder, batchRegionOp.getLoc(), newFunc.getName(),
                           newFunc.getResultTypes(), callOperands);
  batchRegionOp.replaceAllUsesWith(callOp.getResults());
  batchRegionOp.erase();
  return success();
}

LogicalResult handleCallLikeOp(
    Operation *callLike, StringRef symbolAttrName, OpBuilder &builder,
    IRMapping &mapper, llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  auto calledFunc = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      callLike, callLike->getAttrOfType<SymbolRefAttr>(symbolAttrName));

  if (!calledFunc) {
    return failure();
  }

  std::string fnName = "batched_" + calledFunc.getName().str();
  FunctionOpInterface batchedFunc = batchCloneFunction(
      builder, calledFunc, fnName, batchSizes, batchedFunctionCache);

  if (!batchedFunc) {
    return failure();
  }

  // Create new call operation to the batched function
  SmallVector<Value> newOperands;
  for (auto operand : callLike->getOperands())
    newOperands.push_back(mapper.lookup(operand));

  SmallVector<Type> resultTypes =
      llvm::map_to_vector(callLike->getResultTypes(), [&](Type Ty) -> Type {
        return applyBatchSizes(Ty, batchSizes);
      });

  Operation *newCallLike =
      builder.create(callLike->getLoc(), callLike->getName().getIdentifier(),
                     newOperands, resultTypes, callLike->getAttrs());

  newCallLike->setAttr(symbolAttrName, SymbolRefAttr::get(batchedFunc));

  for (auto [oldRes, newRes] :
       llvm::zip_equal(callLike->getResults(), newCallLike->getResults())) {
    mapper.map(oldRes, newRes);
  }

  return success();
}

LogicalResult handleCallOp(
    func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  return handleCallLikeOp(callOp, "callee", builder, mapper, batchSizes,
                          batchedFunctionCache);
}

void batchCloneBlock(
    OpBuilder &builder, Block *blk, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    bool withoutTerminator) {
  for (auto &src : *blk) {
    if (auto callOp = dyn_cast<func::CallOp>(&src)) {
      if (succeeded(handleCallOp(callOp, builder, mapper, batchSizes,
                                 batchedFunctionCache)))
        continue;
    }

    if (isa<enzyme::ForwardDiffOp, enzyme::AutoDiffOp>(&src)) {
      if (succeeded(handleCallLikeOp(&src, "fn", builder, mapper, batchSizes,
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

    if (withoutTerminator && src.hasTrait<OpTrait::IsTerminator>()) {
      // map the operands and the results
      for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
        mapper.map(src.getResult(i), operands[i]);
      continue;
    }

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
        mlir::PropertyRef(), successors, src.getNumRegions());

    // Clone the regions.
    for (auto &&[oldReg, newReg] :
         llvm::zip(src.getRegions(), newOp->getRegions())) {
      batchCloneRegion(builder, &oldReg, &newReg, mapper, batchSizes,
                       batchedFunctionCache);
    }

    // Remember the mapping of any results.
    for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
      mapper.map(src.getResult(i), newOp->getResult(i));

    builder.insert(newOp);
  }
}

void batchCloneRegion(
    OpBuilder &builder, Region *src, Region *dest, IRMapping &mapper,
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
    IRRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToEnd(&newBlk);
    batchCloneBlock(builder, &blk, mapper, batchSizes, batchedFunctionCache,
                    false);
  }
}

FunctionOpInterface batchCloneFunction(
    OpBuilder &builder, FunctionOpInterface F, Twine name,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  assert(!F.getFunctionBody().empty());

  // Add the function to the cache BEFORE processing its body to support
  // recursion.
  BatchCacheKey key{F,
                    SmallVector<int64_t>(batchSizes.begin(), batchSizes.end())};

  auto it = batchedFunctionCache.find(key);
  if (it != batchedFunctionCache.end())
    return it->second;

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

  FunctionType newFTy = builder.getFunctionType(ArgTypes, RetTypes);

  auto NewF = cast<FunctionOpInterface>(F->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, name.str());
  NewF.setType(newFTy);

  Operation *parent = F->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  batchedFunctionCache[key] = NewF;

  auto &origReg = F.getFunctionBody();
  auto &newReg = NewF.getFunctionBody();

  IRMapping mapper;
  batchCloneRegion(builder, &origReg, &newReg, mapper, batchSizes,
                   batchedFunctionCache);

  return NewF;
}

template <typename T>
FunctionOpInterface batchOperationWithoutInsertingCallOp(
    OpBuilder &builder, T CI, FunctionOpInterface fn,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  enzyme::batchutils::BatchCacheKey key{
      fn, SmallVector<int64_t>(CI.getBatchShape().begin(),
                               CI.getBatchShape().end())};

  // Check if we already have a batched version
  auto it = batchedFunctionCache.find(key);
  FunctionOpInterface newFunc;

  if (it != batchedFunctionCache.end()) {
    return it->second;
  } else {
    // Create new batched function and store in cache
    std::string newFnName = "batched_" + fn.getName().str();
    newFunc = batchCloneFunction(builder, fn, newFnName, CI.getBatchShape(),
                                 batchedFunctionCache);
    return newFunc;
  }
}

} // namespace batchutils
} // namespace enzyme
} // namespace mlir

namespace {

struct BatchPass : public enzyme::impl::BatchPassBase<BatchPass> {
  void runOnOperation() override;

  void lowerEnzymeBatchCalls(
      SymbolTableCollection &symbolTable, FunctionOpInterface op,
      std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
          &BatchedFunctionCache) {
    {
      SmallVector<Operation *> toLower;
      op->walk([&](Operation *nested) {
        if (isa<enzyme::BatchOp, enzyme::BatchRegionOp>(nested))
          toLower.push_back(nested);
      });

      OpBuilder builder(op);

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::BatchOp>(T)) {
          auto res = enzyme::batchutils::batchOperation(symbolTable, builder, F,
                                                        BatchedFunctionCache);
          if (!res.succeeded()) {
            signalPassFailure();
            return;
          }
        } else if (auto R = dyn_cast<enzyme::BatchRegionOp>(T)) {
          auto res = enzyme::batchutils::batchOperation(builder, R,
                                                        BatchedFunctionCache);
          if (!res.succeeded()) {
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

void BatchPass::runOnOperation() {
  std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
      BatchedFunctionCache;
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  getOperation()->walk([&](FunctionOpInterface op) {
    lowerEnzymeBatchCalls(symbolTable, op, BatchedFunctionCache);
  });
}
