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

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "enzyme-batch"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {

static mlir::TensorType applyBatchSizes(mlir::Type Ty,
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

static void batchCloneRegion(Region *src, Region *dest, IRMapping &mapper,
                             llvm::ArrayRef<int64_t> batchSizes) {
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
        batchCloneRegion(&oldReg, &newReg, mapper, batchSizes);
      }

      // Remember the mapping of any results.
      for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
        mapper.map(src.getResult(i), newOp->getResult(i));

      builder.insert(newOp);
    }
  }
}

static FunctionOpInterface
batchCloneFunction(FunctionOpInterface F, Twine name,
                   llvm::ArrayRef<int64_t> batchSizes) {
  assert(!F.getFunctionBody().empty());

  auto FTy = F.getFunctionType().cast<FunctionType>();

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

  auto &origReg = F.getFunctionBody();
  auto &newReg = NewF.getFunctionBody();

  IRMapping mapper;
  batchCloneRegion(&origReg, &newReg, mapper, batchSizes);

  return NewF;
}

struct BatchPass : public BatchPassBase<BatchPass> {
  void runOnOperation() override;

  template <typename T>
  LogicalResult HandleBatch(SymbolTableCollection &symbolTable, T CI) {
    SmallVector<mlir::Value, 2> args;

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    FunctionOpInterface newFunc =
        batchCloneFunction(fn, "batched_" + fn.getName(), CI.getBatchShape());

    if (!newFunc)
      return failure();

    OpBuilder builder(CI);
    auto dCI =
        builder.create<func::CallOp>(CI.getLoc(), newFunc.getName(),
                                     newFunc.getResultTypes(), CI.getInputs());
    CI.replaceAllUsesWith(dCI);
    CI->erase();
    return success();
  }

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
          auto res = HandleBatch(symbolTable, F);
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
