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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <set>

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

namespace {

using BatchAnalysisKey = std::pair<Operation *, SmallVector<bool>>;

/// Collect a map from forwarded operand indices to the successor values
/// (block arguments or parent results) they populate for a region branch op or
/// its terminator.
void collectRegionBranchOperandTargets(
    RegionBranchOpInterface regionIface, Operation *op,
    llvm::SmallDenseMap<unsigned, SmallVector<Value, 2>> &operandToTargets) {
  auto recordTargets = [&](OperandRange operands, ValueRange targets) {
    if (operands.empty())
      return;
    unsigned base = operands.getBeginOperandIndex();
    if (operands.size() == targets.size()) {
      for (unsigned i = 0, e = operands.size(); i < e; ++i)
        operandToTargets[base + i].push_back(targets[i]);
    } else {
      for (unsigned i = 0, e = operands.size(); i < e; ++i)
        for (Value target : targets)
          operandToTargets[base + i].push_back(target);
    }
  };

  SmallVector<RegionSuccessor> successors;
  if (op == regionIface.getOperation()) {
    regionIface.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (auto &successor : successors) {
      OperandRange operands = regionIface.getEntrySuccessorOperands(successor);
      ValueRange targets = successor.getSuccessor() == nullptr
                               ? regionIface->getResults()
                               : regionIface.getSuccessorInputs(successor);
      recordTargets(operands, targets);
    }
  } else if (auto termIface = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
    termIface.getSuccessorRegions(
        SmallVector<Attribute>(termIface->getNumOperands(), Attribute()),
        successors);
    for (auto &successor : successors) {
      OperandRange operands = termIface.getSuccessorOperands(successor);
      ValueRange targets = successor.getSuccessor() == nullptr
                               ? regionIface->getResults()
                               : regionIface.getSuccessorInputs(successor);
      recordTargets(operands, targets);
    }
  }
}

void computeBatchedValuesImpl(
    FunctionOpInterface F, ArrayRef<bool> batchedArgs,
    DenseSet<Value> &batchedValues, std::set<BatchAnalysisKey> &visiting,
    std::map<BatchAnalysisKey, SmallVector<bool>> &resultCache);

/// Compute which return values of \p F are batched given \p batchedArgs,
/// handling recursive calls conservatively via \p visiting.
SmallVector<bool> computeCalleeResultBatching(
    FunctionOpInterface F, ArrayRef<bool> batchedArgs,
    std::set<BatchAnalysisKey> &visiting,
    std::map<BatchAnalysisKey, SmallVector<bool>> &resultCache) {
  BatchAnalysisKey key{F.getOperation(), SmallVector<bool>(batchedArgs.begin(),
                                                           batchedArgs.end())};
  if (auto it = resultCache.find(key); it != resultCache.end())
    return it->second;

  bool anyArgBatched = llvm::any_of(batchedArgs, [](bool b) { return b; });
  if (!anyArgBatched)
    return SmallVector<bool>(F.getNumResults(), false);

  if (F.getFunctionBody().empty() || visiting.count(key))
    return SmallVector<bool>(F.getNumResults(), true);

  visiting.insert(key);
  DenseSet<Value> calleeBatchedValues;
  computeBatchedValuesImpl(F, batchedArgs, calleeBatchedValues, visiting,
                           resultCache);
  visiting.erase(key);

  SmallVector<bool> batchedResults(F.getNumResults(), false);
  for (Block &blk : F.getFunctionBody()) {
    if (blk.empty())
      continue;
    Operation *term = blk.getTerminator();
    if (!term->hasTrait<OpTrait::ReturnLike>())
      continue;
    for (auto [i, retVal] : llvm::enumerate(term->getOperands())) {
      if (i < batchedResults.size() && calleeBatchedValues.contains(retVal))
        batchedResults[i] = true;
    }
  }

  resultCache[key] = batchedResults;
  return batchedResults;
}

/// Propagate batch dependence to a fixed point inside \p F starting from the
/// entry block arguments flagged in \p batchedArgs.
void computeBatchedValuesImpl(
    FunctionOpInterface F, ArrayRef<bool> batchedArgs,
    DenseSet<Value> &batchedValues, std::set<BatchAnalysisKey> &visiting,
    std::map<BatchAnalysisKey, SmallVector<bool>> &resultCache) {
  if (F.getFunctionBody().empty())
    return;

  Block &entryBlock = F.getFunctionBody().front();
  for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments())) {
    if (i < batchedArgs.size() && batchedArgs[i])
      batchedValues.insert(arg);
  }

  auto isBatched = [&](Value v) { return batchedValues.contains(v); };

  bool changed = true;
  while (changed) {
    changed = false;
    F.getFunctionBody().walk([&](Operation *op) {
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        SmallVector<bool> calleeBatchedArgs;
        calleeBatchedArgs.reserve(callOp.getNumOperands());
        for (Value operand : callOp.getOperands())
          calleeBatchedArgs.push_back(isBatched(operand));

        if (!llvm::any_of(calleeBatchedArgs, [](bool b) { return b; }))
          return;

        auto calledFunc =
            SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
                callOp, callOp.getCalleeAttr());
        if (calledFunc) {
          SmallVector<bool> calleeBatchedRes = computeCalleeResultBatching(
              calledFunc, calleeBatchedArgs, visiting, resultCache);
          for (auto [res, resBatched] :
               llvm::zip_equal(callOp.getResults(), calleeBatchedRes)) {
            if (resBatched)
              changed |= batchedValues.insert(res).second;
          }
        } else {
          for (Value res : callOp.getResults())
            changed |= batchedValues.insert(res).second;
        }
        return;
      }

      if (isa<enzyme::ForwardDiffOp, enzyme::AutoDiffOp>(op)) {
        if (llvm::any_of(op->getOperands(), isBatched)) {
          for (Value res : op->getResults())
            changed |= batchedValues.insert(res).second;
        }
        return;
      }

      if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
        bool nonForwardedBatched = false;
        for (OpOperand &opnd : op->getOpOperands()) {
          if (!branchOp.getSuccessorBlockArgument(opnd.getOperandNumber()) &&
              isBatched(opnd.get())) {
            nonForwardedBatched = true;
            break;
          }
        }
        for (OpOperand &opnd : op->getOpOperands()) {
          if (auto targetArg =
                  branchOp.getSuccessorBlockArgument(opnd.getOperandNumber())) {
            if (nonForwardedBatched || isBatched(opnd.get()))
              changed |= batchedValues.insert(*targetArg).second;
          }
        }
        return;
      }

      if (auto regionIface =
              dyn_cast_or_null<RegionBranchOpInterface>(op->getParentOp())) {
        if (isa<RegionBranchTerminatorOpInterface>(op)) {
          llvm::SmallDenseMap<unsigned, SmallVector<Value, 2>> operandToTargets;
          collectRegionBranchOperandTargets(regionIface, op, operandToTargets);

          bool nonForwardedBatched = false;
          for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
            auto it = operandToTargets.find(i);
            if (it == operandToTargets.end()) {
              if (isBatched(op->getOperand(i)))
                nonForwardedBatched = true;
              continue;
            }
            if (isBatched(op->getOperand(i)) ||
                llvm::any_of(it->second, isBatched)) {
              for (Value target : it->second)
                changed |= batchedValues.insert(target).second;
            }
          }
          if (nonForwardedBatched) {
            for (Value res : regionIface->getResults())
              changed |= batchedValues.insert(res).second;
            for (Region &reg : regionIface->getRegions())
              for (Block &blk : reg)
                for (BlockArgument arg : blk.getArguments())
                  changed |= batchedValues.insert(arg).second;
          }
          return;
        }
      }

      if (auto regionIface = dyn_cast<RegionBranchOpInterface>(op)) {
        llvm::SmallDenseMap<unsigned, SmallVector<Value, 2>> operandToTargets;
        collectRegionBranchOperandTargets(regionIface, op, operandToTargets);

        bool nonForwardedBatched = false;
        for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
          auto it = operandToTargets.find(i);
          if (it == operandToTargets.end()) {
            if (isBatched(op->getOperand(i)))
              nonForwardedBatched = true;
            continue;
          }
          if (isBatched(op->getOperand(i)) ||
              llvm::any_of(it->second, isBatched)) {
            for (Value target : it->second)
              changed |= batchedValues.insert(target).second;
          }
        }
        if (nonForwardedBatched) {
          for (Value res : op->getResults())
            changed |= batchedValues.insert(res).second;
          for (Region &reg : op->getRegions())
            for (Block &blk : reg)
              for (BlockArgument arg : blk.getArguments())
                changed |= batchedValues.insert(arg).second;
        }
        return;
      }

      if (op->hasTrait<OpTrait::ReturnLike>())
        return;

      bool anyInputBatched = llvm::any_of(op->getOperands(), isBatched);
      if (!anyInputBatched && op->getNumRegions() > 0) {
        mlir::visitUsedValuesDefinedAbove(op->getRegions(),
                                          [&](OpOperand *use) {
                                            if (isBatched(use->get()))
                                              anyInputBatched = true;
                                          });
      }

      if (anyInputBatched) {
        for (Value res : op->getResults())
          changed |= batchedValues.insert(res).second;
        for (Region &reg : op->getRegions())
          for (Block &blk : reg)
            for (BlockArgument arg : blk.getArguments())
              changed |= batchedValues.insert(arg).second;
      }
    });
  }
}

/// Return true if \p use requires a batched value for its operand.
bool operandNeedsBatched(OpOperand &use, const DenseSet<Value> &batchedValues,
                         ArrayRef<bool> batchedResults, Region *funcBody) {
  Operation *user = use.getOwner();
  unsigned opIdx = use.getOperandNumber();
  auto isBatched = [&](Value v) { return batchedValues.contains(v); };

  if (user->hasTrait<OpTrait::ReturnLike>() &&
      user->getParentRegion() == funcBody) {
    return opIdx < batchedResults.size() && batchedResults[opIdx];
  }

  if (auto branchOp = dyn_cast<BranchOpInterface>(user)) {
    if (auto targetArg = branchOp.getSuccessorBlockArgument(opIdx))
      return isBatched(*targetArg);
    return false;
  }

  if (auto regionIface =
          dyn_cast_or_null<RegionBranchOpInterface>(user->getParentOp())) {
    if (isa<RegionBranchTerminatorOpInterface>(user)) {
      llvm::SmallDenseMap<unsigned, SmallVector<Value, 2>> operandToTargets;
      collectRegionBranchOperandTargets(regionIface, user, operandToTargets);
      auto it = operandToTargets.find(opIdx);
      return it != operandToTargets.end() &&
             llvm::any_of(it->second, isBatched);
    }
  }

  if (auto regionIface = dyn_cast<RegionBranchOpInterface>(user)) {
    llvm::SmallDenseMap<unsigned, SmallVector<Value, 2>> operandToTargets;
    collectRegionBranchOperandTargets(regionIface, user, operandToTargets);
    auto it = operandToTargets.find(opIdx);
    return it != operandToTargets.end() && llvm::any_of(it->second, isBatched);
  }

  if (isa<func::CallOp>(user))
    return isBatched(use.get());

  if (isa<enzyme::ForwardDiffOp, enzyme::AutoDiffOp>(user))
    return llvm::any_of(user->getOperands(), isBatched);

  if (llvm::any_of(user->getOperands(), isBatched))
    return true;

  if (user->getNumRegions() > 0) {
    bool capturedBatched = false;
    mlir::visitUsedValuesDefinedAbove(user->getRegions(),
                                      [&](OpOperand *captured) {
                                        if (isBatched(captured->get()))
                                          capturedBatched = true;
                                      });
    return capturedBatched;
  }

  return false;
}

/// Collect all values defined inside \p blk (recursively) into \p values.
void collectAllValuesInBlock(Block *blk, DenseSet<Value> &values) {
  for (BlockArgument arg : blk->getArguments())
    values.insert(arg);
  blk->walk([&](Operation *op) {
    for (Value res : op->getResults())
      values.insert(res);
    for (Region &reg : op->getRegions())
      for (Block &innerBlk : reg)
        for (BlockArgument arg : innerBlk.getArguments())
          values.insert(arg);
  });
}

} // namespace

void computeBatchedValues(FunctionOpInterface F,
                          llvm::ArrayRef<bool> batchedArgs,
                          llvm::DenseSet<Value> &batchedValues) {
  std::set<BatchAnalysisKey> visiting;
  std::map<BatchAnalysisKey, SmallVector<bool>> resultCache;
  computeBatchedValuesImpl(F, batchedArgs, batchedValues, visiting,
                           resultCache);
}

Value getBatchedValue(Value origVal, OpBuilder &builder, IRMapping &mapper,
                      llvm::ArrayRef<int64_t> batchSizes,
                      const llvm::DenseSet<Value> &batchedValues,
                      llvm::DenseMap<Value, Value> &broadcastCache,
                      bool insertAtCurrentPoint) {
  if (batchedValues.contains(origVal))
    return mapper.lookup(origVal);

  if (auto it = broadcastCache.find(origVal); it != broadcastCache.end())
    return it->second;

  Value unbatchedVal = mapper.lookup(origVal);
  auto targetTy = applyBatchSizes(origVal.getType(), batchSizes);
  if (unbatchedVal.getType() == targetTy) {
    broadcastCache[origVal] = unbatchedVal;
    return unbatchedVal;
  }

  OpBuilder::InsertionGuard guard(builder);
  bool canCache = !insertAtCurrentPoint;
  if (!insertAtCurrentPoint) {
    if (Operation *defOp = unbatchedVal.getDefiningOp()) {
      if (!defOp->hasTrait<OpTrait::IsTerminator>())
        builder.setInsertionPointAfter(defOp);
      else
        canCache = false;
    } else if (auto bArg = dyn_cast<BlockArgument>(unbatchedVal)) {
      builder.setInsertionPointToStart(bArg.getOwner());
    }
  }

  Value batchedVal;
  if (Operation *origDefOp = origVal.getDefiningOp()) {
    if (origDefOp->getNumOperands() == 0 && origDefOp->getNumResults() == 1) {
      if (auto ifaceOp = dyn_cast<BatchOpInterface>(origDefOp)) {
        IRMapping localMapper;
        if (succeeded(ifaceOp.createBatch(builder, localMapper, batchSizes)))
          batchedVal = localMapper.lookup(origVal);
      }
    }
  }

  if (!batchedVal) {
    batchedVal = enzyme::BroadcastOp::create(
        builder, unbatchedVal.getLoc(), targetTy, unbatchedVal,
        builder.getDenseI64ArrayAttr(batchSizes));
  }

  if (canCache)
    broadcastCache[origVal] = batchedVal;
  return batchedVal;
}

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

LogicalResult handleCallLikeOp(
    Operation *callLike, StringRef symbolAttrName, OpBuilder &builder,
    IRMapping &mapper, llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    const llvm::DenseSet<Value> &batchedValues,
    llvm::DenseMap<Value, Value> &broadcastCache,
    bool insertBroadcastsLocally = false) {
  bool anyOperandBatched = llvm::any_of(callLike->getOperands(), [&](Value v) {
    return batchedValues.contains(v);
  });

  if (!anyOperandBatched) {
    SmallVector<Value> newOperands;
    newOperands.reserve(callLike->getNumOperands());
    for (Value operand : callLike->getOperands())
      newOperands.push_back(mapper.lookup(operand));

    Operation *newCallLike = builder.create(
        callLike->getLoc(), callLike->getName().getIdentifier(), newOperands,
        callLike->getResultTypes(), callLike->getAttrs());
    for (auto [oldRes, newRes] :
         llvm::zip_equal(callLike->getResults(), newCallLike->getResults()))
      mapper.map(oldRes, newRes);
    return success();
  }

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

  SmallVector<Value> newOperands;
  newOperands.reserve(callLike->getNumOperands());
  for (auto operand : callLike->getOperands())
    newOperands.push_back(getBatchedValue(operand, builder, mapper, batchSizes,
                                          batchedValues, broadcastCache,
                                          insertBroadcastsLocally));

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

LogicalResult
handleCallOp(func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
             llvm::ArrayRef<int64_t> batchSizes,
             std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
             const llvm::DenseSet<Value> &batchedValues,
             llvm::DenseMap<Value, Value> &broadcastCache) {
  SmallVector<bool> calleeBatchedArgs;
  calleeBatchedArgs.reserve(callOp.getNumOperands());
  for (Value operand : callOp.getOperands())
    calleeBatchedArgs.push_back(batchedValues.contains(operand));

  if (!llvm::any_of(calleeBatchedArgs, [](bool b) { return b; })) {
    SmallVector<Value> newOperands;
    newOperands.reserve(callOp.getNumOperands());
    for (Value operand : callOp.getOperands())
      newOperands.push_back(mapper.lookup(operand));

    Operation *newCall = builder.create(
        callOp.getLoc(), callOp->getName().getIdentifier(), newOperands,
        callOp.getResultTypes(), callOp->getAttrs());
    for (auto [oldRes, newRes] :
         llvm::zip_equal(callOp.getResults(), newCall->getResults()))
      mapper.map(oldRes, newRes);
    return success();
  }

  auto calledFunc = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      callOp, callOp.getCalleeAttr());
  if (!calledFunc || calledFunc.getFunctionBody().empty())
    return failure();

  SmallVector<bool> calleeBatchedResults;
  calleeBatchedResults.reserve(callOp.getNumResults());
  for (Value res : callOp.getResults())
    calleeBatchedResults.push_back(batchedValues.contains(res));

  std::string fnName = "batched_" + calledFunc.getName().str();
  FunctionOpInterface batchedFunc = batchCloneFunction(
      builder, calledFunc, fnName, batchSizes, calleeBatchedArgs,
      calleeBatchedResults, batchedFunctionCache);
  if (!batchedFunc)
    return failure();

  SmallVector<Value> newOperands;
  newOperands.reserve(callOp.getNumOperands());
  for (Value operand : callOp.getOperands())
    newOperands.push_back(mapper.lookup(operand));

  SmallVector<Type> resultTypes;
  resultTypes.reserve(callOp.getNumResults());
  for (auto [res, isBatchedRes] :
       llvm::zip_equal(callOp.getResults(), calleeBatchedResults)) {
    resultTypes.push_back(isBatchedRes
                              ? applyBatchSizes(res.getType(), batchSizes)
                              : res.getType());
  }

  Operation *newCall =
      builder.create(callOp.getLoc(), callOp->getName().getIdentifier(),
                     newOperands, resultTypes, callOp->getAttrs());
  newCall->setAttr("callee", SymbolRefAttr::get(batchedFunc));

  for (auto [oldRes, newRes] :
       llvm::zip_equal(callOp.getResults(), newCall->getResults()))
    mapper.map(oldRes, newRes);

  return success();
}

LogicalResult handleCallOp(
    func::CallOp callOp, OpBuilder &builder, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  DenseSet<Value> allBatched;
  for (Value operand : callOp.getOperands())
    allBatched.insert(operand);
  for (Value res : callOp.getResults())
    allBatched.insert(res);
  DenseMap<Value, Value> broadcastCache;
  return handleCallOp(callOp, builder, mapper, batchSizes, batchedFunctionCache,
                      allBatched, broadcastCache);
}

void batchCloneBlock(
    OpBuilder &builder, Block *blk, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    bool withoutTerminator, const llvm::DenseSet<Value> &batchedValues,
    llvm::ArrayRef<bool> batchedResults,
    llvm::DenseMap<Value, Value> &broadcastCache, Region *funcBody,
    bool insertBroadcastsLocally) {
  for (auto &src : *blk) {
    if (auto callOp = dyn_cast<func::CallOp>(&src)) {
      if (succeeded(handleCallOp(callOp, builder, mapper, batchSizes,
                                 batchedFunctionCache, batchedValues,
                                 broadcastCache)))
        continue;
    }

    if (isa<enzyme::ForwardDiffOp, enzyme::AutoDiffOp>(&src)) {
      if (succeeded(handleCallLikeOp(&src, "fn", builder, mapper, batchSizes,
                                     batchedFunctionCache, batchedValues,
                                     broadcastCache, insertBroadcastsLocally)))
        continue;
    }

    if (auto ifaceOp = dyn_cast<BatchOpInterface>(&src)) {
      bool shouldUseBatchIface = false;
      if (src.getNumOperands() == 0 && src.getNumResults() == 1) {
        Value res = src.getResult(0);
        shouldUseBatchIface =
            batchedValues.contains(res) ||
            (!res.use_empty() &&
             llvm::all_of(res.getUses(), [&](OpOperand &use) {
               return operandNeedsBatched(use, batchedValues, batchedResults,
                                          funcBody);
             }));
      } else {
        shouldUseBatchIface = llvm::any_of(src.getOperands(), [&](Value v) {
          return batchedValues.contains(v);
        });
      }

      if (shouldUseBatchIface) {
        IRMapping opMapper(mapper);
        for (Value operand : src.getOperands()) {
          if (!batchedValues.contains(operand)) {
            opMapper.map(operand,
                         getBatchedValue(operand, builder, mapper, batchSizes,
                                         batchedValues, broadcastCache,
                                         insertBroadcastsLocally));
          }
        }
        auto res = ifaceOp.createBatch(builder, opMapper, batchSizes);
        if (res.succeeded()) {
          for (Value oldRes : src.getResults())
            mapper.map(oldRes, opMapper.lookup(oldRes));
          continue;
        }
      }
    }

    SmallVector<Value, 8> operands;
    SmallVector<Block *, 2> successors;

    operands.reserve(src.getNumOperands());
    for (OpOperand &opnd : src.getOpOperands()) {
      if (operandNeedsBatched(opnd, batchedValues, batchedResults, funcBody)) {
        operands.push_back(getBatchedValue(
            opnd.get(), builder, mapper, batchSizes, batchedValues,
            broadcastCache, insertBroadcastsLocally));
      } else {
        operands.push_back(mapper.lookup(opnd.get()));
      }
    }

    if (withoutTerminator && src.hasTrait<OpTrait::IsTerminator>()) {
      for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
        mapper.map(src.getResult(i), operands[i]);
      continue;
    }

    successors.reserve(src.getNumSuccessors());
    for (Block *successor : src.getSuccessors())
      successors.push_back(mapper.lookup(successor));

    SmallVector<Type> resultTypes;
    resultTypes.reserve(src.getNumResults());
    for (Value res : src.getResults()) {
      resultTypes.push_back(batchedValues.contains(res)
                                ? applyBatchSizes(res.getType(), batchSizes)
                                : res.getType());
    }

    Operation *newOp = Operation::create(
        src.getLoc(), src.getName(), resultTypes, operands, src.getAttrs(),
        mlir::PropertyRef(), successors, src.getNumRegions());

    for (auto &&[oldReg, newReg] :
         llvm::zip(src.getRegions(), newOp->getRegions())) {
      batchCloneRegion(builder, &oldReg, &newReg, mapper, batchSizes,
                       batchedFunctionCache, batchedValues, batchedResults,
                       broadcastCache, funcBody);
    }

    for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
      mapper.map(src.getResult(i), newOp->getResult(i));

    builder.insert(newOp);
  }
}

void batchCloneBlock(
    OpBuilder &builder, Block *blk, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    bool withoutTerminator) {
  DenseSet<Value> allBatched;
  collectAllValuesInBlock(blk, allBatched);
  SmallVector<bool> allResultsBatched;
  if (!blk->empty() && blk->getTerminator()->hasTrait<OpTrait::ReturnLike>())
    allResultsBatched.assign(blk->getTerminator()->getNumOperands(), true);
  DenseMap<Value, Value> broadcastCache;
  batchCloneBlock(builder, blk, mapper, batchSizes, batchedFunctionCache,
                  withoutTerminator, allBatched, allResultsBatched,
                  broadcastCache, blk->getParent());
}

void batchCloneRegion(
    OpBuilder &builder, Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache,
    const llvm::DenseSet<Value> &batchedValues,
    llvm::ArrayRef<bool> batchedResults,
    llvm::DenseMap<Value, Value> &broadcastCache, Region *funcBody) {
  for (auto &blk : *src) {
    auto newBlk = new Block();
    dest->push_back(newBlk);

    mapper.map(&blk, newBlk);

    for (auto arg : blk.getArguments()) {
      Type argTy = batchedValues.contains(arg)
                       ? applyBatchSizes(arg.getType(), batchSizes)
                       : arg.getType();
      Value newArg = newBlk->addArgument(argTy, arg.getLoc());
      mapper.map(arg, newArg);
    }
  }

  for (auto &&[blk, newBlk] : llvm::zip(*src, *dest)) {
    IRRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToEnd(&newBlk);
    batchCloneBlock(builder, &blk, mapper, batchSizes, batchedFunctionCache,
                    false, batchedValues, batchedResults, broadcastCache,
                    funcBody);
  }
}

void batchCloneRegion(
    OpBuilder &builder, Region *src, Region *dest, IRMapping &mapper,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  DenseSet<Value> allBatched;
  for (Block &blk : *src)
    collectAllValuesInBlock(&blk, allBatched);
  SmallVector<bool> allResultsBatched;
  if (auto parentFunc =
          dyn_cast_or_null<FunctionOpInterface>(src->getParentOp()))
    allResultsBatched.assign(parentFunc.getNumResults(), true);
  DenseMap<Value, Value> broadcastCache;
  batchCloneRegion(builder, src, dest, mapper, batchSizes, batchedFunctionCache,
                   allBatched, allResultsBatched, broadcastCache, src);
}

FunctionOpInterface batchCloneFunction(
    OpBuilder &builder, FunctionOpInterface F, Twine name,
    llvm::ArrayRef<int64_t> batchSizes, llvm::ArrayRef<bool> batchedArgs,
    llvm::ArrayRef<bool> batchedResults,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  assert(!F.getFunctionBody().empty());

  BatchCacheKey key{
      F, SmallVector<int64_t>(batchSizes.begin(), batchSizes.end()),
      SmallVector<bool>(batchedArgs.begin(), batchedArgs.end()),
      SmallVector<bool>(batchedResults.begin(), batchedResults.end())};

  auto it = batchedFunctionCache.find(key);
  if (it != batchedFunctionCache.end())
    return it->second;

  DenseSet<Value> batchedValues;
  computeBatchedValues(F, batchedArgs, batchedValues);

  for (Block &blk : F.getFunctionBody()) {
    if (blk.empty())
      continue;
    Operation *term = blk.getTerminator();
    if (!term->hasTrait<OpTrait::ReturnLike>())
      continue;
    for (auto [i, retVal] : llvm::enumerate(term->getOperands())) {
      if (i < batchedResults.size() && !batchedResults[i] &&
          batchedValues.contains(retVal)) {
        F.emitError() << "return value #" << i
                      << " depends on batched arguments and must have batched "
                         "result type "
                      << applyBatchSizes(retVal.getType(), batchSizes);
        return FunctionOpInterface();
      }
    }
  }

  auto FTy = cast<FunctionType>(F.getFunctionType());

  llvm::SmallVector<mlir::Type> RetTypes;
  RetTypes.reserve(FTy.getNumResults());

  for (auto [i, Ty] : llvm::enumerate(FTy.getResults())) {
    bool isBatchedRes = i < batchedResults.size() ? batchedResults[i] : true;
    RetTypes.push_back(isBatchedRes ? applyBatchSizes(Ty, batchSizes) : Ty);
  }

  SmallVector<mlir::Type, 4> ArgTypes;
  ArgTypes.reserve(FTy.getNumInputs());

  for (auto [i, Ty] : llvm::enumerate(FTy.getInputs())) {
    bool isBatchedArg = i < batchedArgs.size() ? batchedArgs[i] : true;
    ArgTypes.push_back(isBatchedArg ? applyBatchSizes(Ty, batchSizes) : Ty);
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
  DenseMap<Value, Value> broadcastCache;
  batchCloneRegion(builder, &origReg, &newReg, mapper, batchSizes,
                   batchedFunctionCache, batchedValues, batchedResults,
                   broadcastCache, &origReg);

  return NewF;
}

FunctionOpInterface batchCloneFunction(
    OpBuilder &builder, FunctionOpInterface F, Twine name,
    llvm::ArrayRef<int64_t> batchSizes,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  auto FTy = cast<FunctionType>(F.getFunctionType());
  SmallVector<bool> allBatchedArgs(FTy.getNumInputs(), true);
  SmallVector<bool> allBatchedResults(FTy.getNumResults(), true);
  return batchCloneFunction(builder, F, name, batchSizes, allBatchedArgs,
                            allBatchedResults, batchedFunctionCache);
}

template <typename T>
FunctionOpInterface batchOperationWithoutInsertingCallOp(
    OpBuilder &builder, T CI, FunctionOpInterface fn,
    std::map<BatchCacheKey, FunctionOpInterface> &batchedFunctionCache) {
  SmallVector<bool> batchedArgs;
  batchedArgs.reserve(fn.getNumArguments());
  for (auto [actualVal, origTy] :
       llvm::zip_equal(CI.getInputs(), fn.getArgumentTypes()))
    batchedArgs.push_back(actualVal.getType() != origTy);

  SmallVector<bool> batchedResults;
  batchedResults.reserve(fn.getNumResults());
  for (auto [actualRes, origTy] :
       llvm::zip_equal(CI.getOutputs(), fn.getResultTypes()))
    batchedResults.push_back(actualRes.getType() != origTy);

  enzyme::batchutils::BatchCacheKey key{
      fn,
      SmallVector<int64_t>(CI.getBatchShape().begin(),
                           CI.getBatchShape().end()),
      batchedArgs, batchedResults};

  auto it = batchedFunctionCache.find(key);
  if (it != batchedFunctionCache.end())
    return it->second;

  std::string newFnName = "batched_" + fn.getName().str();
  return batchCloneFunction(builder, fn, newFnName, CI.getBatchShape(),
                            batchedArgs, batchedResults, batchedFunctionCache);
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
      op->walk([&](enzyme::BatchOp dop) { toLower.push_back(dop); });

      OpBuilder builder(op);

      for (auto T : toLower) {
        if (auto F = dyn_cast<enzyme::BatchOp>(T)) {
          auto res = enzyme::batchutils::batchOperation(symbolTable, builder, F,
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
