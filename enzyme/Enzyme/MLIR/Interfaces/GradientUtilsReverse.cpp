//===- GradientUtilsReverse.cpp - Utilities for gradient interfaces
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/GradientUtilsReverse.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "CloneFunction.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

using namespace mlir;
using namespace mlir::enzyme;

mlir::enzyme::MGradientUtilsReverse::MGradientUtilsReverse(
    MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
    IRMapping invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_,
    ArrayRef<DIFFE_TYPE> ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
    IRMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode_, unsigned width)
    : MDiffeGradientUtils(Logic, newFunc_, oldFunc_, TA_, /*MTypeResults*/ {},
                          invertedPointers_, constantvalues_, activevals_,
                          ReturnActivity, ArgDiffeTypes_, originalToNewFn_,
                          originalToNewFnOps_, mode_, width, /*omp*/ false) {}

Type mlir::enzyme::MGradientUtilsReverse::getIndexCacheType() {
  Type indexType = getIndexType();
  return getCacheType(indexType);
}

Type mlir::enzyme::MGradientUtilsReverse::getIndexType() {
  return mlir::IntegerType::get(initializationBlock->begin()->getContext(), 32);
}

Value mlir::enzyme::MGradientUtilsReverse::insertInit(Type t) {
  OpBuilder builder(initializationBlock, initializationBlock->begin());
  return builder.create<enzyme::InitOp>(
      (initializationBlock->rbegin())->getLoc(), t);
}

// Cache
Type mlir::enzyme::MGradientUtilsReverse::getCacheType(Type t) {
  Type cacheType =
      CacheType::get(initializationBlock->begin()->getContext(), t);
  return cacheType;
}

void MGradientUtilsReverse::registerCacheCreatorHook(
    std::function<std::pair<Value, Value>(Type)> hook) {
  if (hook != nullptr) {
    cacheCreatorHook.push_back(hook);
  }
}

void MGradientUtilsReverse::deregisterCacheCreatorHook(
    std::function<std::pair<Value, Value>(Type)> hook) {
  if (hook != nullptr) {
    cacheCreatorHook.pop_back();
  }
}

std::pair<Value, Value> MGradientUtilsReverse::getNewCache(Type t) {
  if (cacheCreatorHook.empty()) {
    Value cache = insertInit(t);
    return {cache, cache};
  }
  return cacheCreatorHook.back()(t);
}

// We assume that caches will only be written to at one location. The returned
// cache is (might be) "pop only"
Value MGradientUtilsReverse::initAndPushCache(Value v, OpBuilder &builder) {
  auto [pushCache, popCache] = getNewCache(getCacheType(v.getType()));
  builder.create<enzyme::PushOp>(v.getLoc(), pushCache, v);
  return popCache;
}

Value MGradientUtilsReverse::popCache(Value cache, OpBuilder &builder) {
  return builder.create<enzyme::PopOp>(
      cache.getLoc(), cast<enzyme::CacheType>(cache.getType()).getType(),
      cache);
}

Operation *
mlir::enzyme::MGradientUtilsReverse::cloneWithNewOperands(OpBuilder &B,
                                                          Operation *op) {
  IRMapping map;
  for (auto operand : op->getOperands())
    map.map(operand, getNewFromOriginal(operand));
  return B.clone(*op, map);
}

void mlir::enzyme::MGradientUtilsReverse::addToDiffe(Value oldGradient,
                                                     Value addedGradient,
                                                     OpBuilder &builder) {
  assert(!isConstantValue(oldGradient));
  Value operandGradient = diffe(oldGradient, builder);
  auto iface = cast<AutoDiffTypeInterface>(addedGradient.getType());
  auto added = iface.createAddOp(builder, oldGradient.getLoc(), operandGradient,
                                 addedGradient);
  setDiffe(oldGradient, added, builder);
}

void MGradientUtilsReverse::createReverseModeBlocks(Region &oldFunc,
                                                    Region &newFunc) {
  for (auto it = oldFunc.getBlocks().rbegin(); it != oldFunc.getBlocks().rend();
       ++it) {
    Block *block = &*it;
    Block *reverseBlock = new Block();
    newFunc.getBlocks().insert(newFunc.end(), reverseBlock);
    mapReverseModeBlocks.map(block, reverseBlock);
  }
}

MGradientUtilsReverse *MGradientUtilsReverse::CreateFromClone(
    MEnzymeLogic &Logic, DerivativeMode mode_, unsigned width,
    FunctionOpInterface todiff, MTypeAnalysis &TA, MFnTypeInfo &oldTypeInfo,
    const std::vector<bool> &returnPrimals,
    const std::vector<bool> &returnShadows, ArrayRef<DIFFE_TYPE> retType,
    ArrayRef<DIFFE_TYPE> constant_args, mlir::Type additionalArg) {
  std::string prefix;

  switch (mode_) {
  case DerivativeMode::ForwardMode:
  case DerivativeMode::ForwardModeSplit:
  case DerivativeMode::ForwardModeError:
    assert(false);
    break;
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
    prefix = "diffe";
    break;
  case DerivativeMode::ReverseModePrimal:
    llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
  }

  if (width > 1)
    prefix += std::to_string(width);

  IRMapping originalToNew;
  std::map<Operation *, Operation *> originalToNewOps;

  SmallPtrSet<mlir::Value, 1> returnvals;
  SmallPtrSet<mlir::Value, 1> constant_values;
  SmallPtrSet<mlir::Value, 1> nonconstant_values;
  IRMapping invertedPointers;
  FunctionOpInterface newFunc = CloneFunctionWithReturns(
      mode_, width, todiff, invertedPointers, constant_args, constant_values,
      nonconstant_values, returnvals, returnPrimals, returnShadows, retType,
      prefix + todiff.getName(), originalToNew, originalToNewOps,
      additionalArg);

  return new MGradientUtilsReverse(Logic, newFunc, todiff, TA, invertedPointers,
                                   constant_values, nonconstant_values, retType,
                                   constant_args, originalToNew,
                                   originalToNewOps, mode_, width);
}
