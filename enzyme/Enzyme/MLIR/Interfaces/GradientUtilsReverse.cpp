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
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

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
    const SmallPtrSetImpl<mlir::Value> &activevals_, DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_, IRMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode_, unsigned width, SymbolTableCollection &symbolTable_)
    : newFunc(newFunc_), Logic(Logic), mode(mode_), oldFunc(oldFunc_), TA(TA_),
      width(width), ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_), symbolTable(symbolTable_) {

  initInitializationBlock(invertedPointers_, ArgDiffeTypes_);
}

// for(auto x : v.getUsers()){x->dump();} DEBUG

bool MGradientUtilsReverse::onlyUsedInParentBlock(Value v) {
  return !v.isUsedOutsideOfBlock(v.getParentBlock());
}

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
  auto pushOp = builder.create<enzyme::PushOp>(v.getLoc(), pushCache, v);
  return popCache;
}

Value MGradientUtilsReverse::popCache(Value cache, OpBuilder &builder) {
  return builder.create<enzyme::PopOp>(
      cache.getLoc(), cast<enzyme::CacheType>(cache.getType()).getType(),
      cache);
}

// Gradient
Type mlir::enzyme::MGradientUtilsReverse::getGradientType(Value v) {
  Type valueType = v.getType();
  return GradientType::get(v.getContext(), valueType);
}

Value mlir::enzyme::MGradientUtilsReverse::insertInitGradient(
    mlir::Value v, OpBuilder &builder) {
  Type gradientType = getGradientType(v);
  OpBuilder initBuilder(initializationBlock, initializationBlock->begin());
  Value gradient = initBuilder.create<enzyme::InitOp>(v.getLoc(), gradientType);
  return gradient;
}

// Shadow Gradient
Type mlir::enzyme::MGradientUtilsReverse::getShadowedGradientType(Value v) {
  Type valueType = v.getType();
  return ShadowedGradientType::get(v.getContext(), valueType);
}

Value mlir::enzyme::MGradientUtilsReverse::insertInitShadowedGradient(
    mlir::Value v, OpBuilder &builder) {
  Type gradientType = getShadowedGradientType(v);
  OpBuilder initBuilder(initializationBlock, initializationBlock->begin());
  Value gradient = initBuilder.create<enzyme::InitOp>(v.getLoc(), gradientType);
  return gradient;
}

Value mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(
    const mlir::Value originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new val from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Block *mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(
    mlir::Block *originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new blk from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Operation *mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(
    Operation *originst) const {
  auto found = originalToNewFnOps.find(originst);
  if (found == originalToNewFnOps.end()) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    for (auto &pair : originalToNewFnOps) {
      llvm::errs() << " map[" << pair.first << "] = " << pair.second << "\n";
    }
    llvm::errs() << originst << " - " << *originst << "\n";
    llvm_unreachable("Could not get new op from original");
  }
  return found->second;
}

Operation *
mlir::enzyme::MGradientUtilsReverse::cloneWithNewOperands(OpBuilder &B,
                                                          Operation *op) {
  IRMapping map;
  for (auto operand : op->getOperands())
    map.map(operand, getNewFromOriginal(operand));
  return B.clone(*op, map);
}

bool mlir::enzyme::MGradientUtilsReverse::isConstantValue(Value v) const {
  if (isa<mlir::IntegerType>(v.getType()))
    return true;
  if (isa<mlir::IndexType>(v.getType()))
    return true;

  if (matchPattern(v, m_Constant()))
    return true;

  // TODO
  return false;
}

bool mlir::enzyme::MGradientUtilsReverse::requiresShadow(Type t) {
  if (auto iface = dyn_cast<AutoDiffTypeInterface>(t)) {
    return iface.requiresShadow();
  }
  return false;
}

/*
The value v must have an invert pointer
*/
Value mlir::enzyme::MGradientUtilsReverse::invertPointerM(Value v,
                                                          OpBuilder &builder) {
  if (invertedPointersGlobal.contains(v)) {
    Value gradient = invertedPointersGlobal.lookupOrNull(v);
    Type type = gradient.getType();

    if (GradientType gType = dyn_cast<GradientType>(type)) {
      Value ret = builder.create<enzyme::GetOp>(v.getLoc(), gType.getBasetype(),
                                                gradient);
      return ret;
    } else {
      llvm_unreachable("found invalid type");
    }
  } else if (invertedPointersShadow.contains(v)) {
    Value gradient = invertedPointersShadow.lookupOrNull(v);
    Type type = gradient.getType();

    if (ShadowedGradientType gType =
            dyn_cast<enzyme::ShadowedGradientType>(type)) {
      Value ret = builder.create<enzyme::GetOp>(v.getLoc(), gType.getBasetype(),
                                                gradient);
      return ret;
    } else {
      llvm_unreachable("found invalid type");
    }
  }

  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

void mlir::enzyme::MGradientUtilsReverse::mapInvertPointer(
    mlir::Value v, mlir::Value invertValue, OpBuilder &builder) {
  if (!invertedPointersGlobal.contains(v)) {
    Value g = insertInitGradient(v, builder);
    invertedPointersGlobal.map(v, g);
  }
  Value gradient = invertedPointersGlobal.lookupOrNull(v);
  builder.create<enzyme::SetOp>(v.getLoc(), gradient, invertValue);
}

Value mlir::enzyme::MGradientUtilsReverse::getShadowValue(mlir::Value v) {
  return shadowValues.lookupOrNull(v);
}

void mlir::enzyme::MGradientUtilsReverse::mapShadowValue(mlir::Value v,
                                                         mlir::Value shadow,
                                                         OpBuilder &builder) {
  assert(!invertedPointersShadow.contains(
      v)); // Shadow Values must only be mapped exactly once

  Value cache = insertInitShadowedGradient(v, builder);
  invertedPointersShadow.map(v, cache);

  builder.create<enzyme::PushOp>(v.getLoc(), cache, shadow);

  shadowValues.map(v, shadow);
}

void mlir::enzyme::MGradientUtilsReverse::clearValue(mlir::Value v,
                                                     OpBuilder &builder) {
  if (invertedPointersGlobal.contains(v)) {
    if (!onlyUsedInParentBlock(v)) { // TODO is this necessary?
      Value gradient = invertedPointersGlobal.lookupOrNull(v);
      Type type = cast<GradientType>(gradient.getType()).getBasetype();
      if (auto iface = dyn_cast<AutoDiffTypeInterface>(type)) {
        Value zero = iface.createNullValue(builder, v.getLoc());
        builder.create<enzyme::SetOp>(v.getLoc(), gradient, zero);
      } else {
        llvm_unreachable(
            "Type does not have an associated AutoDiffTypeInterface");
      }
    }
  } else if (invertedPointersShadow.contains(v)) {
    Value gradient = invertedPointersShadow.lookupOrNull(v);
    builder.create<enzyme::ClearOp>(v.getLoc(), gradient);
  }
}

bool mlir::enzyme::MGradientUtilsReverse::hasInvertPointer(mlir::Value v) {
  return (invertedPointersGlobal.contains(v)) ||
         (invertedPointersShadow.contains(v));
}

void MGradientUtilsReverse::initInitializationBlock(
    IRMapping invertedPointers_, ArrayRef<DIFFE_TYPE> argDiffeTypes) {
  initializationBlock = &*(this->newFunc.getFunctionBody().begin());

  OpBuilder initializationBuilder(
      &*(this->newFunc.getFunctionBody().begin()),
      this->newFunc.getFunctionBody().begin()->begin());

  for (const auto &[val, diffe_type] : llvm::zip(
           this->oldFunc.getFunctionBody().getArguments(), argDiffeTypes)) {
    if (diffe_type != DIFFE_TYPE::OUT_DIFF) {
      continue;
    }
    auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType());
    if (!iface) {
      llvm_unreachable(
          "Type does not have an associated AutoDiffTypeInterface");
    }
    Value zero = iface.createNullValue(initializationBuilder, val.getLoc());
    mapInvertPointer(val, zero, initializationBuilder);
  }
  for (auto const &x : invertedPointers_.getValueMap()) {
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(x.first.getType())) {
      if (iface.requiresShadow()) {
        mapShadowValue(x.first, x.second,
                       initializationBuilder); // This may create an unnecessary
                                               // ShadowedGradient which could
                                               // be avoidable TODO
      } else {
        mapInvertPointer(x.first, x.second, initializationBuilder);
      }
    } else {
      llvm_unreachable("TODO not implemented");
    }
  }
}

void MGradientUtilsReverse::createReverseModeBlocks(Region &oldFunc,
                                                    Region &newFunc,
                                                    bool isParentRegion) {
  for (auto it = oldFunc.getBlocks().rbegin(); it != oldFunc.getBlocks().rend();
       ++it) {
    Block *block = &*it;
    Block *reverseBlock = new Block();

    SmallVector<std::pair<Value, Value>>
        reverseModeArguments; // Argument, Assigned value (2. is technically not
                              // necessary but simplifies code a lot)

    // Add reverse mode Arguments to Block
    Operation *term = block->getTerminator();
    mlir::BranchOpInterface brOp = dyn_cast<mlir::BranchOpInterface>(term);
    bool returnLike = term->hasTrait<mlir::OpTrait::ReturnLike>();
    if (brOp) {
      for (int i = 0; i < (int)term->getNumSuccessors(); i++) {
        SuccessorOperands sOps = brOp.getSuccessorOperands(i);
        Block *successorBlock = term->getSuccessor(i);

        assert(successorBlock->getNumArguments() == sOps.size());
        for (int j = 0; j < (int)sOps.size(); j++) {
          // Check if the argument needs a gradient
          if (auto iface = successorBlock->getArgument(j)
                               .getType()
                               .dyn_cast<AutoDiffTypeInterface>()) {
            reverseModeArguments.push_back(std::pair<Value, Value>(
                successorBlock->getArgument(j), sOps[j]));
          }
        }
      }
      for (auto it : reverseModeArguments) {
        reverseBlock->addArgument(it.second.getType(), it.second.getLoc());
      }

      mapBlockArguments[block] = reverseModeArguments;
    } else if (returnLike) {
      if (!isParentRegion) {
        for (OpOperand &operand : term->getOpOperands()) {
          Value val = operand.get();
          if (auto iface = val.getType().dyn_cast<AutoDiffTypeInterface>()) {
            reverseBlock->addArgument(val.getType(), val.getLoc());
          }
        }
      }
    }

    mapReverseModeBlocks.map(block, reverseBlock);
    newFunc.getBlocks().insert(newFunc.end(), reverseBlock);
  }
}

MGradientUtilsReverse *MGradientUtilsReverse::CreateFromClone(
    MEnzymeLogic &Logic, DerivativeMode mode_, unsigned width,
    FunctionOpInterface todiff, MTypeAnalysis &TA, MFnTypeInfo &oldTypeInfo,
    DIFFE_TYPE retType, bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
    ReturnType returnValue, mlir::Type additionalArg,
    SymbolTableCollection &symbolTable_) {
  std::string prefix;

  switch (mode_) {
  case DerivativeMode::ForwardMode:
  case DerivativeMode::ForwardModeSplit:
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
      nonconstant_values, returnvals, returnValue, retType,
      prefix + todiff.getName(), originalToNew, originalToNewOps,
      diffeReturnArg, additionalArg);

  return new MGradientUtilsReverse(
      Logic, newFunc, todiff, TA, invertedPointers, constant_values,
      nonconstant_values, retType, constant_args, originalToNew,
      originalToNewOps, mode_, width, symbolTable_);
}