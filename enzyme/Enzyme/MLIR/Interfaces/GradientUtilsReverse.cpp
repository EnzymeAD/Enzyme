//===- GradientUtilsReverse.cpp - Utilities for gradient interfaces --------------===//
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

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "mlir/IR/Dominance.h"
#include "CloneFunction.h"

using namespace mlir;
using namespace mlir::enzyme;

mlir::enzyme::MGradientUtilsReverse::MGradientUtilsReverse(
    MEnzymeLogic &Logic, 
    FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, 
    MTypeAnalysis &TA_, 
    MTypeResults TR_,
    BlockAndValueMapping invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_, 
    DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_, 
    BlockAndValueMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode_, 
    unsigned width, 
    BlockAndValueMapping mapReverseModeBlocks_, 
    DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments_,
    SymbolTableCollection &symbolTable_)
    : newFunc(newFunc_), 
      Logic(Logic), 
      mode(mode_), 
      oldFunc(oldFunc_), 
      TA(TA_),
      TR(TR_), 
      width(width), 
      ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_),
      mapReverseModeBlocks(mapReverseModeBlocks_),
      mapBlockArguments(mapBlockArguments_),
      symbolTable(symbolTable_){

  initInitializationBlock(invertedPointers_);
  initReturnBlocks();
}

//for(auto x : v.getUsers()){x->dump();} DEBUG

bool MGradientUtilsReverse::onlyUsedInParentBlock(Value v){
  return !v.isUsedOutsideOfBlock(v.getParentBlock());
}

Type mlir::enzyme::MGradientUtilsReverse::getIndexCacheType(){
  Type indexType = getIndexType();
  return getCacheType(indexType);
}

Type mlir::enzyme::MGradientUtilsReverse::getIndexType(){
  return mlir::IntegerType::get(initializationBlock->begin()->getContext(), 32);
}

Value mlir::enzyme::MGradientUtilsReverse::insertInit(Type t){
  OpBuilder builder(initializationBlock, initializationBlock->begin());
  return builder.create<enzyme::InitOp>((initializationBlock->rbegin())->getLoc(), t);
}

//Cache
Type mlir::enzyme::MGradientUtilsReverse::getCacheType(Type t){
  Type cacheType = CacheType::get(initializationBlock->begin()->getContext(), t);
  return cacheType;
}

Value MGradientUtilsReverse::initAndPushCache(Value v, OpBuilder& builder){
  Value cache = insertInit(getCacheType(v.getType()));
  builder.create<enzyme::PushOp>(v.getLoc(), cache, v);
  return cache;
}

Value MGradientUtilsReverse::popCache(Value cache, OpBuilder& builder){
  return builder.create<enzyme::PopOp>(cache.getLoc(), cast<enzyme::CacheType>(cache.getType()).getType(), cache);
}

//Gradient
Type mlir::enzyme::MGradientUtilsReverse::getGradientType(Value v){
  Type valueType = v.getType();
  return GradientType::get(v.getContext(), valueType);
}

Value mlir::enzyme::MGradientUtilsReverse::insertInitGradient(mlir::Value v, OpBuilder &builder){
  Type gradientType = getGradientType(v);
  OpBuilder initBuilder(initializationBlock, initializationBlock->begin());
  Value gradient = initBuilder.create<enzyme::InitOp>(v.getLoc(), gradientType);
  return gradient;
}

//Shadow Gradient
Type mlir::enzyme::MGradientUtilsReverse::getShadowGradientType(Value v){
  Type valueType = v.getType();
  return ShadowGradientType::get(v.getContext(), valueType);
}

Value mlir::enzyme::MGradientUtilsReverse::insertInitShadowGradient(mlir::Value v, OpBuilder &builder){
  Type gradientType = getShadowGradientType(v);
  OpBuilder initBuilder(initializationBlock, initializationBlock->begin());
  Value gradient = initBuilder.create<enzyme::InitOp>(v.getLoc(), gradientType);
  return gradient;
}

Value mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(const mlir::Value originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new val from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Block * mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(mlir::Block *originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new blk from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Operation * mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(Operation *originst) const {
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

Operation *mlir::enzyme::MGradientUtilsReverse::cloneWithNewOperands(OpBuilder &B, Operation *op) {
  BlockAndValueMapping map;
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

bool mlir::enzyme::MGradientUtilsReverse::requiresShadow(Type t){
  if(auto iface = dyn_cast<AutoDiffTypeInterface>(t)){
    return iface.requiresShadow();
  }
  return false;
}

/*
The value v must have an invert pointer
*/
Value mlir::enzyme::MGradientUtilsReverse::invertPointerM(Value v, OpBuilder &builder){
  if (invertedPointers.contains(v)){
    assert(onlyUsedInParentBlock(v));
    return invertedPointers.lookupOrNull(v);
  }
  else if(invertedPointersGlobal.contains(v)){
    assert(!onlyUsedInParentBlock(v));
    Value gradient = invertedPointersGlobal.lookupOrNull(v);
    Type type = gradient.getType();
    
    if (GradientType gType = dyn_cast<GradientType>(type)) {
      Value ret = builder.create<enzyme::GetOp>(v.getLoc(), gType.getBasetype(), gradient);
      return ret;
    }
    else{
      llvm_unreachable("found invalid type");
    }
  }
  else if(invertedPointersShadow.contains(v)){
    Value gradient = invertedPointersShadow.lookupOrNull(v);
    Type type = gradient.getType();
    
    if (ShadowGradientType gType = dyn_cast<enzyme::ShadowGradientType>(type)) {
      Value ret = builder.create<enzyme::GetOp>(v.getLoc(), gType.getBasetype(), gradient);
      return ret;
    }
    else{
      llvm_unreachable("found invalid type");
    }
  }

  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}


void mlir::enzyme::MGradientUtilsReverse::mapInvertPointer(mlir::Value v, mlir::Value invertValue, OpBuilder &builder){
  if (!invertedPointersGlobal.contains(v) && onlyUsedInParentBlock(v)){
    invertedPointers.map(v, invertValue);
  }
  else{
    if(!invertedPointersGlobal.contains(v)){
      Value g = insertInitGradient(v, builder);
      invertedPointersGlobal.map(v, g);
    }
    Value gradient = invertedPointersGlobal.lookupOrNull(v);
    builder.create<enzyme::SetOp>(v.getLoc(), gradient, invertValue);
  }
}

Value mlir::enzyme::MGradientUtilsReverse::getShadowValue(mlir::Value v){
  return shadowValues.lookupOrNull(v);
}

void mlir::enzyme::MGradientUtilsReverse::mapShadowValue(mlir::Value v, mlir::Value shadow, OpBuilder &builder){
  assert(!invertedPointersShadow.contains(v)); //Shadow Values must only be mapped exactly once

  Value cache = insertInitShadowGradient(v, builder);
  invertedPointersShadow.map(v, cache);

  builder.create<enzyme::PushOp>(v.getLoc(), cache, shadow);

  shadowValues.map(v, shadow);
}

void mlir::enzyme::MGradientUtilsReverse::clearValue(mlir::Value v, OpBuilder &builder){
  if (invertedPointers.contains(v)){
    // Do nothing
  }
  else if(invertedPointersGlobal.contains(v)){
    Value gradient = invertedPointersGlobal.lookupOrNull(v);
    Type type = cast<GradientType>(gradient.getType()).getBasetype();
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(type)){
      Value zero = iface.createNullValue(builder, v.getLoc());
      builder.create<enzyme::SetOp>(v.getLoc(), gradient, zero);
    }
    else{
      llvm_unreachable("Type does not have an associated AutoDiffTypeInterface");
    }
  }
  else if(invertedPointersShadow.contains(v)){
    Value gradient = invertedPointersShadow.lookupOrNull(v);
    builder.create<enzyme::ClearOp>(v.getLoc(), gradient);
  }
}

bool mlir::enzyme::MGradientUtilsReverse::hasInvertPointer(mlir::Value v){
  return (invertedPointers.contains(v)) || (invertedPointersGlobal.contains(v)) || (invertedPointersShadow.contains(v));
}

void MGradientUtilsReverse::initInitializationBlock(BlockAndValueMapping invertedPointers_){
  initializationBlock = &*(this->newFunc.getFunctionBody().begin());

  OpBuilder initializationBuilder(&*(this->newFunc.getFunctionBody().begin()), this->newFunc.getFunctionBody().begin()->begin());
  
  for (auto const& x : invertedPointers_.getValueMap()){
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(x.first.getType())){
      if(iface.requiresShadow()){
        mapShadowValue(x.first, x.second, initializationBuilder); //This may create an unnecessary ShadowGradient which could be avoidable TODO
      }
      else{
        mapInvertPointer(x.first, x.second, initializationBuilder);
      }
    }
    else{
      llvm_unreachable("TODO not implemented");
    }
  }
}

void MGradientUtilsReverse::initReturnBlocks(){
  returnBlocks.clear();
  for(auto it = oldFunc.getFunctionBody().getBlocks().begin(); it != oldFunc.getFunctionBody().getBlocks().end(); it++){
    Block * b = &*it;
    if(b->getNumSuccessors() == 0){
      returnBlocks.push_back(b);
    }
  }
}

std::pair<BlockAndValueMapping, DenseMap<Block *, SmallVector<std::pair<Value, Value>>>> createReverseModeBlocks(FunctionOpInterface & oldFunc, FunctionOpInterface & newFunc){
  BlockAndValueMapping mapReverseModeBlocks;
  DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapReverseBlockArguments;
  for (auto it = oldFunc.getFunctionBody().getBlocks().rbegin(); it != oldFunc.getFunctionBody().getBlocks().rend(); ++it) {
    Block * block = &*it;
    Block * reverseBlock = new Block();

    SmallVector<std::pair<Value, Value>> reverseModeArguments; // Argument, Assigned value (2. is technically not necessary but simplifies code a lot)

    //Add reverse mode Arguments to Block
    Operation * term = block->getTerminator();
    mlir::BranchOpInterface brOp = dyn_cast<mlir::BranchOpInterface>(term);
    if(brOp){
      for (int i = 0; i < (int)term->getNumSuccessors(); i++){
        SuccessorOperands sOps = brOp.getSuccessorOperands(i);
        Block * successorBlock = term->getSuccessor(i);
        
        assert(successorBlock->getNumArguments() == sOps.size());
        for (int j = 0; j < (int)sOps.size(); j++){
          reverseModeArguments.push_back(std::pair<Value,Value>(successorBlock->getArgument(j), sOps[j]));
        }
      }
      for (auto it : reverseModeArguments){
        reverseBlock->addArgument(it.second.getType(), it.second.getLoc());
      }

      mapReverseBlockArguments[block] = reverseModeArguments;
    }

    mapReverseModeBlocks.map(block, reverseBlock);
    newFunc.getFunctionBody().getBlocks().insert(newFunc.getFunctionBody().end(), reverseBlock);
  }
  return std::pair<BlockAndValueMapping, DenseMap<Block *, SmallVector<std::pair<Value, Value>>>> (mapReverseModeBlocks, mapReverseBlockArguments);
}


MDiffeGradientUtilsReverse::MDiffeGradientUtilsReverse(MEnzymeLogic &Logic, 
                      FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, 
                      MTypeAnalysis &TA,
                      MTypeResults TR, 
                      BlockAndValueMapping invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode_, 
                      unsigned width, 
                      BlockAndValueMapping mapReverseModeBlocks_, 
                      DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments_,
                      SymbolTableCollection &symbolTable_)
      : MGradientUtilsReverse(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_, constantvalues_, returnvals_, ActiveReturn, constant_values, origToNew_, origToNewOps_, mode, width, mapReverseModeBlocks_, mapBlockArguments_, symbolTable_) {}

MDiffeGradientUtilsReverse * MDiffeGradientUtilsReverse::CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode_, unsigned width, FunctionOpInterface todiff, MTypeAnalysis &TA, MFnTypeInfo &oldTypeInfo, DIFFE_TYPE retType, bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args, ReturnType returnValue, mlir::Type additionalArg, SymbolTableCollection &symbolTable_) {
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

    BlockAndValueMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    BlockAndValueMapping invertedPointers;
    FunctionOpInterface newFunc = CloneFunctionWithReturns(
        mode_, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);

    auto reverseModeBlockMapPair = createReverseModeBlocks(todiff, newFunc);
    BlockAndValueMapping mapReverseModeBlocks = reverseModeBlockMapPair.first;
    DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments = reverseModeBlockMapPair.second;


    MTypeResults TR; // TODO
    return new MDiffeGradientUtilsReverse(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values, nonconstant_values, retType, constant_args, originalToNew, originalToNewOps, mode_, width, mapReverseModeBlocks, mapBlockArguments, symbolTable_);
  }