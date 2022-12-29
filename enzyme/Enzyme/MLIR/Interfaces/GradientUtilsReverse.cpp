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
    BlockAndValueMapping &invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_, 
    DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_, 
    BlockAndValueMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode, 
    unsigned width, 
    bool omp, 
    BlockAndValueMapping mapReverseModeBlocks_, 
    DenseMap<Block *, SmallVector<Value>> mapBlockArguments_)
    : newFunc(newFunc_), 
      Logic(Logic), 
      mode(mode), 
      oldFunc(oldFunc_), 
      TA(TA_),
      TR(TR_), 
      omp(omp), 
      width(width), 
      ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_),
      invertedPointers(invertedPointers_),
      mapReverseModeBlocks(mapReverseModeBlocks_),
      mapBlockArguments(mapBlockArguments_){
  
  auto valueMap = invertedPointers.getValueMap();
  assert(invertedPointers.getBlockMap().begin() == invertedPointers.getBlockMap().end());
  for(auto it = valueMap.begin(); it != valueMap.end(); it++){
    mapInvertPointer(it->first, it->second);
  }
  
  initializationBlock = insertInitializationBlock();
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
      // llvm::errs() << " map[" << pair.first << "] = " << pair.second << "
      // -- " << *pair.first << " " << *pair.second << "\n";
    }
    llvm::errs() << originst << " - " << *originst << "\n";
    llvm_unreachable("Could not get new op from original");
  }
  return found->second;
}

Operation *mlir::enzyme::MGradientUtilsReverse::cloneWithNewOperands(OpBuilder &B,
                                                              Operation *op) {
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

Value mlir::enzyme::MGradientUtilsReverse::invertPointerM(Value v,
                                                   OpBuilder &Builder2) {
  // TODO
  if (invertedPointers.contains(v))
    return invertedPointers.lookupOrNull(v);

  if (isConstantValue(v)) {
    if (auto iface = v.getType().cast<AutoDiffTypeInterface>()) {
      OpBuilder::InsertionGuard guard(Builder2);
      Builder2.setInsertionPoint(getNewFromOriginal(v.getDefiningOp()));
      Value dv = iface.createNullValue(Builder2, v.getLoc());
      mapInvertPointer(v, dv);
      return dv;
    }
    return getNewFromOriginal(v);
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

Value mlir::enzyme::MGradientUtilsReverse::invertPointerReverseM(Value v, Block * askingOp) {
  if (invertedPointersReverse.count(v) != 0){
    auto values = (invertedPointersReverse.find(v))->second;
    for (auto it = values.rbegin(); it != values.rend(); it++){
      if(mlir::DominanceInfo().dominates(it->getParentBlock(), askingOp)){
        return *it;
      }
    }
    llvm::errs() << "could not find in vector " << v << "\n";
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

Optional<Value> mlir::enzyme::MGradientUtilsReverse::invertPointerReverseMOptional(Value v, Block * askingOp) {
  if (invertedPointersReverse.count(v) != 0){
    auto values = (invertedPointersReverse.find(v))->second;
    for (auto it = values.rbegin(); it != values.rend(); it++){
      if(mlir::DominanceInfo().dominates(it->getParentBlock(), askingOp)){
        return *it;
      }
    }
  }
  return Optional<Value>();
}

void mlir::enzyme::MGradientUtilsReverse::mapInvertPointer(mlir::Value v, mlir::Value invertValue){
  invertedPointers.map(v, invertValue);
  if (invertedPointersReverse.count(v) == 0){
    invertedPointersReverse[v] = SmallVector<mlir::Value, 4>();
  }
  invertedPointersReverse[v].push_back(invertValue);
}

bool mlir::enzyme::MGradientUtilsReverse::hasInvertPointer(mlir::Value v){
  return invertedPointersReverse.count(v) != 0;
}

void mlir::enzyme::MGradientUtilsReverse::forceAugmentedReturnsReverse() {
  assert(mode == DerivativeMode::ReverseModeGradient);

  oldFunc.walk([&](Block *blk) {
    if (blk == &oldFunc.getBody().getBlocks().front())
      return;
    auto nblk = getNewFromOriginal(blk);
    for (auto val : llvm::reverse(blk->getArguments())) {
      if (isConstantValue(val))
        continue;
      auto i = val.getArgNumber();
      mlir::Value dval;
      if (i == blk->getArguments().size() - 1)
        dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
      else
        dval = nblk->insertArgument(nblk->args_begin() + i + 1,
                                    getShadowType(val.getType()), val.getLoc());

      mapInvertPointer(val, dval);
    }
  });

  int index = oldFunc.getNumArguments() - 1;
  auto argument = oldFunc.getArgument(index);
  oldFunc.walk([&](Block *blk) {
    auto terminator = blk->getTerminator();
    if (terminator->hasTrait<OpTrait::ReturnLike>()) {
      auto nblk = getNewFromOriginal(blk);
      mapInvertPointer(terminator->getOperand(0), argument);
    }
  });
}

void mlir::enzyme::MGradientUtilsReverse::forceAugmentedReturns() {

  oldFunc.walk([&](Block *blk) {
    if (blk == &oldFunc.getBody().getBlocks().front())
      return;
    auto nblk = getNewFromOriginal(blk);
    for (auto val : llvm::reverse(blk->getArguments())) {
      if (isConstantValue(val))
        continue;
      auto i = val.getArgNumber();
      mlir::Value dval;
      if (i == blk->getArguments().size() - 1)
        dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
      else
        dval = nblk->insertArgument(nblk->args_begin() + i + 1,
                                    getShadowType(val.getType()), val.getLoc());

      mapInvertPointer(val, dval);
    }
  });

  oldFunc.walk([&](Operation *inst) {
    if (inst == oldFunc)
      return;
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      OpBuilder BuilderZ(getNewFromOriginal(inst));
      for (auto res : inst->getResults()) {
        if (!isConstantValue(res)) {
          mlir::Type antiTy = getShadowType(res.getType());
          auto anti = BuilderZ.create<enzyme::PlaceholderOp>(res.getLoc(),
                                                             res.getType());
          mapInvertPointer(res, anti);
        }
      }
      return;
    }
  });
}

LogicalResult MGradientUtilsReverse::visitChildReverse(Operation *op, OpBuilder &builder) {
  if (mode == DerivativeMode::ReverseModeGradient) {
    if (auto binst = dyn_cast<BranchOpInterface>(op)) {
      
    }
    else if (auto binst = dyn_cast<func::ReturnOp>(op)) {
      
    }
    else if (auto iface = dyn_cast<AutoDiffOpInterfaceReverse>(op)) {
      return iface.createReverseModeAdjoint(builder, this);
    }
  }
  return success();
}

Block * MGradientUtilsReverse::insertInitializationBlock(){
  Block *initializationBlock = new Block();
  int numArgs = this->newFunc.getNumArguments();
  for (int i = 0; i < numArgs; i++){
    BlockArgument oldArg = this->newFunc.getArgument(i);
    BlockArgument newArg = initializationBlock->addArgument(oldArg.getType(), oldArg.getLoc());
    oldArg.replaceAllUsesWith(newArg);
  }
  for (int i = 0; i < (numArgs-1) / 2; i++){
    this->mapInvertPointer(this->oldFunc.getArgument(i), initializationBlock->getArgument(2*i+1));
  }
  Block * oldEntry = &*(this->newFunc.getBody().begin());
  this->newFunc.getBody().begin()->eraseArguments(0,numArgs);
  auto initializationPos = this->newFunc.getBody().begin();
  this->newFunc.getBody().getBlocks().insert(initializationPos, initializationBlock);
  OpBuilder initializationBuilder(&*(this->newFunc.getBody().begin()), this->newFunc.getBody().begin()->begin());
  initializationBuilder.create<cf::BranchOp>(oldEntry->begin()->getLoc(), oldEntry);
  return initializationBlock;
}

std::pair<BlockAndValueMapping, DenseMap<Block *, SmallVector<Value>>> createReverseModeBlocks(FunctionOpInterface & oldFunc, FunctionOpInterface & newFunc){
  BlockAndValueMapping mapReverseModeBlocks;
  DenseMap<Block *, SmallVector<Value>> mapReverseBlockArguments;
  for (auto it = oldFunc.getBody().getBlocks().rbegin(); it != oldFunc.getBody().getBlocks().rend(); ++it) {
    Block *block = &*it;
    Block *newBlock = new Block();

    //Add reverse mode Arguments to Block
    Operation * term = block->getTerminator();
    mlir::BranchOpInterface brOp = dyn_cast<mlir::BranchOpInterface>(term);
    SmallVector<Value> reverseModeArguments;
    if(brOp){
      DenseMap<Value, int> valueToIndex;
      int index = 0;
      // Taking the number of successors might be a false assumtpion on the BranchOpInterface
      // TODO make nicer algorithm for this!
      for (int i = 0; i < term->getNumSuccessors(); i++){
        SuccessorOperands sOps = brOp.getSuccessorOperands(i);
        for (int j = 0; j < sOps.size(); j++){
          if (valueToIndex.count(sOps[i]) == 0){
            valueToIndex[sOps[i]] = index;
            index++;
          }
        }
      }
      reverseModeArguments.resize(index);
      for (auto it : valueToIndex){
        reverseModeArguments[it.second] = it.first;
        newBlock->addArgument(it.first.getType(), it.first.getLoc());
      }
      mapReverseBlockArguments[block] = reverseModeArguments;
    }
    //

    mapReverseModeBlocks.map(block, newBlock);
    auto regionPos = newFunc.getBody().end();
    newFunc.getBody().getBlocks().insert(regionPos, newBlock);
  }
  return std::pair<BlockAndValueMapping, DenseMap<Block *, SmallVector<Value>>> (mapReverseModeBlocks, mapReverseBlockArguments);
}


MDiffeGradientUtilsReverse::MDiffeGradientUtilsReverse(MEnzymeLogic &Logic, 
                      FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, 
                      MTypeAnalysis &TA,
                      MTypeResults TR, 
                      BlockAndValueMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, 
                      unsigned width, 
                      bool omp,
                      BlockAndValueMapping mapReverseModeBlocks_, 
                      DenseMap<Block *, SmallVector<Value>> mapBlockArguments_)
      : MGradientUtilsReverse(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_, constantvalues_, returnvals_, ActiveReturn, constant_values, origToNew_, origToNewOps_, mode, width, omp, mapReverseModeBlocks_, mapBlockArguments_) {}

MDiffeGradientUtilsReverse * MDiffeGradientUtilsReverse::CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode, unsigned width,
                  FunctionOpInterface todiff, MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, mlir::Type additionalArg, bool omp) {
    std::string prefix;

    switch (mode) {
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      prefix = "fwddiffe";
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
        mode, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);

    auto reverseModeBlockMapPair = createReverseModeBlocks(todiff, newFunc);
    BlockAndValueMapping mapReverseModeBlocks = reverseModeBlockMapPair.first;
    DenseMap<Block *, SmallVector<Value>> mapBlockArguments = reverseModeBlockMapPair.second;

    MTypeResults TR; // TODO
    return new MDiffeGradientUtilsReverse(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values, nonconstant_values, retType, constant_args, originalToNew, originalToNewOps, mode, width, omp, mapReverseModeBlocks, mapBlockArguments);
  }