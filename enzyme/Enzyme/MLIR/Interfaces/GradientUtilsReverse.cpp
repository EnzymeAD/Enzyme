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
    MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_, MTypeResults TR_,
    BlockAndValueMapping &invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_, DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_, BlockAndValueMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode, unsigned width, bool omp)
    : newFunc(newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_), TA(TA_),
      TR(TR_), omp(omp), width(width), ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_),
      invertedPointers(invertedPointers_) {
  
  auto valueMap = invertedPointers.getValueMap();
  assert(invertedPointers.getBlockMap().begin() == invertedPointers.getBlockMap().end());
  for(auto it = valueMap.begin(); it != valueMap.end(); it++){
    mapInvertPointer(it->first, it->second);
  }

  /*
  for (BasicBlock &BB : *oldFunc) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        originalCalls.push_back(CI);
      }
    }
  }
  */

  /*
  for (BasicBlock &oBB : *oldFunc) {
    for (Instruction &oI : oBB) {
      newToOriginalFn[originalToNewFn[&oI]] = &oI;
    }
    newToOriginalFn[originalToNewFn[&oBB]] = &oBB;
  }
  for (Argument &oArg : oldFunc->args()) {
    newToOriginalFn[originalToNewFn[&oArg]] = &oArg;
  }
  */
  /*
  for (BasicBlock &BB : *newFunc) {
    originalBlocks.emplace_back(&BB);
  }
  tape = nullptr;
  tapeidx = 0;
  assert(originalBlocks.size() > 0);

  SmallVector<BasicBlock *, 4> ReturningBlocks;
  for (BasicBlock &BB : *oldFunc) {
    if (isa<ReturnInst>(BB.getTerminator()))
      ReturningBlocks.push_back(&BB);
  }
  for (BasicBlock &BB : *oldFunc) {
    bool legal = true;
    for (auto BRet : ReturningBlocks) {
      if (!(BRet == &BB || OrigDT.dominates(&BB, BRet))) {
        legal = false;
        break;
      }
    }
    if (legal)
      BlocksDominatingAllReturns.insert(&BB);
  }
  */
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

Block *
mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(mlir::Block *originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new blk from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Operation *
mlir::enzyme::MGradientUtilsReverse::getNewFromOriginal(Operation *originst) const {
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
  // TODO also block arguments
  // assert(TR.getFunction() == oldFunc);

  // Don't create derivatives for code that results in termination
  // if (notForAnalysis.find(&oBB) != notForAnalysis.end())
  //  continue;

  // LoopContext loopContext;
  // getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

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
    /*

    if (inst->getType()->isFPOrFPVectorTy())
      continue; //! op->getType()->isPointerTy() &&
                //! !op->getType()->isIntegerTy()) {

    if (!TR.query(inst)[{-1}].isPossiblePointer())
      continue;

    if (isa<LoadInst>(inst)) {
      IRBuilder<> BuilderZ(inst);
      getForwardBuilder(BuilderZ);
      Type *antiTy = getShadowType(inst->getType());
      PHINode *anti =
          BuilderZ.CreatePHI(antiTy, 1, inst->getName() + "'il_phi");
      invertedPointers.insert(std::make_pair(
          (const Value *)inst, InvertedPointerVH(this, anti)));
      continue;
    }

    if (!isa<CallInst>(inst)) {
      continue;
    }

    if (isa<IntrinsicInst>(inst)) {
      continue;
    }

    if (isConstantValue(inst)) {
      continue;
    }

    CallInst *op = cast<CallInst>(inst);
    Function *called = op->getCalledFunction();

    IRBuilder<> BuilderZ(inst);
    getForwardBuilder(BuilderZ);
    Type *antiTy = getShadowType(inst->getType());

    PHINode *anti =
        BuilderZ.CreatePHI(antiTy, 1, op->getName() + "'ip_phi");
    invertedPointers.insert(
        std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));

    if (called && isAllocationFunction(called->getName(), TLI)) {
      anti->setName(op->getName() + "'mi");
    }
    */
  });
}

LogicalResult MGradientUtilsReverse::visitChildReverse(Operation *op,
                                                OpBuilder &builder) {
  if (mode == DerivativeMode::ReverseModeGradient) {
    if (auto binst = dyn_cast<BranchOpInterface>(op)) {
      
    }
    else if (auto binst = dyn_cast<func::ReturnOp>(op)) {
      
    }
    else if (auto iface = dyn_cast<AutoDiffOpInterface>(op)) {
      //return iface.createReverseModeAdjoint(builder, this);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//

std::pair<BlockAndValueMapping, DenseMap<Block *, SmallVector<Value>>> createReverseModeBlocks(MGradientUtilsReverse *gutils){
  BlockAndValueMapping mapReverseModeBlocks;
  DenseMap<Block *, SmallVector<Value>> mapReverseBlockArguments;
  for (auto it = gutils->oldFunc.getBody().getBlocks().rbegin(); it != gutils->oldFunc.getBody().getBlocks().rend(); ++it) {
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
    auto regionPos = gutils->newFunc.getBody().end();
    gutils->newFunc.getBody().getBlocks().insert(regionPos, newBlock);
  }
  return std::pair<BlockAndValueMapping, DenseMap<Block *, SmallVector<Value>>> (mapReverseModeBlocks, mapReverseBlockArguments);
}

Block * insertInitializationBlock(MGradientUtilsReverse *gutils){
  Block *initializationBlock = new Block();
  int numArgs = gutils->newFunc.getNumArguments();
  for (int i = 0; i < numArgs; i++){
    BlockArgument oldArg = gutils->newFunc.getArgument(i);
    BlockArgument newArg = initializationBlock->addArgument(oldArg.getType(), oldArg.getLoc());
    oldArg.replaceAllUsesWith(newArg);
  }
  for (int i = 0; i < (numArgs-1) / 2; i++){
    gutils->mapInvertPointer(gutils->oldFunc.getArgument(i), initializationBlock->getArgument(2*i+1));
  }
  Block * oldEntry = &*(gutils->newFunc.getBody().begin());
  gutils->newFunc.getBody().begin()->eraseArguments(0,numArgs);
  auto initializationPos = gutils->newFunc.getBody().begin();
  gutils->newFunc.getBody().getBlocks().insert(initializationPos, initializationBlock);
  OpBuilder initializationBuilder(&*(gutils->newFunc.getBody().begin()), gutils->newFunc.getBody().begin()->begin());
  initializationBuilder.create<cf::BranchOp>(oldEntry->begin()->getLoc(), oldEntry);
  return initializationBlock;
}

Value initializeBackwardCacheValue(Type t, Block * initializationBlock){
  OpBuilder builder(initializationBlock, initializationBlock->begin());
  return builder.create<enzyme::CreateCacheOp>((initializationBlock->rbegin())->getLoc(), t);
}

SmallVector<mlir::Block*> getDominatorToposort(MGradientUtilsReverse *gutils){
  SmallVector<mlir::Block*> dominatorToposortBlocks;
  auto dInfo = mlir::detail::DominanceInfoBase<false>(nullptr);
  llvm::DominatorTreeBase<Block, false> & dt = dInfo.getDomTree(&(gutils->oldFunc.getBody()));
  auto root = dt.getNode(&*(gutils->oldFunc.getBody().begin()));

  for(llvm::DomTreeNodeBase<mlir::Block> * node : llvm::breadth_first(root)){
    dominatorToposortBlocks.push_back(node->getBlock());
  }
  
  return dominatorToposortBlocks;
}

FunctionOpInterface mlir::enzyme::MEnzymeLogic::CreateReverseDiff(
    FunctionOpInterface fn, DIFFE_TYPE retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented) {
  if (fn.getBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }

  bool retActive = retType != DIFFE_TYPE::CONSTANT;
  ReturnType returnValue = ReturnType::Tape;
  auto gutils = MDiffeGradientUtilsReverse::CreateFromClone(
      *this, mode, width, fn, TA, type_args, retType, /*diffeReturnArg*/ true,
      constants, returnValue, addedType, /*omp*/ false);

  const SmallPtrSet<mlir::Block *, 4> guaranteedUnreachable;
  gutils->forceAugmentedReturnsReverse();

  // Insert reversemode blocks
  auto reverseModeStore = createReverseModeBlocks(gutils);
  BlockAndValueMapping mapReverseModeBlocks = reverseModeStore.first;
  DenseMap<Block *, SmallVector<Value>> mapBlockArguments = reverseModeStore.second;
  
  // Insert initialization for caches TODO 
  Block *initializationBlock = insertInitializationBlock(gutils);

  // Get dominator tree and create topological sorting
  SmallVector<mlir::Block*> dominatorToposortBlocks = getDominatorToposort(gutils);

  //Iterate blocks and create reverse mode blocks
  for (auto it = dominatorToposortBlocks.rbegin(); it != dominatorToposortBlocks.rend(); ++it){
    Block& oBB = **it;

    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = gutils->getNewFromOriginal(&oBB);

      SmallVector<Operation *, 4> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : llvm::reverse(toerase)) {
        gutils->eraseIfUnused(I, true, false);
      }
      OpBuilder builder(gutils->oldFunc.getContext());
      builder.setInsertionPointToEnd(newBB);
      builder.create<LLVM::UnreachableOp>(gutils->oldFunc.getLoc());
      continue;
    }

    auto newBB = gutils->getNewFromOriginal(&oBB);
    if (oBB.getNumSuccessors() == 0){
      //gutils->oldFunc.getBody().getBlocks().front();
      OpBuilder forwardToBackwardBuilder(&*(newBB->rbegin())->getContext());
      forwardToBackwardBuilder.setInsertionPoint(gutils->getNewFromOriginal(&*(oBB.rbegin())));
      auto revBlock = mapReverseModeBlocks.lookupOrNull(&oBB);
      Operation * newBranchOp = forwardToBackwardBuilder.create<cf::BranchOp>(gutils->getNewFromOriginal(&*(oBB.rbegin()))->getLoc(), revBlock);
      
      Operation * returnStatement = newBB->getTerminator();
      gutils->mapInvertPointer(oBB.getTerminator()->getOperand(0), gutils->newFunc.getArgument(gutils->newFunc.getNumArguments() - 1)); 
      
      Operation * retVal = oBB.getTerminator();
      gutils->originalToNewFnOps[retVal] = newBranchOp;
      gutils->erase(returnStatement);
    }
    
    OpBuilder revBuilder(mapReverseModeBlocks.lookupOrNull(&oBB), mapReverseModeBlocks.lookupOrNull(&oBB)->begin());
    if (!oBB.empty()){
      auto first = oBB.rbegin();
      auto last = oBB.rend();
      for (auto it = first; it != last; ++it) {
        (void)gutils->visitChildReverse(&*it, revBuilder);
      }
    }

    if (oBB.hasNoPredecessors()){
      auto revBlock = mapReverseModeBlocks.lookupOrNull(&oBB);

      OpBuilder revBuilder(revBlock, revBlock->end());
      SmallVector<mlir::Value, 2> retargs;
      for (auto attribute : gutils->oldFunc.getBody().getArguments()) {
        auto attributeGradient = gutils->invertPointerReverseM(attribute, revBlock);
        retargs.push_back(attributeGradient);
      }
      
      revBuilder.create<func::ReturnOp>(oBB.rbegin()->getLoc(), retargs);
    }
    else {
      if (std::next(oBB.getPredecessors().begin()) == oBB.getPredecessors().end()){
        Block * predecessor = *(oBB.getPredecessors().begin());
        Block * predecessorRevMode = mapReverseModeBlocks.lookupOrNull(predecessor);
        // Create op operands
        SmallVector<Value> operands;
        auto argumentsIt = mapBlockArguments.find(predecessor);
        if (argumentsIt != mapBlockArguments.end()){
          for(auto operandOld : argumentsIt->second){
            Optional<Value> invertPointer = gutils->invertPointerReverseMOptional(operandOld, mapReverseModeBlocks.lookupOrNull(&oBB));
            if (invertPointer.has_value()){
              operands.push_back(invertPointer.value());
            }
            else{
              if (auto iface = operandOld.getType().cast<AutoDiffTypeInterface>()) {
                Value nullValue = iface.createNullValue(revBuilder, oBB.rbegin()->getLoc());
                operands.push_back(nullValue);
              }
            }
          }
        }
        ValueRange operandsValueRange(operands);
        revBuilder.create<cf::BranchOp>(gutils->getNewFromOriginal(&*(oBB.rbegin()))->getLoc(), predecessorRevMode, operandsValueRange);
      }
      else{
        Type indexType = mlir::IntegerType::get(initializationBlock->begin()->getContext(), 32);
        Type cacheType = CacheType::get(initializationBlock->begin()->getContext(), indexType);
        Value cache = initializeBackwardCacheValue(cacheType, initializationBlock);
        
        Value flag = revBuilder.create<enzyme::GetCacheOp>(oBB.rbegin()->getLoc(), indexType, cache);
        SmallVector<Block *> blocks;
        SmallVector<APInt> indices;
        SmallVector<ValueRange> arguments;
        ValueRange defaultArguments;
        Block * defaultBlock;
        int i = 0;
        for (auto it = oBB.getPredecessors().begin(); it != oBB.getPredecessors().end(); it++){
          Block * predecessor = *it;
          predecessor = mapReverseModeBlocks.lookupOrNull(predecessor);

          SmallVector<Value> operands;
          auto argumentsIt = mapBlockArguments.find(predecessor);
          if (argumentsIt != mapBlockArguments.end()){
            for(auto operandOld : argumentsIt->second){
              //TODO Create canonical null value if not found!
              Optional<Value> invertPointer = gutils->invertPointerReverseMOptional(operandOld, mapReverseModeBlocks.lookupOrNull(&oBB));
              if (invertPointer.has_value()){
                operands.push_back(invertPointer.value());
              }
              else{
                if (auto iface = operandOld.getType().cast<AutoDiffTypeInterface>()) {
                  Value nullValue = iface.createNullValue(revBuilder, oBB.rbegin()->getLoc());
                  operands.push_back(nullValue);
                }
              }
            }
          }

          if (it != oBB.getPredecessors().begin()){
            blocks.push_back(predecessor);
            indices.push_back(APInt(32, i++));
            arguments.push_back(ValueRange(operands));
          }
          else{
            defaultBlock = predecessor;
            defaultArguments = ValueRange(operands);
          }
        }
        revBuilder.create<cf::SwitchOp>(oBB.rbegin()->getLoc(), flag, defaultBlock, defaultArguments, ArrayRef<APInt>(indices), ArrayRef<Block *>(blocks), ArrayRef<ValueRange>(arguments));
      }
    }
  }

  auto nf = gutils->newFunc;

  llvm::errs() << "nf\n";
  nf.dump();
  llvm::errs() << "nf end\n";
  delete gutils;
  return nf;
}