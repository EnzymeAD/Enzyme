#include "Interfaces/GradientUtils.h"
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

#include "GradientUtils.h"
#include "EnzymeLogic.h"

using namespace mlir;
using namespace mlir::enzyme;

bool onlyUsedInParentBlock(Value v){
  return v.isUsedOutsideOfBlock(v.getParentBlock());
}

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