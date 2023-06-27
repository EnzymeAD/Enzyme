#include "CloneFunction.h"

using namespace mlir;
using namespace mlir::enzyme;

Type getShadowType(Type type, unsigned width) {
  return type.cast<AutoDiffTypeInterface>().getShadowType(width);
}

mlir::FunctionType getFunctionTypeForClone(
    mlir::FunctionType FTy, DerivativeMode mode, unsigned width,
    mlir::Type additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE ReturnType) {
  SmallVector<mlir::Type, 4> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::Return) {
    assert(FTy.getNumResults() == 1);
    if (ReturnType != DIFFE_TYPE::CONSTANT &&
        ReturnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  } else if (returnValue == ReturnType::ArgsWithTwoReturns ||
             returnValue == ReturnType::TwoReturns) {
    assert(FTy.getNumResults() == 1);
    RetTypes.push_back(FTy.getResult(0));
    if (ReturnType != DIFFE_TYPE::CONSTANT &&
        ReturnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  }

  SmallVector<mlir::Type, 4> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;

  for (auto I : FTy.getInputs()) {
    ArgTypes.push_back(I);
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(getShadowType(I, width));
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(I, width));
    }
    ++argno;
  }

  // TODO: Expand for multiple returns
  if (diffeReturnArg) {
    ArgTypes.push_back(getShadowType(FTy.getResult(0), width));
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }

  OpBuilder builder(FTy.getContext());
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn) {
    RetTypes.insert(RetTypes.begin(),
                    LLVM::LLVMPointerType::get(builder.getIntegerType(8)));
  } else if (returnValue == ReturnType::Tape) {
    for (auto I : FTy.getInputs()) {
      RetTypes.push_back(I);
    }
  }

  // Create a new function type...
  return builder.getFunctionType(ArgTypes, RetTypes);
}

Operation *clone(Operation *src, IRMapping &mapper,
                 Operation::CloneOptions options,
                 std::map<Operation *, Operation *> &opMap) {
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;

  // Remap the operands.
  if (options.shouldCloneOperands()) {
    operands.reserve(src->getNumOperands());
    for (auto opValue : src->getOperands())
      operands.push_back(mapper.lookupOrDefault(opValue));
  }

  // Remap the successors.
  successors.reserve(src->getNumSuccessors());
  for (Block *successor : src->getSuccessors())
    successors.push_back(mapper.lookupOrDefault(successor));

  // Create the new operation.
  auto *newOp =
      src->create(src->getLoc(), src->getName(), src->getResultTypes(),
                  operands, src->getAttrs(), successors, src->getNumRegions());

  // Clone the regions.
  if (options.shouldCloneRegions()) {
    for (unsigned i = 0; i != src->getNumRegions(); ++i)
      cloneInto(&src->getRegion(i), &newOp->getRegion(i), mapper, opMap);
  }

  // Remember the mapping of any results.
  for (unsigned i = 0, e = src->getNumResults(); i != e; ++i)
    mapper.map(src->getResult(i), newOp->getResult(i));

  opMap[src] = newOp;
  return newOp;
}

void cloneInto(Region *src, Region *dest, IRMapping &mapper,
               std::map<Operation *, Operation *> &opMap) {
  cloneInto(src, dest, dest->end(), mapper, opMap);
}

/// Clone this region into 'dest' before the given position in 'dest'.
void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               IRMapping &mapper, std::map<Operation *, Operation *> &opMap) {
  assert(src);
  assert(dest && "expected valid region to clone into");
  assert(src != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (src->empty())
    return;

  // The below clone implementation takes special care to be read only for the
  // sake of multi threading. That essentially means not adding any uses to any
  // of the blocks or operation results contained within this region as that
  // would lead to a write in their use-def list. This is unavoidable for
  // 'Value's from outside the region however, in which case it is not read
  // only. Using the BlockAndValueMapper it is possible to remap such 'Value's
  // to ones owned by the calling thread however, making it read only once
  // again.

  // First clone all the blocks and block arguments and map them, but don't yet
  // clone the operations, as they may otherwise add a use to a block that has
  // not yet been mapped
  for (Block &block : *src) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    dest->getBlocks().insert(destPos, newBlock);
  }

  auto newBlocksRange =
      llvm::make_range(Region::iterator(mapper.lookup(&src->front())), destPos);

  // Now follow up with creating the operations, but don't yet clone their
  // regions, nor set their operands. Setting the successors is safe as all have
  // already been mapped. We are essentially just creating the operation results
  // to be able to map them.
  // Cloning the operands and region as well would lead to uses of operations
  // not yet mapped.
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto zippedBlocks : llvm::zip(*src, newBlocksRange)) {
    Block &sourceBlock = std::get<0>(zippedBlocks);
    Block &clonedBlock = std::get<1>(zippedBlocks);
    // Clone and remap the operations within this block.
    for (Operation &op : sourceBlock) {
      clonedBlock.push_back(clone(&op, mapper, cloneOptions, opMap));
    }
  }

  // Finally now that all operation results have been mapped, set the operands
  // and clone the regions.
  SmallVector<Value> operands;
  for (auto zippedBlocks : llvm::zip(*src, newBlocksRange)) {
    for (auto ops :
         llvm::zip(std::get<0>(zippedBlocks), std::get<1>(zippedBlocks))) {
      Operation &source = std::get<0>(ops);
      Operation &clone = std::get<1>(ops);

      operands.resize(source.getNumOperands());
      llvm::transform(
          source.getOperands(), operands.begin(),
          [&](Value operand) { return mapper.lookupOrDefault(operand); });
      clone.setOperands(operands);

      for (auto regions : llvm::zip(source.getRegions(), clone.getRegions()))
        cloneInto(&std::get<0>(regions), &std::get<1>(regions), mapper, opMap);
    }
  }
}

FunctionOpInterface CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, FunctionOpInterface F,
    IRMapping &ptrInputs, ArrayRef<DIFFE_TYPE> constant_args,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals, ReturnType returnValue,
    DIFFE_TYPE ReturnType, Twine name, IRMapping &VMap,
    std::map<Operation *, Operation *> &OpMap, bool diffeReturnArg,
    mlir::Type additionalArg) {
  assert(!F.getFunctionBody().empty());
  // F = preprocessForClone(F, mode);
  // llvm::ValueToValueMapTy VMap;
  auto FTy = getFunctionTypeForClone(
      F.getFunctionType().cast<mlir::FunctionType>(), mode, width,
      additionalArg, constant_args, diffeReturnArg, returnValue, ReturnType);

  /*
  for (Block &BB : F.getFunctionBody().getBlocks()) {
    if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (auto rv = ri->getReturnValue()) {
        returnvals.insert(rv);
      }
    }
  }
  */

  // Create the new function. This needs to go through the raw Operation API
  // instead of a concrete builder for genericity.
  auto NewF = cast<FunctionOpInterface>(F->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, name.str());
  NewF.setType(FTy);

  Operation *parent = F->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  cloneInto(&F.getFunctionBody(), &NewF.getFunctionBody(), VMap, OpMap);

  {
    auto &blk = NewF.getFunctionBody().front();
    for (ssize_t i = constant_args.size() - 1; i >= 0; i--) {
      mlir::Value oval = F.getFunctionBody().front().getArgument(i);
      if (constant_args[i] == DIFFE_TYPE::CONSTANT)
        constants.insert(oval);
      else if (constant_args[i] == DIFFE_TYPE::OUT_DIFF)
        nonconstants.insert(oval);
      else if (constant_args[i] == DIFFE_TYPE::DUP_ARG ||
               constant_args[i] == DIFFE_TYPE::DUP_NONEED) {
        nonconstants.insert(oval);
        mlir::Value val = blk.getArgument(i);
        mlir::Value dval;
        if (i == constant_args.size() - 1)
          dval = blk.addArgument(val.getType(), val.getLoc());
        else
          dval = blk.insertArgument(blk.args_begin() + i + 1, val.getType(),
                                    val.getLoc());
        ptrInputs.map(oval, dval);
      }
    }
    // TODO: Add support for mulitple outputs?
    if (diffeReturnArg) {
      auto location = blk.getArgument(blk.getNumArguments() - 1).getLoc();
      auto val = F.getFunctionType().cast<mlir::FunctionType>().getResult(0);
      mlir::Value dval = blk.addArgument(val, location);
    }
  }

  return NewF;
}