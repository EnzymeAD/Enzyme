#include "llvm/ADT/APSInt.h"

#include "mlir/IR/BuiltinTypes.h"

#include "CloneFunction.h"

using namespace mlir;
using namespace mlir::enzyme;

Type getShadowType(Type type, unsigned width) {
  if (auto iface = dyn_cast<AutoDiffTypeInterface>(type))
    return iface.getShadowType(width);
  llvm::errs() << " type does not have autodifftypeinterface: " << type << "\n";
  exit(1);
}

template <typename T>
mlir::FunctionType
getFunctionTypeForClone(T FTy, DerivativeMode mode, unsigned width,
                        mlir::Type additionalArg,
                        const std::vector<bool> &returnPrimals,
                        const std::vector<bool> &returnShadows,
                        llvm::ArrayRef<DIFFE_TYPE> ReturnActivity,
                        llvm::ArrayRef<DIFFE_TYPE> ArgActivity) {
  static_assert(llvm::is_one_of<T, FunctionType, LLVM::LLVMFunctionType>::value,
                "Expected FunctionType or LLVMFunctionType");
  SmallVector<mlir::Type, 4> RetTypes;
  ArrayRef<Type> origInputTypes, origResultTypes;
  if constexpr (std::is_same<T, LLVM::LLVMFunctionType>::value) {
    origInputTypes = FTy.getParams();
    origResultTypes = FTy.getReturnTypes();
  } else {
    origInputTypes = FTy.getInputs();
    origResultTypes = FTy.getResults();
  }

  for (auto &&[Ty, returnPrimal, returnShadow, activity] : llvm::zip(
           origResultTypes, returnPrimals, returnShadows, ReturnActivity)) {
    if (returnPrimal) {
      RetTypes.push_back(Ty);
    }
    if (returnShadow) {
      assert(activity != DIFFE_TYPE::CONSTANT);
      assert(activity != DIFFE_TYPE::OUT_DIFF);
      RetTypes.push_back(getShadowType(Ty, width));
    }
  }

  SmallVector<mlir::Type, 4> ArgTypes;

  for (auto &&[ITy, act] : llvm::zip(origInputTypes, ArgActivity)) {
    ArgTypes.push_back(ITy);
    if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(getShadowType(ITy, width));
    } else if (act == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(ITy, width));
    }
  }

  for (auto &&[Ty, activity] : llvm::zip(origResultTypes, ReturnActivity)) {
    if (activity == DIFFE_TYPE::OUT_DIFF) {
      ArgTypes.push_back(getShadowType(Ty, width));
    }
  }

  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }

  // Create a new function type...
  OpBuilder builder(FTy.getContext());
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
  Operation *newOp = nullptr;

  if (!newOp) {
    SmallVector<Type> resultTypes(src->getResultTypes().begin(),
                                  src->getResultTypes().end());
    newOp = Operation::create(
        src->getLoc(), src->getName(), resultTypes, operands, src->getAttrs(),
        OpaqueProperties(nullptr), successors, src->getNumRegions());

    // Clone the regions.
    if (options.shouldCloneRegions()) {
      for (unsigned i = 0; i != src->getNumRegions(); ++i)
        cloneInto(&src->getRegion(i), &newOp->getRegion(i), mapper, opMap);
    }
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
      if (!mapper.contains(arg)) {
        auto Ty = arg.getType();
        mapper.map(arg, newBlock->addArgument(Ty, arg.getLoc()));
      }

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
    IRMapping &ptrInputs, ArrayRef<DIFFE_TYPE> ArgActivity,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals,
    const std::vector<bool> &returnPrimals,
    const std::vector<bool> &returnShadows, ArrayRef<DIFFE_TYPE> RetActivity,
    Twine name, IRMapping &VMap, std::map<Operation *, Operation *> &OpMap,
    mlir::Type additionalArg) {
  assert(!F.getFunctionBody().empty());
  // F = preprocessForClone(F, mode);
  // llvm::ValueToValueMapTy VMap;
  FunctionType FTy;
  if (auto llFTy = dyn_cast<LLVM::LLVMFunctionType>(F.getFunctionType())) {
    FTy = getFunctionTypeForClone(llFTy, mode, width, additionalArg,
                                  returnPrimals, returnShadows, RetActivity,
                                  ArgActivity);
  } else {
    FTy = getFunctionTypeForClone(cast<mlir::FunctionType>(F.getFunctionType()),
                                  mode, width, additionalArg, returnPrimals,
                                  returnShadows, RetActivity, ArgActivity);
  }

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
  SmallVector<Type> resultTypes(FTy.getResults());
  if (isa<LLVM::LLVMFuncOp>(F)) {
    if (resultTypes.empty()) {
      // llvm.func ops that return no results need to explicitly return
      // LLVMVoidType
      resultTypes.push_back(LLVM::LLVMVoidType::get(FTy.getContext()));
    } else if (resultTypes.size() > 1) {
      auto structType =
          LLVM::LLVMStructType::getLiteral(FTy.getContext(), resultTypes);
      resultTypes.clear();
      resultTypes.push_back(structType);
    }
  }
  NewF.setType(F.cloneTypeWith(FTy.getInputs(), resultTypes));

  Operation *parent = F->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  cloneInto(&F.getFunctionBody(), &NewF.getFunctionBody(), VMap, OpMap);

  {
    auto &blk = NewF.getFunctionBody().front();
    assert(F.getFunctionBody().front().getNumArguments() == ArgActivity.size());
    for (ssize_t i = ArgActivity.size() - 1; i >= 0; i--) {
      mlir::Value oval = F.getFunctionBody().front().getArgument(i);
      if (ArgActivity[i] == DIFFE_TYPE::CONSTANT)
        constants.insert(oval);
      else if (ArgActivity[i] == DIFFE_TYPE::OUT_DIFF)
        nonconstants.insert(oval);
      else if (ArgActivity[i] == DIFFE_TYPE::DUP_ARG ||
               ArgActivity[i] == DIFFE_TYPE::DUP_NONEED) {
        nonconstants.insert(oval);
        mlir::Value val = blk.getArgument(i);
        mlir::Value dval;
        if ((size_t)i == ArgActivity.size() - 1)
          dval = blk.addArgument(getShadowType(val.getType(), width),
                                 val.getLoc());
        else
          dval = blk.insertArgument(blk.args_begin() + i + 1,
                                    getShadowType(val.getType(), width),
                                    val.getLoc());
        ptrInputs.map(oval, dval);
      }
    }
    auto retloc = blk.getTerminator()->getLoc();
    ArrayRef<Type> resultTypes;
    if (auto llFTy = dyn_cast<LLVM::LLVMFunctionType>(F.getFunctionType()))
      resultTypes = llFTy.getReturnTypes();
    else
      resultTypes = cast<mlir::FunctionType>(F.getFunctionType()).getResults();

    for (auto &&[Ty, activity] : llvm::zip(resultTypes, RetActivity)) {
      if (activity == DIFFE_TYPE::OUT_DIFF) {
        blk.addArgument(getShadowType(Ty, width), retloc);
      }
    }
  }

  std::string ToClone[] = {
      "bufferization.writable",
      "mhlo.sharding",
      "mhlo.layout_mode",
      "xla_framework.input_mapping",
      "xla_framework.result_mapping",
  };
  size_t newxlacnt = 0;
  {
    size_t oldi = 0;
    size_t newi = 0;
    while (oldi < F.getNumResults()) {
      if (returnPrimals[oldi]) {
        for (auto attrName : ToClone) {
          auto attrNameS = StringAttr::get(F->getContext(), attrName);
          NewF.removeResultAttr(newi, attrNameS);
          if (auto attr = F.getResultAttr(oldi, attrName)) {
            if (attrName == "xla_framework.result_mapping") {
              auto iattr = cast<IntegerAttr>(attr);
              APSInt nc(iattr.getValue());
              nc = newxlacnt;
              attr = IntegerAttr::get(F->getContext(), nc);
              newxlacnt++;
            }
            NewF.setResultAttr(newi, attrNameS, attr);
          }
        }
        newi++;
      }
      if (returnShadows[oldi]) {
        for (auto attrName : ToClone) {
          auto attrNameS = StringAttr::get(F->getContext(), attrName);
          NewF.removeResultAttr(newi, attrNameS);
          if (auto attr = F.getResultAttr(oldi, attrName)) {
            if (attrName == "xla_framework.result_mapping") {
              auto iattr = cast<IntegerAttr>(attr);
              APSInt nc(iattr.getValue());
              nc = newxlacnt;
              attr = IntegerAttr::get(F->getContext(), nc);
              newxlacnt++;
            }
            NewF.setResultAttr(newi, attrNameS, attr);
          }
        }
        newi++;
      }
      oldi++;
    }
  }
  {
    size_t oldi = 0;
    size_t newi = 0;
    while (oldi < F.getNumArguments()) {
      for (auto attrName : ToClone) {
        NewF.removeArgAttr(newi, attrName);
        if (auto attr = F.getArgAttr(oldi, attrName)) {
          if (attrName == "xla_framework.input_mapping") {
            auto iattr = cast<IntegerAttr>(attr);
            APSInt nc(iattr.getValue());
            nc = newxlacnt;
            attr = IntegerAttr::get(F->getContext(), nc);
            newxlacnt++;
          }
          NewF.setArgAttr(newi, attrName, attr);
        }
      }

      newi++;
      if (ArgActivity[oldi] == DIFFE_TYPE::DUP_ARG ||
          ArgActivity[oldi] == DIFFE_TYPE::DUP_NONEED) {

        for (auto attrName : ToClone) {
          NewF.removeArgAttr(newi, attrName);
          if (auto attr = F.getArgAttr(oldi, attrName)) {
            if (attrName == "xla_framework.input_mapping") {
              auto iattr = cast<IntegerAttr>(attr);
              APSInt nc(iattr.getValue());
              nc = newxlacnt;
              attr = IntegerAttr::get(F->getContext(), nc);
              newxlacnt++;
            }
            NewF.setArgAttr(newi, attrName, attr);
          }
        }
        newi++;
      }
      oldi++;
    }
  }

  return NewF;
}
