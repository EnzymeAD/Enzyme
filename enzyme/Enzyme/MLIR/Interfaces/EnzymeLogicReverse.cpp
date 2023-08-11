#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

#include "EnzymeLogic.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::enzyme;

SmallVector<mlir::Block *>
MEnzymeLogic::getDominatorToposort(MGradientUtilsReverse *gutils,
                                   Region &region) {
  SmallVector<mlir::Block *> dominatorToposortBlocks;
  if (region.hasOneBlock()) {
    dominatorToposortBlocks.push_back(&*(region.begin()));
  } else {
    auto dInfo = mlir::detail::DominanceInfoBase<false>(nullptr);
    llvm::DominatorTreeBase<Block, false> &dt =
        dInfo.getDomTree(&(gutils->oldFunc.getFunctionBody()));
    auto root = dt.getNode(&*(region.begin()));

    for (llvm::DomTreeNodeBase<mlir::Block> *node : llvm::breadth_first(root)) {
      dominatorToposortBlocks.push_back(node->getBlock());
    }
  }
  return dominatorToposortBlocks;
}

void MEnzymeLogic::mapInvertArguments(Block *oBB, Block *reverseBB,
                                      MGradientUtilsReverse *gutils) {
  OpBuilder builder(reverseBB, reverseBB->begin());
  for (int i = 0; i < (int)gutils->mapBlockArguments[oBB].size(); i++) {
    auto x = gutils->mapBlockArguments[oBB][i];
    if (auto iface = x.second.getType().dyn_cast<AutoDiffTypeInterface>()) {
      Value added = reverseBB->getArgument(i);
      if (gutils->hasInvertPointer(x.second)) {
        added = iface.createAddOp(builder, x.second.getLoc(), added,
                                  gutils->invertPointerM(x.second, builder));
      }
      gutils->mapInvertPointer(x.second, added, builder);
    }
  }
}

void MEnzymeLogic::handleReturns(Block *oBB, Block *newBB, Block *reverseBB,
                                 MGradientUtilsReverse *gutils,
                                 bool parentRegion) {
  if (oBB->getNumSuccessors() == 0) {
    if (parentRegion) {
      Operation *returnStatement = newBB->getTerminator();
      gutils->erase(returnStatement);

      OpBuilder forwardToBackwardBuilder(newBB, newBB->end());
      gutils->mapInvertPointer(
          oBB->getTerminator()->getOperand(0),
          gutils->newFunc.getArgument(gutils->newFunc.getNumArguments() - 1),
          forwardToBackwardBuilder); // TODO handle multiple return values
      Operation *newBranchOp = forwardToBackwardBuilder.create<cf::BranchOp>(
          oBB->getTerminator()->getLoc(), reverseBB);

      gutils->originalToNewFnOps[oBB->getTerminator()] = newBranchOp;
    } else {
      Operation *terminator = oBB->getTerminator();
      OpBuilder builder(reverseBB, reverseBB->begin());

      int i = 0;
      for (OpOperand &operand : terminator->getOpOperands()) {
        Value val = operand.get();
        if (auto iface = val.getType().dyn_cast<AutoDiffTypeInterface>()) {
          gutils->mapInvertPointer(val, reverseBB->getArgument(i), builder);
          i++;
        }
      }
    }
  }
}

bool MEnzymeLogic::visitChildCustom(Operation *op, OpBuilder &builder,
                                    MGradientUtilsReverse *gutils) {
  std::string nameDiffe = "diffe_" + op->getName().getDialectNamespace().str() +
                          "_" + op->getName().stripDialect().str();
  std::string nameStore = "store_" + op->getName().getDialectNamespace().str() +
                          "_" + op->getName().stripDialect().str();

  StringRef srDiffe(nameDiffe);
  StringRef srStore(nameStore);

  OperationName opNameDiffe(srDiffe, op->getContext());
  OperationName opNameStore(srStore, op->getContext());

  Operation *symbolDiffe = gutils->symbolTable.lookupNearestSymbolFrom(
      op, opNameDiffe.getIdentifier());
  Operation *symbolStore = gutils->symbolTable.lookupNearestSymbolFrom(
      op, opNameStore.getIdentifier());

  if (symbolDiffe != nullptr) {
    SmallVector<Value> caches;
    if (symbolStore != nullptr) {
      Operation *newOp = gutils->getNewFromOriginal(op);

      func::FuncOp funcStore = cast<func::FuncOp>(symbolStore);

      SmallVector<Type, 2> storeResultTypes;
      for (auto x : funcStore.getFunctionType().getResults()) {
        storeResultTypes.push_back(x);
      }

      SmallVector<Value, 2> storeArgs;
      for (auto x : newOp->getOperands()) {
        storeArgs.push_back(x);
      }

      OpBuilder storeBuilder(newOp);
      func::CallOp storeCI = storeBuilder.create<func::CallOp>(
          op->getLoc(), srStore, storeResultTypes, storeArgs);
      for (auto x : storeCI.getResults()) {
        caches.push_back(gutils->initAndPushCache(x, storeBuilder));
      }
    }

    SmallVector<Value> args;
    for (Value opResult : op->getResults()) {
      if (gutils->hasInvertPointer(opResult)) {
        Value invertValue = gutils->invertPointerM(opResult, builder);
        args.push_back(invertValue);
      }
    }
    for (Value cache : caches) {
      args.push_back(gutils->popCache(cache, builder));
    }

    SmallVector<Type, 2> resultTypes;
    for (auto x : op->getOperands()) {
      resultTypes.push_back(x.getType());
    }

    func::CallOp dCI =
        builder.create<func::CallOp>(op->getLoc(), srDiffe, resultTypes, args);
    for (int i = 0; i < (int)op->getNumOperands(); i++) {
      gutils->mapInvertPointer(op->getOperand(i), dCI.getResult(i), builder);
    }

    return true;
  }
  return false;
}

/*
Create reverse mode adjoint for an operation.
*/
void MEnzymeLogic::visitChild(Operation *op, OpBuilder &builder,
                              MGradientUtilsReverse *gutils) {
  if (auto ifaceOp = dyn_cast<ReverseAutoDiffOpInterface>(op)) {
    SmallVector<Value> caches = ifaceOp.cacheValues(gutils);
    ifaceOp.createReverseModeAdjoint(builder, gutils, caches);

    for (int indexResult = 0; indexResult < (int)op->getNumResults();
         indexResult++) {
      Value result = op->getResult(indexResult);
      gutils->clearValue(result, builder);
    }
  }
}

void MEnzymeLogic::visitChildren(Block *oBB, Block *reverseBB,
                                 MGradientUtilsReverse *gutils) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  if (!oBB->empty()) {
    auto first = oBB->rbegin();
    auto last = oBB->rend();
    for (auto it = first; it != last; ++it) {
      Operation *op = &*it;
      bool customFound = visitChildCustom(op, revBuilder, gutils);
      if (!customFound) {
        visitChild(op, revBuilder, gutils);
      }
    }
  }
}

void MEnzymeLogic::handlePredecessors(
    Block *oBB, Block *newBB, Block *reverseBB, MGradientUtilsReverse *gutils,
    llvm::function_ref<buildReturnFunction> buildReturnOp, bool parentRegion) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  if (oBB->hasNoPredecessors()) {
    SmallVector<mlir::Value> retargs;
    // We need different handling on the top level due to
    // the presence of duplicated args since we don't yet have activity analysis
    if (parentRegion) {
      assert(gutils->ArgDiffeTypes.size() ==
                 gutils->oldFunc.getNumArguments() &&
             "Mismatch of activity array size vs # original function args");
      for (const auto &[diffeType, oldArg] :
           llvm::zip(gutils->ArgDiffeTypes, oBB->getArguments())) {
        if (diffeType == DIFFE_TYPE::OUT_DIFF) {
          retargs.push_back(gutils->invertPointerM(oldArg, revBuilder));
        }
      }
    } else {
      for (auto arg : oBB->getArguments()) {
        if (gutils->hasInvertPointer(arg)) {
          retargs.push_back(gutils->invertPointerM(arg, revBuilder));
        }
      }
    }
    buildReturnOp(revBuilder, oBB->rbegin()->getLoc(), retargs);
  } else {
    SmallVector<Block *> blocks;
    SmallVector<APInt> indices;
    SmallVector<ValueRange> arguments;
    SmallVector<Value> defaultArguments;
    Block *defaultBlock;
    int i = 1;
    for (Block *predecessor : oBB->getPredecessors()) {
      Block *predecessorRevMode =
          gutils->mapReverseModeBlocks.lookupOrNull(predecessor);

      SmallVector<Value> operands;
      auto argumentsIt = gutils->mapBlockArguments.find(predecessor);
      if (argumentsIt != gutils->mapBlockArguments.end()) {
        for (auto operandOld : argumentsIt->second) {
          if (oBB == operandOld.first.getParentBlock() &&
              gutils->hasInvertPointer(operandOld.first)) {
            operands.push_back(
                gutils->invertPointerM(operandOld.first, revBuilder));
          } else {
            if (auto iface = operandOld.first.getType()
                                 .dyn_cast<AutoDiffTypeInterface>()) {
              Value nullValue =
                  iface.createNullValue(revBuilder, oBB->rbegin()->getLoc());
              operands.push_back(nullValue);
            } else {
              llvm_unreachable("no canonial null value found");
            }
          }
        }
      }
      if (predecessor != *(oBB->getPredecessors().begin())) {
        blocks.push_back(predecessorRevMode);
        indices.push_back(APInt(32, i++));
        arguments.push_back(operands);
      } else {
        defaultBlock = predecessorRevMode;
        defaultArguments = operands;
      }
    }

    // Clear invert pointers of all arguments with gradient
    for (auto argument : oBB->getArguments()) {
      if (gutils->hasInvertPointer(argument)) {
        auto iface = argument.getType().cast<AutoDiffTypeInterface>();
        Value nullValue = iface.createNullValue(revBuilder, argument.getLoc());
        gutils->mapInvertPointer(argument, nullValue, revBuilder);
      }
    }

    Location loc = oBB->rbegin()->getLoc();
    // Remove Dependency to CF dialect
    if (std::next(oBB->getPredecessors().begin()) ==
        oBB->getPredecessors().end()) {
      // If there is only one block we can directly create a branch for
      // simplicity sake
      revBuilder.create<cf::BranchOp>(loc, defaultBlock, defaultArguments);
    } else {
      Value cache = gutils->insertInit(gutils->getIndexCacheType());
      Value flag =
          revBuilder.create<enzyme::PopOp>(loc, gutils->getIndexType(), cache);

      revBuilder.create<cf::SwitchOp>(
          loc, flag, defaultBlock, defaultArguments, ArrayRef<APInt>(indices),
          ArrayRef<Block *>(blocks), ArrayRef<ValueRange>(arguments));

      Value origin = newBB->addArgument(gutils->getIndexType(), loc);

      OpBuilder newBuilder(newBB, newBB->begin());
      newBuilder.create<enzyme::PushOp>(loc, cache, origin);

      int j = 0;
      for (Block *predecessor : oBB->getPredecessors()) {
        Block *newPredecessor = gutils->getNewFromOriginal(predecessor);

        OpBuilder predecessorBuilder(newPredecessor,
                                     std::prev(newPredecessor->end()));
        Value indicator =
            predecessorBuilder.create<arith::ConstantIntOp>(loc, j++, 32);

        Operation *terminator = newPredecessor->getTerminator();
        if (auto binst = dyn_cast<BranchOpInterface>(terminator)) {
          for (unsigned i = 0; i < terminator->getNumSuccessors(); i++) {
            if (terminator->getSuccessor(i) == newBB) {
              SuccessorOperands sOps = binst.getSuccessorOperands(i);
              sOps.append(indicator);
            }
          }
        } else {
          llvm_unreachable("invalid terminator");
        }
      }
    }
  }
}

void MEnzymeLogic::initializeShadowValues(
    SmallVector<mlir::Block *> &dominatorToposortBlocks,
    MGradientUtilsReverse *gutils) {
  for (auto it = dominatorToposortBlocks.begin();
       it != dominatorToposortBlocks.end(); ++it) {
    Block *oBB = *it;

    if (!oBB->empty()) {
      for (auto it = oBB->begin(); it != oBB->end(); ++it) {
        Operation *op = &*it;
        Operation *newOp = gutils->getNewFromOriginal(op);

        if (auto ifaceOp = dyn_cast<ReverseAutoDiffOpInterface>(op)) {
          OpBuilder builder(newOp);
          ifaceOp.createShadowValues(builder, gutils);
        }
      }
    }
  }
}

void MEnzymeLogic::differentiate(
    MGradientUtilsReverse *gutils, Region &oldRegion, Region &newRegion,
    bool parentRegion,
    llvm::function_ref<buildReturnFunction> buildFuncReturnOp,
    std::function<std::pair<Value, Value>(Type)> cacheCreator) {
  gutils->registerCacheCreatorHook(cacheCreator);
  auto scope = llvm::make_scope_exit(
      [&]() { gutils->deregisterCacheCreatorHook(cacheCreator); });

  gutils->createReverseModeBlocks(oldRegion, newRegion, parentRegion);

  SmallVector<mlir::Block *> dominatorToposortBlocks =
      getDominatorToposort(gutils, oldRegion);
  initializeShadowValues(dominatorToposortBlocks, gutils);

  for (auto it = dominatorToposortBlocks.rbegin();
       it != dominatorToposortBlocks.rend(); ++it) {
    Block *oBB = *it;
    Block *newBB = gutils->getNewFromOriginal(oBB);
    Block *reverseBB = gutils->mapReverseModeBlocks.lookupOrNull(oBB);

    mapInvertArguments(oBB, reverseBB, gutils);
    handleReturns(oBB, newBB, reverseBB, gutils, parentRegion);
    visitChildren(oBB, reverseBB, gutils);
    handlePredecessors(oBB, newBB, reverseBB, gutils, buildFuncReturnOp,
                       parentRegion);
  }
}

FunctionOpInterface MEnzymeLogic::CreateReverseDiff(
    FunctionOpInterface fn, DIFFE_TYPE retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented,
    SymbolTableCollection &symbolTable) {

  if (fn.getFunctionBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }

  ReturnType returnValue = ReturnType::Args;
  MGradientUtilsReverse *gutils = MGradientUtilsReverse::CreateFromClone(
      *this, mode, width, fn, TA, type_args, retType, /*diffeReturnArg*/ true,
      constants, returnValue, addedType, symbolTable);

  Region &oldRegion = gutils->oldFunc.getFunctionBody();
  Region &newRegion = gutils->newFunc.getFunctionBody();

  auto buildFuncReturnOp = [](OpBuilder &builder, Location loc,
                              SmallVector<Value> retargs) {
    builder.create<func::ReturnOp>(loc, retargs);
    return;
  };

  differentiate(gutils, oldRegion, newRegion, true, buildFuncReturnOp, nullptr);

  auto nf = gutils->newFunc;

  // llvm::errs() << "nf\n";
  // nf.dump();
  // llvm::errs() << "nf end\n";

  delete gutils;
  return nf;
}