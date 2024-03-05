#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "EnzymeLogic.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::enzyme;

void handleReturns(Block *oBB, Block *newBB, Block *reverseBB,
                   MGradientUtilsReverse *gutils) {
  if (oBB->getNumSuccessors() == 0) {
    Operation *returnStatement = newBB->getTerminator();
    gutils->erase(returnStatement);

    OpBuilder forwardToBackwardBuilder(newBB, newBB->end());

    Operation *newBranchOp = forwardToBackwardBuilder.create<cf::BranchOp>(
        oBB->getTerminator()->getLoc(), reverseBB);

    gutils->originalToNewFnOps[oBB->getTerminator()] = newBranchOp;
  }
}

/*
Create reverse mode adjoint for an operation.
*/
void MEnzymeLogic::visitChild(Operation *op, OpBuilder &builder,
                              MGradientUtilsReverse *gutils) {
  if ((op->getBlock()->getTerminator() != op) &&
      llvm::all_of(op->getResults(),
                   [gutils](Value v) { return gutils->isConstantValue(v); }) &&
      gutils->isConstantInstruction(op)) {
    return;
  }
  if (auto ifaceOp = dyn_cast<ReverseAutoDiffOpInterface>(op)) {
    SmallVector<Value> caches = ifaceOp.cacheValues(gutils);
    ifaceOp.createReverseModeAdjoint(builder, gutils, caches);
    return;
    /*
    for (int indexResult = 0; indexResult < (int)op->getNumResults();
         indexResult++) {
      Value result = op->getResult(indexResult);
      gutils->clearValue(result, builder);
    }
    */
  }
  op->emitError() << "could not compute the adjoint for this operation " << *op;
}

void MEnzymeLogic::visitChildren(Block *oBB, Block *reverseBB,
                                 MGradientUtilsReverse *gutils) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  if (!oBB->empty()) {
    auto first = oBB->rbegin();
    auto last = oBB->rend();
    for (auto it = first; it != last; ++it) {
      Operation *op = &*it;
      visitChild(op, revBuilder, gutils);
    }
  }
}

void MEnzymeLogic::handlePredecessors(
    Block *oBB, Block *newBB, Block *reverseBB, MGradientUtilsReverse *gutils,
    llvm::function_ref<buildReturnFunction> buildReturnOp) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  if (oBB->hasNoPredecessors()) {
    buildReturnOp(revBuilder, oBB);
  } else {
    Location loc = oBB->rbegin()->getLoc();
    // TODO remove dependency on CF dialect

    Value cache = gutils->insertInit(gutils->getIndexCacheType());

    Value flag =
        revBuilder.create<enzyme::PopOp>(loc, gutils->getIndexType(), cache);

    Block *defaultBlock = nullptr;

    SmallVector<Block *> blocks;
    SmallVector<APInt> indices;

    OpBuilder newBuilder(newBB, newBB->begin());

    SmallVector<Value, 1> diffes;
    for (auto arg : oBB->getArguments()) {
      if (!gutils->isConstantValue(arg) &&
          !cast<AutoDiffTypeInterface>(arg.getType()).isMutable()) {
        diffes.push_back(gutils->diffe(arg, revBuilder));
        gutils->zeroDiffe(arg, revBuilder);
        continue;
      }
      diffes.push_back(nullptr);
    }

    for (auto [idx, pred] : llvm::enumerate(oBB->getPredecessors())) {
      auto reversePred = gutils->mapReverseModeBlocks.lookupOrNull(pred);

      Block *newPred = gutils->getNewFromOriginal(pred);

      OpBuilder predecessorBuilder(newPred->getTerminator());

      Value pred_idx_c =
          predecessorBuilder.create<arith::ConstantIntOp>(loc, idx - 1, 32);
      predecessorBuilder.create<enzyme::PushOp>(loc, cache, pred_idx_c);

      if (idx == 0) {
        defaultBlock = reversePred;

      } else {
        indices.push_back(APInt(32, idx - 1));
        blocks.push_back(reversePred);
      }

      auto term = pred->getTerminator();
      if (auto iface = dyn_cast<BranchOpInterface>(term)) {
        for (auto &op : term->getOpOperands())
          if (auto blk_idx =
                  iface.getSuccessorBlockArgument(op.getOperandNumber()))
            if ((*blk_idx).getOwner() == oBB) {
              auto idx = (*blk_idx).getArgNumber();
              if (diffes[idx]) {

                Value rev_idx_c =
                    revBuilder.create<arith::ConstantIntOp>(loc, idx - 1, 32);

                auto to_prop = revBuilder.create<arith::SelectOp>(
                    loc,
                    revBuilder.create<arith::CmpIOp>(
                        loc, arith::CmpIPredicate::eq, flag, rev_idx_c),
                    diffes[idx],
                    cast<AutoDiffTypeInterface>(diffes[idx].getType())
                        .createNullValue(revBuilder, loc));

                gutils->addToDiffe(op.get(), to_prop, revBuilder);
              }
            }
      } else {
        assert(0 && "predecessor did not implement branch op interface");
      }
    }

    revBuilder.create<cf::SwitchOp>(
        loc, flag, defaultBlock, ArrayRef<Value>(), ArrayRef<APInt>(indices),
        ArrayRef<Block *>(blocks),
        SmallVector<ValueRange>(indices.size(), ValueRange()));
  }
}

void MEnzymeLogic::differentiate(
    MGradientUtilsReverse *gutils, Region &oldRegion, Region &newRegion,
    llvm::function_ref<buildReturnFunction> buildFuncReturnOp,
    std::function<std::pair<Value, Value>(Type)> cacheCreator) {
  gutils->registerCacheCreatorHook(cacheCreator);
  auto scope = llvm::make_scope_exit(
      [&]() { gutils->deregisterCacheCreatorHook(cacheCreator); });

  gutils->createReverseModeBlocks(oldRegion, newRegion);

  for (auto &oBB : oldRegion) {
    Block *newBB = gutils->getNewFromOriginal(&oBB);
    Block *reverseBB = gutils->mapReverseModeBlocks.lookupOrNull(&oBB);
    handleReturns(&oBB, newBB, reverseBB, gutils);
    visitChildren(&oBB, reverseBB, gutils);
    handlePredecessors(&oBB, newBB, reverseBB, gutils, buildFuncReturnOp);
  }
}

FunctionOpInterface MEnzymeLogic::CreateReverseDiff(
    FunctionOpInterface fn, DIFFE_TYPE retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented) {

  if (fn.getFunctionBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }

  ReturnType returnValue = ReturnType::Args;
  MGradientUtilsReverse *gutils = MGradientUtilsReverse::CreateFromClone(
      *this, mode, width, fn, TA, type_args, retType, /*diffeReturnArg*/ true,
      constants, returnValue, addedType);

  Region &oldRegion = gutils->oldFunc.getFunctionBody();
  Region &newRegion = gutils->newFunc.getFunctionBody();

  auto buildFuncReturnOp = [&](OpBuilder &builder, Block *oBB) {
    SmallVector<mlir::Value> retargs;
    for (auto [arg, cv] : llvm::zip(oBB->getArguments(), constants)) {
      if (cv == DIFFE_TYPE::OUT_DIFF) {
        retargs.push_back(gutils->diffe(arg, builder));
      }
    }
    builder.create<func::ReturnOp>(oBB->rbegin()->getLoc(), retargs);
    return;
  };

  gutils->forceAugmentedReturns();

  differentiate(gutils, oldRegion, newRegion, buildFuncReturnOp, nullptr);

  auto nf = gutils->newFunc;

  // llvm::errs() << "nf\n";
  // nf.dump();
  // llvm::errs() << "nf end\n";

  delete gutils;
  return nf;
}
