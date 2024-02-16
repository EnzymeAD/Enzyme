#include "Dialect/Ops.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"

using namespace mlir;
using namespace mlir::enzyme;

void createTerminator(MGradientUtils *gutils, mlir::Block *oBB,
                      DIFFE_TYPE retType, ReturnType retVal) {
  auto inst = oBB->getTerminator();

  mlir::Block *nBB = gutils->getNewFromOriginal(inst->getBlock());
  assert(nBB);
  auto newInst = nBB->getTerminator();

  OpBuilder nBuilder(inst);
  nBuilder.setInsertionPointToEnd(nBB);

  if (auto binst = dyn_cast<BranchOpInterface>(inst)) {
    mlir::enzyme::detail::branchingForwardHandler(inst, nBuilder, gutils);
    return;
  }

  // In forward mode we only need to update the return value
  if (!inst->hasTrait<OpTrait::ReturnLike>())
    return;

  SmallVector<mlir::Value, 2> retargs;

  switch (retVal) {
  case ReturnType::Return: {
    for (size_t i = 0; i < inst->getNumOperands(); i++) {
      auto ret = inst->getOperand(i);

      mlir::Value toret;
      if (retType == DIFFE_TYPE::CONSTANT) {
        toret = gutils->getNewFromOriginal(ret);
      } else if (!isa<mlir::FloatType>(ret.getType()) &&
                 true /*type analysis*/) {
        toret = gutils->invertPointerM(ret, nBuilder);
      } else if (!gutils->isConstantValue(ret)) {
        toret = gutils->invertPointerM(ret, nBuilder);
      } else {
        Type retTy =
            ret.getType().cast<AutoDiffTypeInterface>().getShadowType();
        toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(
            nBuilder, ret.getLoc());
      }
      retargs.push_back(toret);
    }

    break;
  }
  case ReturnType::TwoReturns: {
    if (retType == DIFFE_TYPE::CONSTANT)
      assert(false && "Invalid return type");
    for (size_t i = 0; i < inst->getNumOperands(); i++) {
      auto ret = inst->getOperand(i);

      retargs.push_back(gutils->getNewFromOriginal(ret));

      mlir::Value toret;
      if (retType == DIFFE_TYPE::CONSTANT) {
        toret = gutils->getNewFromOriginal(ret);
      } else if (!isa<mlir::FloatType>(ret.getType()) &&
                 true /*type analysis*/) {
        toret = gutils->invertPointerM(ret, nBuilder);
      } else if (!gutils->isConstantValue(ret)) {
        toret = gutils->invertPointerM(ret, nBuilder);
      } else {
        Type retTy =
            ret.getType().cast<AutoDiffTypeInterface>().getShadowType();
        toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(
            nBuilder, ret.getLoc());
      }
      retargs.push_back(toret);
    }
    break;
  }
  case ReturnType::Void: {
    break;
  }
  default: {
    llvm::errs() << "Invalid return type: "
                 << "for function: \n"
                 << gutils->newFunc << "\n";
    assert(false && "Invalid return type for function");
    return;
  }
  }

  nBB->push_back(
      newInst->create(newInst->getLoc(), newInst->getName(), TypeRange(),
                      retargs, newInst->getAttrs(), OpaqueProperties(nullptr),
                      newInst->getSuccessors(), newInst->getNumRegions()));
  gutils->erase(newInst);
  return;
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

FunctionOpInterface mlir::enzyme::MEnzymeLogic::CreateForwardDiff(
    FunctionOpInterface fn, DIFFE_TYPE retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented) {
  if (fn.getFunctionBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }
  assert(fn.getFunctionBody().front().getNumArguments() == constants.size());
  assert(fn.getFunctionBody().front().getNumArguments() ==
         volatile_args.size());

  MForwardCacheKey tup = {
      fn, retType, constants,
      // std::map<Argument *, bool>(_uncacheable_args.begin(),
      //                           _uncacheable_args.end()),
      returnUsed, mode, static_cast<unsigned>(width), addedType, type_args};

  if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
    return ForwardCachedFunctions.find(tup)->second;
  }
  bool retActive = retType != DIFFE_TYPE::CONSTANT;
  ReturnType returnValue =
      returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                 : (retActive ? ReturnType::Return : ReturnType::Void);
  auto gutils = MDiffeGradientUtils::CreateFromClone(
      *this, mode, width, fn, TA, type_args, retType,
      /*diffeReturnArg*/ false, constants, returnValue, addedType,
      /*omp*/ false);
  ForwardCachedFunctions[tup] = gutils->newFunc;

  insert_or_assign2<MForwardCacheKey, FunctionOpInterface>(
      ForwardCachedFunctions, tup, gutils->newFunc);

  // gutils->FreeMemory = freeMemory;

  const SmallPtrSet<mlir::Block *, 4> guaranteedUnreachable;
  // = getGuaranteedUnreachable(gutils->oldFunc);

  // gutils->forceActiveDetection();
  gutils->forceAugmentedReturns();
  /*

  // TODO populate with actual unnecessaryInstructions once the dependency
  // cycle with activity analysis is removed
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
  for (auto BB : guaranteedUnreachable) {
    for (auto &I : *BB)
      unnecessaryInstructionsTmp.insert(&I);
  }
  if (mode == DerivativeMode::ForwardModeSplit)
    gutils->computeGuaranteedFrees();

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(
      *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
  returnUsed, mode, gutils, TLI, constant_args, guaranteedUnreachable);
  gutils->unnecessaryValuesP = &unnecessaryValues;

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);
                                  */

  bool valid = true;
  for (Block &oBB : gutils->oldFunc.getFunctionBody().getBlocks()) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = gutils->getNewFromOriginal(&oBB);

      for (auto &I : make_early_inc_range(reverse(oBB))) {
        gutils->eraseIfUnused(&I, /*erase*/ true, /*check*/ false);
      }

      OpBuilder builder(gutils->oldFunc.getContext());
      builder.setInsertionPointToEnd(newBB);
      builder.create<LLVM::UnreachableOp>(gutils->oldFunc.getLoc());
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);

    auto first = oBB.begin();
    auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
    for (auto it = first; it != last; ++it) {
      // TODO: propagate errors.
      auto res = gutils->visitChild(&*it);
      valid &= res.succeeded();
    }

    createTerminator(gutils, &oBB, retType, returnValue);
  }

  // if (mode == DerivativeMode::ForwardModeSplit && augmenteddata)
  //  restoreCache(gutils, augmenteddata->tapeIndices, guaranteedUnreachable);

  // gutils->eraseFictiousPHIs();

  // mlir::Block *entry = &gutils->newFunc.getFunctionBody().front();

  // cleanupInversionAllocs(gutils, entry);
  // clearFunctionAttributes(gutils->newFunc);

  /*
  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }
  */

  auto nf = gutils->newFunc;
  delete gutils;

  if (!valid)
    return nullptr;

  // if (PostOpt)
  //  PPC.optimizeIntermediate(nf);
  // if (EnzymePrint) {
  //  llvm::errs() << nf << "\n";
  //}
  return nf;
}
