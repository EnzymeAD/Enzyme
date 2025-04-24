#include "Dialect/Ops.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/BreadthFirstIterator.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"

using namespace mlir;
using namespace mlir::enzyme;

void createTerminator(MGradientUtils *gutils, mlir::Block *oBB,
                      const ArrayRef<bool> returnPrimals,
                      const ArrayRef<bool> returnShadows) {
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

  for (auto &&[ret, returnPrimal, returnShadow] :
       llvm::zip(inst->getOperands(), returnPrimals, returnShadows)) {
    if (returnPrimal) {
      retargs.push_back(gutils->getNewFromOriginal(ret));
    }
    if (returnShadow) {
      if (!gutils->isConstantValue(ret)) {
        retargs.push_back(gutils->invertPointerM(ret, nBuilder));
      } else {
        Type retTy = cast<AutoDiffTypeInterface>(ret.getType()).getShadowType();
        auto toret = cast<AutoDiffTypeInterface>(retTy).createNullValue(
            nBuilder, ret.getLoc());
        retargs.push_back(toret);
      }
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
    FunctionOpInterface fn, std::vector<DIFFE_TYPE> RetActivity,
    std::vector<DIFFE_TYPE> ArgActivity, MTypeAnalysis &TA,
    std::vector<bool> returnPrimals, DerivativeMode mode, bool freeMemory,
    size_t width, mlir::Type addedType, MFnTypeInfo type_args,
    std::vector<bool> volatile_args, void *augmented, bool omp,
    llvm::StringRef postpasses) {
  if (fn.getFunctionBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }
  assert(fn.getFunctionBody().front().getNumArguments() == ArgActivity.size());
  assert(fn.getFunctionBody().front().getNumArguments() ==
         volatile_args.size());

  MForwardCacheKey tup = {
      fn, RetActivity, ArgActivity,
      // std::map<Argument *, bool>(_uncacheable_args.begin(),
      //                           _uncacheable_args.end()),
      returnPrimals, mode, static_cast<unsigned>(width), addedType, type_args,
      omp};

  if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
    return ForwardCachedFunctions.find(tup)->second;
  }
  std::vector<bool> returnShadows;
  for (auto act : RetActivity) {
    returnShadows.push_back(act != DIFFE_TYPE::CONSTANT);
  }
  SmallVector<bool> returnPrimalsP(returnPrimals.begin(), returnPrimals.end());
  SmallVector<bool> returnShadowsP(returnShadows.begin(), returnShadows.end());
  auto gutils = MDiffeGradientUtils::CreateFromClone(
      *this, mode, width, fn, TA, type_args, returnPrimalsP, returnShadowsP,
      RetActivity, ArgActivity, addedType,
      /*omp*/ false, postpasses);
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

    assert(oBB.getTerminator());

    auto first = oBB.begin();
    auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
    for (auto it = first; it != last; ++it) {
      // TODO: propagate errors.
      auto res = gutils->visitChild(&*it);
      valid &= res.succeeded();
    }

    createTerminator(gutils, &oBB, returnPrimalsP, returnShadowsP);
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

  if (postpasses != "") {
    mlir::PassManager pm(nf->getContext());
    std::string error_message;
    // llvm::raw_string_ostream error_stream(error_message);
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (mlir::failed(result)) {
      return nullptr;
    }

    if (!mlir::succeeded(pm.run(nf))) {
      return nullptr;
    }
  }

  return nf;
}
