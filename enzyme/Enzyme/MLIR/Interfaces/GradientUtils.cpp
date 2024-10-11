//===- GradientUtils.cpp - Utilities for gradient interfaces --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/GradientUtils.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/CloneFunction.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

using namespace mlir;
using namespace mlir::enzyme;

mlir::enzyme::MGradientUtils::MGradientUtils(
    MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_, MTypeResults TR_,
    IRMapping &invertedPointers_, const llvm::ArrayRef<bool> returnPrimals,
    const llvm::ArrayRef<bool> returnShadows,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_,
    ArrayRef<DIFFE_TYPE> ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
    IRMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeMode mode, unsigned width, bool omp)
    : newFunc(newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_),
      invertedPointers(invertedPointers_), originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_), blocksNotForAnalysis(),
      activityAnalyzer(std::make_unique<enzyme::ActivityAnalyzer>(
          blocksNotForAnalysis, constantvalues_, activevals_, ReturnActivity)),
      TA(TA_), TR(TR_), omp(omp), returnPrimals(returnPrimals),
      returnShadows(returnShadows), width(width), ArgDiffeTypes(ArgDiffeTypes_),
      RetDiffeTypes(ReturnActivity) {}

mlir::Value mlir::enzyme::MGradientUtils::getNewFromOriginal(
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
mlir::enzyme::MGradientUtils::getNewFromOriginal(mlir::Block *originst) const {
  if (!originalToNewFn.contains(originst)) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    llvm::errs() << originst << "\n";
    llvm_unreachable("Could not get new blk from original");
  }
  return originalToNewFn.lookupOrNull(originst);
}

Operation *
mlir::enzyme::MGradientUtils::getNewFromOriginal(Operation *originst) const {
  assert(originst);
  auto found = originalToNewFnOps.find(originst);
  if (found == originalToNewFnOps.end()) {
    llvm::errs() << oldFunc << "\n";
    llvm::errs() << newFunc << "\n";
    for (auto &pair : originalToNewFnOps) {
      llvm::errs() << " map[" << pair.first << "] = " << pair.second << "\n";
      llvm::errs() << " map[" << *pair.first << "] = " << *pair.second << "\n";
    }
    llvm::errs() << originst << " - " << *originst << "\n";
    llvm_unreachable("Could not get new op from original");
  }
  return found->second;
}

Operation *mlir::enzyme::MGradientUtils::cloneWithNewOperands(OpBuilder &B,
                                                              Operation *op) {
  IRMapping map;
  for (auto operand : op->getOperands())
    map.map(operand, getNewFromOriginal(operand));
  return B.clone(*op, map);
}

bool mlir::enzyme::MGradientUtils::isConstantInstruction(Operation *op) const {
  return activityAnalyzer->isConstantOperation(TR, op);
}
bool mlir::enzyme::MGradientUtils::isConstantValue(Value v) const {
  return activityAnalyzer->isConstantValue(TR, v);
}

mlir::Value mlir::enzyme::MGradientUtils::invertPointerM(mlir::Value v,
                                                         OpBuilder &Builder2) {
  // TODO
  if (invertedPointers.contains(v))
    return invertedPointers.lookupOrNull(v);

  if (isConstantValue(v)) {
    if (auto iface = v.getType().dyn_cast<AutoDiffTypeInterface>()) {
      OpBuilder::InsertionGuard guard(Builder2);
      if (auto op = v.getDefiningOp())
        Builder2.setInsertionPoint(getNewFromOriginal(op));
      else {
        auto ba = cast<BlockArgument>(v);
        Builder2.setInsertionPointToStart(getNewFromOriginal(ba.getOwner()));
      }
      Value dv = iface.createNullValue(Builder2, v.getLoc());
      invertedPointers.map(v, dv);
      return dv;
    }
    return getNewFromOriginal(v);
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
}

mlir::Value
mlir::enzyme::MDiffeGradientUtils::getDifferential(mlir::Value oval) {
  auto found = differentials.lookupOrNull(oval);
  if (found != nullptr)
    return found;

  auto shadowty = getShadowType(oval.getType());
  OpBuilder builder(oval.getContext());
  builder.setInsertionPointToStart(initializationBlock);

  auto shadow = builder.create<enzyme::InitOp>(
      oval.getLoc(), enzyme::GradientType::get(oval.getContext(), shadowty));
  auto toset = cast<AutoDiffTypeInterface>(shadowty).createNullValue(
      builder, oval.getLoc());
  builder.create<enzyme::SetOp>(oval.getLoc(), shadow, toset);

  differentials.map(oval, shadow);
  return shadow;
}

void mlir::enzyme::MDiffeGradientUtils::setDiffe(mlir::Value oval,
                                                 mlir::Value toset,
                                                 OpBuilder &BuilderM) {
  assert(!isConstantValue(oval));
  auto iface = oval.getType().cast<AutoDiffTypeInterface>();
  if (!iface.isMutable()) {
    auto shadow = getDifferential(oval);
    BuilderM.create<enzyme::SetOp>(oval.getLoc(), shadow, toset);
  } else {
    MGradientUtils::setDiffe(oval, toset, BuilderM);
  }
}

void mlir::enzyme::MDiffeGradientUtils::zeroDiffe(mlir::Value oval,
                                                  OpBuilder &BuilderM) {
  assert(!isConstantValue(oval));
  auto iface = getShadowType(oval.getType()).cast<AutoDiffTypeInterface>();
  assert(!iface.isMutable());
  setDiffe(oval, iface.createNullValue(BuilderM, oval.getLoc()), BuilderM);
}

mlir::Value mlir::enzyme::MDiffeGradientUtils::diffe(mlir::Value oval,
                                                     OpBuilder &BuilderM) {

  auto shadow = getDifferential(oval);
  return BuilderM.create<enzyme::GetOp>(oval.getLoc(),
                                        getShadowType(oval.getType()), shadow);
}

void mlir::enzyme::MGradientUtils::setDiffe(mlir::Value val, mlir::Value toset,
                                            OpBuilder &BuilderM) {
  /*
 if (auto arg = dyn_cast<Argument>(val))
   assert(arg->getParent() == oldFunc);
 if (auto inst = dyn_cast<Instruction>(val))
   assert(inst->getParent()->getParent() == oldFunc);
   */
  if (isConstantValue(val)) {
    llvm::errs() << newFunc << "\n";
    llvm::errs() << val << "\n";
  }
  assert(!isConstantValue(val));
  if (mode == DerivativeMode::ForwardMode ||
      mode == DerivativeMode::ForwardModeSplit) {
    assert(getShadowType(val.getType()) == toset.getType());
    auto found = invertedPointers.lookupOrNull(val);
    assert(found != nullptr);
    auto placeholder = found.getDefiningOp<enzyme::PlaceholderOp>();
    invertedPointers.erase(val);
    // replaceAWithB(placeholder, toset);
    placeholder.replaceAllUsesWith(toset);
    erase(placeholder);
    invertedPointers.map(val, toset);
    return;
  }
  /*
  Value *tostore = getDifferential(val);
  if (toset->getType() != tostore->getType()->getPointerElementType()) {
    llvm::errs() << "toset:" << *toset << "\n";
    llvm::errs() << "tostore:" << *tostore << "\n";
  }
  assert(toset->getType() == tostore->getType()->getPointerElementType());
  BuilderM.CreateStore(toset, tostore);
  */
}

void mlir::enzyme::MGradientUtils::forceAugmentedReturns() {
  // TODO also block arguments
  // assert(TR.getFunction() == oldFunc);

  // Don't create derivatives for code that results in termination
  // if (notForAnalysis.find(&oBB) != notForAnalysis.end())
  //  continue;

  // LoopContext loopContext;
  // getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

  oldFunc.walk([&](Block *blk) {
    if (blk == &oldFunc.getFunctionBody().getBlocks().front())
      return;
    auto nblk = getNewFromOriginal(blk);
    for (auto val : llvm::reverse(blk->getArguments())) {
      if (isConstantValue(val))
        continue;
      auto i = val.getArgNumber();
      if (mode == DerivativeMode::ForwardMode ||
          mode == DerivativeMode::ForwardModeSplit ||
          cast<AutoDiffTypeInterface>(val.getType()).isMutable()) {
        mlir::Value dval;
        if (i == blk->getArguments().size() - 1)
          dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
        else
          dval =
              nblk->insertArgument(nblk->args_begin() + i + 1,
                                   getShadowType(val.getType()), val.getLoc());

        invertedPointers.map(val, dval);
      }
    }
  });

  oldFunc.walk([&](Operation *inst) {
    if (inst == oldFunc)
      return;

    OpBuilder BuilderZ(getNewFromOriginal(inst));
    for (auto res : inst->getResults()) {
      if (isConstantValue(res))
        continue;

      if (!(mode == DerivativeMode::ForwardMode ||
            mode == DerivativeMode::ForwardModeSplit ||
            cast<AutoDiffTypeInterface>(res.getType()).isMutable()))
        continue;
      mlir::Type antiTy = getShadowType(res.getType());
      auto anti = BuilderZ.create<enzyme::PlaceholderOp>(res.getLoc(), antiTy);
      invertedPointers.map(res, anti);
    }
  });
}

LogicalResult MGradientUtils::visitChild(Operation *op) {
  if (mode == DerivativeMode::ForwardMode) {
    if ((op->getBlock()->getTerminator() != op) &&
        llvm::all_of(op->getResults(),
                     [this](Value v) { return isConstantValue(v); }) &&
        /*iface.hasNoEffect()*/ activityAnalyzer->isConstantOperation(TR, op)) {
      return success();
    }
    // }
    if (auto iface = dyn_cast<AutoDiffOpInterface>(op)) {
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(getNewFromOriginal(op));
      return iface.createForwardModeTangent(builder, this);
    }
  }
  return op->emitError() << "could not compute the adjoint for this operation "
                         << *op;
}
