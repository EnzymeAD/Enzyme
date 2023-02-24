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

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

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
    FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
    BlockAndValueMapping &invertedPointers_,
    const SmallPtrSetImpl<mlir::Value> &constantvalues_,
    const SmallPtrSetImpl<mlir::Value> &activevals_,
    DIFFE_TYPE_MLIR ReturnActivity, ArrayRef<DIFFE_TYPE_MLIR> ArgDiffeTypes_,
    BlockAndValueMapping &originalToNewFn_,
    std::map<Operation *, Operation *> &originalToNewFnOps_,
    DerivativeModeMLIR mode, unsigned width, bool omp)
    : newFunc(newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_), TA(TA_),
      omp(omp), width(width), ArgDiffeTypes(ArgDiffeTypes_),
      originalToNewFn(originalToNewFn_),
      originalToNewFnOps(originalToNewFnOps_),
      invertedPointers(invertedPointers_) {

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

Value mlir::enzyme::MGradientUtils::getNewFromOriginal(
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

Operation *mlir::enzyme::MGradientUtils::cloneWithNewOperands(OpBuilder &B,
                                                              Operation *op) {
  BlockAndValueMapping map;
  for (auto operand : op->getOperands())
    map.map(operand, getNewFromOriginal(operand));
  return B.clone(*op, map);
}

bool mlir::enzyme::MGradientUtils::isConstantValue(Value v) const {
  if (isa<mlir::IntegerType>(v.getType()))
    return true;
  if (isa<mlir::IndexType>(v.getType()))
    return true;

  if (matchPattern(v, m_Constant()))
    return true;

  // TODO
  return false;
}

Value mlir::enzyme::MGradientUtils::invertPointerM(Value v,
                                                   OpBuilder &Builder2) {
  // TODO
  if (invertedPointers.contains(v))
    return invertedPointers.lookupOrNull(v);

  if (isConstantValue(v)) {
    if (auto iface = v.getType().cast<AutoDiffTypeInterface>()) {
      OpBuilder::InsertionGuard guard(Builder2);
      Builder2.setInsertionPoint(getNewFromOriginal(v.getDefiningOp()));
      Value dv = iface.createNullValue(Builder2, v.getLoc());
      invertedPointers.map(v, dv);
      return dv;
    }
    return getNewFromOriginal(v);
  }
  llvm::errs() << " could not invert pointer v " << v << "\n";
  llvm_unreachable("could not invert pointer");
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
  if (mode == DerivativeModeMLIR::ForwardMode ||
      mode == DerivativeModeMLIR::ForwardModeSplit) {
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
      mlir::Value dval;
      if (i == blk->getArguments().size() - 1)
        dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
      else
        dval = nblk->insertArgument(nblk->args_begin() + i + 1,
                                    getShadowType(val.getType()), val.getLoc());

      invertedPointers.map(val, dval);
    }
  });

  oldFunc.walk([&](Operation *inst) {
    if (inst == oldFunc)
      return;
    if (mode == DerivativeModeMLIR::ForwardMode ||
        mode == DerivativeModeMLIR::ForwardModeSplit) {
      OpBuilder BuilderZ(getNewFromOriginal(inst));
      for (auto res : inst->getResults()) {
        if (!isConstantValue(res)) {
          mlir::Type antiTy = getShadowType(res.getType());
          auto anti = BuilderZ.create<enzyme::PlaceholderOp>(res.getLoc(),
                                                             res.getType());
          invertedPointers.map(res, anti);
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

LogicalResult MGradientUtils::visitChild(Operation *op) {
  if (mode == DerivativeModeMLIR::ForwardMode) {
    if (auto iface = dyn_cast<AutoDiffOpInterface>(op)) {
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(getNewFromOriginal(op));
      return iface.createForwardModeTangent(builder, this);
    }
  }
  return failure();
}
