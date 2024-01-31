//===- CoreDialectsAutoDiffImplementations.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common utilities for the external model implementation of
// the automatic differentiation op interfaces for upstream MLIR dialects.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"

using namespace mlir;
using namespace mlir::enzyme;

LogicalResult mlir::enzyme::detail::controlFlowForwardHandler(
    Operation *op, OpBuilder &builder, MGradientUtils *gutils) {
  // For all active results, add shadow types.
  // For now, assuming all results are relevant.
  Operation *newOp = gutils->getNewFromOriginal(op);
  SmallVector<Type> newOpResultTypes;
  newOpResultTypes.reserve(op->getNumResults() * 2);
  for (Value result : op->getResults()) {
    // TODO only if used (can we DCE the primal after having done the
    // derivative).
    newOpResultTypes.push_back(result.getType());
    if (gutils->isConstantValue(result))
      continue;
    auto typeIface = dyn_cast<AutoDiffTypeInterface>(result.getType());
    if (!typeIface)
      return failure();
    newOpResultTypes.push_back(typeIface.getShadowType());
  }

  // For all operands that are forwarded to the body, if they are active, also
  // add the shadow as operand.
  auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op);
  if (!regionBranchOp)
    return failure();

  SmallVector<RegionSuccessor> successors;
  // TODO: consider getEntrySuccessorRegions variation that cares about
  // activity and stuff.
  // TODO: we may need to record, for every successor, which of its inputs
  // need a shadow to recreate the body correctly.
  // TODO: support region-to-region control flow as opposed to
  // entry-to-region-to-exit (e.g., scf.while).
  llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
  regionBranchOp.getSuccessorRegions(RegionBranchPoint::parent(), successors);
  for (const RegionSuccessor &successor : successors) {
    if (!successor.isParent() && successor.getSuccessor()->empty())
      continue;

    OperandRange operandRange =
        regionBranchOp.getEntrySuccessorOperands(successor);

    // Need to know which of the arguments are being forwarded to from
    // operands.
    for (auto &&[i, regionValue, operand] :
         llvm::enumerate(successor.getSuccessorInputs(), operandRange)) {
      if (gutils->isConstantValue(regionValue))
        continue;
      operandPositionsToShadow.insert(operandRange.getBeginOperandIndex() + i);
    }
  }
  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands() + operandPositionsToShadow.size());
  for (OpOperand &operand : op->getOpOperands()) {
    newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    if (operandPositionsToShadow.contains(operand.getOperandNumber()))
      newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
  }
  // We are assuming the op can forward additional operands, listed
  // immediately after the original operands, to the same regions.
  // ^^
  // Our interface guarantees this.
  // We also assume that the region-holding op returns all of the values
  // yielded by terminators, and only those values.

  auto iface = dyn_cast<ControlFlowAutoDiffOpInterface>(op);
  if (!iface)
    return failure();
  Operation *replacement = iface.createWithShadows(
      builder, gutils, op, newOperands, newOpResultTypes);
  for (auto &&[region, replacementRegion] :
       llvm::zip(newOp->getRegions(), replacement->getRegions())) {
    replacementRegion.takeBody(region);
  }

  // Inject the mapping for the new results into GradientUtil's shadow
  // table.
  SmallVector<Value> reps;
  size_t idx = 0;
  for (Value r : op->getResults()) {
    // TODO only if used
    reps.push_back(replacement->getResult(idx));
    idx++;
    if (!gutils->isConstantValue(r)) {
      auto inverted = gutils->invertedPointers.lookupOrNull(r);
      assert(inverted);
      gutils->invertedPointers.map(r, replacement->getResult(idx));
      inverted.replaceAllUsesWith(replacement->getResult(idx));
      gutils->erase(inverted.getDefiningOp());
      idx++;
    }
  }

  // Differentiate body.
  for (auto &origRegion : op->getRegions()) {
    for (auto &origBlock : origRegion) {
      for (Operation &o : origBlock.without_terminator()) {
        if (failed(gutils->visitChild(&o)))
          return failure();
      }
    }
  }

  for (auto &&[origRegion, replRegion] :
       llvm::zip(op->getRegions(), replacement->getRegions())) {
    for (auto &&[origBlock, replBlock] : llvm::zip(origRegion, replRegion)) {
      Operation *origTerminator = origBlock.getTerminator();
      auto termIface =
          dyn_cast<RegionBranchTerminatorOpInterface>(origTerminator);
      if (!termIface)
        continue;

      SmallVector<RegionSuccessor> successors;
      termIface.getSuccessorRegions({}, successors);

      llvm::SmallDenseSet<unsigned> operandsToShadow;
      for (auto &successor : successors) {
        OperandRange operandRange = termIface.getSuccessorOperands(successor);
        ValueRange targetValues = successor.isParent()
                                      ? op->getResults()
                                      : successor.getSuccessorInputs();
        assert(operandRange.size() == targetValues.size());
        for (auto &&[i, target] : llvm::enumerate(targetValues)) {
          if (!gutils->isConstantValue(target))
            operandsToShadow.insert(operandRange.getBeginOperandIndex() + i);
        }
      }
      SmallVector<Value> newOperands;
      newOperands.reserve(termIface->getNumOperands() +
                          operandsToShadow.size());
      for (OpOperand &operand : termIface->getOpOperands()) {
        newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
        if (operandsToShadow.contains(operand.getOperandNumber()))
          newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
      }

      // Assuming shadows following the originals are fine.
      // TODO: consider extending to have a ShadowableTerminatorOpInterface
      Operation *replTerminator = replBlock.getTerminator();
      builder.setInsertionPointToEnd(&replBlock);
      Operation *newTerminator = builder.clone(*replTerminator);
      newTerminator->setOperands(newOperands);
      gutils->erase(replTerminator);
    }
  }

  // Replace all uses of original results
  gutils->replaceOrigOpWith(op, reps);
  gutils->erase(newOp);

  return success();
}
