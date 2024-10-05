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
#include "Interfaces/GradientUtilsReverse.h"

using namespace mlir;
using namespace mlir::enzyme;

mlir::TypedAttr mlir::enzyme::getConstantAttr(mlir::Type type,
                                              llvm::StringRef value) {
  using namespace mlir;
  if (auto T = dyn_cast<TensorType>(type)) {
    size_t num = 1;
    for (auto sz : T.getShape())
      num *= sz;
    APFloat apvalue(T.getElementType().cast<FloatType>().getFloatSemantics(),
                    value);
    SmallVector<APFloat> supportedValues(num, apvalue);
    return DenseFPElementsAttr::get(type.cast<ShapedType>(), supportedValues);
  }
  auto T = cast<FloatType>(type);
  APFloat apvalue(T.getFloatSemantics(), value);
  return FloatAttr::get(T, apvalue);
}

void mlir::enzyme::detail::branchingForwardHandler(Operation *inst,
                                                   OpBuilder &builder,
                                                   MGradientUtils *gutils) {
  auto newInst = gutils->getNewFromOriginal(inst);

  auto binst = cast<BranchOpInterface>(inst);

  // TODO generalize to cloneWithNewBlockArgs interface
  SmallVector<Value> newVals;

  SmallVector<int32_t> segSizes;
  // Keep non-differentiated, non-forwarded operands
  size_t non_forwarded = 0;
  for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
    auto ops = binst.getSuccessorOperands(i).getForwardedOperands();
    if (ops.empty())
      continue;
    non_forwarded = ops.getBeginOperandIndex();
    break;
  }

  for (size_t i = 0; i < non_forwarded; i++)
    newVals.push_back(gutils->getNewFromOriginal(binst->getOperand(i)));

  segSizes.push_back(newVals.size());
  for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
    size_t cur = newVals.size();
    auto ops = binst.getSuccessorOperands(i).getForwardedOperands();
    for (auto &&[idx, op] : llvm::enumerate(ops)) {
      auto arg =
          *binst.getSuccessorBlockArgument(ops.getBeginOperandIndex() + idx);
      newVals.push_back(gutils->getNewFromOriginal(op));
      if (!gutils->isConstantValue(arg)) {
        if (!gutils->isConstantValue(op)) {
          newVals.push_back(gutils->invertPointerM(op, builder));
        } else {
          Type retTy =
              arg.getType().cast<AutoDiffTypeInterface>().getShadowType();
          auto toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(
              builder, op.getLoc());
          newVals.push_back(toret);
        }
      }
    }
    cur = newVals.size() - cur;
    segSizes.push_back(cur);
  }

  SmallVector<NamedAttribute> attrs(newInst->getAttrs());
  bool has_cases = false;
  for (auto &attr : attrs) {
    if (attr.getName() == "case_operand_segments") {
      has_cases = true;
    }
  }
  for (auto &attr : attrs) {
    if (attr.getName() == "operandSegmentSizes") {
      if (!has_cases) {
        attr.setValue(builder.getDenseI32ArrayAttr(segSizes));
      } else {
        SmallVector<int32_t> segSlices2(segSizes.begin(), segSizes.begin() + 2);
        segSlices2.push_back(0);
        for (size_t i = 2; i < segSizes.size(); i++)
          segSlices2[2] += segSizes[i];
        attr.setValue(builder.getDenseI32ArrayAttr(segSlices2));
      }
    }
    if (attr.getName() == "case_operand_segments") {
      SmallVector<int32_t> segSlices2(segSizes.begin() + 2, segSizes.end());
      attr.setValue(builder.getDenseI32ArrayAttr(segSlices2));
    }
  }

  gutils->getNewFromOriginal(inst->getBlock())
      ->push_back(
          newInst->create(newInst->getLoc(), newInst->getName(), TypeRange(),
                          newVals, attrs, OpaqueProperties(nullptr),
                          newInst->getSuccessors(), newInst->getNumRegions()));
  gutils->erase(newInst);
  return;
}

static bool contains(ArrayRef<int> ar, int v) {
  for (auto a : ar) {
    if (a == v) {
      return true;
    }
  }
  return false;
}

LogicalResult mlir::enzyme::detail::memoryIdentityForwardHandler(
    Operation *orig, OpBuilder &builder, MGradientUtils *gutils,
    ArrayRef<int> storedVals) {
  auto iface = cast<ActivityOpInterface>(orig);

  SmallVector<Value> newOperands;
  newOperands.reserve(orig->getNumOperands());
  for (OpOperand &operand : orig->getOpOperands()) {
    if (iface.isArgInactive(operand.getOperandNumber())) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    } else {
      if (gutils->isConstantValue(operand.get())) {

        if (contains(storedVals, operand.getOperandNumber()) ||
            contains(storedVals, -1)) {
          if (auto iface =
                  dyn_cast<AutoDiffTypeInterface>(operand.get().getType())) {
            if (!iface.isMutable()) {
              Type retTy = iface.getShadowType();
              auto toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(
                  builder, operand.get().getLoc());
              newOperands.push_back(toret);
              continue;
            }
          }
        }
        orig->emitError()
            << "Unsupported constant arg to memory identity forward "
               "handler(opidx="
            << operand.getOperandNumber() << ", op=" << operand.get() << ")\n";
        return failure();
      }
      newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
    }
  }

  // Assuming shadows following the originals are fine.
  // TODO: consider extending to have a ShadowableTerminatorOpInterface
  Operation *primal = gutils->getNewFromOriginal(orig);
  Operation *shadow = builder.clone(*primal);
  shadow->setOperands(newOperands);
  for (auto &&[oval, sval] :
       llvm::zip(orig->getResults(), shadow->getResults())) {
    gutils->setDiffe(oval, sval, builder);
  }

  return success();
}

LogicalResult mlir::enzyme::detail::allocationForwardHandler(
    Operation *orig, OpBuilder &builder, MGradientUtils *gutils, bool zero) {

  Operation *primal = gutils->getNewFromOriginal(orig);
  Operation *shadow = builder.clone(*primal);

  Value shadowRes = shadow->getResult(0);

  gutils->setDiffe(orig->getResult(0), shadowRes, builder);
  gutils->eraseIfUnused(orig);

  if (zero) {
    // Fill with zeros
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(shadowRes.getType())) {
      return iface.zeroInPlace(builder, orig->getLoc(), shadowRes);
    } else {
      orig->emitError() << "Type " << shadowRes.getType()
                        << " does not implement "
                           "AutoDiffTypeInterface";
      return failure();
    }
  }
  return success();
}

void mlir::enzyme::detail::returnReverseHandler(Operation *op,
                                                OpBuilder &builder,
                                                MGradientUtilsReverse *gutils) {
  size_t num_out = 0;
  for (auto act : gutils->RetDiffeTypes) {
    if (act == DIFFE_TYPE::OUT_DIFF)
      num_out++;
  }

  size_t idx = 0;
  auto args = gutils->newFunc->getRegions().begin()->begin()->getArguments();

  for (auto &&[op, act] : llvm::zip(op->getOperands(), gutils->RetDiffeTypes)) {
    if (act == DIFFE_TYPE::OUT_DIFF) {
      if (!gutils->isConstantValue(op)) {
        auto d_out = args[args.size() - num_out + idx];
        gutils->addToDiffe(op, d_out, builder);
      }
      idx++;
    }
  }
}

void mlir::enzyme::detail::regionTerminatorForwardHandler(
    Operation *origTerminator, OpBuilder &builder, MGradientUtils *gutils) {
  auto parentOp = origTerminator->getParentOp();

  llvm::SmallDenseSet<unsigned> operandsToShadow;
  if (auto termIface =
          dyn_cast<RegionBranchTerminatorOpInterface>(origTerminator)) {
    SmallVector<RegionSuccessor> successors;
    termIface.getSuccessorRegions(
        SmallVector<Attribute>(origTerminator->getNumOperands(), Attribute()),
        successors);

    for (auto &successor : successors) {
      OperandRange operandRange = termIface.getSuccessorOperands(successor);
      ValueRange targetValues = successor.isParent()
                                    ? parentOp->getResults()
                                    : successor.getSuccessorInputs();
      assert(operandRange.size() == targetValues.size());
      for (auto &&[i, target] : llvm::enumerate(targetValues)) {
        if (!gutils->isConstantValue(target))
          operandsToShadow.insert(operandRange.getBeginOperandIndex() + i);
      }
    }
  } else {
    assert(parentOp->getNumResults() == origTerminator->getNumOperands());
    for (auto res : parentOp->getResults()) {
      if (!gutils->isConstantValue(res))
        operandsToShadow.insert(res.getResultNumber());
    }
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(origTerminator->getNumOperands() +
                      operandsToShadow.size());
  for (OpOperand &operand : origTerminator->getOpOperands()) {
    newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
    if (operandsToShadow.contains(operand.getOperandNumber()))
      newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
  }

  // Assuming shadows following the originals are fine.
  // TODO: consider extending to have a ShadowableTerminatorOpInterface
  Operation *replTerminator = gutils->getNewFromOriginal(origTerminator);
  replTerminator->setOperands(newOperands);
}

LogicalResult mlir::enzyme::detail::controlFlowForwardHandler(
    Operation *op, OpBuilder &builder, MGradientUtils *gutils) {

  // For all operands that are forwarded to the body, if they are active, also
  // add the shadow as operand.
  auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op);
  if (!regionBranchOp) {
    op->emitError() << " RegionBranchOpInterface not implemented for " << *op
                    << "\n";
    return failure();
  }

  // TODO: we may need to record, for every successor, which of its inputs
  // need a shadow to recreate the body correctly.
  llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
  llvm::SmallDenseSet<unsigned> resultPositionsToShadow;

  SmallVector<RegionSuccessor> entrySuccessors;
  regionBranchOp.getEntrySuccessorRegions(
      SmallVector<Attribute>(op->getNumOperands(), Attribute()),
      entrySuccessors);

  for (const RegionSuccessor &successor : entrySuccessors) {

    OperandRange operandRange =
        regionBranchOp.getEntrySuccessorOperands(successor);

    ValueRange targetValues = successor.isParent()
                                  ? op->getResults()
                                  : successor.getSuccessorInputs();

    // Need to know which of the arguments are being forwarded to from
    // operands.
    for (auto &&[i, regionValue, operand] :
         llvm::enumerate(targetValues, operandRange)) {
      if (gutils->isConstantValue(regionValue))
        continue;
      operandPositionsToShadow.insert(operandRange.getBeginOperandIndex() + i);
      if (successor.isParent())
        resultPositionsToShadow.insert(i);
    }
  }

  for (auto res : op->getResults())
    if (!gutils->isConstantValue(res))
      resultPositionsToShadow.insert(res.getResultNumber());

  return controlFlowForwardHandler(
      op, builder, gutils, operandPositionsToShadow, resultPositionsToShadow);
}

LogicalResult mlir::enzyme::detail::controlFlowForwardHandler(
    Operation *op, OpBuilder &builder, MGradientUtils *gutils,
    const llvm::SmallDenseSet<unsigned> &operandPositionsToShadow,
    const llvm::SmallDenseSet<unsigned> &resultPositionsToShadow) {
  // For all active results, add shadow types.
  // For now, assuming all results are relevant.
  Operation *newOp = gutils->getNewFromOriginal(op);
  SmallVector<Type> newOpResultTypes;
  newOpResultTypes.reserve(op->getNumResults() * 2);
  for (auto result : op->getResults()) {
    // TODO only if used (can we DCE the primal after having done the
    // derivative).
    newOpResultTypes.push_back(result.getType());
    if (!gutils->isConstantValue(result)) {
      assert(resultPositionsToShadow.count(result.getResultNumber()));
    }
    if (!resultPositionsToShadow.count(result.getResultNumber()))
      continue;
    auto typeIface = dyn_cast<AutoDiffTypeInterface>(result.getType());
    if (!typeIface) {
      op->emitError() << " AutoDiffTypeInterface not implemented for "
                      << result.getType() << "\n";
      return failure();
    }
    newOpResultTypes.push_back(typeIface.getShadowType());
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
  if (!iface) {
    op->emitError() << " ControlFlowAutoDiffOpInterface not implemented for "
                    << *op << "\n";
    return failure();
  }
  Operation *replacement = iface.createWithShadows(
      builder, gutils, op, newOperands, newOpResultTypes);
  assert(replacement->getNumResults() == newOpResultTypes.size());
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
      for (Operation &o : origBlock) {
        if (failed(gutils->visitChild(&o))) {
          return failure();
        }
      }
    }
  }

  // Replace all uses of original results
  gutils->replaceOrigOpWith(op, reps);
  gutils->erase(newOp);
  gutils->originalToNewFnOps[op] = replacement;

  return success();
}

void mlir::enzyme::registerCoreDialectAutodiffInterfaces(
    DialectRegistry &registry) {
  enzyme::registerAffineDialectAutoDiffInterface(registry);
  enzyme::registerArithDialectAutoDiffInterface(registry);
  enzyme::registerBuiltinDialectAutoDiffInterface(registry);
  enzyme::registerComplexDialectAutoDiffInterface(registry);
  enzyme::registerLLVMDialectAutoDiffInterface(registry);
  enzyme::registerNVVMDialectAutoDiffInterface(registry);
  enzyme::registerMathDialectAutoDiffInterface(registry);
  enzyme::registerMemRefDialectAutoDiffInterface(registry);
  enzyme::registerComplexDialectAutoDiffInterface(registry);
  enzyme::registerSCFDialectAutoDiffInterface(registry);
  enzyme::registerCFDialectAutoDiffInterface(registry);
  enzyme::registerLinalgDialectAutoDiffInterface(registry);
  enzyme::registerFuncDialectAutoDiffInterface(registry);
}
