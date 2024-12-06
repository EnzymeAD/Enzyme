//===- EnzymeOps.cpp - Enzyme dialect ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ops.h"
#include "Dialect.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace enzyme;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> InitOp::getPromotableSlots() {
  auto Ty = this->getType();
  if (isa<CacheType>(Ty))
    return {};

  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  auto gTy = cast<GradientType>(Ty);
  MemorySlot slot = {this->getResult(), gTy.getBasetype()};

  return {slot};
}

Value InitOp::getDefaultValue(const MemorySlot &slot, OpBuilder &builder) {
  auto gTy = cast<GradientType>(this->getType());
  return cast<AutoDiffTypeInterface>(gTy.getBasetype())
      .createNullValue(builder, this->getLoc());
}

void InitOp::handleBlockArgument(const MemorySlot &slot, BlockArgument argument,
                                 OpBuilder &builder) {}

std::optional<mlir::PromotableAllocationOpInterface>
InitOp::handlePromotionComplete(const MemorySlot &slot, Value defaultValue,
                                OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//

bool GetOp::loadsFrom(const MemorySlot &slot) {
  return this->getGradient() == slot.ptr;
}

bool GetOp::storesTo(const MemorySlot &slot) { return false; }

Value GetOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                       Value reachingDef, const DataLayout &dataLayout) {
  return {};
}

bool GetOp::canUsesBeRemoved(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses,
    llvm::SmallVectorImpl<OpOperand *> &newBlockingUses,
    const mlir::DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getGradient() == slot.ptr;
}

DeletionKind GetOp::removeBlockingUses(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder,
    Value reachingDefinition, const DataLayout &dataLayout) {
  this->getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

llvm::LogicalResult GetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, llvm::SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(slot.ptr == getGradient());
}

//===----------------------------------------------------------------------===//
// SetOp
//===----------------------------------------------------------------------===//

bool SetOp::loadsFrom(const MemorySlot &slot) { return false; }

bool SetOp::storesTo(const MemorySlot &slot) {
  return this->getGradient() == slot.ptr;
}

Value SetOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                       Value reachingDef, const DataLayout &dataLayout) {
  return this->getValue();
}

bool SetOp::canUsesBeRemoved(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses,
    llvm::SmallVectorImpl<OpOperand *> &newBlockingUses,
    const mlir::DataLayout &dataLayout) {
  return true;
}

DeletionKind SetOp::removeBlockingUses(
    const MemorySlot &slot,
    const llvm::SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder,
    Value reachingDefinition, const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

llvm::LogicalResult SetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, llvm::SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(slot.ptr == getGradient());
}

//===----------------------------------------------------------------------===//
// GetFuncOp
//===----------------------------------------------------------------------===//

LogicalResult
ForwardDiffOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

LogicalResult AutoDiffOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

LogicalResult BatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

void BroadcastOp::build(OpBuilder &builder, OperationState &result, Value input, int64_t width) {
  auto widthAttr = builder.getI64IntegerAttr(width);
  RankedTensorType output;
  // TODO: support things other than scalars and ranked tensors, maybe reuse getShadowType here?
  if (auto tensorType = input.getType().dyn_cast<TensorType>()) {
      auto shape = tensorType.getShape();
      SmallVector<int64_t, 4> newShape;
      newShape.push_back(width);
      newShape.append(shape.begin(), shape.end());
      output = RankedTensorType::get(newShape, tensorType.getElementType());
  } else {
    output = RankedTensorType::get({width}, input.getType());
  }
  build(builder, result, output, input, widthAttr);
}
