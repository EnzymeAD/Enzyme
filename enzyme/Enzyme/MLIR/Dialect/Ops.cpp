//===- EnzymeOps.cpp - Enzyme dialect ops -----------------------*- C++ -*-===//
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
#include "mlir/IR/PatternMatch.h"
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

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

//===----------------------------------------------------------------------===//
// ForwardDiffOp
//===----------------------------------------------------------------------===//

// Helper: check if any input is mutable.
static inline bool isMutable(Type type) {
  if (isa<mlir::MemRefType>(type) || isa<mlir::UnrankedMemRefType>(type) ||
      isa<mlir::LLVM::LLVMPointerType>(type)) {
    return true;
  }

  return false;
}

/**
 *
 * Modifies return activites for the FwdDiffOp
 * The activity promotion flow is as follows
 * (depending on variable use):
 *
 *           -----> enzyme_dupnoneed ----
 *          /                            \
 * enzyme_dup                             ---> enzyme_constnoneed
 *          \                           /
 *           ------> enzyme_const -----
 *
 */
class FwdRetOpt final : public OpRewritePattern<ForwardDiffOp> {
public:
  using OpRewritePattern<ForwardDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForwardDiffOp uop,
                                PatternRewriter &rewriter) const override {

    if (uop.getOutputs().size() == 0)
      return failure();

    auto retActivity = uop.getRetActivity();
    auto out_idx = 0;
    SmallVector<mlir::Value, 2> outs_args;
    SmallVector<Type, 2> out_ty;
    SmallVector<ActivityAttr, 2> newRetActivityArgs;
    bool changed = false;

    for (auto [idx, act] : llvm::enumerate(retActivity)) {
      auto iattr = cast<ActivityAttr>(act);
      auto val = iattr.getValue();
      mlir::Value res = uop.getOutputs()[out_idx];

      switch (val) {
      case Activity::enzyme_active:
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;

      case mlir::enzyme::Activity::enzyme_const:
        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          newRetActivityArgs.push_back(iattr);
        } else {
          changed = true;
          auto new_constnn = mlir::enzyme::ActivityAttr::get(
              rewriter.getContext(),
              mlir::enzyme::Activity::enzyme_constnoneed);
          newRetActivityArgs.push_back(new_constnn);
        }
        break;
      case Activity::enzyme_dupnoneed:

        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
          newRetActivityArgs.push_back(iattr);
        } else {
          if (!isMutable(res.getType())) {
            changed = true;
            auto new_constnn = mlir::enzyme::ActivityAttr::get(
                rewriter.getContext(),
                mlir::enzyme::Activity::enzyme_constnoneed);
            newRetActivityArgs.push_back(new_constnn);
          }
        }
        break;
      case Activity::enzyme_constnoneed:
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;
      case Activity::enzyme_activenoneed:
        outs_args.push_back(res);
        out_ty.push_back(res.getType());
        newRetActivityArgs.push_back(iattr);
        break;
      case Activity::enzyme_dup: {
        ActivityAttr new_dup = iattr;
        if (!res.use_empty()) {
          outs_args.push_back(res);
          out_ty.push_back(res.getType());
        } else {
          changed = true;
          // discard return, change attr
          new_dup = ActivityAttr::get(rewriter.getContext(),
                                      Activity::enzyme_dupnoneed);
        }

        out_idx++;

        // derivative
        res = uop.getOutputs()[out_idx];
        if (!res.use_empty()) {
          // activity arg doesn't update
          out_ty.push_back(res.getType());
          outs_args.push_back(res);
        } else {
          // no uses, can discard
          if (!isMutable(res.getType())) {
            changed = true;
            // check if primal is used
            if (new_dup.getValue() == Activity::enzyme_dupnoneed) {
              new_dup = ActivityAttr::get(rewriter.getContext(),
                                          Activity::enzyme_constnoneed);
            } else {
              new_dup = ActivityAttr::get(rewriter.getContext(),
                                          Activity::enzyme_const);
            }
          }
        }
        newRetActivityArgs.push_back(new_dup);
        break;
      }
      default:
        llvm_unreachable("unexpected activity arg");
      }

      out_idx++;
    }

    if (!changed)
      return failure();

    ArrayAttr newRetActivity =
        ArrayAttr::get(rewriter.getContext(),
                       llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                 newRetActivityArgs.end()));

    ForwardDiffOp newOp = rewriter.create<ForwardDiffOp>(
        uop.getLoc(), out_ty, uop.getFnAttr(), uop.getInputs(),
        uop.getActivityAttr(), newRetActivity, uop.getWidthAttr());

    // Map old uses of uop to newOp
    auto oldIdx = 0;
    auto newIdx = 0;
    for (auto [idx, old_act, new_act] :
         llvm::enumerate(retActivity, newRetActivityArgs)) {

      auto iattr = cast<ActivityAttr>(old_act);
      auto old_val = iattr.getValue();
      auto new_val = new_act.getValue();

      if (old_val == new_val) {
        // replace use
        uop.getOutputs()[oldIdx++].replaceAllUsesWith(
            newOp.getOutputs()[newIdx++]);
        if (old_val == Activity::enzyme_dup) {
          // 2nd replacement for derivative
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        }
      } else {
        // handle all substitutions
        if (new_val == Activity::enzyme_dupnoneed &&
            old_val == Activity::enzyme_dup) {
          ++oldIdx; // skip primal
          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
        } else if (new_val == mlir::enzyme::Activity::enzyme_constnoneed &&
                   old_val == mlir::enzyme::Activity::enzyme_const) {
          ++oldIdx; // skip const
        } else if (new_val == mlir::enzyme::Activity::enzyme_constnoneed &&
                   old_val == mlir::enzyme::Activity::enzyme_dupnoneed) {
          ++oldIdx; // skip gradient too
        } else if (new_val == Activity::enzyme_const &&
                   old_val == Activity::enzyme_dup) {

          uop.getOutputs()[oldIdx++].replaceAllUsesWith(
              newOp.getOutputs()[newIdx++]);
          ++oldIdx; // skip derivative
        } else if (new_val == Activity::enzyme_constnoneed &&
                   old_val == Activity::enzyme_dup) {
          ++oldIdx; // skip primal
          ++oldIdx; // skip derivative
        }
      }
    }

    rewriter.eraseOp(uop);
    return success();
  }
};

void ForwardDiffOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {

  patterns.add<FwdRetOpt>(context);
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

void BroadcastOp::build(OpBuilder &builder, OperationState &result, Value input,
                        ArrayRef<int64_t> shape) {
  auto shapeAttr = builder.getDenseI64ArrayAttr(shape);
  auto resultTy = input.getType();
  for (auto s : llvm::reverse(shape)) {
    resultTy = cast<AutoDiffTypeInterface>(resultTy).getShadowType(s);
  }
  build(builder, result, resultTy, input, shapeAttr);
}

//===----------------------------------------------------------------------===//
// SampleOp
//===----------------------------------------------------------------------===//

LogicalResult SampleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// TraceOp
//===----------------------------------------------------------------------===//

LogicalResult TraceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// SimulateOp
//===----------------------------------------------------------------------===//

LogicalResult SimulateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}