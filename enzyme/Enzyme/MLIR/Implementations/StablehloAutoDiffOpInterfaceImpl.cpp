//===- StablehloAutoDiffOpInterfaceImpl.cpp - StableHLO AD models ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// External model implementations of the reverse-mode automatic differentiation
// op interfaces for the StableHLO dialect.
//
// Currently implements: stablehlo::WhileOp
//   - Online_r2 (Stumm-Walther 2010) checkpointing when
//     enzyme.online_checkpointing is present.
//   - Full-tape fallback via enzyme.push/pop when no checkpointing is
//     requested.
//
// XLA constraint: stablehlo.while cond/body regions may only contain
// StableHLO ops.  All arith.*, scf.*, memref.* ops must be kept OUTSIDE
// the stablehlo.while regions (in the gradient function body, which is
// not compiled through XLA's StableHLO→HLO path).
//
// Design for Online_r2 forward scaffold:
//   - The SM arrays (ch, ord_ch, num_rep) are tensor<smSize x i64> iter-args.
//   - Scalar SM state (check, capo, …) is tensor<i64> iter-args.
//   - Checkpoint buffers are memref allocated OUTSIDE the while, stored to
//     from the body via memref.store — BUT memref ops inside stablehlo.while
//     body are also invalid in XLA.
//
// Therefore we use a HYBRID architecture:
//   - An OUTER scf.while (arith/scf/memref allowed) drives the SM step.
//   - For each FORWARD action, it invokes the primal stablehlo.while for
//     exactly one body step, re-entering/exiting it.
//   - For STORE actions, it writes to a memref checkpoint buffer.
//
// This keeps ALL non-StableHLO ops outside stablehlo.while regions.
//
// The outer scf.while carries: (numFwd, SM_scalars..., SM_arrays..., primalState...)
// and uses OnlineProgressOp + memref stores directly — all valid in scf context.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Implementations/LoopCheckpointing.h"
#include "Implementations/WhileCheckpointing.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/EnzymeLogic.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

// StableHLO dialect headers — available in the Bazel/Reactant build only.
// In a standalone LLVM build (without StableHLO), this file compiles as an
// empty translation unit via the ENZYME_ENABLE_STABLEHLO guard below.
#ifdef ENZYME_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

// ---------------------------------------------------------------------------
// Scaffold utility instantiation
// ---------------------------------------------------------------------------
struct StablehloWhileScaffoldHelper
    : public LoopCheckpointing<StablehloWhileScaffoldHelper, scf::ForOp> {
  static std::optional<int64_t> getConstantNumberOfIterations(scf::ForOp) {
    return std::nullopt;
  }
  static Value materializeLowerBound(OpBuilder &b, Location loc,
                                     scf::ForOp op,
                                     MGradientUtilsReverse *gutils) {
    return gutils->getNewFromOriginal(op.getLowerBound());
  }
  static Value materializeUpperBound(OpBuilder &b, Location loc,
                                     scf::ForOp op,
                                     MGradientUtilsReverse *gutils) {
    return gutils->getNewFromOriginal(op.getUpperBound());
  }
  static Value materializeStep(OpBuilder &b, Location loc, scf::ForOp op,
                                MGradientUtilsReverse *gutils) {
    return gutils->getNewFromOriginal(op.getStep());
  }
  static int64_t getConstantStart(scf::ForOp op) { return 0; }
  static int64_t getConstantStep(scf::ForOp op) { return 1; }
  static LogicalResult requireSingleResultBounds(scf::ForOp) {
    return success();
  }
  static void cloneOp(OpBuilder &b, Operation &op, IRMapping &m) {
    b.clone(op, m);
  }
  static scf::ForOp createConstantScaffoldLoop(OpBuilder &b, Location loc,
                                               int64_t lb, int64_t ub,
                                               int64_t step, ValueRange inits) {
    auto lbV  = arith::ConstantIndexOp::create(b, loc, lb);
    auto ubV  = arith::ConstantIndexOp::create(b, loc, ub);
    auto stV  = arith::ConstantIndexOp::create(b, loc, step);
    return scf::ForOp::create(b, loc, lbV, ubV, stV, inits);
  }
  static scf::ForOp createDynamicScaffoldLoop(OpBuilder &b, Location loc,
                                              Value lb, Value ub, Value step,
                                              ValueRange inits) {
    return scf::ForOp::create(b, loc, lb, ub, step, inits);
  }
};
using H = StablehloWhileScaffoldHelper;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pack an index scalar to 0-d tensor<i64> for StableHLO.
static Value packScalar(OpBuilder &b, Location loc, Value scalar) {
  auto i64Ty = b.getI64Type();
  auto tensorTy = RankedTensorType::get({}, i64Ty);
  Value asI64 = arith::IndexCastOp::create(b, loc, i64Ty, scalar);
  return tensor::FromElementsOp::create(b, loc, tensorTy, ValueRange{asI64});
}

/// Unpack 0-d tensor<i64> to index scalar.
static Value unpackScalar(OpBuilder &b, Location loc, Value t) {
  Value asI64 = tensor::ExtractOp::create(b, loc, t, ValueRange{});
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), asI64);
}

/// Copy ops from a stablehlo region block (except terminator).
/// `gutils` may be null — if so, only `mapping` is used for operand resolution.
static void copyStablehloBlock(OpBuilder &builder, Block *b,
                                MGradientUtilsReverse *gutils,
                                IRMapping &mapping) {
  for (auto &op : b->without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (!mapping.contains(operand) && gutils) {
        if (Value nv = gutils->originalToNewFn.lookupOrNull(operand))
          if (nv)
            mapping.map(operand, nv);
      }
    }
    if (builder.getInsertionBlock())
      builder.clone(op, mapping);
  }
  if (gutils) {
    for (auto [oldVal, newVal] : mapping.getValueMap())
      if (newVal)
        gutils->originalToNewFn.map(oldVal, newVal);
    for (auto [oldOp, newOp] : mapping.getOperationMap())
      if (newOp)
        gutils->originalToNewFnOps[oldOp] = newOp;
  }
}

/// Copy ops from a stablehlo region block WITHOUT touching gutils mappings.
/// Used when cloning primal body steps inside the scaffold — avoids polluting
/// originalToNewFn with mappings from the scaffold context.
/// Clones in order so each op's operands are either:
///   (a) in `mapping` (from block arg remapping or previous cloned ops), or
///   (b) defined OUTSIDE the block (valid to use directly in the new context).
/// Values defined INSIDE the block but NOT yet cloned should not appear as
/// operands of earlier ops (SSA form guarantees this for well-formed blocks).
static void copyStablehloBlockClean(OpBuilder &builder, Block *b,
                                     IRMapping &mapping) {
  for (auto &op : b->without_terminator()) {
    if (!builder.getInsertionBlock()) break;
    // For each operand: if it's defined inside this block, it must already
    // be in `mapping` (from a previous clone in this loop). If it's defined
    // outside the block, use it directly (it dominates the insertion point).
    // OpBuilder::clone handles both cases correctly.
    builder.clone(op, mapping);
  }
}

/// Evaluate the primal cond region, return scalar i1.
static Value evalCondRegion(OpBuilder &builder, Location loc,
                             stablehlo::WhileOp whileOp,
                             MGradientUtilsReverse *gutils,
                             ValueRange iterArgs,
                             IRMapping &outMapping) {
  Block *condBlock = &whileOp.getCond().front();
  for (auto &&[old, nw] :
       llvm::zip_equal(condBlock->getArguments(), iterArgs))
    outMapping.map(old, nw);
  // Use clean copy (no gutils lookup) to avoid polluting originalToNewFn.
  copyStablehloBlockClean(builder, condBlock, outMapping);

  auto retOp = cast<stablehlo::ReturnOp>(condBlock->getTerminator());
  return outMapping.lookupOrDefault(retOp.getOperand(0)); // tensor<i1>
}

/// Run one body step, return new iter-args.
static SmallVector<Value> runBodyStep(OpBuilder &builder, Location loc,
                                       stablehlo::WhileOp whileOp,
                                       MGradientUtilsReverse *gutils,
                                       ValueRange iterArgs,
                                       IRMapping &outMapping) {
  Block *bodyBlock = &whileOp.getBody().front();
  // Map body block arguments to the provided iter-args.
  // iterArgs count must match bodyBlock argument count.
  if (bodyBlock->getNumArguments() != iterArgs.size())
    return {};
  for (auto &&[old, nw] :
       llvm::zip_equal(bodyBlock->getArguments(), iterArgs))
    outMapping.map(old, nw);
  if (!builder.getInsertionBlock())
    return {};
  for (auto &op : bodyBlock->without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (outMapping.contains(operand))
        continue;
      if (gutils) {
        if (Value nv = gutils->originalToNewFn.lookupOrNull(operand))
          if (nv) { outMapping.map(operand, nv); continue; }
      }
      if (Operation *defOp = operand.getDefiningOp()) {
        if (defOp->getNumOperands() == 0 &&
            defOp->getParentRegion() != builder.getInsertionBlock()->getParent()) {
          auto *localClone = builder.clone(*defOp);
          auto opRes = llvm::cast<mlir::OpResult>(operand);
          outMapping.map(operand, localClone->getResult(opRes.getResultNumber()));
        }
      }
    }
    if (!builder.getInsertionBlock())
      break;
    builder.clone(op, outMapping);
  }

  auto retOp = cast<stablehlo::ReturnOp>(bodyBlock->getTerminator());
  SmallVector<Value> newArgs;
  for (auto v : retOp.getOperands())
    newArgs.push_back(outMapping.lookupOrDefault(v));
  return newArgs;
}

// ---------------------------------------------------------------------------
// Cache-layout for while online checkpointing
// ---------------------------------------------------------------------------
struct SHWhileOnlineCacheLayout {
  size_t numCarried = 0;

  size_t ckptBuf(size_t i) const { return i; }
  size_t idxBuf() const { return numCarried; }
  size_t numFwdBuf() const { return numCarried + 1; }
  size_t size() const { return numCarried + 2; }
};

// ---------------------------------------------------------------------------
// Pure StableHLO helpers for the Online_r2 SM
// ---------------------------------------------------------------------------

// 0-d tensor<i64> constant.
static Value shC(OpBuilder &b, Location loc, int64_t v) {
  auto ty = RankedTensorType::get({}, b.getI64Type());
  return stablehlo::ConstantOp::create(b, loc, ty,
      DenseElementsAttr::get(ty, b.getI64IntegerAttr(v)));
}
// 0-d tensor<i1> constant.
static Value shBool(OpBuilder &b, Location loc, bool v) {
  auto ty = RankedTensorType::get({}, b.getI1Type());
  return stablehlo::ConstantOp::create(b, loc, ty,
      DenseElementsAttr::get(ty, b.getIntegerAttr(b.getI1Type(), (int64_t)v)));
}
// stablehlo.add on 0-d tensor<i64>.
static Value shAdd(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::AddOp::create(b, loc, l, r);
}
// stablehlo.subtract.
static Value shSub(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::SubtractOp::create(b, loc, l, r);
}
// stablehlo.compare (signed) → tensor<i1>.
static Value shCmpEQ(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::CompareOp::create(
      b, loc, (mlir::Value)l, (mlir::Value)r,
      stablehlo::ComparisonDirection::EQ);
}
static Value shCmpNE(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::CompareOp::create(
      b, loc, (mlir::Value)l, (mlir::Value)r,
      stablehlo::ComparisonDirection::NE);
}
static Value shCmpLT(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::CompareOp::create(
      b, loc, (mlir::Value)l, (mlir::Value)r,
      stablehlo::ComparisonDirection::LT);
}
static Value shCmpLE(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::CompareOp::create(
      b, loc, (mlir::Value)l, (mlir::Value)r,
      stablehlo::ComparisonDirection::LE);
}
static Value shCmpGE(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::CompareOp::create(
      b, loc, (mlir::Value)l, (mlir::Value)r,
      stablehlo::ComparisonDirection::GE);
}
// stablehlo.select.
static Value shSel(OpBuilder &b, Location loc, Value cond, Value t, Value f) {
  return stablehlo::SelectOp::create(b, loc, cond, t, f);
}
// stablehlo.and / stablehlo.or / stablehlo.not on tensor<i1>.
static Value shAnd(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::AndOp::create(b, loc, l, r);
}
static Value shOr(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::OrOp::create(b, loc, l, r);
}
static Value shNot(OpBuilder &b, Location loc, Value v) {
  return stablehlo::NotOp::create(b, loc, v);
}
// stablehlo.remainder (signed) on tensor<i64>.
static Value shRem(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::RemOp::create(b, loc, l, r);
}
// stablehlo.divide (signed) on tensor<i64>.
static Value shDiv(OpBuilder &b, Location loc, Value l, Value r) {
  return stablehlo::DivOp::create(b, loc, l, r);
}

// Load element at 0-d tensor<i64> slot from 1-d tensor<n×i64> arr.
// Returns 0-d tensor<i64>.
static Value shLoad(OpBuilder &b, Location loc, Value arr, Value slot) {
  // slot is tensor<i64> (0-d); dynamic_slice needs start indices as 0-d tensors.
  auto sliceTy = RankedTensorType::get({1}, b.getI64Type());
  Value slice = stablehlo::DynamicSliceOp::create(b, loc, sliceTy, arr,
      ValueRange{slot}, b.getDenseI64ArrayAttr({1}));
  return stablehlo::ReshapeOp::create(b, loc,
      RankedTensorType::get({}, b.getI64Type()), slice);
}

// Store 0-d tensor<i64> val at slot in 1-d tensor<n×i64> arr.
// Returns updated arr.
static Value shStore(OpBuilder &b, Location loc, Value arr, Value slot, Value val) {
  auto sliceTy = RankedTensorType::get({1}, b.getI64Type());
  Value reshaped = stablehlo::ReshapeOp::create(b, loc, sliceTy, val);
  return stablehlo::DynamicUpdateSliceOp::create(b, loc, arr.getType(),
      arr, reshaped, ValueRange{slot});
}

// Load element of arbitrary type from 1-d tensor at i64 slot.
static Value shLoadTyped(OpBuilder &b, Location loc, Value arr, Value slot,
                          RankedTensorType elemTy0d) {
  auto n = cast<RankedTensorType>(arr.getType()).getDimSize(0);
  (void)n;
  auto sliceTy = RankedTensorType::get({1}, elemTy0d.getElementType());
  Value slice = stablehlo::DynamicSliceOp::create(b, loc, sliceTy, arr,
      ValueRange{slot}, b.getDenseI64ArrayAttr({1}));
  return stablehlo::ReshapeOp::create(b, loc, elemTy0d, slice);
}
static Value shStoreTyped(OpBuilder &b, Location loc, Value arr, Value slot, Value val) {
  auto elemTy = cast<RankedTensorType>(val.getType()).getElementType();
  auto sliceTy = RankedTensorType::get({1}, elemTy);
  Value reshaped = stablehlo::ReshapeOp::create(b, loc, sliceTy, val);
  return stablehlo::DynamicUpdateSliceOp::create(b, loc, arr.getType(),
      arr, reshaped, ValueRange{slot});
}

// Build stacked checkpoint buffer type: prepend nSlots to the original shape.
// E.g. tensor<4x8xf64> with nSlots=3 → tensor<3x4x8xf64>.
static RankedTensorType stackedCkptType(int64_t nSlots, RankedTensorType origTy) {
  SmallVector<int64_t> shape;
  shape.push_back(nSlots);
  shape.append(origTy.getShape().begin(), origTy.getShape().end());
  return RankedTensorType::get(shape, origTy.getElementType());
}

// Load one tensor from a stacked checkpoint buffer at the given slot.
// arr has type tensor<nSlots × d1 × ... × dk × elemTy>, slot is tensor<i64>.
// Returns tensor<d1 × ... × dk × elemTy>.
static Value stackedLoad(OpBuilder &b, Location loc, Value arr, Value slot,
                          RankedTensorType origTy) {
  auto arrTy = cast<RankedTensorType>(arr.getType());
  int64_t rank = arrTy.getRank();
  // start_indices: [slot, 0, 0, ..., 0]
  SmallVector<Value> startIndices;
  startIndices.push_back(slot);
  Value zero64 = stablehlo::ConstantOp::create(b, loc,
      RankedTensorType::get({}, b.getI64Type()),
      DenseElementsAttr::get(RankedTensorType::get({}, b.getI64Type()),
                             b.getI64IntegerAttr(0)));
  for (int64_t i = 1; i < rank; ++i)
    startIndices.push_back(zero64);
  // slice_sizes: [1, d1, d2, ..., dk]
  SmallVector<int64_t> sliceSizes;
  sliceSizes.push_back(1);
  sliceSizes.append(origTy.getShape().begin(), origTy.getShape().end());
  // Result of dynamic_slice is tensor<1 × d1 × ... × dk>
  SmallVector<int64_t> sliceShape(sliceSizes);
  auto sliceTy = RankedTensorType::get(sliceShape, arrTy.getElementType());
  Value slice = stablehlo::DynamicSliceOp::create(b, loc, sliceTy, arr,
      startIndices, b.getDenseI64ArrayAttr(sliceSizes));
  return stablehlo::ReshapeOp::create(b, loc, origTy, slice);
}

// Store one tensor into a stacked checkpoint buffer at the given slot.
// arr has type tensor<nSlots × d1 × ... × dk × elemTy>, slot is tensor<i64>.
// val has type tensor<d1 × ... × dk × elemTy>.
// Returns the updated buffer.
static Value stackedStore(OpBuilder &b, Location loc, Value arr, Value slot,
                           Value val) {
  auto arrTy = cast<RankedTensorType>(arr.getType());
  int64_t rank = arrTy.getRank();
  auto valTy = cast<RankedTensorType>(val.getType());
  // Reshape val: tensor<d1×...×dk> → tensor<1×d1×...×dk>
  SmallVector<int64_t> expandedShape;
  expandedShape.push_back(1);
  expandedShape.append(valTy.getShape().begin(), valTy.getShape().end());
  auto expandedTy = RankedTensorType::get(expandedShape, valTy.getElementType());
  Value reshaped = stablehlo::ReshapeOp::create(b, loc, expandedTy, val);
  // start_indices: [slot, 0, 0, ..., 0]
  SmallVector<Value> startIndices;
  startIndices.push_back(slot);
  Value zero64 = stablehlo::ConstantOp::create(b, loc,
      RankedTensorType::get({}, b.getI64Type()),
      DenseElementsAttr::get(RankedTensorType::get({}, b.getI64Type()),
                             b.getI64IntegerAttr(0)));
  for (int64_t i = 1; i < rank; ++i)
    startIndices.push_back(zero64);
  return stablehlo::DynamicUpdateSliceOp::create(b, loc, arr.getType(),
      arr, reshaped, startIndices);
}

// Build a stablehlo.while loop from [lb, ub) carrying `initArgs`.
// condFn receives (builder, loc, args) and returns tensor<i1>.
// bodyFn receives (builder, loc, args) and returns new args via stablehlo.return.
static Operation *shWhile(OpBuilder &b, Location loc, ValueRange initArgs,
    llvm::function_ref<Value(OpBuilder &, Location, ValueRange)> condFn,
    llvm::function_ref<SmallVector<Value>(OpBuilder &, Location, ValueRange)> bodyFn) {
  SmallVector<Type> types;
  for (auto v : initArgs) types.push_back(v.getType());
  auto w = stablehlo::WhileOp::create(b, loc, types, initArgs);

  // cond region
  {
    OpBuilder::InsertionGuard g(b);
    Block *condBlk = b.createBlock(&w.getCond());
    for (auto ty : types) condBlk->addArgument(ty, loc);
    b.setInsertionPointToEnd(condBlk);
    Value c = condFn(b, loc, condBlk->getArguments());
    stablehlo::ReturnOp::create(b, loc, ValueRange{c});
  }
  // body region
  {
    OpBuilder::InsertionGuard g(b);
    Block *bodyBlk = b.createBlock(&w.getBody());
    for (auto ty : types) bodyBlk->addArgument(ty, loc);
    b.setInsertionPointToEnd(bodyBlk);
    SmallVector<Value> newArgs = bodyFn(b, loc, bodyBlk->getArguments());
    stablehlo::ReturnOp::create(b, loc, newArgs);
  }
  return w;
}

// stablehlo.if returning `types`. thenFn/elseFn return vectors via stablehlo.return.
static stablehlo::IfOp shIf(OpBuilder &b, Location loc,
    Value cond, TypeRange types,
    llvm::function_ref<SmallVector<Value>(OpBuilder &, Location)> thenFn,
    llvm::function_ref<SmallVector<Value>(OpBuilder &, Location)> elseFn) {
  auto ifOp = stablehlo::IfOp::create(b, loc, types, cond);
  // IfOp::create produces two empty regions; create blocks explicitly.
  {
    OpBuilder::InsertionGuard g(b);
    b.createBlock(&ifOp.getTrueBranch());
    SmallVector<Value> thenVals = thenFn(b, loc);
    stablehlo::ReturnOp::create(b, loc, thenVals);
  }
  {
    OpBuilder::InsertionGuard g(b);
    b.createBlock(&ifOp.getFalseBranch());
    SmallVector<Value> elseVals = elseFn(b, loc);
    stablehlo::ReturnOp::create(b, loc, elseVals);
  }
  return ifOp;
}

// Shift ord_ch right: ord_ch[k] = ord_ch[k-1] for k = snaps-1 downto kLow+1.
// Returns updated ord_ch.
static Value shOrdChShiftRight(OpBuilder &b, Location loc,
                                Value snaps, Value kLow, Value ord_ch,
                                int64_t smSize) {
  // Loop: i = 0 .. (snaps-1-kLow), k = snaps-1-i, km1 = snaps-2-i
  // Carry: (i, ord_ch)
  auto i64ty0 = RankedTensorType::get({}, b.getI64Type());
  auto arrTy  = cast<RankedTensorType>(ord_ch.getType());
  Value zero = shC(b, loc, 0);
  Value one  = shC(b, loc, 1);
  Value two  = shC(b, loc, 2);
  Value snapsM1 = shSub(b, loc, snaps, one);
  Value loopUB  = shSub(b, loc, snapsM1, kLow); // snaps-1-kLow iterations

  // stablehlo.while loop: carry (i: ti64, ord_ch: arr)
  SmallVector<Value> initArgs = {zero, ord_ch};
  SmallVector<Type>  initTypes = {i64ty0, arrTy};

  auto w = shWhile(b, loc, initArgs,
    [&](OpBuilder &b2, Location l2, ValueRange args) -> Value {
      // cond: i < loopUB
      return shCmpLT(b2, l2, args[0], loopUB);
    },
    [&](OpBuilder &b2, Location l2, ValueRange args) -> SmallVector<Value> {
      Value i = args[0], ch = args[1];
      // k = snaps-1-i, km1 = snaps-2-i
      Value snapsM1b = shSub(b2, l2, snaps, shC(b2, l2, 1));
      Value snapsM2b = shSub(b2, l2, snaps, shC(b2, l2, 2));
      Value k   = shSub(b2, l2, snapsM1b, i);
      Value km1 = shSub(b2, l2, snapsM2b, i);
      Value val = shLoad(b2, l2, ch, km1);
      Value newCh = shStore(b2, l2, ch, k, val);
      return {shAdd(b2, l2, i, shC(b2, l2, 1)), newCh};
    });
  return w->getResult(1); // updated ord_ch
}

// Struct holding SM output: 11 scalars + 3 arrays.
struct SMResult {
  Value action;     // tensor<i64>: 0=terminate, 1=takeshot, 2=advance
  Value check, capo, oldcapoO, t, incr, iter, oldind, ind, old_f, offset;
  Value ch, ord_ch, num_rep; // tensor<smSize×i64>
};

// Inline the Online_r2 state machine in pure StableHLO.
// Flat implementation — all scalar outputs via shSel, array mutations via
// sequential shStore calls. No nested builder aliasing.
static SMResult shInlineSM(OpBuilder &b, Location loc,
    Value check, Value capo, Value oldcapoO, Value t, Value incr,
    Value iter, Value oldind, Value ind, Value old_f, Value offset,
    Value snaps, Value ch, Value ord_ch, Value num_rep, int64_t smSize) {

  Value zero  = shC(b, loc, 0);
  Value one   = shC(b, loc, 1);
  Value two   = shC(b, loc, 2);
  Value three = shC(b, loc, 3);

  Value actTerminate = zero;
  Value actTakeshot  = one;
  Value actAdvance   = two;

  Value snapsM1 = shSub(b, loc, snaps, one);
  Value snapsP1 = shAdd(b, loc, snaps, one);
  Value snapsP2 = shAdd(b, loc, snaps, two);

  // ── Top-level condition: takeshot r=1? ─────────────────────────────────────
  Value checkLtZero = shCmpLT(b, loc, check, zero);
  Value safeCheck   = shSel(b, loc, checkLtZero, zero, check);
  Value chAtCheck   = shLoad(b, loc, ch, safeCheck);
  Value chNeqCapo    = shCmpNE(b, loc, chAtCheck, capo);
  Value capoLeSnapsM1 = shCmpLE(b, loc, capo, snapsM1);
  Value isTakeshotR1 = shOr(b, loc, checkLtZero,
                            shAnd(b, loc, chNeqCapo, capoLeSnapsM1));

  // ── Pre-compute everything we'll need for all branches ─────────────────────
  // These are all emitted unconditionally; shSel picks the right ones.

  // --- Branch 2: advance r=1 ---
  Value capoLtSnapsM1 = shCmpLT(b, loc, capo, snapsM1);
  Value b2Capo = shAdd(b, loc, oldcapoO, one);  // capo = oldcapo_o + 1

  // --- Branch 3: r=2 ---
  Value chCheck = shLoad(b, loc, ch, check);
  Value r2AdvCond = shCmpEQ(b, loc, chCheck, capo);

  // Branch 3a: advance r=2 (ch[check] == capo)
  // case 1: capo = very large sentinel
  Value infCapo = shC(b, loc, 4611686018427387903LL); // max/4

  // case 2: capo = ch[1] + incr
  Value ch1 = shLoad(b, loc, ch, one);
  Value case2Capo = shAdd(b, loc, ch1, incr);

  // case 3: dispatch on iter==0
  Value iterIsZero = shCmpEQ(b, loc, iter, zero);
  Value chOldind = shLoad(b, loc, ch, oldind);
  // iter==0 path: loop (t+1)/2+1 times; use shWhile
  Value tP1Div2 = shDiv(b, loc, shAdd(b, loc, t, one), two);
  Value loopUBc3 = shAdd(b, loc, tP1Div2, one);
  // Build the loop: carry (i, capo_acc, incr_acc, iter_acc)
  auto c3Loop = shWhile(b, loc,
      SmallVector<Value>{zero, chOldind, incr, iter},
      [&](OpBuilder &b2, Location l2, ValueRange a) -> Value {
        return shCmpLT(b2, l2, a[0], loopUBc3);
      },
      [&](OpBuilder &b2, Location l2, ValueRange a) -> SmallVector<Value> {
        return {shAdd(b2, l2, a[0], shC(b2, l2, 1)),
                shAdd(b2, l2, a[1], a[2]),
                shAdd(b2, l2, a[2], shC(b2, l2, 1)),
                shAdd(b2, l2, a[3], shC(b2, l2, 1))};
      });
  Value c3LoopCapo = c3Loop->getResult(1);
  Value c3LoopIncr = c3Loop->getResult(2);
  Value c3LoopIter = c3Loop->getResult(3);

  // iter!=0 path: capo=ch[ind]+incr; iter++; incr++
  Value chInd = shLoad(b, loc, ch, ind);
  Value c3ne0Capo = shAdd(b, loc, chInd, incr);
  Value c3ne0Incr = shAdd(b, loc, incr, one);
  Value c3ne0Iter = shAdd(b, loc, iter, one);

  Value case3Capo = shSel(b, loc, iterIsZero, c3LoopCapo, c3ne0Capo);
  Value case3Incr = shSel(b, loc, iterIsZero, c3LoopIncr, c3ne0Incr);
  Value case3Iter = shSel(b, loc, iterIsZero, c3LoopIter, c3ne0Iter);

  // 3a default: complex with capoEqSnapsM1/tIsZero/iterLtOffset
  Value capoEqSnapsM1_d = shCmpEQ(b, loc, capo, snapsM1);
  Value tIsZero_d        = shCmpEQ(b, loc, t, zero);
  Value iterLtOffset_d   = shCmpLT(b, loc, iter, offset);
  Value offsetIs1_d      = shCmpEQ(b, loc, offset, one);
  Value defCapoSnapsM1 = shAdd(b, loc, capo, two);
  Value defCapoT0      = shSel(b, loc, iterLtOffset_d,
                            shAdd(b, loc, capo, one), shAdd(b, loc, capo, two));
  Value defTNew        = shSel(b, loc, offsetIs1_d, shAdd(b, loc, t, one), t);
  Value defActD        = shSel(b, loc, capoEqSnapsM1_d, actAdvance,
                            shSel(b, loc, tIsZero_d, actAdvance, actTerminate));
  Value defCapoD       = shSel(b, loc, capoEqSnapsM1_d, defCapoSnapsM1,
                            shSel(b, loc, tIsZero_d, defCapoT0, capo));
  Value defIndD        = shSel(b, loc, capoEqSnapsM1_d, snapsM1, ind);
  Value defTD          = shSel(b, loc, capoEqSnapsM1_d, t,
                            shSel(b, loc, tIsZero_d, defTNew, t));

  Value is1 = shCmpEQ(b, loc, snaps, one);
  Value is2 = shCmpEQ(b, loc, snaps, two);
  Value is3 = shCmpEQ(b, loc, snaps, three);
  Value notIs123 = shAnd(b, loc,
                     shAnd(b, loc, shCmpNE(b, loc, snaps, one),
                                   shCmpNE(b, loc, snaps, two)),
                     shCmpNE(b, loc, snaps, three));

  Value a3aFinalCapo = shSel(b, loc, is1, infCapo,
                         shSel(b, loc, is2, case2Capo,
                         shSel(b, loc, is3, case3Capo, defCapoD)));
  Value a3aFinalIncr = shSel(b, loc, is3, case3Incr, incr);
  Value a3aFinalIter = shSel(b, loc, is3, case3Iter, iter);
  Value a3aFinalInd  = shSel(b, loc, notIs123, defIndD, ind);
  Value a3aFinalT    = shSel(b, loc, notIs123, defTD, t);
  Value a3aFinalAct  = shSel(b, loc, notIs123, defActD, actAdvance);
  // ch/ord_ch/num_rep unchanged in 3a advance

  // Branch 3b: takeshot r=2 (ch[check] != capo)
  // case 2: ch[1]=capo; incr++
  Value b3b_is2_ch  = shStore(b, loc, ch, one, capo);  // ch[1]=capo
  Value b3b_is2_incr = shAdd(b, loc, incr, one);

  // case 3: ch[ind]=capo; check=ind; nr1-based updates
  Value b3b_is3_ch    = shStore(b, loc, ch, ind, capo);  // ch[ind]=capo
  Value numRep1       = shLoad(b, loc, num_rep, one);
  Value iterEqNR1     = shCmpEQ(b, loc, iter, numRep1);
  Value newNR1        = shAdd(b, loc, numRep1, one);
  Value newNr_upd     = shStore(b, loc, num_rep, one, newNR1);
  Value remVal        = shRem(b, loc, newNR1, two);
  Value newIndNR1     = shSub(b, loc, two, remVal);
  Value nr1_iter      = shSel(b, loc, iterEqNR1, zero, iter);
  Value nr1_t         = shSel(b, loc, iterEqNR1, shAdd(b, loc, t, one), t);
  Value nr1_oldind    = shSel(b, loc, iterEqNR1, ind, oldind);
  Value nr1_ind       = shSel(b, loc, iterEqNR1, newIndNR1, ind);
  Value nr1_incr      = shSel(b, loc, iterEqNR1, one, incr);
  Value nr1_num_rep   = shSel(b, loc, iterEqNR1, newNr_upd, num_rep);
  Value b3b_is3_check = ind;  // check=ind
  // ch: b3b_is3_ch (ch[ind]=capo)

  // default (snaps>=4): sub-case A (capo < snaps+2)
  Value isDefault_b = shAnd(b, loc, shCmpNE(b, loc, snaps, two),
                                    shCmpNE(b, loc, snaps, three));
  Value capoLtSnapsP2 = shCmpLT(b, loc, capo, snapsP2);
  Value capoEqSnapsP1 = shCmpEQ(b, loc, capo, snapsP1);

  // subA: ch[ind]=capo
  Value defA_ch = shStore(b, loc, ch, ind, capo);

  // rot1: ordSnapsM1=ord_ch[snaps-1]; shift; ord_ch[1]=ordSnapsM1; check=ind; incr=2; ind=2
  Value ordSnapsM1 = shLoad(b, loc, ord_ch, snapsM1);
  // Shift ord_ch right for rot1 (kLow=1): ord_ch[k]=ord_ch[k-1] for k=snaps-1 downto 2
  Value rot1OrdCh = shOrdChShiftRight(b, loc, snaps, one, ord_ch, smSize);
  Value rot1OrdCh2 = shStore(b, loc, rot1OrdCh, one, ordSnapsM1);
  Value rot1Check = ind;
  Value rot1Incr = two;
  Value rot1Oldind = ordSnapsM1;
  Value rot1Ind = two;
  // subA check/incr/oldind/ind: pick rot1 if capo==snaps+1, else keep (ind, incr, oldind, ind)
  Value defA_check  = shSel(b, loc, capoEqSnapsP1, rot1Check, ind);
  Value defA_incr   = shSel(b, loc, capoEqSnapsP1, rot1Incr, incr);
  Value defA_oldind = shSel(b, loc, capoEqSnapsP1, rot1Oldind, oldind);
  Value defA_ind    = shSel(b, loc, capoEqSnapsP1, rot1Ind, ind);
  Value defA_ordCh  = shSel(b, loc, capoEqSnapsP1, rot1OrdCh2, ord_ch);

  // subB (capo>=snaps+2): rot2 or simple store or terminate
  Value tIsZero_b = shCmpEQ(b, loc, t, zero);
  Value iterEqOff_b = shCmpEQ(b, loc, iter, offset);
  // rot2: ord_ch[snaps-1]=capo; shift(kLow=incr); ord_ch[incr]=ord_ch[snaps-1]; incr++; ind=newIncr; iter=1
  Value rot2Ch = shStore(b, loc, ch, ordSnapsM1, capo);
  Value rot2OrdCh  = shOrdChShiftRight(b, loc, snaps, incr, ord_ch, smSize);
  Value rot2OrdCh2 = shStore(b, loc, rot2OrdCh, incr, ordSnapsM1);
  Value rot2Incr = shAdd(b, loc, incr, one);
  Value rot2Ind  = rot2Incr;

  // simple store: ch[ord_ch[ind]]=capo; check=ord_ch[ind]; iter++; ind++
  Value ordChInd_b = shLoad(b, loc, ord_ch, ind);
  Value simCh = shStore(b, loc, ch, ordChInd_b, capo);
  Value simIter = shAdd(b, loc, iter, one);
  Value simInd  = shAdd(b, loc, ind, one);

  // subB results (tIsZero && iterEqOff → rot2; tIsZero && !iterEqOff → simple; else terminate)
  Value tIsZeroAndIterEqOff = shAnd(b, loc, tIsZero_b, iterEqOff_b);
  Value tIsZeroAndNotOff    = shAnd(b, loc, tIsZero_b, shNot(b, loc, iterEqOff_b));
  // tIsZero, iterEqOff → rot2
  // tIsZero, !iterEqOff → simple store
  // !tIsZero → terminate
  Value defB_act   = shSel(b, loc, tIsZero_b, actTakeshot, actTerminate);
  Value defB_check = shSel(b, loc, tIsZeroAndIterEqOff, ordSnapsM1,
                       shSel(b, loc, tIsZeroAndNotOff, ordChInd_b, check));
  Value defB_incr  = shSel(b, loc, tIsZeroAndIterEqOff, rot2Incr, incr);
  Value defB_iter  = shSel(b, loc, tIsZeroAndIterEqOff, one,
                       shSel(b, loc, tIsZeroAndNotOff, simIter, iter));
  Value defB_oldind = shSel(b, loc, tIsZeroAndIterEqOff, ordSnapsM1, oldind);
  Value defB_ind   = shSel(b, loc, tIsZeroAndIterEqOff, rot2Ind,
                       shSel(b, loc, tIsZeroAndNotOff, simInd, ind));
  Value defB_t     = t; // t unchanged in all subB paths
  Value defB_ch    = shSel(b, loc, tIsZeroAndIterEqOff, rot2Ch,
                       shSel(b, loc, tIsZeroAndNotOff, simCh, ch));
  Value defB_ordCh = shSel(b, loc, tIsZeroAndIterEqOff, rot2OrdCh2, ord_ch);

  // subA vs subB (capoLtSnapsP2 → subA, else → subB), all within isDefault_b
  Value def_act   = shSel(b, loc, capoLtSnapsP2, actTakeshot, defB_act);
  Value def_check = shSel(b, loc, capoLtSnapsP2, defA_check, defB_check);
  Value def_incr  = shSel(b, loc, capoLtSnapsP2, defA_incr, defB_incr);
  Value def_iter  = shSel(b, loc, capoLtSnapsP2, iter, defB_iter);
  Value def_oldind = shSel(b, loc, capoLtSnapsP2, defA_oldind, defB_oldind);
  Value def_ind   = shSel(b, loc, capoLtSnapsP2, defA_ind, defB_ind);
  Value def_t     = shSel(b, loc, capoLtSnapsP2, t, defB_t);
  Value def_ch    = shSel(b, loc, capoLtSnapsP2, defA_ch, defB_ch);
  Value def_ordCh = shSel(b, loc, capoLtSnapsP2, defA_ordCh, defB_ordCh);

  // rot2 offset decrement (only in rot2 path within default)
  Value capoGeSnapsP2 = shCmpGE(b, loc, capo, snapsP2);
  Value rot2Cond = shAnd(b, loc,
                     shAnd(b, loc, isDefault_b, capoGeSnapsP2),
                     shAnd(b, loc, tIsZero_b, iterEqOff_b));
  Value def_offset = shSel(b, loc, rot2Cond, shSub(b, loc, offset, one), offset);

  // 3b: merge case2 / case3 / default
  // ch merges: case2 updates ch[1], case3 updates ch[ind], default updates various
  // Use is2/is3/isDefault to pick:
  Value b3b_ch = shSel(b, loc, is2, b3b_is2_ch,
                   shSel(b, loc, is3, b3b_is3_ch,
                   shSel(b, loc, isDefault_b, def_ch, ch)));
  Value b3b_ordCh = shSel(b, loc, isDefault_b, def_ordCh, ord_ch);
  Value b3b_numRep = shSel(b, loc, is3, nr1_num_rep, num_rep);

  Value is1b = shCmpEQ(b, loc, snaps, one);
  Value b3b_act   = shSel(b, loc, is1b, actAdvance,
                      shSel(b, loc, isDefault_b, def_act, actTakeshot));
  Value b3b_check = shSel(b, loc, is3, b3b_is3_check,
                      shSel(b, loc, isDefault_b, def_check, check));
  Value b3b_incr  = shSel(b, loc, is2, b3b_is2_incr,
                      shSel(b, loc, is3, nr1_incr,
                      shSel(b, loc, isDefault_b, def_incr, incr)));
  Value b3b_iter  = shSel(b, loc, is3, nr1_iter,
                      shSel(b, loc, isDefault_b, def_iter, iter));
  Value b3b_oldind = shSel(b, loc, is3, nr1_oldind,
                       shSel(b, loc, isDefault_b, def_oldind, oldind));
  Value b3b_ind   = shSel(b, loc, is3, nr1_ind,
                      shSel(b, loc, isDefault_b, def_ind, ind));
  Value b3b_t     = shSel(b, loc, is3, nr1_t,
                      shSel(b, loc, isDefault_b, def_t, t));
  Value b3b_offset = shSel(b, loc, isDefault_b, def_offset, offset);

  // ── Branch 1: TAKESHOT r=1 ─────────────────────────────────────────────
  Value b1_newCheck = shAdd(b, loc, check, one);
  Value b1_ch = shStore(b, loc, ch, b1_newCheck, capo);

  // Fill num_rep and ord_ch arrays using shWhile loops.
  // Loop 1: fill with value 2 (for snaps<4)
  auto fill2 = shWhile(b, loc,
      SmallVector<Value>{zero, num_rep, ord_ch},
      [&](OpBuilder &b2, Location l2, ValueRange a) -> Value {
        return shCmpLT(b2, l2, a[0], snaps);
      },
      [&](OpBuilder &b2, Location l2, ValueRange a) -> SmallVector<Value> {
        Value i = a[0], nr = a[1], oc = a[2];
        return {shAdd(b2, l2, i, shC(b2, l2, 1)),
                shStore(b2, l2, nr, i, shC(b2, l2, 2)),
                shStore(b2, l2, oc, i, i)};
      });
  Value fill2_nr = fill2->getResult(1);
  Value fill2_oc = fill2->getResult(2);

  // Loop 2: fill with value 1 (for snaps>=4)
  auto fill1 = shWhile(b, loc,
      SmallVector<Value>{zero, num_rep, ord_ch},
      [&](OpBuilder &b2, Location l2, ValueRange a) -> Value {
        return shCmpLT(b2, l2, a[0], snaps);
      },
      [&](OpBuilder &b2, Location l2, ValueRange a) -> SmallVector<Value> {
        Value i = a[0], nr = a[1], oc = a[2];
        return {shAdd(b2, l2, i, shC(b2, l2, 1)),
                shStore(b2, l2, nr, i, shC(b2, l2, 1)),
                shStore(b2, l2, oc, i, i)};
      });
  Value fill1_nr = fill1->getResult(1);
  Value fill1_oc = fill1->getResult(2);

  Value snapsLt4 = shCmpLT(b, loc, snaps, shC(b, loc, 4));
  Value b1_numRep = shSel(b, loc, snapsLt4, fill2_nr, fill1_nr);
  Value b1_ordCh  = shSel(b, loc, snapsLt4, fill2_oc, fill1_oc);
  Value b1_incr   = shSel(b, loc, snapsLt4, two, one);
  Value b1_iter   = one;
  Value b1_oldind = shSel(b, loc, snapsLt4, snapsM1, one);
  Value b1_offset = shSel(b, loc, snapsLt4, offset, snapsM1);
  Value capoEqSnapsM1_b1 = shCmpEQ(b, loc, capo, snapsM1);
  Value b1_ind   = shSel(b, loc, capoEqSnapsM1_b1, two, ind);
  Value b1_oldF  = shSel(b, loc, capoEqSnapsM1_b1, one, old_f);

  // ── Merge all branches ──────────────────────────────────────────────────
  // Branch 3 (r=2): r2AdvCond → 3a, else → 3b
  Value brIsNotR1 = shNot(b, loc, isTakeshotR1);
  // Among the not-r1 cases: branch2 or branch3
  // branch2 cond: !isTakeshotR1 && capoLtSnapsM1
  // branch3 cond: !isTakeshotR1 && !capoLtSnapsM1; then r2AdvCond → 3a, else → 3b

  // 3a scalars: a3aFinalCapo, a3aFinalIncr, a3aFinalIter, a3aFinalInd, a3aFinalT, a3aFinalAct
  // 3b scalars: b3b_act, b3b_check, b3b_incr, b3b_iter, b3b_oldind, b3b_ind, b3b_t, b3b_offset
  // Note: 3a doesn't change check, oldind, offset (from original)

  Value inR2 = shAnd(b, loc, brIsNotR1, shNot(b, loc, capoLtSnapsM1));
  Value r3_act   = shSel(b, loc, r2AdvCond, a3aFinalAct, b3b_act);
  Value r3_check = shSel(b, loc, r2AdvCond, check, b3b_check);
  Value r3_capo  = shSel(b, loc, r2AdvCond, a3aFinalCapo, capo);
  Value r3_t     = shSel(b, loc, r2AdvCond, a3aFinalT, b3b_t);
  Value r3_incr  = shSel(b, loc, r2AdvCond, a3aFinalIncr, b3b_incr);
  Value r3_iter  = shSel(b, loc, r2AdvCond, a3aFinalIter, b3b_iter);
  Value r3_oldind = shSel(b, loc, r2AdvCond, oldind, b3b_oldind);
  Value r3_ind   = shSel(b, loc, r2AdvCond, a3aFinalInd, b3b_ind);
  Value r3_offset = shSel(b, loc, r2AdvCond, offset, b3b_offset);
  Value r3_ch    = shSel(b, loc, r2AdvCond, ch, b3b_ch);     // 3a: ch unchanged
  Value r3_ordCh = shSel(b, loc, r2AdvCond, ord_ch, b3b_ordCh);
  Value r3_numRep = shSel(b, loc, r2AdvCond, num_rep, b3b_numRep);

  // Branch 2 vs Branch 3
  Value notR1_act   = shSel(b, loc, capoLtSnapsM1, actAdvance, r3_act);
  Value notR1_check = shSel(b, loc, capoLtSnapsM1, check, r3_check);
  Value notR1_capo  = shSel(b, loc, capoLtSnapsM1, b2Capo, r3_capo);
  Value notR1_t     = shSel(b, loc, capoLtSnapsM1, t, r3_t);
  Value notR1_incr  = shSel(b, loc, capoLtSnapsM1, incr, r3_incr);
  Value notR1_iter  = shSel(b, loc, capoLtSnapsM1, iter, r3_iter);
  Value notR1_oldind = shSel(b, loc, capoLtSnapsM1, oldind, r3_oldind);
  Value notR1_ind   = shSel(b, loc, capoLtSnapsM1, ind, r3_ind);
  Value notR1_offset = shSel(b, loc, capoLtSnapsM1, offset, r3_offset);
  Value notR1_ch    = shSel(b, loc, capoLtSnapsM1, ch, r3_ch);
  Value notR1_ordCh = shSel(b, loc, capoLtSnapsM1, ord_ch, r3_ordCh);
  Value notR1_numRep = shSel(b, loc, capoLtSnapsM1, num_rep, r3_numRep);

  // Branch 1 vs (Branch 2/3)
  SMResult res;
  res.action   = shSel(b, loc, isTakeshotR1, actTakeshot, notR1_act);
  res.check    = shSel(b, loc, isTakeshotR1, b1_newCheck, notR1_check);
  res.capo     = shSel(b, loc, isTakeshotR1, capo, notR1_capo);
  res.oldcapoO = shSel(b, loc, isTakeshotR1, capo, oldcapoO); // b1: oldcapo_o=capo
  res.t        = shSel(b, loc, isTakeshotR1, zero, notR1_t);  // b1: t=0
  res.incr     = shSel(b, loc, isTakeshotR1, b1_incr, notR1_incr);
  res.iter     = shSel(b, loc, isTakeshotR1, b1_iter, notR1_iter);
  res.oldind   = shSel(b, loc, isTakeshotR1, b1_oldind, notR1_oldind);
  res.ind      = shSel(b, loc, isTakeshotR1, b1_ind, notR1_ind);
  res.old_f    = shSel(b, loc, isTakeshotR1, b1_oldF, old_f);
  res.offset   = shSel(b, loc, isTakeshotR1, b1_offset, notR1_offset);
  res.ch       = shSel(b, loc, isTakeshotR1, b1_ch, notR1_ch);
  res.ord_ch   = shSel(b, loc, isTakeshotR1, b1_ordCh, notR1_ordCh);
  res.num_rep  = shSel(b, loc, isTakeshotR1, b1_numRep, notR1_numRep);
  return res;
}

// ---------------------------------------------------------------------------
// cacheWhileOnlineStablehlo — pure StableHLO version
// ---------------------------------------------------------------------------
// All SM state, SM arrays, and checkpoint buffers are carried as tensor
// iter-args of the scaffold stablehlo.while. No arith/scf/memref ops anywhere.
//
// Scaffold iter-args:
//   [0]      numFwd      : tensor<i64>
//   [1..10]  SM scalars  : tensor<i64> × 10  (check, capo, ..., offset)
//   [11]     ch          : tensor<smSize × i64>
//   [12]     ord_ch      : tensor<smSize × i64>
//   [13]     num_rep     : tensor<smSize × i64>
//   [14..14+N-1]  primalState : original tensor types
//   [14+N..14+N+N]  ckptBuf_i : tensor<budget+1 × elemTy_i>
//   [14+N+N+1]  ckptIdxBuf  : tensor<budget+1 × i64>
//   [14+N+N+2]  done          : tensor<i1>
//   [14+N+N+3]  stepsRemaining : tensor<i64>  (multi-step FORWARD counter)

static SmallVector<Value>
cacheWhileOnlineStablehloV2(stablehlo::WhileOp whileOp, int64_t budget,
                              MGradientUtilsReverse *gutils) {
  Location loc = whileOp.getLoc();
  auto newWhileOp = cast<stablehlo::WhileOp>(gutils->getNewFromOriginal(whileOp));
  OpBuilder builder(newWhileOp);

  auto i64ty0 = RankedTensorType::get({}, builder.getI64Type());
  auto i1ty0  = RankedTensorType::get({}, builder.getI1Type());
  int64_t smSize = std::max(budget, (int64_t)3);
  auto arrTy = RankedTensorType::get({smSize}, builder.getI64Type());

  ValueRange primalInits = newWhileOp.getOperands();
  size_t numCarried = primalInits.size();

  // Stacked checkpoint buffer: tensor<(budget+1) × original_shape...>
  auto ckptTy = [&](Value init) -> RankedTensorType {
    return stackedCkptType(budget + 1, cast<RankedTensorType>(init.getType()));
  };
  auto ckptIdxTy = RankedTensorType::get({budget + 1}, builder.getI64Type());

  // Build initial values ---------------------------------------------------
  SmallVector<Value> initArgs;
  // [0] numFwd
  initArgs.push_back(shC(builder, loc, 0));
  // [1..10] SM scalars: check=-1, rest=0
  initArgs.push_back(shC(builder, loc, -1)); // check
  for (int i = 0; i < 9; ++i)
    initArgs.push_back(shC(builder, loc, 0));
  // [11..13] SM arrays: ch, ord_ch, num_rep (all zeros initially)
  for (int i = 0; i < 3; ++i) {
    initArgs.push_back(stablehlo::ConstantOp::create(builder, loc, arrTy,
        DenseElementsAttr::get(arrTy, builder.getI64IntegerAttr(0))));
  }
  // [14..14+N-1] primal state
  initArgs.append(primalInits.begin(), primalInits.end());
  // [14+N..14+N+N] checkpoint buffers (zeros)
  for (auto init : primalInits) {
    auto ct = ckptTy(init);
    auto elemTy = ct.getElementType();
    initArgs.push_back(stablehlo::ConstantOp::create(builder, loc, ct,
        DenseElementsAttr::get(ct, builder.getZeroAttr(elemTy))));
  }
  // [14+N+N+1] checkpoint index buffer (all -1 = uninitialized)
  initArgs.push_back(stablehlo::ConstantOp::create(builder, loc, ckptIdxTy,
      DenseElementsAttr::get(ckptIdxTy, builder.getI64IntegerAttr(-1))));
  // [14+N+N+2] done = false
  initArgs.push_back(shBool(builder, loc, false));
  // [14+N+N+3] stepsRemaining = 0 (multi-step FORWARD counter)
  initArgs.push_back(shC(builder, loc, 0));

  SmallVector<Type> scaffoldTypes;
  for (auto v : initArgs) scaffoldTypes.push_back(v.getType());

  // Index constants for iter-arg positions.
  const size_t IDX_NUM_FWD   = 0;
  const size_t IDX_SM_START  = 1;   // check at [1]
  const size_t IDX_CH        = 11;
  const size_t IDX_ORD_CH    = 12;
  const size_t IDX_NUM_REP   = 13;
  const size_t IDX_PRIMAL    = 14;
  const size_t IDX_CKPT      = 14 + numCarried;
  const size_t IDX_CKPT_IDX  = 14 + numCarried + numCarried;
  const size_t IDX_DONE      = 14 + numCarried + numCarried + 1;
  const size_t IDX_STEPS_REM = 14 + numCarried + numCarried + 2;

  // Create scaffold stablehlo.while ----------------------------------------
  auto scaffoldWhile = stablehlo::WhileOp::create(
      builder, loc, scaffoldTypes, initArgs);
  // Mark as checkpoint segment so the AD pass does not try to differentiate
  // the scaffold (it is gradient code, not primal code).
  scaffoldWhile->setAttr("enzymexla.checkpoint_segment",
                          mlir::UnitAttr::get(builder.getContext()));

  // Cond region: !done  (pure stablehlo — just XOR the done tensor<i1> with true)
  {
    OpBuilder::InsertionGuard g(builder);
    Block *condBlk = builder.createBlock(&scaffoldWhile.getCond());
    for (auto ty : scaffoldTypes) condBlk->addArgument(ty, loc);
    builder.setInsertionPointToEnd(condBlk);
    Value doneTensor = condBlk->getArgument(IDX_DONE);
    // !done = stablehlo.not done
    Value notDoneT = shNot(builder, loc, doneTensor);
    stablehlo::ReturnOp::create(builder, loc, ValueRange{notDoneT});
  }

  // Body region
  {
    OpBuilder::InsertionGuard g(builder);
    Block *bodyBlk = builder.createBlock(&scaffoldWhile.getBody());
    for (auto ty : scaffoldTypes) bodyBlk->addArgument(ty, loc);
    builder.setInsertionPointToEnd(bodyBlk);

    auto ba = bodyBlk->getArguments();
    Value numFwd  = ba[IDX_NUM_FWD];
    Value smCheck = ba[IDX_SM_START + 0];
    Value smCapo  = ba[IDX_SM_START + 1];
    Value smOldcapoO = ba[IDX_SM_START + 2];
    Value smT     = ba[IDX_SM_START + 3];
    Value smIncr  = ba[IDX_SM_START + 4];
    Value smIter  = ba[IDX_SM_START + 5];
    Value smOldind = ba[IDX_SM_START + 6];
    Value smInd   = ba[IDX_SM_START + 7];
    Value smOldF  = ba[IDX_SM_START + 8];
    Value smOffset = ba[IDX_SM_START + 9];
    Value smCh    = ba[IDX_CH];
    Value smOrdCh = ba[IDX_ORD_CH];
    Value smNumRep = ba[IDX_NUM_REP];
    auto primalCarried = ba.slice(IDX_PRIMAL, numCarried);
    auto ckptBufs      = ba.slice(IDX_CKPT, numCarried);
    Value ckptIdxBuf   = ba[IDX_CKPT_IDX];

    // Multi-step FORWARD support: read remaining step count from prior iteration.
    Value stepsRem = ba[IDX_STEPS_REM];
    Value hasRemaining = shCmpLT(builder, loc, shC(builder, loc, 0), stepsRem);

    Value snaps = shC(builder, loc, budget);

    // Run SM
    SMResult sm = shInlineSM(builder, loc,
        smCheck, smCapo, smOldcapoO, smT, smIncr, smIter,
        smOldind, smInd, smOldF, smOffset, snaps,
        smCh, smOrdCh, smNumRep, smSize);
    // Capacity check: if action==0 and numFwd > beta, we have a problem.
    // We express it as: done=true (terminate loop, reconstruction will fail).
    // (A proper assert is not expressible in stablehlo; we rely on the outer
    // gradient to detect the wrong result. For now just terminate.)
    int64_t beta = (int64_t)(budget + 2) * (budget + 1) / 2 - 1;
    Value betaV = shC(builder, loc, beta);
    Value actionIsZero = shCmpEQ(builder, loc, sm.action, shC(builder, loc, 0));
    Value numFwdGtBeta = shCmpLT(builder, loc, betaV, numFwd); // numFwd > beta
    Value capacityExceeded = shAnd(builder, loc, actionIsZero, numFwdGtBeta);

    // Evaluate primal condition
    IRMapping condMap;
    // evalCondRegion returns tensor<i1> directly — no wrapping needed.
    Value primalCondT = evalCondRegion(builder, loc, newWhileOp, gutils,
                                       primalCarried, condMap);

    // Pre-run the primal body step BEFORE any dispatchIf, in the main body scope.
    // This avoids cloning primal ops inside a stablehlo.if branch lambda (which
    // causes MLIR clone() to crash due to insertion-point inconsistency inside
    // stablehlo.if regions). The stepped state is then selected via shSel.
    IRMapping bodyMap;
    SmallVector<Value> steppedState = runBodyStep(builder, loc, newWhileOp, gutils,
                                                   primalCarried, bodyMap);

    // action == 1 → STORE
    Value isStore = shCmpEQ(builder, loc, sm.action, shC(builder, loc, 1));
    // action == 0 → TERMINATE
    Value isTerminate = shCmpEQ(builder, loc, sm.action, shC(builder, loc, 0));

    // When hasRemaining (continuing a multi-step FORWARD from a prior SM call),
    // force FORWARD behavior: suppress STORE, TERMINATE, and capacityExceeded.
    Value notHasRem = shNot(builder, loc, hasRemaining);
    isStore = shAnd(builder, loc, isStore, notHasRem);
    isTerminate = shAnd(builder, loc, isTerminate, notHasRem);
    capacityExceeded = shAnd(builder, loc, capacityExceeded, notHasRem);

    // doStep = primalCond && !isTerminate
    Value doStep = shAnd(builder, loc, primalCondT, shNot(builder, loc, isTerminate));

    // Select stepped vs original state based on doStep.
    SmallVector<Value> selectedState;
    for (size_t i = 0; i < numCarried; ++i)
      selectedState.push_back(shSel(builder, loc, doStep, steppedState[i], primalCarried[i]));

    Value stepTaken = shSel(builder, loc, doStep, shC(builder, loc, 1), shC(builder, loc, 0));
    Value newNumFwdFwd = shAdd(builder, loc, numFwd, stepTaken);

    // Build result types for the dispatch: (numFwd, primalState..., ckptBufs..., ckptIdxBuf)
    SmallVector<Type> dispTys = {i64ty0}; // numFwd
    for (auto v : primalCarried) dispTys.push_back(v.getType());
    for (auto v : ckptBufs)      dispTys.push_back(v.getType());
    dispTys.push_back(ckptIdxTy); // ckptIdxBuf

    auto dispatchIf = shIf(builder, loc, isStore, dispTys,
      // STORE branch: write checkpoint tensors at slot new_check
      [&](OpBuilder &b2, Location l2) -> SmallVector<Value> {
        Value slot = sm.check; // new_check is tensor<i64>
        SmallVector<Value> res = {numFwd}; // numFwd unchanged
        res.append(primalCarried.begin(), primalCarried.end()); // primal unchanged
        // Update each ckptBuf[slot] = primalCarried[i]
        SmallVector<Value> newCkpt(ckptBufs.begin(), ckptBufs.end());
        for (size_t i = 0; i < numCarried; ++i)
          newCkpt[i] = stackedStore(b2, l2, newCkpt[i], slot, primalCarried[i]);
        for (auto v : newCkpt) res.push_back(v);
        // Update ckptIdxBuf[slot] = numFwd
        Value newIdxBuf = shStore(b2, l2, ckptIdxBuf, slot, numFwd);
        res.push_back(newIdxBuf);
        return res;
      },
      // FORWARD / TERMINATE branch: use pre-computed stepped state.
      [&](OpBuilder &b2, Location l2) -> SmallVector<Value> {
        SmallVector<Value> res = {newNumFwdFwd};
        res.append(selectedState.begin(), selectedState.end());
        res.append(ckptBufs.begin(), ckptBufs.end()); // ckptBufs unchanged
        res.push_back(ckptIdxBuf);
        return res;
      });

    Value newNumFwd = dispatchIf.getResult(0);
    auto newPrimal  = dispatchIf.getResults().slice(1, numCarried);
    auto newCkpts   = dispatchIf.getResults().slice(1 + numCarried, numCarried);
    Value newIdxBuf = dispatchIf.getResult(1 + numCarried + numCarried);

    // Multi-step FORWARD: compute new stepsRemaining.
    // stepsNeeded = sm.capo - numFwd (numFwd is the pre-step primal position;
    // sm.capo is where the SM wants the primal to reach).
    Value isFwd = shCmpEQ(builder, loc, sm.action, shC(builder, loc, 2));
    Value stepsNeeded = shSub(builder, loc, sm.capo, numFwd);
    Value stepsMinusOne = shSub(builder, loc, stepsNeeded, shC(builder, loc, 1));
    Value stepsPositive = shCmpLT(builder, loc, shC(builder, loc, 0), stepsMinusOne);
    Value newStepsFromSM = shSel(builder, loc, stepsPositive, stepsMinusOne,
                                  shC(builder, loc, 0));
    Value stepsFromDecr = shSub(builder, loc, stepsRem, shC(builder, loc, 1));
    Value smBasedSteps = shSel(builder, loc, isFwd, newStepsFromSM,
                                shC(builder, loc, 0));
    Value newStepsRem = shSel(builder, loc, hasRemaining, stepsFromDecr,
                               smBasedSteps);

    // done = !primalCond || isTerminate || capacityExceeded
    Value notPrimalCond = shNot(builder, loc, primalCondT);
    Value newDone = shOr(builder, loc, notPrimalCond,
                     shOr(builder, loc, isTerminate, capacityExceeded));

    // Build yield — when hasRemaining, preserve old SM state (don't advance SM).
    SmallVector<Value> yields;
    yields.push_back(newNumFwd);
    yields.push_back(shSel(builder, loc, hasRemaining, smCheck,    sm.check));
    yields.push_back(shSel(builder, loc, hasRemaining, smCapo,     sm.capo));
    yields.push_back(shSel(builder, loc, hasRemaining, smOldcapoO, sm.oldcapoO));
    yields.push_back(shSel(builder, loc, hasRemaining, smT,        sm.t));
    yields.push_back(shSel(builder, loc, hasRemaining, smIncr,     sm.incr));
    yields.push_back(shSel(builder, loc, hasRemaining, smIter,     sm.iter));
    yields.push_back(shSel(builder, loc, hasRemaining, smOldind,   sm.oldind));
    yields.push_back(shSel(builder, loc, hasRemaining, smInd,      sm.ind));
    yields.push_back(shSel(builder, loc, hasRemaining, smOldF,     sm.old_f));
    yields.push_back(shSel(builder, loc, hasRemaining, smOffset,   sm.offset));
    yields.push_back(shSel(builder, loc, hasRemaining, smCh,       sm.ch));
    yields.push_back(shSel(builder, loc, hasRemaining, smOrdCh,    sm.ord_ch));
    yields.push_back(shSel(builder, loc, hasRemaining, smNumRep,   sm.num_rep));
    yields.append(newPrimal.begin(), newPrimal.end());
    yields.append(newCkpts.begin(), newCkpts.end());
    yields.push_back(newIdxBuf);
    yields.push_back(newDone);
    yields.push_back(newStepsRem);
    stablehlo::ReturnOp::create(builder, loc, yields);
  }

  // After scaffold: extract results ----------------------------------------
  builder.setInsertionPointAfter(scaffoldWhile);
  auto wr = scaffoldWhile.getResults();
  Value finalNumFwd      = wr[IDX_NUM_FWD];
  auto finalPrimalState  = wr.slice(IDX_PRIMAL, numCarried);
  auto finalCkpts        = wr.slice(IDX_CKPT, numCarried);
  Value finalCkptIdxBuf  = wr[IDX_CKPT_IDX];

  // Build caches: checkpoint tensors + idx buffer + numFwd
  // Cache layout mirrors SHWhileOnlineCacheLayout: ckptBuf(i), idxBuf, numFwdBuf
  SmallVector<Value> caches;
  for (auto ckpt : finalCkpts)
    caches.push_back(gutils->initAndPushCache(ckpt, builder));
  caches.push_back(gutils->initAndPushCache(finalCkptIdxBuf, builder));
  caches.push_back(gutils->initAndPushCache(finalNumFwd, builder));

  // Replace original while results
  gutils->replaceOrigOpWith(whileOp,
      SmallVector<Value>(finalPrimalState.begin(), finalPrimalState.end()));
  gutils->erase(newWhileOp);
  gutils->originalToNewFnOps[whileOp] = scaffoldWhile;

  return caches;
}

// ---------------------------------------------------------------------------
// cacheWhileOnlineStablehlo
// ---------------------------------------------------------------------------
// Hybrid architecture: the Online_r2 SM (OnlineProgressOp + memref stores)
// runs in an OUTER scf.while (where arith/scf/memref are valid).  For each
// FORWARD action the outer while enters the primal stablehlo.while for one
// body step, then exits.
//
// Outer scf.while iter-args:
//   (numFwd: index, SM_scalars[10]: index..., primalState..., running: i1)
//
// The SM arrays (ch, ord_ch, num_rep) and checkpoint buffers are memrefs
// allocated outside both loops.
//
// On action==STORE: write checkpoint to memref, don't advance primal.
// On action==FORWARD: if primal cond holds, run one stablehlo.while body step.
// On action==TERMINATE (0): set running=false to exit outer while.

static SmallVector<Value>
cacheWhileOnlineStablehlo(stablehlo::WhileOp whileOp, int64_t budget,
                           MGradientUtilsReverse *gutils) {
  Location loc = whileOp.getLoc();
  auto newWhileOp =
      cast<stablehlo::WhileOp>(gutils->getNewFromOriginal(whileOp));
  OpBuilder builder(newWhileOp);

  Type idxTy = builder.getIndexType();
  Type i1Ty  = builder.getI1Type();

  Value c0    = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1    = arith::ConstantIndexOp::create(builder, loc, 1);
  Value cNeg1 = arith::ConstantIndexOp::create(builder, loc, -1);
  Value budgetV = arith::ConstantIndexOp::create(builder, loc, budget);

  // Primal iter-arg types and init values.
  ValueRange primalInits = newWhileOp.getOperands();
  size_t numCarried = primalInits.size();

  // Checkpoint buffers (memref, OUTSIDE any while loop).
  SmallVector<Value> stores;
  for (auto init : primalInits)
    stores.push_back(memref::AllocOp::create(
        builder, loc, MemRefType::get({budget + 1}, init.getType())));
  stores.push_back(memref::AllocOp::create(
      builder, loc, MemRefType::get({budget + 1}, idxTy)));
  const size_t idxStore = stores.size() - 1;

  // SM arrays (memref, OUTSIDE any while loop).
  int64_t smSize = std::max(budget, (int64_t)3);
  auto smArrTy = MemRefType::get({smSize}, idxTy);
  Value chBuf     = memref::AllocOp::create(builder, loc, smArrTy);
  Value ordChBuf  = memref::AllocOp::create(builder, loc, smArrTy);
  Value numRepBuf = memref::AllocOp::create(builder, loc, smArrTy);

  // Outer scf.while iter-args:
  //   [0]       numFwd       : index
  //   [1]       check        : index
  //   [2]       capo         : index
  //   [3]       oldcapoO     : index
  //   [4]       t            : index
  //   [5]       incr         : index
  //   [6]       iter         : index
  //   [7]       oldind       : index
  //   [8]       ind          : index
  //   [9]       old_f        : index
  //   [10]      offset       : index
  //   [11..11+numCarried-1]  primalState
  //   [11+numCarried]  running : i1

  SmallVector<Value> outerInit;
  outerInit.push_back(c0);      // numFwd
  outerInit.push_back(cNeg1);  // check = -1
  for (int i = 0; i < 9; ++i)
    outerInit.push_back(c0);   // capo..offset
  outerInit.append(primalInits.begin(), primalInits.end());
  outerInit.push_back(arith::ConstantIntOp::create(builder, loc, i1Ty, 1)); // running=true

  SmallVector<Type> outerTypes;
  for (auto v : outerInit)
    outerTypes.push_back(v.getType());

  auto outerWhile = scf::WhileOp::create(builder, loc, outerTypes, outerInit);

  // ── Before block (condition) ─────────────────────────────────────────────
  {
    Block &bb = outerWhile.getBefore().emplaceBlock();
    for (auto ty : outerTypes)
      bb.addArgument(ty, loc);
    OpBuilder bld(&bb, bb.end());

    // running is last arg
    Value running = bb.getArguments().back();
    scf::ConditionOp::create(bld, loc, running, bb.getArguments());
  }

  // ── After block (body) ──────────────────────────────────────────────────
  {
    Block &ab = outerWhile.getAfter().emplaceBlock();
    for (auto ty : outerTypes)
      ab.addArgument(ty, loc);
    OpBuilder bld(&ab, ab.end());

    auto aa      = ab.getArguments();
    Value numFwd = aa[0];
    Value smCheck    = aa[1];
    Value smCapo     = aa[2];
    Value smOldcapoO = aa[3];
    Value smT        = aa[4];
    Value smIncr     = aa[5];
    Value smIter     = aa[6];
    Value smOldind   = aa[7];
    Value smInd      = aa[8];
    Value smOldF     = aa[9];
    Value smOffset   = aa[10];
    auto primalCarried = aa.slice(11, numCarried);
    // aa.back() = running (don't use here, will be set below)

    // Run OnlineProgressOp (valid here — we're in scf.while after-block).
    auto prog = enzyme::OnlineProgressOp::create(
        bld, loc,
        smCheck, smCapo, smOldcapoO, smT, smIncr, smIter,
        smOldind, smInd, smOldF, smOffset, budgetV,
        chBuf, ordChBuf, numRepBuf);

    // Capacity assertion.
    {
      int64_t beta = (int64_t)(budget + 2) * (budget + 1) / 2 - 1;
      Value betaV = arith::ConstantIndexOp::create(bld, loc, beta);
      Value actionIsZero = arith::CmpIOp::create(
          bld, loc, arith::CmpIPredicate::eq, prog.getAction(), c0);
      Value belowBeta = arith::CmpIOp::create(
          bld, loc, arith::CmpIPredicate::sle, numFwd, betaV);
      Value notAZ = arith::XOrIOp::create(bld, loc, actionIsZero,
          arith::ConstantIntOp::create(bld, loc, i1Ty, 1));
      Value ok = arith::OrIOp::create(bld, loc, notAZ, belowBeta);
      cf::AssertOp::create(bld, loc, ok,
          "Online_r2: loop exceeds maximum supported iterations for this "
          "checkpoint budget.  Increase enzyme.checkpoint_period.");
    }

    // action == 1 → STORE;  action == 0 → TERMINATE;  action == 2 → FORWARD
    Value isStore = arith::CmpIOp::create(
        bld, loc, arith::CmpIPredicate::eq, prog.getAction(),
        arith::ConstantIndexOp::create(bld, loc, 1));
    Value isTerminate = arith::CmpIOp::create(
        bld, loc, arith::CmpIPredicate::eq, prog.getAction(), c0);

    // Evaluate primal condition (needed for FORWARD and termination detection).
    IRMapping condMap;
    Value primalCond = evalCondRegion(bld, loc, newWhileOp, gutils,
                                      primalCarried, condMap);

    // New SM scalar values (from OnlineProgressOp results).
    Value newCheck    = prog.getNewCheck();
    Value newCapo     = prog.getNewCapo();
    Value newOldcapoO = prog.getNewOldcapoO();
    Value newT        = prog.getNewT();
    Value newIncr     = prog.getNewIncr();
    Value newIter2    = prog.getNewIter();
    Value newOldind   = prog.getNewOldind();
    Value newInd      = prog.getNewInd();
    Value newOldF     = prog.getNewOldF();
    Value newOffset   = prog.getNewOffset();

    // Result types for the dispatch if: (newNumFwd, newPrimalCarried...)
    SmallVector<Type> dispTys = {idxTy};
    for (auto init : primalInits)
      dispTys.push_back(init.getType());

    auto storeIf = scf::IfOp::create(bld, loc, dispTys, isStore,
                                     /*withElse=*/true);
    // STORE branch
    {
      OpBuilder::InsertionGuard g(bld);
      bld.setInsertionPointToStart(storeIf.thenBlock());
      Value checkSlot = newCheck;
      for (size_t i = 0; i < numCarried; ++i)
        memref::StoreOp::create(bld, loc, primalCarried[i],
                                stores[i], ValueRange{checkSlot});
      memref::StoreOp::create(bld, loc, numFwd, stores[idxStore],
                              ValueRange{checkSlot});
      SmallVector<Value> thenY = {numFwd};
      thenY.append(primalCarried.begin(), primalCarried.end());
      scf::YieldOp::create(bld, loc, thenY);
    }
    // FORWARD / TERMINATE branch
    {
      OpBuilder::InsertionGuard g(bld);
      bld.setInsertionPointToStart(storeIf.elseBlock());

      // If primal cond is true and action==FORWARD, advance primal one step.
      Value doStep = arith::AndIOp::create(bld, loc, primalCond,
          arith::XOrIOp::create(bld, loc, isTerminate,
              arith::ConstantIntOp::create(bld, loc, i1Ty, 1)));

      SmallVector<Type> stepTys;
      for (auto init : primalInits)
        stepTys.push_back(init.getType());
      stepTys.push_back(idxTy); // stepTaken

      auto stepIf = scf::IfOp::create(bld, loc, stepTys, doStep,
                                       /*withElse=*/true);
      {
        OpBuilder::InsertionGuard g2(bld);
        bld.setInsertionPointToStart(stepIf.thenBlock());
        IRMapping bodyMap;
        SmallVector<Value> ns = runBodyStep(bld, loc, newWhileOp, gutils,
                                            primalCarried, bodyMap);
        ns.push_back(c1);
        scf::YieldOp::create(bld, loc, ns);
      }
      {
        OpBuilder::InsertionGuard g2(bld);
        bld.setInsertionPointToStart(stepIf.elseBlock());
        SmallVector<Value> unchanged(primalCarried.begin(), primalCarried.end());
        unchanged.push_back(c0);
        scf::YieldOp::create(bld, loc, unchanged);
      }
      auto stepRes = stepIf.getResults();
      auto newState = stepRes.drop_back(1);
      Value stepTaken = stepRes.back();
      Value newNumFwd = arith::AddIOp::create(bld, loc, numFwd, stepTaken);

      SmallVector<Value> elseY = {newNumFwd};
      elseY.append(newState.begin(), newState.end());
      scf::YieldOp::create(bld, loc, elseY);
    }

    auto dr = storeIf.getResults();
    Value finalNumFwd = dr[0];
    auto finalCarried = dr.drop_front(1);

    // running = !isTerminate && primalCond (keep running unless terminated or primal done)
    Value notTerminate = arith::XOrIOp::create(bld, loc, isTerminate,
        arith::ConstantIntOp::create(bld, loc, i1Ty, 1));
    Value newRunning = arith::AndIOp::create(bld, loc, notTerminate, primalCond);

    // Build yield: (numFwd, SM_scalars[10], primalState, running)
    SmallVector<Value> yields;
    yields.push_back(finalNumFwd);
    yields.push_back(newCheck);
    yields.push_back(newCapo);
    yields.push_back(newOldcapoO);
    yields.push_back(newT);
    yields.push_back(newIncr);
    yields.push_back(newIter2);
    yields.push_back(newOldind);
    yields.push_back(newInd);
    yields.push_back(newOldF);
    yields.push_back(newOffset);
    yields.append(finalCarried.begin(), finalCarried.end());
    yields.push_back(newRunning);
    scf::YieldOp::create(bld, loc, yields);
  }

  // After the outer while: extract numFwd and primal state.
  builder.setInsertionPointAfter(outerWhile);
  auto wr = outerWhile.getResults();
  Value finalNumFwd    = wr[0];
  auto finalPrimalState = wr.slice(11, numCarried);

  // Free SM arrays.
  memref::DeallocOp::create(builder, loc, chBuf);
  memref::DeallocOp::create(builder, loc, ordChBuf);
  memref::DeallocOp::create(builder, loc, numRepBuf);

  // Build caches: checkpoint buffers + numFwd.
  SmallVector<Value> caches;
  for (auto store : stores)
    caches.push_back(gutils->initAndPushCache(store, builder));
  caches.push_back(gutils->initAndPushCache(finalNumFwd, builder));

  // Replace original while results with final primal state.
  gutils->replaceOrigOpWith(whileOp,
      SmallVector<Value>(finalPrimalState.begin(), finalPrimalState.end()));
  gutils->erase(newWhileOp);
  gutils->originalToNewFnOps[whileOp] = outerWhile;

  return caches;
}

// ---------------------------------------------------------------------------
// cacheWhileFullTapeStablehlo
// ---------------------------------------------------------------------------
// Full-tape (no checkpointing) forward for stablehlo.while.
// Uses an scf.while scaffold wrapping the stablehlo.while body.
static SmallVector<Value>
cacheWhileFullTapeStablehlo(stablehlo::WhileOp whileOp,
                             MGradientUtilsReverse *gutils) {
  Location loc = whileOp.getLoc();
  auto newWhileOp =
      cast<stablehlo::WhileOp>(gutils->getNewFromOriginal(whileOp));
  OpBuilder builder(newWhileOp);

  size_t numCarried = whileOp.getNumOperands();

  SmallVector<bool> operandsActive;
  for (size_t i = 0; i < numCarried; ++i) {
    bool active = !gutils->isConstantValue(whileOp->getOperand(i)) ||
                  !gutils->isConstantValue(whileOp.getResult(i));
    operandsActive.push_back(active);
  }

  SmallVector<Value> tapePushCaches, tapePopCaches;
  Block *condBlock = &newWhileOp.getCond().front();
  for (size_t i = 0; i < numCarried; ++i) {
    if (operandsActive[i]) {
      auto ct = gutils->getCacheType(condBlock->getArgument(i).getType());
      auto [push, pop] = gutils->getNewCache(ct);
      tapePushCaches.push_back(push);
      tapePopCaches.push_back(pop);
    }
  }
  auto niCt = gutils->getCacheType(builder.getIndexType());
  auto [niPushCache, niPopCache] = gutils->getNewCache(niCt);

  ValueRange primalInits = newWhileOp.getOperands();
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

  SmallVector<Value> scaffoldInit(primalInits.begin(), primalInits.end());
  scaffoldInit.push_back(c0);

  SmallVector<Type> afterTypes;
  for (auto v : primalInits)
    afterTypes.push_back(v.getType());
  afterTypes.push_back(builder.getIndexType());

  auto scaffoldWhile = scf::WhileOp::create(builder, loc, afterTypes, scaffoldInit);

  {
    Block &bb = scaffoldWhile.getBefore().emplaceBlock();
    for (size_t i = 0; i < numCarried; ++i)
      bb.addArgument(condBlock->getArgument(i).getType(), loc);
    bb.addArgument(builder.getIndexType(), loc);
    OpBuilder bld(&bb, bb.end());

    auto bbArgs       = bb.getArguments();
    Value numItersArg = bbArgs.back();
    auto primalArgs   = bbArgs.drop_back(1);

    size_t tapeIdx = 0;
    for (size_t i = 0; i < numCarried; ++i)
      if (operandsActive[i])
        enzyme::PushOp::create(bld, loc, tapePushCaches[tapeIdx++], primalArgs[i]);

    IRMapping condMap;
    Value condScalar = evalCondRegion(bld, loc, newWhileOp, gutils,
                                      primalArgs, condMap);

    SmallVector<Value> condArgs(primalArgs.begin(), primalArgs.end());
    condArgs.push_back(numItersArg);
    scf::ConditionOp::create(bld, loc, condScalar, condArgs);
  }

  {
    Block &ab = scaffoldWhile.getAfter().emplaceBlock();
    for (size_t i = 0; i < numCarried; ++i)
      ab.addArgument(condBlock->getArgument(i).getType(), loc);
    ab.addArgument(builder.getIndexType(), loc);
    OpBuilder bld(&ab, ab.end());

    auto abArgs    = ab.getArguments();
    Value numIters = abArgs.back();
    auto condPassed = abArgs.drop_back(1);

    IRMapping bodyMap;
    SmallVector<Value> newState = runBodyStep(bld, loc, newWhileOp, gutils,
                                              condPassed, bodyMap);
    newState.push_back(arith::AddIOp::create(bld, loc, numIters, c1));
    scf::YieldOp::create(bld, loc, newState);
  }

  builder.setInsertionPointAfter(scaffoldWhile);
  auto scaffoldResults = scaffoldWhile.getResults();
  Value finalNumIters  = scaffoldResults.back();
  auto primalResults   = scaffoldResults.drop_back(1);
  enzyme::PushOp::create(builder, loc, niPushCache, finalNumIters);

  gutils->replaceOrigOpWith(whileOp, primalResults);
  gutils->erase(newWhileOp);
  gutils->originalToNewFnOps[whileOp] = scaffoldWhile;

  SmallVector<Value> resultCaches;
  resultCaches.push_back(niPopCache);
  resultCaches.append(tapePopCaches.begin(), tapePopCaches.end());
  return resultCaches;
}

// ---------------------------------------------------------------------------
// reverseWhileOnlineStablehlo
// ---------------------------------------------------------------------------
// Reverse uses outer scf.for + scan + inner scf.while Revolve —
// all outside stablehlo.while, so arith/scf/memref are fine.
static LogicalResult
reverseWhileOnlineStablehlo(stablehlo::WhileOp whileOp, int64_t budget,
                              OpBuilder &builder, MGradientUtilsReverse *gutils,
                              SmallVector<Value> caches,
                              ArrayRef<bool> operandsActive,
                              ArrayRef<Value> incomingGradients) {
  Location loc = whileOp.getLoc();
  size_t numCarried = whileOp.getNumOperands();

  SHWhileOnlineCacheLayout layout;
  layout.numCarried = numCarried;

  SmallVector<Value> stores;
  for (size_t j = 0; j < numCarried; ++j)
    stores.push_back(gutils->popCache(caches[layout.ckptBuf(j)], builder));
  stores.push_back(gutils->popCache(caches[layout.idxBuf()], builder));
  const size_t idxStore = stores.size() - 1;

  Value numItersV  = gutils->popCache(caches[layout.numFwdBuf()], builder);
  Value c0         = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1         = arith::ConstantIndexOp::create(builder, loc, 1);
  Value totalBudget = arith::ConstantIndexOp::create(builder, loc, budget);
  Value negOne     = arith::ConstantIndexOp::create(builder, loc, -1);

  SmallVector<Value> outerInit(incomingGradients.begin(), incomingGradients.end());
  auto revOuter = H::createScaffoldForLoop(builder, loc, c0, numItersV, c1,
                                           outerInit);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(revOuter.body);

    Value ivO      = revOuter.getIV();
    auto adjArgs   = revOuter.args();
    SmallVector<Value> liveStores(stores);

    Value currentRevStep = H::emitSub(builder, loc, numItersV, ivO);
    Value targetStep     = H::emitSub(builder, loc, currentRevStep, c1);

    // Step→slot scan (Julia's storemap).
    auto scanLoop = scf::ForOp::create(builder, loc, c0, totalBudget, c1,
                                       SmallVector<Value>{c0, negOne});
    if (!scanLoop.getBody()->empty())
      scanLoop.getBody()->front().erase();
    {
      OpBuilder::InsertionGuard sg(builder);
      builder.setInsertionPointToStart(scanLoop.getBody());
      Value slot     = scanLoop.getInductionVar();
      Value bestCapo = scanLoop.getRegionIterArgs()[0];
      Value bestStep = scanLoop.getRegionIterArgs()[1];
      Value stepAtSlot = H::loadSlot(builder, loc, liveStores[idxStore],
                                     slot, builder.getIndexType());
      Value withinTarget = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sle, stepAtSlot, targetStep);
      Value isBetter = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sgt, stepAtSlot, bestStep);
      Value update = arith::AndIOp::create(builder, loc, withinTarget, isBetter);
      Value newBestCapo = arith::SelectOp::create(builder, loc, update, slot, bestCapo);
      Value newBestStep = arith::SelectOp::create(builder, loc, update, stepAtSlot, bestStep);
      scf::YieldOp::create(builder, loc, ValueRange{newBestCapo, newBestStep});
    }
    Value capo = scanLoop.getResult(0);

    // Load checkpoint state.
    SmallVector<Value> ckptState;
    Block *condBlock = &whileOp.getCond().front();
    for (auto &&[store, arg] : llvm::zip_equal(
             ArrayRef<Value>(liveStores).drop_back(),
             condBlock->getArguments()))
      ckptState.push_back(H::loadSlot(builder, loc, store, capo, arg.getType()));
    Value ckptStep = H::loadSlot(builder, loc, liveStores[idxStore], capo,
                                 builder.getIndexType());

    // Inner Revolve reconstruction while loop.
    SmallVector<Value> whileInitInner;
    whileInitInner.push_back(ckptStep);
    whileInitInner.push_back(capo);
    whileInitInner.append(ckptState.begin(), ckptState.end());

    auto revWhile = H::createScaffoldWhileLoop(
        builder, loc, whileInitInner,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value posPlus1 = H::emitAdd(b, l, args[0], c1);
          return H::emitCmpLT(b, l, posPlus1, currentRevStep);
        });

    {
      H::setInsertionPointToBodyEnd(builder, revWhile.body);
      auto wargs  = revWhile.args();
      Value pos   = wargs[0];
      Value acapo = wargs[1];
      auto astate = wargs.slice(2, numCarried);

      Value remaining  = H::emitSub(builder, loc, currentRevStep, pos);
      Value tBudget    = arith::ConstantIndexOp::create(builder, loc, budget);
      Value budgetRem  = H::emitSub(builder, loc, tBudget, acapo);
      budgetRem = H::emitMin(builder, loc, budgetRem, remaining);
      Value split = enzyme::BinomialProgressOp::create(
          builder, loc, builder.getIndexType(), remaining, budgetRem);

      for (auto &&[i, val] : llvm::enumerate(astate))
        H::storeSlot(builder, loc, liveStores[i], acapo, val);
      H::storeSlot(builder, loc, liveStores[idxStore], acapo, pos);

      Value posPlusSplit = H::emitAdd(builder, loc, pos, split);
      Value isLast = H::emitCmpEQ(builder, loc, posPlusSplit, currentRevStep);
      Value rematUB = H::emitSelect(builder, loc, isLast,
                                    H::emitSub(builder, loc, posPlusSplit, c1),
                                    posPlusSplit);

      auto innerRemat = H::createScaffoldForLoop(
          builder, loc, pos, rematUB, c1,
          SmallVector<Value>(astate.begin(), astate.end()));

      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(innerRemat.body);

        IRMapping condMap, bodyMap;
        evalCondRegion(builder, loc, whileOp, gutils,
                       innerRemat.args(), condMap);
        SmallVector<Value> newState = runBodyStep(
            builder, loc, whileOp, gutils, innerRemat.args(), bodyMap);
        H::finalizeScaffoldLoop(builder, loc, innerRemat, newState);
      }

      Value newCapo = H::emitAdd(builder, loc, acapo, c1);
      SmallVector<Value> afterYields;
      afterYields.push_back(posPlusSplit);
      afterYields.push_back(newCapo);
      auto rematResults = innerRemat.results();
      afterYields.append(rematResults.begin(), rematResults.end());
      H::finalizeScaffoldLoop(builder, loc, revWhile, afterYields);
    }

    H::setInsertionPointToBodyEnd(builder, revOuter.body);
    auto whileResults = revWhile.results();
    auto reconState   = whileResults.slice(2, numCarried);

    IRMapping adjCondMap, adjBodyMap;
    evalCondRegion(builder, loc, whileOp, gutils, reconState, adjCondMap);

    Block *bodyBlock = &whileOp.getBody().front();
    for (auto &&[old, nw] :
         llvm::zip_equal(bodyBlock->getArguments(), reconState))
      adjBodyMap.map(old, nw);
    copyStablehloBlock(builder, bodyBlock, gutils, adjBodyMap);

    auto retOp = cast<stablehlo::ReturnOp>(bodyBlock->getTerminator());
    unsigned revIdx = 0;
    for (auto &&[active, operand] :
         llvm::zip_equal(operandsActive, retOp.getOperands())) {
      if (active) {
        Value mappedOperand = adjBodyMap.lookupOrDefault(operand);
        gutils->addToDiffe(mappedOperand, adjArgs[revIdx], builder);
        revIdx++;
      }
    }

    bool valid = true;
    {
      H::SegmentCacheCreatorGuard cacheGuard(gutils, revWhile.op);
      auto visitBlock = [&](Block *b) {
        for (auto it = b->rbegin(); it != b->rend(); ++it) {
          if (&*it == b->getTerminator())
            continue;
          valid &= gutils->Logic.visitChild(&*it, builder, gutils).succeeded();
        }
      };
      visitBlock(bodyBlock);
      visitBlock(&whileOp.getCond().front());
    }

    SmallVector<Value> newAdjoints;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, bodyBlock->getArguments())) {
      if (active) {
        newAdjoints.push_back(gutils->diffe(arg, builder));
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, builder);
      }
    }

    SmallVector<Value> outerYields(newAdjoints.begin(), newAdjoints.end());
    H::finalizeScaffoldLoop(builder, loc, revOuter, outerYields);

    builder.setInsertionPointAfter(revOuter.op);

    revIdx = 0;
    auto revOuterResults = revOuter.results();
    for (auto &&[active, initArg] :
         llvm::zip_equal(operandsActive, whileOp.getOperands())) {
      if (active) {
        if (!gutils->isConstantValue(initArg))
          gutils->addToDiffe(initArg, revOuterResults[revIdx], builder);
        revIdx++;
      }
    }

    for (auto store : stores)
      H::destroyStore(builder, loc, store);
  }

  return success(true);
}

// ---------------------------------------------------------------------------
// reverseWhileOnlineStablehloV2 — works with tensor-based caches from V2
// ---------------------------------------------------------------------------
static LogicalResult
reverseWhileOnlineStablehloV2(stablehlo::WhileOp whileOp, int64_t budget,
                               OpBuilder &builder, MGradientUtilsReverse *gutils,
                               SmallVector<Value> caches,
                               ArrayRef<bool> operandsActive,
                               ArrayRef<Value> incomingGradients) {
  Location loc = whileOp.getLoc();
  size_t numCarried = whileOp.getNumOperands();

  // Cache layout (from cacheWhileOnlineStablehloV2):
  //   caches[0..N-1]  = checkpoint tensors  tensor<budget+1 × elemTy_i>
  //   caches[N]       = checkpoint idx buf  tensor<budget+1 × i64>
  //   caches[N+1]     = numFwd              tensor<i64> (0-d, cached as index)
  SmallVector<Value> ckptTensors;
  for (size_t i = 0; i < numCarried; ++i)
    ckptTensors.push_back(gutils->popCache(caches[i], builder));
  Value ckptIdxTensor = gutils->popCache(caches[numCarried], builder);
  Value numItersV = gutils->popCache(caches[numCarried + 1], builder);

  // numItersV may be tensor<i64> (if cacheWhileOnlineStablehloV2 pushed it as such)
  // or index (if enzyme's cache wrapping converts it). Convert to index.
  Value numItersIdx;
  if (numItersV.getType().isIndex()) {
    numItersIdx = numItersV;
  } else {
    // tensor<i64> → index
    Value asI64 = tensor::ExtractOp::create(builder, loc, numItersV, ValueRange{});
    numItersIdx = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), asI64);
  }

  Value c0  = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1  = arith::ConstantIndexOp::create(builder, loc, 1);
  Value totalBudgetIdx = arith::ConstantIndexOp::create(builder, loc, budget);
  Value negOneIdx = arith::ConstantIndexOp::create(builder, loc, -1);

  // Helper: load tensor<i64> at index slot from tensor<n×i64>.
  // Returns tensor<i64> (0-d).
  auto loadIdxTensor = [&](Value arr, Value slotIdx) -> Value {
    // slotIdx is index; convert to tensor<i64> for shLoad
    Value asI64 = arith::IndexCastOp::create(builder, loc, builder.getI64Type(), slotIdx);
    auto slotT = RankedTensorType::get({}, builder.getI64Type());
    Value slot64 = tensor::FromElementsOp::create(builder, loc, slotT, ValueRange{asI64});
    Value val64 = shLoad(builder, loc, arr, slot64);
    // Convert back to index
    Value valI64 = tensor::ExtractOp::create(builder, loc, val64, ValueRange{});
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(), valI64);
  };

  // Helper: load a full tensor from a stacked checkpoint buffer at index slot.
  auto loadTypedAt = [&](Value arr, Value slotIdx, RankedTensorType origTy) -> Value {
    Value asI64 = arith::IndexCastOp::create(builder, loc, builder.getI64Type(), slotIdx);
    auto slotT = RankedTensorType::get({}, builder.getI64Type());
    Value slot64 = tensor::FromElementsOp::create(builder, loc, slotT, ValueRange{asI64});
    return stackedLoad(builder, loc, arr, slot64, origTy);
  };

  // Collect original primal carried types for checkpoint reshape.
  SmallVector<RankedTensorType> primalCarriedTys;
  for (auto ty : whileOp.getOperandTypes())
    primalCarriedTys.push_back(cast<RankedTensorType>(ty));

  // Outer reverse for loop over all numIters steps.
  SmallVector<Value> outerInit(incomingGradients.begin(), incomingGradients.end());
  // Carry checkpoint tensors through outer loop so inner Revolve can update them.
  outerInit.append(ckptTensors.begin(), ckptTensors.end());
  outerInit.push_back(ckptIdxTensor);

  auto revOuter = H::createScaffoldForLoop(builder, loc, c0, numItersIdx, c1, outerInit);

  bool valid = true;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(revOuter.body);

    Value ivO     = revOuter.getIV();
    auto allArgs  = revOuter.args();
    auto adjArgs  = allArgs.slice(0, incomingGradients.size());
    auto liveCkpt = allArgs.slice(incomingGradients.size(), numCarried);
    Value liveIdxBuf = allArgs[incomingGradients.size() + numCarried];

    Value currentRevStep = H::emitSub(builder, loc, numItersIdx, ivO);
    Value targetStep     = H::emitSub(builder, loc, currentRevStep, c1);

    // Step→slot scan: find slot with highest step ≤ targetStep.
    auto scanLoop = scf::ForOp::create(builder, loc, c0, totalBudgetIdx, c1,
                                       SmallVector<Value>{c0, negOneIdx});
    if (!scanLoop.getBody()->empty())
      scanLoop.getBody()->front().erase();
    {
      OpBuilder::InsertionGuard sg(builder);
      builder.setInsertionPointToStart(scanLoop.getBody());
      Value slot     = scanLoop.getInductionVar();
      Value bestCapo = scanLoop.getRegionIterArgs()[0];
      Value bestStep = scanLoop.getRegionIterArgs()[1];
      // Load step stored at this slot from liveIdxBuf.
      Value stepAtSlot = loadIdxTensor(liveIdxBuf, slot);
      Value withinTarget = arith::CmpIOp::create(builder, loc,
          arith::CmpIPredicate::sle, stepAtSlot, targetStep);
      Value isBetter = arith::CmpIOp::create(builder, loc,
          arith::CmpIPredicate::sgt, stepAtSlot, bestStep);
      Value update = arith::AndIOp::create(builder, loc, withinTarget, isBetter);
      Value newBestCapo = arith::SelectOp::create(builder, loc, update, slot, bestCapo);
      Value newBestStep = arith::SelectOp::create(builder, loc, update, stepAtSlot, bestStep);
      scf::YieldOp::create(builder, loc, ValueRange{newBestCapo, newBestStep});
    }
    Value capo     = scanLoop.getResult(0);
    Value ckptStep = loadIdxTensor(liveIdxBuf, capo);

    // Load checkpoint state at capo.
    SmallVector<Value> ckptState;
    for (size_t i = 0; i < numCarried; ++i)
      ckptState.push_back(loadTypedAt(liveCkpt[i], capo, primalCarriedTys[i]));

    // Inner Revolve reconstruction while loop.
    // Carry: (pos: index, acapo: index, primalState..., ckptTensors..., ckptIdxBuf)
    SmallVector<Value> whileInitInner;
    whileInitInner.push_back(ckptStep);
    whileInitInner.push_back(capo);
    whileInitInner.append(ckptState.begin(), ckptState.end());
    SmallVector<Value> innerLiveCkpt(liveCkpt.begin(), liveCkpt.end());
    whileInitInner.append(innerLiveCkpt.begin(), innerLiveCkpt.end());
    whileInitInner.push_back(liveIdxBuf);

    auto revWhile = H::createScaffoldWhileLoop(
        builder, loc, whileInitInner,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value posPlus1 = H::emitAdd(b, l, args[0], c1);
          return H::emitCmpLT(b, l, posPlus1, currentRevStep);
        });

    {
      H::setInsertionPointToBodyEnd(builder, revWhile.body);
      auto wargs   = revWhile.args();
      Value pos    = wargs[0];
      Value acapo  = wargs[1];
      auto astate  = wargs.slice(2, numCarried);
      auto wCkpt   = wargs.slice(2 + numCarried, numCarried);
      Value wIdxBuf = wargs[2 + numCarried + numCarried];

      Value remaining = H::emitSub(builder, loc, currentRevStep, pos);
      Value tBudget   = arith::ConstantIndexOp::create(builder, loc, budget);
      Value budgetRem = H::emitSub(builder, loc, tBudget, acapo);
      budgetRem = H::emitMin(builder, loc, budgetRem, remaining);
      Value split = enzyme::BinomialProgressOp::create(
          builder, loc, builder.getIndexType(), remaining, budgetRem);

      // Store current state at acapo in checkpoint tensors.
      SmallVector<Value> updCkpt(wCkpt.begin(), wCkpt.end());
      for (size_t i = 0; i < numCarried; ++i) {
        Value asI64 = arith::IndexCastOp::create(builder, loc,
            builder.getI64Type(), acapo);
        auto slotT = RankedTensorType::get({}, builder.getI64Type());
        Value slot64 = tensor::FromElementsOp::create(builder, loc, slotT,
            ValueRange{asI64});
        updCkpt[i] = stackedStore(builder, loc, updCkpt[i], slot64, astate[i]);
      }
      // Store pos as step index.
      {
        Value asI64 = arith::IndexCastOp::create(builder, loc,
            builder.getI64Type(), acapo);
        auto slotT = RankedTensorType::get({}, builder.getI64Type());
        Value slot64 = tensor::FromElementsOp::create(builder, loc, slotT,
            ValueRange{asI64});
        Value posI64 = arith::IndexCastOp::create(builder, loc,
            builder.getI64Type(), pos);
        auto posT = RankedTensorType::get({}, builder.getI64Type());
        Value pos64 = tensor::FromElementsOp::create(builder, loc, posT,
            ValueRange{posI64});
        wIdxBuf = shStore(builder, loc, wIdxBuf, slot64, pos64);
      }

      Value posPlusSplit = H::emitAdd(builder, loc, pos, split);
      Value isLast = H::emitCmpEQ(builder, loc, posPlusSplit, currentRevStep);
      Value rematUB = H::emitSelect(builder, loc, isLast,
          H::emitSub(builder, loc, posPlusSplit, c1), posPlusSplit);

      auto innerRemat = H::createScaffoldForLoop(
          builder, loc, pos, rematUB, c1,
          SmallVector<Value>(astate.begin(), astate.end()));

      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(innerRemat.body);
        IRMapping condMap, bodyMap;
        evalCondRegion(builder, loc, whileOp, gutils, innerRemat.args(), condMap);
        SmallVector<Value> newState = runBodyStep(builder, loc, whileOp, gutils,
                                                   innerRemat.args(), bodyMap);
        H::finalizeScaffoldLoop(builder, loc, innerRemat, newState);
      }

      Value newCapo = H::emitAdd(builder, loc, acapo, c1);
      SmallVector<Value> afterYields;
      afterYields.push_back(posPlusSplit);
      afterYields.push_back(newCapo);
      auto rematResults = innerRemat.results();
      afterYields.append(rematResults.begin(), rematResults.end());
      afterYields.append(updCkpt.begin(), updCkpt.end());
      afterYields.push_back(wIdxBuf);
      H::finalizeScaffoldLoop(builder, loc, revWhile, afterYields);
    }

    H::setInsertionPointToBodyEnd(builder, revOuter.body);
    auto whileResults  = revWhile.results();
    auto reconState    = whileResults.slice(2, numCarried);
    auto newLiveCkpt   = whileResults.slice(2 + numCarried, numCarried);
    Value newLiveIdxBuf = whileResults[2 + numCarried + numCarried];

    // Adjoint of one primal step using reconstructed state.
    IRMapping adjCondMap, adjBodyMap;
    evalCondRegion(builder, loc, whileOp, gutils, reconState, adjCondMap);

    Block *bodyBlock = &whileOp.getBody().front();
    for (auto &&[old, nw] : llvm::zip_equal(bodyBlock->getArguments(), reconState))
      adjBodyMap.map(old, nw);
    copyStablehloBlock(builder, bodyBlock, gutils, adjBodyMap);

    auto retOp = cast<stablehlo::ReturnOp>(bodyBlock->getTerminator());
    unsigned revIdx = 0;
    for (auto &&[active, operand] :
         llvm::zip_equal(operandsActive, retOp.getOperands())) {
      if (active) {
        Value mappedOperand = adjBodyMap.lookupOrDefault(operand);
        gutils->addToDiffe(mappedOperand, adjArgs[revIdx], builder);
        revIdx++;
      }
    }

    {
      H::SegmentCacheCreatorGuard cacheGuard(gutils, revWhile.op);
      auto visitBlock = [&](Block *b) {
        for (auto it = b->rbegin(); it != b->rend(); ++it) {
          if (&*it == b->getTerminator())
            continue;
          valid &= gutils->Logic.visitChild(&*it, builder, gutils).succeeded();
        }
      };
      visitBlock(bodyBlock);
      visitBlock(&whileOp.getCond().front());
    }

    SmallVector<Value> newAdjoints;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, bodyBlock->getArguments())) {
      if (active) {
        newAdjoints.push_back(gutils->diffe(arg, builder));
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, builder);
      }
    }

    // Outer yields: (gradients..., newLiveCkpt..., newLiveIdxBuf)
    SmallVector<Value> outerYields(newAdjoints.begin(), newAdjoints.end());
    outerYields.append(newLiveCkpt.begin(), newLiveCkpt.end());
    outerYields.push_back(newLiveIdxBuf);
    H::finalizeScaffoldLoop(builder, loc, revOuter, outerYields);

    builder.setInsertionPointAfter(revOuter.op);

    revIdx = 0;
    auto revOuterResults = revOuter.results();
    for (auto &&[active, initArg] :
         llvm::zip_equal(operandsActive, whileOp.getOperands())) {
      if (active) {
        if (!gutils->isConstantValue(initArg))
          gutils->addToDiffe(initArg, revOuterResults[revIdx], builder);
        revIdx++;
      }
    }
  }

  return success(valid);
}

// ---------------------------------------------------------------------------
// reverseWhileFullTapeStablehlo
// ---------------------------------------------------------------------------
static LogicalResult
reverseWhileFullTapeStablehlo(stablehlo::WhileOp whileOp, OpBuilder &builder,
                               MGradientUtilsReverse *gutils,
                               SmallVector<Value> caches,
                               ArrayRef<bool> operandsActive,
                               ArrayRef<Value> incomingGradients) {
  Location loc = whileOp.getLoc();
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);

  Value numIters = gutils->popCache(caches[0], builder);

  Block *condBlock = &whileOp.getCond().front();
  Block *bodyBlock = &whileOp.getBody().front();

  // Discard extra top push (fires on failing condition check).
  size_t tapeIdx1 = 1;
  for (auto &&[active, arg] :
       llvm::zip_equal(operandsActive, condBlock->getArguments())) {
    if (active)
      gutils->popCache(caches[tapeIdx1++], builder);
  }

  auto revFor = H::createScaffoldForLoop(
      builder, loc, c0, numIters, c1,
      SmallVector<Value>(incomingGradients.begin(), incomingGradients.end()));

  bool valid = true;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(revFor.body);
    auto adjArgs = revFor.args();

    IRMapping mapping;
    size_t tapeIdx = 1;
    for (auto &&[active, arg, initOp] :
         llvm::zip_equal(operandsActive, condBlock->getArguments(),
                         whileOp.getOperands())) {
      if (active) {
        Value popped = gutils->popCache(caches[tapeIdx++], builder);
        mapping.map(arg, popped);
        gutils->originalToNewFn.map(arg, popped);
      } else {
        if (Value nv = gutils->originalToNewFn.lookupOrNull(initOp))
          mapping.map(arg, nv);
      }
    }

    copyStablehloBlock(builder, condBlock, gutils, mapping);

    IRMapping bodyMap;
    for (auto &&[condArg, bodyArg] :
         llvm::zip_equal(condBlock->getArguments(), bodyBlock->getArguments()))
      bodyMap.map(bodyArg, mapping.lookupOrDefault(condArg));

    copyStablehloBlock(builder, bodyBlock, gutils, bodyMap);

    auto retOp = cast<stablehlo::ReturnOp>(bodyBlock->getTerminator());
    unsigned revIdx = 0;
    for (auto &&[active, operand] :
         llvm::zip_equal(operandsActive, retOp.getOperands())) {
      if (active) {
        Value mappedOperand = bodyMap.lookupOrDefault(operand);
        gutils->addToDiffe(mappedOperand, adjArgs[revIdx], builder);
        revIdx++;
      }
    }

    {
      H::SegmentCacheCreatorGuard cacheGuard(gutils, revFor.op);
      auto visitBlock = [&](Block *b) {
        for (auto it = b->rbegin(); it != b->rend(); ++it) {
          if (&*it == b->getTerminator())
            continue;
          valid &= gutils->Logic.visitChild(&*it, builder, gutils).succeeded();
        }
      };
      visitBlock(bodyBlock);
      visitBlock(condBlock);
    }

    SmallVector<Value> newAdjoints;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, bodyBlock->getArguments())) {
      if (active) {
        newAdjoints.push_back(gutils->diffe(arg, builder));
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, builder);
      }
    }
    H::finalizeScaffoldLoop(builder, loc, revFor, newAdjoints);
  }

  builder.setInsertionPointAfter(revFor.op);
  unsigned revIdx = 0;
  auto revForResults = revFor.results();
  for (auto &&[active, initArg] :
       llvm::zip_equal(operandsActive, whileOp.getOperands())) {
    if (active) {
      if (!gutils->isConstantValue(initArg))
        gutils->addToDiffe(initArg, revForResults[revIdx], builder);
      revIdx++;
    }
  }
  return success(valid);
}

// ---------------------------------------------------------------------------
// StablehloWhileOpInterfaceReverse
// ---------------------------------------------------------------------------
struct StablehloWhileOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          StablehloWhileOpInterfaceReverse, stablehlo::WhileOp> {

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto whileOp = cast<stablehlo::WhileOp>(op);
    if (mlir::enzyme::whileNeedsOnlineCheckpointing(op)) {
      auto budget = mlir::enzyme::whileGetCheckpointBudget(op);
      if (budget && *budget >= 1)
        return cacheWhileOnlineStablehloV2(whileOp, *budget, gutils);
      op->emitError() << "online checkpointing requires enzyme.checkpoint_period >= 1";
      return {};
    }
    return cacheWhileFullTapeStablehlo(whileOp, gutils);
  }

  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto whileOp = cast<stablehlo::WhileOp>(op);

    SmallVector<bool> operandsActive;
    SmallVector<Value> incomingGradients;
    for (auto &&[init, res] :
         llvm::zip_equal(whileOp.getOperands(), op->getResults())) {
      bool active = !gutils->isConstantValue(init) ||
                    !gutils->isConstantValue(res);
      operandsActive.push_back(active);
      if (active) {
        incomingGradients.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    if (mlir::enzyme::whileNeedsOnlineCheckpointing(op)) {
      auto budget = mlir::enzyme::whileGetCheckpointBudget(op);
      if (!budget || *budget < 1)
        return op->emitError()
               << "online checkpointing requires enzyme.checkpoint_period >= 1";
      return reverseWhileOnlineStablehloV2(whileOp, *budget, builder, gutils,
                                          caches, operandsActive,
                                          incomingGradients);
    }
    return reverseWhileFullTapeStablehlo(whileOp, builder, gutils, caches,
                                          operandsActive, incomingGradients);
  }

  LogicalResult createShadowValues(Operation *, OpBuilder &,
                                   MGradientUtilsReverse *) const {
    return success();
  }
};

} // namespace

void mlir::enzyme::registerStablehloDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, stablehlo::StablehloDialect *) {
        stablehlo::WhileOp::attachInterface<StablehloWhileOpInterfaceReverse>(
            *ctx);
      });
}

#endif // ENZYME_ENABLE_STABLEHLO
// DIAGNOSTIC: check mapping before clone
