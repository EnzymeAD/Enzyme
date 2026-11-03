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
static void copyStablehloBlock(OpBuilder &builder, Block *b,
                                MGradientUtilsReverse *gutils,
                                IRMapping &mapping) {
  for (auto &op : b->without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (!mapping.contains(operand)) {
        if (Value nv = gutils->originalToNewFn.lookupOrNull(operand))
          mapping.map(operand, nv);
      }
    }
    builder.clone(op, mapping);
  }
  for (auto [oldVal, newVal] : mapping.getValueMap())
    gutils->originalToNewFn.map(oldVal, newVal);
  for (auto [oldOp, newOp] : mapping.getOperationMap())
    gutils->originalToNewFnOps[oldOp] = newOp;
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
  copyStablehloBlock(builder, condBlock, gutils, outMapping);

  auto retOp = cast<stablehlo::ReturnOp>(condBlock->getTerminator());
  Value condTensor = outMapping.lookupOrDefault(retOp.getOperand(0));
  return tensor::ExtractOp::create(builder, loc, condTensor, ValueRange{});
}

/// Run one body step, return new iter-args.
static SmallVector<Value> runBodyStep(OpBuilder &builder, Location loc,
                                       stablehlo::WhileOp whileOp,
                                       MGradientUtilsReverse *gutils,
                                       ValueRange iterArgs,
                                       IRMapping &outMapping) {
  Block *bodyBlock = &whileOp.getBody().front();
  for (auto &&[old, nw] :
       llvm::zip_equal(bodyBlock->getArguments(), iterArgs))
    outMapping.map(old, nw);
  copyStablehloBlock(builder, bodyBlock, gutils, outMapping);

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
  auto ty = RankedTensorType::get({}, b.getI1Type());
  return stablehlo::CompareOp::create(b, loc, ty, l, r,
      stablehlo::ComparisonDirection::EQ, stablehlo::ComparisonType::SIGNED);
}
static Value shCmpNE(OpBuilder &b, Location loc, Value l, Value r) {
  auto ty = RankedTensorType::get({}, b.getI1Type());
  return stablehlo::CompareOp::create(b, loc, ty, l, r,
      stablehlo::ComparisonDirection::NE, stablehlo::ComparisonType::SIGNED);
}
static Value shCmpLT(OpBuilder &b, Location loc, Value l, Value r) {
  auto ty = RankedTensorType::get({}, b.getI1Type());
  return stablehlo::CompareOp::create(b, loc, ty, l, r,
      stablehlo::ComparisonDirection::LT, stablehlo::ComparisonType::SIGNED);
}
static Value shCmpLE(OpBuilder &b, Location loc, Value l, Value r) {
  auto ty = RankedTensorType::get({}, b.getI1Type());
  return stablehlo::CompareOp::create(b, loc, ty, l, r,
      stablehlo::ComparisonDirection::LE, stablehlo::ComparisonType::SIGNED);
}
static Value shCmpGE(OpBuilder &b, Location loc, Value l, Value r) {
  auto ty = RankedTensorType::get({}, b.getI1Type());
  return stablehlo::CompareOp::create(b, loc, ty, l, r,
      stablehlo::ComparisonDirection::GE, stablehlo::ComparisonType::SIGNED);
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
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&ifOp.getTrueBranch().front());
    SmallVector<Value> thenVals = thenFn(b, loc);
    stablehlo::ReturnOp::create(b, loc, thenVals);
  }
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&ifOp.getFalseBranch().front());
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
// All inputs/outputs are tensor<i64> (scalars) or tensor<smSize×i64> (arrays).
// snaps = shC(budget), budget_v = shC(budget).
static SMResult shInlineSM(OpBuilder &b, Location loc,
    Value check, Value capo, Value oldcapoO, Value t, Value incr,
    Value iter, Value oldind, Value ind, Value old_f, Value offset,
    Value snaps, Value ch, Value ord_ch, Value num_rep, int64_t smSize) {

  auto i64ty0 = RankedTensorType::get({}, b.getI64Type());
  auto i1ty0  = RankedTensorType::get({}, b.getI1Type());
  auto arrTy  = RankedTensorType::get({smSize}, b.getI64Type());
  SmallVector<Type> smTys(11, i64ty0);
  smTys.push_back(arrTy); smTys.push_back(arrTy); smTys.push_back(arrTy);
  // 14 total: 11 scalars + 3 arrays

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

  // ch[check] with safe (non-negative) index.
  Value checkLtZero = shCmpLT(b, loc, check, zero);
  Value safeCheck   = shSel(b, loc, checkLtZero, zero, check);
  Value chAtCheck   = shLoad(b, loc, ch, safeCheck);

  // Top-level: takeshot r=1 when (check<0) OR (ch[check]!=capo AND capo<=snaps-1)
  Value chNeqCapo    = shCmpNE(b, loc, chAtCheck, capo);
  Value capoLeSnapsM1 = shCmpLE(b, loc, capo, snapsM1);
  Value isTakeshotR1 = shOr(b, loc, checkLtZero,
                            shAnd(b, loc, chNeqCapo, capoLeSnapsM1));

  // ── Outer if: takeshot r=1 vs else ─────────────────────────────────────────
  auto outerIf = shIf(b, loc, isTakeshotR1, smTys,

    // ── Branch 1: TAKESHOT r=1 ─────────────────────────────────────────────
    [&](OpBuilder &b2, Location l2) -> SmallVector<Value> {
      Value newCheck = shAdd(b2, l2, check, shC(b2, l2, 1));
      Value newCh    = shStore(b2, l2, ch, newCheck, capo);

      // Init arrays: depends on snaps < 4.
      Value snapsLt4 = shCmpLT(b2, l2, snaps, shC(b2, l2, 4));

      // Both branches fill num_rep and ord_ch with loops.
      // We build them with stablehlo.while.
      // Carry: (i, num_rep, ord_ch)
      Value snapsI = snaps; // tensor<i64>
      auto arrTy2 = arrTy;
      auto fillLoop = [&](Value fillVal) -> std::pair<Value,Value> {
        // Fill num_rep[i]=fillVal, ord_ch[i]=i for i in [0, snaps)
        SmallVector<Value> initA = {shC(b2, l2, 0), num_rep, newCh/*ord_ch placeholder*/};
        // Actually we need separate loops for num_rep and ord_ch since they're
        // separate tensors. Use a single loop carrying (i, num_rep, ord_ch).
        SmallVector<Value> initB = {shC(b2, l2, 0), num_rep, ord_ch};
        auto w2 = shWhile(b2, l2, initB,
          [&](OpBuilder &b3, Location l3, ValueRange a) -> Value {
            return shCmpLT(b3, l3, a[0], snapsI);
          },
          [&](OpBuilder &b3, Location l3, ValueRange a) -> SmallVector<Value> {
            Value i = a[0], nr = a[1], oc = a[2];
            Value newNr = shStore(b3, l3, nr, i, fillVal);
            Value newOc = shStore(b3, l3, oc, i, i); // ord_ch[i]=i
            return {shAdd(b3, l3, i, shC(b3, l3, 1)), newNr, newOc};
          });
        return {w2->getResult(1), w2->getResult(2)};
      };

      // For snaps<4: fillVal=2, incr=2, iter=1, oldind=snaps-1
      // For snaps>=4: fillVal=1, incr=1, iter=1, oldind=1, offset=snaps-1
      auto initIfOp = shIf(b2, l2, snapsLt4,
          {i64ty0, i64ty0, i64ty0, i64ty0, arrTy, arrTy},
          [&](OpBuilder &b3, Location l3) -> SmallVector<Value> {
            auto [newNr, newOc] = fillLoop(shC(b3, l3, 2));
            return {shC(b3, l3, 2)/*incr*/, shC(b3, l3, 1)/*iter*/,
                    shSub(b3, l3, snaps, shC(b3, l3, 1))/*oldind*/,
                    offset/*unchanged*/, newNr, newOc};
          },
          [&](OpBuilder &b3, Location l3) -> SmallVector<Value> {
            auto [newNr, newOc] = fillLoop(shC(b3, l3, 1));
            return {shC(b3, l3, 1)/*incr*/, shC(b3, l3, 1)/*iter*/,
                    shC(b3, l3, 1)/*oldind*/,
                    shSub(b3, l3, snaps, shC(b3, l3, 1))/*offset=snaps-1*/,
                    newNr, newOc};
          });

      Value newIncr   = initIfOp.getResult(0);
      Value newIter   = initIfOp.getResult(1);
      Value newOldind = initIfOp.getResult(2);
      Value newOffset = initIfOp.getResult(3);
      Value newNumRep = initIfOp.getResult(4);
      Value newOrdCh  = initIfOp.getResult(5);

      // if capo == snaps-1: ind=2, old_f=1
      Value capoEqSnapsM1 = shCmpEQ(b2, l2, capo, shSub(b2, l2, snaps, shC(b2, l2, 1)));
      Value newInd  = shSel(b2, l2, capoEqSnapsM1, shC(b2, l2, 2), ind);
      Value newOldF = shSel(b2, l2, capoEqSnapsM1, shC(b2, l2, 1), old_f);

      return {actTakeshot,
              newCheck, capo, capo/*oldcapo_o=capo*/, zero/*t=0*/,
              newIncr, newIter, newOldind, newInd, newOldF, newOffset,
              newCh, newOrdCh, newNumRep};
    },

    // ── Else: advance r=1 or r=2 ──────────────────────────────────────────
    [&](OpBuilder &b2, Location l2) -> SmallVector<Value> {
      // Branch 2: capo < snaps-1 → advance r=1
      Value capoLtSnapsM1 = shCmpLT(b2, l2, capo, shSub(b2, l2, snaps, shC(b2, l2, 1)));

      auto branch23 = shIf(b2, l2, capoLtSnapsM1, smTys,
        // Branch 2: advance r=1
        [&](OpBuilder &b3, Location l3) -> SmallVector<Value> {
          return {actAdvance,
                  check, shAdd(b3, l3, oldcapoO, shC(b3, l3, 1)), oldcapoO,
                  t, incr, iter, oldind, ind, old_f, offset,
                  ch, ord_ch, num_rep};
        },
        // Branch 3: r=2 — dispatch on ch[check] == capo
        [&](OpBuilder &b3, Location l3) -> SmallVector<Value> {
          Value chCheck   = shLoad(b3, l3, ch, check);
          Value r2AdvCond = shCmpEQ(b3, l3, chCheck, capo);

          auto r2If = shIf(b3, l3, r2AdvCond, smTys,

            // Branch 3a: advance r=2
            [&](OpBuilder &b4, Location l4) -> SmallVector<Value> {
              Value is1 = shCmpEQ(b4, l4, snaps, shC(b4, l4, 1));
              Value is2 = shCmpEQ(b4, l4, snaps, shC(b4, l4, 2));
              Value is3 = shCmpEQ(b4, l4, snaps, shC(b4, l4, 3));

              // case 1: capo = very large sentinel
              Value infCapo = shC(b4, l4, std::numeric_limits<int64_t>::max() / 4);

              // case 2: capo = ch[1] + incr
              Value ch1      = shLoad(b4, l4, ch, shC(b4, l4, 1));
              Value case2Capo = shAdd(b4, l4, ch1, incr);

              // case 3: iter==0 → loop; iter!=0 → ch[ind]+incr
              Value iterIsZero = shCmpEQ(b4, l4, iter, shC(b4, l4, 0));
              auto case3If = shIf(b4, l4, iterIsZero,
                  {i64ty0, i64ty0, i64ty0},
                  [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                    // iter==0: capo=ch[oldind]; loop (t+1)/2+1 times
                    Value chOldind = shLoad(b5, l5, ch, oldind);
                    Value tP1Div2  = shDiv(b5, l5, shAdd(b5, l5, t, shC(b5, l5, 1)),
                                                   shC(b5, l5, 2));
                    Value loopUBc3 = shAdd(b5, l5, tP1Div2, shC(b5, l5, 1));
                    // Loop carry: (i, capo, incr, iter)
                    SmallVector<Value> initC3 = {shC(b5, l5, 0), chOldind, incr, iter};
                    auto wC3 = shWhile(b5, l5, initC3,
                      [&](OpBuilder &b6, Location l6, ValueRange a) -> Value {
                        return shCmpLT(b6, l6, a[0], loopUBc3);
                      },
                      [&](OpBuilder &b6, Location l6, ValueRange a) -> SmallVector<Value> {
                        return {shAdd(b6, l6, a[0], shC(b6, l6, 1)),
                                shAdd(b6, l6, a[1], a[2]),
                                shAdd(b6, l6, a[2], shC(b6, l6, 1)),
                                shAdd(b6, l6, a[3], shC(b6, l6, 1))};
                      });
                    return {wC3->getResult(1), wC3->getResult(2), wC3->getResult(3)};
                  },
                  [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                    // iter!=0: capo=ch[ind]+incr
                    Value chInd = shLoad(b5, l5, ch, ind);
                    return {shAdd(b5, l5, chInd, incr),
                            shAdd(b5, l5, incr, shC(b5, l5, 1)),
                            shAdd(b5, l5, iter, shC(b5, l5, 1))};
                  });
              Value case3Capo = case3If.getResult(0);
              Value case3Incr = case3If.getResult(1);
              Value case3Iter = case3If.getResult(2);

              // default: complex with capoEqSnapsM1 / tIsZero / iterLtOffset
              Value capoEqSnapsM1b = shCmpEQ(b4, l4, capo, shSub(b4, l4, snaps, shC(b4, l4, 1)));
              Value tIsZero_d      = shCmpEQ(b4, l4, t, shC(b4, l4, 0));
              Value iterLtOffset   = shCmpLT(b4, l4, iter, offset);
              Value offsetIs1      = shCmpEQ(b4, l4, offset, shC(b4, l4, 1));

              Value defCapoSnapsM1 = shAdd(b4, l4, capo, shC(b4, l4, 2));
              Value defCapoT0      = shSel(b4, l4, iterLtOffset,
                                      shAdd(b4, l4, capo, shC(b4, l4, 1)),
                                      shAdd(b4, l4, capo, shC(b4, l4, 2)));
              Value defTNew        = shSel(b4, l4, offsetIs1,
                                      shAdd(b4, l4, t, shC(b4, l4, 1)), t);
              Value defAction      = shSel(b4, l4, capoEqSnapsM1b, actAdvance,
                                      shSel(b4, l4, tIsZero_d, actAdvance, actTerminate));
              Value defCapo        = shSel(b4, l4, capoEqSnapsM1b, defCapoSnapsM1,
                                      shSel(b4, l4, tIsZero_d, defCapoT0, capo));
              Value defInd_d       = shSel(b4, l4, capoEqSnapsM1b,
                                      shSub(b4, l4, snaps, shC(b4, l4, 1)), ind);
              Value defT_d         = shSel(b4, l4, capoEqSnapsM1b, t,
                                      shSel(b4, l4, tIsZero_d, defTNew, t));

              Value notIs123 = shAnd(b4, l4,
                                 shAnd(b4, l4, shCmpNE(b4, l4, snaps, shC(b4, l4, 1)),
                                              shCmpNE(b4, l4, snaps, shC(b4, l4, 2))),
                                 shCmpNE(b4, l4, snaps, shC(b4, l4, 3)));

              Value finalCapo = shSel(b4, l4, is1, infCapo,
                                shSel(b4, l4, is2, case2Capo,
                                shSel(b4, l4, is3, case3Capo, defCapo)));
              Value finalIncr = shSel(b4, l4, is3, case3Incr, incr);
              Value finalIter = shSel(b4, l4, is3, case3Iter, iter);
              Value finalInd  = shSel(b4, l4, notIs123, defInd_d, ind);
              Value finalT    = shSel(b4, l4, notIs123, defT_d, t);
              Value finalAct  = shSel(b4, l4, notIs123, defAction, actAdvance);

              return {finalAct,
                      check, finalCapo, oldcapoO, finalT, finalIncr, finalIter,
                      oldind, finalInd, old_f, offset,
                      ch, ord_ch, num_rep};
            },

            // Branch 3b: takeshot r=2
            [&](OpBuilder &b4, Location l4) -> SmallVector<Value> {
              Value is1_b = shCmpEQ(b4, l4, snaps, shC(b4, l4, 1));
              Value is2_b = shCmpEQ(b4, l4, snaps, shC(b4, l4, 2));
              Value is3_b = shCmpEQ(b4, l4, snaps, shC(b4, l4, 3));

              // case 2: ch[1]=capo; incr++
              auto case2IfB = shIf(b4, l4, is2_b, {i64ty0, arrTy},
                  [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                    Value newCh5 = shStore(b5, l5, ch, shC(b5, l5, 1), capo);
                    return {shAdd(b5, l5, incr, shC(b5, l5, 1)), newCh5};
                  },
                  [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                    return {incr, ch};
                  });
              Value case2Incr_b = case2IfB.getResult(0);
              Value case2Ch_b   = case2IfB.getResult(1);

              // case 3: ch[ind]=capo; check=ind; maybe update num_rep/ind
              auto case3IfB = shIf(b4, l4, is3_b,
                  {i64ty0/*check*/, i64ty0/*iter*/, i64ty0/*t*/, i64ty0/*oldind*/,
                   i64ty0/*ind*/, i64ty0/*incr*/, arrTy/*ch*/, arrTy/*num_rep*/},
                  [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                    Value newCh5 = shStore(b5, l5, ch, ind, capo);
                    Value numRep1 = shLoad(b5, l5, num_rep, shC(b5, l5, 1));
                    Value iterEqNR1 = shCmpEQ(b5, l5, iter, numRep1);
                    auto nr1If = shIf(b5, l5, iterEqNR1,
                        {i64ty0, i64ty0, i64ty0, i64ty0, i64ty0, arrTy},
                        [&](OpBuilder &b6, Location l6) -> SmallVector<Value> {
                          Value newNR1 = shAdd(b6, l6, numRep1, shC(b6, l6, 1));
                          Value newNr6 = shStore(b6, l6, num_rep, shC(b6, l6, 1), newNR1);
                          Value remVal = shRem(b6, l6, newNR1, shC(b6, l6, 2));
                          Value newInd6 = shSub(b6, l6, shC(b6, l6, 2), remVal);
                          return {shC(b6, l6, 0)/*iter=0*/,
                                  shAdd(b6, l6, t, shC(b6, l6, 1)),
                                  ind/*old ind→oldind*/, newInd6, shC(b6, l6, 1)/*incr=1*/,
                                  newNr6};
                        },
                        [&](OpBuilder &b6, Location l6) -> SmallVector<Value> {
                          return {iter, t, oldind, ind, incr, num_rep};
                        });
                    return {ind/*check=ind*/,
                            nr1If.getResult(0), nr1If.getResult(1), nr1If.getResult(2),
                            nr1If.getResult(3), nr1If.getResult(4),
                            newCh5, nr1If.getResult(5)};
                  },
                  [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                    return {check, iter, t, oldind, ind, incr, ch, num_rep};
                  });
              Value case3Check_b  = case3IfB.getResult(0);
              Value case3Iter_b   = case3IfB.getResult(1);
              Value case3T_b      = case3IfB.getResult(2);
              Value case3Oldind_b = case3IfB.getResult(3);
              Value case3Ind_b    = case3IfB.getResult(4);
              Value case3Incr_b   = case3IfB.getResult(5);
              Value case3Ch_b     = case3IfB.getResult(6);
              Value case3Nr_b     = case3IfB.getResult(7);

              // default (snaps >= 4)
              Value isDefault_b = shAnd(b4, l4,
                                    shCmpNE(b4, l4, snaps, shC(b4, l4, 2)),
                                    shCmpNE(b4, l4, snaps, shC(b4, l4, 3)));
              Value tIsZeroForOffset_b = shCmpEQ(b4, l4, t, shC(b4, l4, 0));
              Value iterEqOffset_b     = shCmpEQ(b4, l4, iter, offset);
              Value capoGeSnapsP2_b    = shCmpGE(b4, l4, capo,
                                            shAdd(b4, l4, snaps, shC(b4, l4, 2)));
              Value rot2Cond_b = shAnd(b4, l4,
                                  shAnd(b4, l4, isDefault_b, capoGeSnapsP2_b),
                                  shAnd(b4, l4, tIsZeroForOffset_b, iterEqOffset_b));

              // defIf: 7 scalars + 2 arrays
              SmallVector<Type> defTys = {i64ty0/*act*/, i64ty0/*check*/,
                  i64ty0/*incr*/, i64ty0/*iter*/, i64ty0/*oldind*/,
                  i64ty0/*ind*/, i64ty0/*t*/, arrTy/*ch*/, arrTy/*ord_ch*/};
              auto defIf = shIf(b4, l4, isDefault_b, defTys,
                [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                  Value snapsP2b = shAdd(b5, l5, snaps, shC(b5, l5, 2));
                  Value snapsP1b = shAdd(b5, l5, snaps, shC(b5, l5, 1));
                  Value capoLtSnapsP2b = shCmpLT(b5, l5, capo, snapsP2b);

                  auto subA = shIf(b5, l5, capoLtSnapsP2b, defTys,
                    [&](OpBuilder &b6, Location l6) -> SmallVector<Value> {
                      // Sub-case A: capo < snaps+2
                      Value newCh6 = shStore(b6, l6, ch, ind, capo);
                      Value capoEqSnapsP1b = shCmpEQ(b6, l6, capo, shAdd(b6, l6, snaps, shC(b6, l6, 1)));

                      auto rot1If = shIf(b6, l6, capoEqSnapsP1b,
                          {i64ty0/*check*/, i64ty0/*incr*/, i64ty0/*oldind*/, i64ty0/*ind*/, arrTy/*ch*/, arrTy/*ord_ch*/},
                          [&](OpBuilder &b7, Location l7) -> SmallVector<Value> {
                            // Rotation 1
                            Value snapsM1b2 = shSub(b7, l7, snaps, shC(b7, l7, 1));
                            Value ordSnapsM1 = shLoad(b7, l7, ord_ch, snapsM1b2);
                            Value newOrdCh7 = shOrdChShiftRight(b7, l7, snaps,
                                shC(b7, l7, 1), ord_ch, smSize);
                            Value newOrdCh7b = shStore(b7, l7, newOrdCh7,
                                shC(b7, l7, 1), ordSnapsM1);
                            return {ind/*check=old ind*/, shC(b7, l7, 2)/*incr*/,
                                    ordSnapsM1/*oldind*/, shC(b7, l7, 2)/*ind*/,
                                    newCh6, newOrdCh7b};
                          },
                          [&](OpBuilder &b7, Location l7) -> SmallVector<Value> {
                            return {ind, incr, oldind, ind, newCh6, ord_ch};
                          });

                      return {actTakeshot,
                              rot1If.getResult(0), rot1If.getResult(1), iter,
                              rot1If.getResult(2), rot1If.getResult(3), t,
                              rot1If.getResult(4), rot1If.getResult(5)};
                    },
                    [&](OpBuilder &b6, Location l6) -> SmallVector<Value> {
                      // Sub-case B: capo >= snaps+2
                      Value tIsZerob6 = shCmpEQ(b6, l6, t, shC(b6, l6, 0));
                      auto tZeroIf = shIf(b6, l6, tIsZerob6, defTys,
                        [&](OpBuilder &b7, Location l7) -> SmallVector<Value> {
                          Value iterEqOff = shCmpEQ(b7, l7, iter, offset);
                          auto iteIf = shIf(b7, l7, iterEqOff, defTys,
                            [&](OpBuilder &b8, Location l8) -> SmallVector<Value> {
                              // rotation2
                              Value snapsM1b3 = shSub(b8, l8, snaps, shC(b8, l8, 1));
                              Value ordSnapsM1b = shLoad(b8, l8, ord_ch, snapsM1b3);
                              Value newCh8 = shStore(b8, l8, ch, ordSnapsM1b, capo);
                              Value newOrdCh8 = shOrdChShiftRight(b8, l8, snaps,
                                  incr, ord_ch, smSize);
                              Value newOrdCh8b = shStore(b8, l8, newOrdCh8, incr, ordSnapsM1b);
                              Value newIncr8 = shAdd(b8, l8, incr, shC(b8, l8, 1));
                              return {actTakeshot,
                                      ordSnapsM1b/*check*/, newIncr8, shC(b8, l8, 1)/*iter=1*/,
                                      ordSnapsM1b/*oldind*/, newIncr8/*ind=newIncr*/,
                                      t, newCh8, newOrdCh8b};
                            },
                            [&](OpBuilder &b8, Location l8) -> SmallVector<Value> {
                              // iter != offset: ch[ord_ch[ind]]=capo; check=ord_ch[ind]; iter++; ind++
                              Value ordChInd = shLoad(b8, l8, ord_ch, ind);
                              Value newCh8 = shStore(b8, l8, ch, ordChInd, capo);
                              return {actTakeshot,
                                      ordChInd/*check*/, incr,
                                      shAdd(b8, l8, iter, shC(b8, l8, 1)),
                                      oldind, shAdd(b8, l8, ind, shC(b8, l8, 1)),
                                      t, newCh8, ord_ch};
                            });
                          return SmallVector<Value>(iteIf.getResults().begin(),
                                                   iteIf.getResults().end());
                        },
                        [&](OpBuilder &b7, Location l7) -> SmallVector<Value> {
                          // t != 0: terminate
                          return {actTerminate, check, incr, iter, oldind, ind, t, ch, ord_ch};
                        });
                      return SmallVector<Value>(tZeroIf.getResults().begin(),
                                               tZeroIf.getResults().end());
                    });
                  return SmallVector<Value>(subA.getResults().begin(),
                                           subA.getResults().end());
                },
                [&](OpBuilder &b5, Location l5) -> SmallVector<Value> {
                  return {actTakeshot, check, incr, iter, oldind, ind, t, ch, ord_ch};
                });

              // Merge case2 / case3 / default
              Value is1b2 = shCmpEQ(b4, l4, snaps, shC(b4, l4, 1));
              Value tsAction_b = shSel(b4, l4, is1b2, actAdvance,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(0), actTakeshot));
              Value tsCheck_b  = shSel(b4, l4, is3_b, case3Check_b,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(1), check));
              Value tsIncr_b   = shSel(b4, l4, is2_b, case2Incr_b,
                                   shSel(b4, l4, is3_b, case3Incr_b,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(2), incr)));
              Value tsIter_b   = shSel(b4, l4, is3_b, case3Iter_b,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(3), iter));
              Value tsOldind_b = shSel(b4, l4, is3_b, case3Oldind_b,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(4), oldind));
              Value tsInd_b    = shSel(b4, l4, is3_b, case3Ind_b,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(5), ind));
              Value tsT_b      = shSel(b4, l4, is3_b, case3T_b,
                                   shSel(b4, l4, isDefault_b, defIf.getResult(6), t));
              Value tsOffset_b = shSel(b4, l4, rot2Cond_b,
                                   shSub(b4, l4, offset, shC(b4, l4, 1)), offset);

              // Merge ch arrays: case2 may update ch[1], case3 may update ch[ind], default updates
              Value tsCh = shSel(b4, l4, is2_b, case2Ch_b,
                             shSel(b4, l4, is3_b, case3Ch_b,
                             shSel(b4, l4, isDefault_b, defIf.getResult(7), ch)));
              Value tsOrdCh = shSel(b4, l4, isDefault_b, defIf.getResult(8), ord_ch);
              Value tsNumRep = shSel(b4, l4, is3_b, case3Nr_b, num_rep);

              return {tsAction_b,
                      tsCheck_b, capo, oldcapoO, tsT_b, tsIncr_b, tsIter_b,
                      tsOldind_b, tsInd_b, old_f, tsOffset_b,
                      tsCh, tsOrdCh, tsNumRep};
            }); // end r2If

          return SmallVector<Value>(r2If.getResults().begin(), r2If.getResults().end());
        }); // end branch23

      return SmallVector<Value>(branch23.getResults().begin(), branch23.getResults().end());
    }); // end outerIf

  SMResult res;
  res.action    = outerIf.getResult(0);
  res.check     = outerIf.getResult(1);
  res.capo      = outerIf.getResult(2);
  res.oldcapoO  = outerIf.getResult(3);
  res.t         = outerIf.getResult(4);
  res.incr      = outerIf.getResult(5);
  res.iter      = outerIf.getResult(6);
  res.oldind    = outerIf.getResult(7);
  res.ind       = outerIf.getResult(8);
  res.old_f     = outerIf.getResult(9);
  res.offset    = outerIf.getResult(10);
  res.ch        = outerIf.getResult(11);
  res.ord_ch    = outerIf.getResult(12);
  res.num_rep   = outerIf.getResult(13);
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
//   [14+N+N+2]  done        : tensor<i1>

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

  // Helper: 1-d tensor<n×elemTy> for checkpoint buffer of a primal init.
  auto ckptTy = [&](Value init) -> RankedTensorType {
    auto elemTy = cast<RankedTensorType>(init.getType()).getElementType();
    return RankedTensorType::get({budget + 1}, elemTy);
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

  // Create scaffold stablehlo.while ----------------------------------------
  auto scaffoldWhile = stablehlo::WhileOp::create(
      builder, loc, scaffoldTypes, initArgs);

  // Cond region: !done
  {
    OpBuilder::InsertionGuard g(builder);
    Block *condBlk = builder.createBlock(&scaffoldWhile.getCond());
    for (auto ty : scaffoldTypes) condBlk->addArgument(ty, loc);
    builder.setInsertionPointToEnd(condBlk);
    Value doneTensor = condBlk->getArgument(IDX_DONE);
    Value doneScalar = shLoadTyped(builder, loc,
        stablehlo::ReshapeOp::create(builder, loc,
            RankedTensorType::get({1}, builder.getI1Type()), doneTensor),
        shC(builder, loc, 0),
        RankedTensorType::get({}, builder.getI1Type()));
    // Extract from 0-d tensor<i1>
    Value doneI1 = tensor::ExtractOp::create(builder, loc, doneTensor, ValueRange{});
    Value notDoneI1 = builder.create<arith::XOrIOp>(loc, doneI1,
        arith::ConstantIntOp::create(builder, loc, builder.getI1Type(), 1));
    // Wrap back to tensor<i1>
    Value notDoneTensor = stablehlo::ConstantOp::create(builder, loc, i1ty0,
        DenseElementsAttr::get(i1ty0, builder.getIntegerAttr(builder.getI1Type(), 0)));
    // Use stablehlo.select: if !done return true_tensor else false_tensor
    // Simpler: build tensor<i1> from the scalar
    Value notDoneT = tensor::FromElementsOp::create(builder, loc, i1ty0,
        ValueRange{notDoneI1});
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
    Value primalCondI1 = evalCondRegion(builder, loc, newWhileOp, gutils,
                                         primalCarried, condMap);
    // Wrap to tensor<i1>
    Value primalCondT = tensor::FromElementsOp::create(builder, loc, i1ty0,
        ValueRange{primalCondI1});

    // action == 1 → STORE
    Value isStore = shCmpEQ(builder, loc, sm.action, shC(builder, loc, 1));
    // action == 0 → TERMINATE
    Value isTerminate = shCmpEQ(builder, loc, sm.action, shC(builder, loc, 0));

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
          newCkpt[i] = shStoreTyped(b2, l2, newCkpt[i], slot, primalCarried[i]);
        for (auto v : newCkpt) res.push_back(v);
        // Update ckptIdxBuf[slot] = numFwd
        Value newIdxBuf = shStore(b2, l2, ckptIdxBuf, slot, numFwd);
        res.push_back(newIdxBuf);
        return res;
      },
      // FORWARD / TERMINATE branch
      [&](OpBuilder &b2, Location l2) -> SmallVector<Value> {
        // Advance primal one step if primalCond && !isTerminate
        Value doStep = shAnd(b2, l2,
            tensor::FromElementsOp::create(b2, l2, i1ty0, ValueRange{primalCondI1}),
            shNot(b2, l2, isTerminate));
        SmallVector<Type> stepTys;
        for (auto v : primalCarried) stepTys.push_back(v.getType());
        stepTys.push_back(i64ty0); // stepTaken

        auto stepIf = shIf(b2, l2, doStep, stepTys,
          [&](OpBuilder &b3, Location l3) -> SmallVector<Value> {
            IRMapping bodyMap;
            SmallVector<Value> ns = runBodyStep(b3, l3, newWhileOp, gutils,
                                                primalCarried, bodyMap);
            ns.push_back(shC(b3, l3, 1));
            return ns;
          },
          [&](OpBuilder &b3, Location l3) -> SmallVector<Value> {
            SmallVector<Value> r(primalCarried.begin(), primalCarried.end());
            r.push_back(shC(b3, l3, 0));
            return r;
          });

        auto stepRes = stepIf.getResults();
        auto newState = stepRes.drop_back(1);
        Value stepTaken = stepRes.back();
        Value newNumFwd = shAdd(b2, l2, numFwd, stepTaken);

        SmallVector<Value> res = {newNumFwd};
        res.append(newState.begin(), newState.end());
        res.append(ckptBufs.begin(), ckptBufs.end()); // ckptBufs unchanged
        res.push_back(ckptIdxBuf);
        return res;
      });

    Value newNumFwd = dispatchIf.getResult(0);
    auto newPrimal  = dispatchIf.getResults().slice(1, numCarried);
    auto newCkpts   = dispatchIf.getResults().slice(1 + numCarried, numCarried);
    Value newIdxBuf = dispatchIf.getResult(1 + numCarried + numCarried);

    // done = !primalCond || isTerminate || capacityExceeded
    Value notPrimalCond = shNot(builder, loc, primalCondT);
    Value newDone = shOr(builder, loc, notPrimalCond,
                     shOr(builder, loc, isTerminate, capacityExceeded));

    // Build yield
    SmallVector<Value> yields;
    yields.push_back(newNumFwd);
    yields.push_back(sm.check);
    yields.push_back(sm.capo);
    yields.push_back(sm.oldcapoO);
    yields.push_back(sm.t);
    yields.push_back(sm.incr);
    yields.push_back(sm.iter);
    yields.push_back(sm.oldind);
    yields.push_back(sm.ind);
    yields.push_back(sm.old_f);
    yields.push_back(sm.offset);
    yields.push_back(sm.ch);
    yields.push_back(sm.ord_ch);
    yields.push_back(sm.num_rep);
    yields.append(newPrimal.begin(), newPrimal.end());
    yields.append(newCkpts.begin(), newCkpts.end());
    yields.push_back(newIdxBuf);
    yields.push_back(newDone);
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

  // Helper: load typed tensor element at index slot.
  auto loadTypedAt = [&](Value arr, Value slotIdx) -> Value {
    Value asI64 = arith::IndexCastOp::create(builder, loc, builder.getI64Type(), slotIdx);
    auto slotT = RankedTensorType::get({}, builder.getI64Type());
    Value slot64 = tensor::FromElementsOp::create(builder, loc, slotT, ValueRange{asI64});
    auto bufTy = cast<RankedTensorType>(arr.getType());
    auto elemTy0d = RankedTensorType::get({}, bufTy.getElementType());
    return shLoadTyped(builder, loc, arr, slot64, elemTy0d);
  };

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
      ckptState.push_back(loadTypedAt(liveCkpt[i], capo));

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
        updCkpt[i] = shStoreTyped(builder, loc, updCkpt[i], slot64, astate[i]);
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
