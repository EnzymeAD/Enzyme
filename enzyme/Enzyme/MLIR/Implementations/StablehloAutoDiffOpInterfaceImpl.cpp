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

// StableHLO dialect headers — available in the Bazel/Reactant build.
// In a standalone build that does not include StableHLO, this translation
// unit is simply not compiled (guarded by the CMake option).
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
        return cacheWhileOnlineStablehlo(whileOp, *budget, gutils);
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
      return reverseWhileOnlineStablehlo(whileOp, *budget, builder, gutils,
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
