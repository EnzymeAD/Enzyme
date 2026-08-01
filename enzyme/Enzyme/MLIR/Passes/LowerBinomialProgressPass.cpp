//===------------------------------------------------------------------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower the enzyme.binomial_progress op (the
// Revolve binomial-checkpointing "split" function) into scf/arith ops.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERBINOMIALPROGRESSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// Lower enzyme.binomial_progress(n, s) to the Revolve advance distance; see
// BinomialProgressOp's description for the derivation.
//
//   %one = arith.constant 1
//   %c = (n <= %one) or (s <= %one)
//   %r = scf.if %c {
//     // n <= 1 yields n (0 or 1); otherwise s <= 1 yields n. Both are n.
//     scf.yield n
//   } else {
//     // smallest t with beta = C(s+t, t) >= n
//     %w:2 = scf.while (%t = %zero, %beta = %one) {
//       %lt = arith.cmpi slt, %beta, n
//       scf.condition(%lt) %t, %beta
//     } do {
//     ^bb0(%t: index, %beta: index):
//       %t2 = %t + %one
//       %beta2 = %beta * (s + %t2) / %t2
//       scf.yield %t2, %beta2
//     }
//     // window [n - beta(s-1,t), beta(s,t-1)], clamped; take the midpoint
//     %lo = maxsi(n - (%beta * s) / (s + %t), %one)
//     %hi = minsi((%beta * %t) / (s + %t), n - %one)
//     scf.yield (%lo + %hi) / 2
//   }
//
// The guard must be a branch, not a select: for s <= 1 the update leaves %beta
// at 1 and the loop would spin forever.
static int64_t binomialProgress(int64_t n, int64_t s) {
  if (n <= 0)
    return 0;
  if (n == 1)
    return 1;
  if (s <= 1)
    return n;
  int64_t t = 0, beta = 1; // beta == C(s + t, t)
  while (beta < n) {
    ++t;
    beta = beta * (s + t) / t;
  }
  int64_t lo = n - beta * s / (s + t);
  int64_t hi = beta * t / (s + t);
  if (lo < 1)
    lo = 1;
  if (hi > n - 1)
    hi = n - 1;
  int64_t m = (lo + hi) / 2;
  int64_t cap = n - (s - 1); // leave a step for each slot still to be placed
  if (m > cap)
    m = cap;
  return m < 1 ? 1 : m;
}

static void lowerBinomialProgress(enzyme::BinomialProgressOp op) {
  // Tensor operands are lowered elsewhere; this pass only handles the
  // scalar integer/index case.
  if (isa<TensorType>(op.getType()))
    return;

  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value n = op.getNumSteps();
  Value s = op.getBudget();
  Type idxTy = op.getType();
  auto constOfType = [&](int64_t v) -> Value {
    return arith::ConstantOp::create(builder, loc,
                                     builder.getIntegerAttr(idxTy, v));
  };

  // Constant fast-path: fold to a plain constant.
  llvm::APInt nCst, sCst;
  if (matchPattern(n, m_ConstantInt(&nCst)) &&
      matchPattern(s, m_ConstantInt(&sCst)) && nCst.getSExtValue() > 0 &&
      sCst.getSExtValue() > 0) {
    Value c =
        constOfType(binomialProgress(nCst.getSExtValue(), sCst.getSExtValue()));
    op.getResult().replaceAllUsesWith(c);
    op->erase();
    return;
  }

  Value zero = constOfType(0);
  Value one = constOfType(1);

  // Guard both degenerate cases. It has to be a branch: with s <= 1 the loop
  // body below leaves beta at 1 and would never terminate.
  Value nSmall =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, n, one);
  Value sSmall =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle, s, one);
  Value cond = arith::OrIOp::create(builder, loc, nSmall, sSmall);

  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{idxTy}, cond,
                                /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    // n <= 1 yields n itself (0 or 1); s <= 1 advances the whole remainder,
    // which is also n. So both degenerate cases are just n.
    scf::YieldOp::create(builder, loc, ValueRange{n});
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.elseBlock());

    // Smallest t with beta = C(s + t, t) >= n, carrying (t, beta).
    auto whileOp = scf::WhileOp::create(builder, loc, TypeRange{idxTy, idxTy},
                                        ValueRange{zero, one});

    // Before region: continue while beta < n.
    {
      Block *before = builder.createBlock(&whileOp.getBefore(), {},
                                          TypeRange{idxTy, idxTy}, {loc, loc});
      builder.setInsertionPointToEnd(before);
      Value beta = before->getArgument(1);
      Value lt = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                       beta, n);
      scf::ConditionOp::create(builder, loc, lt, before->getArguments());
    }

    // After region: t' = t + 1; beta' = beta * (s + t') / t'. That steps
    // C(s+t-1, t-1) to C(s+t, t), so the division is exact.
    {
      Block *after = builder.createBlock(&whileOp.getAfter(), {},
                                         TypeRange{idxTy, idxTy}, {loc, loc});
      builder.setInsertionPointToEnd(after);
      Value t = after->getArgument(0);
      Value beta = after->getArgument(1);
      Value t2 = arith::AddIOp::create(builder, loc, t, one);
      Value sPlusT = arith::AddIOp::create(builder, loc, s, t2);
      Value num = arith::MulIOp::create(builder, loc, beta, sPlusT);
      Value beta2 = arith::DivUIOp::create(builder, loc, num, t2);
      scf::YieldOp::create(builder, loc, ValueRange{t2, beta2});
    }

    builder.setInsertionPointAfter(whileOp);
    Value t = whileOp.getResult(0);
    Value beta = whileOp.getResult(1);

    // beta(s-1, t) = beta * s / (s + t) and beta(s, t-1) = beta * t / (s + t),
    // both exact in integers. Every advance between n - beta(s-1,t) and
    // beta(s, t-1) attains the optimal repetition count; clamp that window to
    // [1, n-1] -- so the caller always progresses and always leaves a tail --
    // and take its midpoint, since either edge can collapse onto the clamp and
    // waste a checkpoint on a one-step advance.
    Value sPlusT = arith::AddIOp::create(builder, loc, s, t);
    Value loNum = arith::MulIOp::create(builder, loc, beta, s);
    Value loRaw = arith::SubIOp::create(
        builder, loc, n, arith::DivUIOp::create(builder, loc, loNum, sPlusT));
    Value hiNum = arith::MulIOp::create(builder, loc, beta, t);
    Value hiRaw = arith::DivUIOp::create(builder, loc, hiNum, sPlusT);

    Value nm1 = arith::SubIOp::create(builder, loc, n, one);
    Value lo = arith::MaxSIOp::create(builder, loc, loRaw, one);
    Value hi = arith::MinSIOp::create(builder, loc, hiRaw, nm1);
    Value two = constOfType(2);
    Value sum = arith::AddIOp::create(builder, loc, lo, hi);
    Value mid = arith::DivUIOp::create(builder, loc, sum, two);

    // Leave a step for each of the s-1 checkpoints still to be placed. Without
    // this the advances can exhaust the interval before the slots run out, and
    // a caller walking one slot per iteration then records slots at a step past
    // the end, holding the final state rather than a checkpoint.
    Value sm1 = arith::SubIOp::create(builder, loc, s, one);
    Value cap = arith::SubIOp::create(builder, loc, n, sm1);
    Value capped = arith::MinSIOp::create(builder, loc, mid, cap);
    Value res = arith::MaxSIOp::create(builder, loc, capped, one);
    scf::YieldOp::create(builder, loc, ValueRange{res});
  }

  op.getResult().replaceAllUsesWith(ifOp.getResult(0));
  op->erase();
}

struct LowerBinomialProgressPass
    : public enzyme::impl::LowerBinomialProgressPassBase<
          LowerBinomialProgressPass> {
  void runOnOperation() override {
    SmallVector<enzyme::BinomialProgressOp> ops;
    getOperation()->walk(
        [&](enzyme::BinomialProgressOp op) { ops.push_back(op); });
    for (auto op : ops)
      lowerBinomialProgress(op);
  }
};
} // end anonymous namespace
