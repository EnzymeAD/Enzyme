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

// Lower enzyme.binomial_progress(n, s) into:
//
//   %one = arith.constant 1 : index
//   %c = (n == 1) or (s == 1)
//   %r = scf.if %c -> index {
//     scf.yield %one
//   } else {
//     %w:2 = scf.while (%j = %one, %binom = s) {
//       %lt = arith.cmpi slt, %binom, n
//       scf.condition(%lt) %j, %binom
//     } do {
//     ^bb0(%j: index, %binom: index):
//       %j2 = %j + %one
//       %binom2 = %binom * (%j2 + s - %one) / %j2
//       scf.yield %j2, %binom2
//     }
//     %eq = arith.cmpi eq, %w#1, n
//     %jm1 = %w#0 - %one
//     %sel = arith.select %eq, %w#0, %jm1
//     scf.yield %sel
//   }
static int64_t binomialProgress(int64_t n, int64_t s) {
  if (s == 1 || n == 1)
    return 1;
  int64_t j = 1, binom = s;
  while (binom < n) {
    ++j;
    binom = binom * (j + s - 1) / j;
  }
  return binom == n ? j : j - 1;
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

  Value one = constOfType(1);

  Value nIsOne =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, n, one);
  Value sIsOne =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, s, one);
  Value cond = arith::OrIOp::create(builder, loc, nIsOne, sIsOne);

  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{idxTy}, cond,
                                /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());
    scf::YieldOp::create(builder, loc, ValueRange{one});
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.elseBlock());

    auto whileOp = scf::WhileOp::create(builder, loc, TypeRange{idxTy, idxTy},
                                        ValueRange{one, s});

    // Before region: continue while binom < n.
    {
      Block *before = builder.createBlock(&whileOp.getBefore(), {},
                                          TypeRange{idxTy, idxTy}, {loc, loc});
      builder.setInsertionPointToEnd(before);
      Value binom = before->getArgument(1);
      Value lt = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                       binom, n);
      scf::ConditionOp::create(builder, loc, lt, before->getArguments());
    }

    // After region: j' = j + 1; binom' = binom * (j' + s - 1) / j'.
    {
      Block *after = builder.createBlock(&whileOp.getAfter(), {},
                                         TypeRange{idxTy, idxTy}, {loc, loc});
      builder.setInsertionPointToEnd(after);
      Value j = after->getArgument(0);
      Value binom = after->getArgument(1);
      Value j2 = arith::AddIOp::create(builder, loc, j, one);
      Value t = arith::AddIOp::create(builder, loc, j2, s);
      t = arith::SubIOp::create(builder, loc, t, one);
      Value num = arith::MulIOp::create(builder, loc, binom, t);
      Value binom2 = arith::DivUIOp::create(builder, loc, num, j2);
      scf::YieldOp::create(builder, loc, ValueRange{j2, binom2});
    }

    builder.setInsertionPointAfter(whileOp);
    Value j = whileOp.getResult(0);
    Value binom = whileOp.getResult(1);
    Value eq =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, binom, n);
    Value jm1 = arith::SubIOp::create(builder, loc, j, one);
    Value sel = arith::SelectOp::create(builder, loc, eq, j, jm1);
    scf::YieldOp::create(builder, loc, ValueRange{sel});
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
