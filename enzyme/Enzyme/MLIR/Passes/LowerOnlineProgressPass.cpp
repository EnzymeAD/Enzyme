//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers enzyme.online_progress to scf/arith/memref ops.
//
// The op encodes one call to Online_r2::revolve() from the PyRevolve library
// (Griewank, Walther, Stumm), which implements the online checkpointing
// algorithm from Stumm & Walther (SIAM J. Sci. Comput. 32(2), 2010).
//
// Source reference: github.com/devitocodes/pyrevolve, src/revolve.cpp,
// function Online_r2::revolve().  The translation is line-for-line faithful
// to that C++ function, with two simplifications:
//
//   1. The ch[] loop-body writes inside the two rotation loops are provably
//      no-ops (after ord_ch[k] = ord_ch[k-1], the LHS ch[ord_ch[k]] and RHS
//      ch[ord_ch[k-1]] index the same slot).  They are omitted.
//   2. Counters advances / takeshots / commands are omitted (not needed in
//      Enzyme's usage).
//
// The switch(snaps) branches are emitted as scf.if chains; when snaps is a
// compile-time constant (always true in Enzyme's usage), arith constant
// folding eliminates all but the live branch.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERONLINEPROGRESSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// Emit the descending rotation:
//   for k = (snaps-1) downto (kLow+1):
//     ord_ch[k] = ord_ch[k-1]
// implemented as ascending j = 0..snaps-2-kLow with k = snaps-1-j.
// The companion ch[] updates are no-ops and are omitted (see file header).
static void emitOrdChShiftRight(OpBuilder &b, Location loc, Value snaps,
                                Value kLow, Value ord_ch) {
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value snapsM1 = arith::SubIOp::create(b, loc, snaps, one);
  Value snapsM2 = arith::SubIOp::create(b, loc, snaps,
                                         arith::ConstantIndexOp::create(b, loc, 2));
  // Number of iterations: snaps-1 - kLow  (k goes snaps-1 downto kLow+1)
  Value loopUB = arith::SubIOp::create(b, loc, snapsM1, kLow);

  auto loop = scf::ForOp::create(b, loc, zero, loopUB, one, ValueRange{});
  if (!loop.getBody()->empty())
    loop.getBody()->front().erase();
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(loop.getBody());
    Value j = loop.getInductionVar();
    // k = snaps-1-j,  k-1 = snaps-2-j
    Value k   = arith::SubIOp::create(b, loc, snapsM1, j);
    Value km1 = arith::SubIOp::create(b, loc, snapsM2, j);
    Value ordChKm1 = memref::LoadOp::create(b, loc, ord_ch, ValueRange{km1});
    memref::StoreOp::create(b, loc, ordChKm1, ord_ch, ValueRange{k});
    scf::YieldOp::create(b, loc);
  }
}

// Lower one enzyme.online_progress op to scf/arith/memref.
//
// The returned SmallVector<Value> has 11 entries in the order the op's results
// are declared:
//   [action, check, capo, oldcapo_o, t, incr, iter, oldind, ind, old_f, offset]
//
// It directly follows Online_r2::revolve(); references to C++ identifiers
// below correspond one-to-one to that function.
static void lowerOnlineProgress(enzyme::OnlineProgressOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();

  Value check    = op.getCheck();
  Value capo     = op.getCapo();
  Value oldcapoO = op.getOldcapoO();
  Value t        = op.getT();
  Value incr     = op.getIncr();
  Value iter     = op.getIter();
  Value oldind   = op.getOldind();
  Value ind      = op.getInd();
  Value old_f    = op.getOldF();
  Value offset   = op.getOffset();
  Value snaps    = op.getSnaps();
  Value ch       = op.getCh();
  Value ord_ch   = op.getOrdCh();
  Value num_rep  = op.getNumRep();

  // Helpers -----------------------------------------------------------------
  auto ci = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  auto add = [&](Value a, Value bv) -> Value {
    return arith::AddIOp::create(b, loc, a, bv);
  };
  auto sub = [&](Value a, Value bv) -> Value {
    return arith::SubIOp::create(b, loc, a, bv);
  };
  auto cmpEQ = [&](Value a, Value bv) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, a, bv);
  };
  auto cmpNE = [&](Value a, Value bv) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, a, bv);
  };
  auto cmpSLT = [&](Value a, Value bv) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, a, bv);
  };
  auto cmpSLE = [&](Value a, Value bv) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, a, bv);
  };
  auto andOp = [&](Value a, Value bv) -> Value {
    return arith::AndIOp::create(b, loc, a, bv);
  };
  auto orOp = [&](Value a, Value bv) -> Value {
    return arith::OrIOp::create(b, loc, a, bv);
  };
  auto sel = [&](Value c, Value tv, Value fv) -> Value {
    return arith::SelectOp::create(b, loc, c, tv, fv);
  };
  auto ld = [&](Value mem, Value idx) -> Value {
    return memref::LoadOp::create(b, loc, mem, ValueRange{idx});
  };
  auto st = [&](Value val, Value mem, Value idx) {
    memref::StoreOp::create(b, loc, val, mem, ValueRange{idx});
  };

  Type idxTy = b.getIndexType();
  // All branches produce 11 index values (action + 10 state fields).
  SmallVector<Type> Tys(11, idxTy);

  Value zero   = ci(0);
  Value one    = ci(1);
  Value two    = ci(2);
  Value three  = ci(3);
  Value snapsM1 = sub(snaps, one);
  Value snapsP1 = add(snaps, one);
  Value snapsP2 = add(snaps, two);

  // Action codes.
  Value actTerminate = zero;
  Value actTakeshot  = one;
  Value actAdvance   = two;

  // ch[check] is only valid when check >= 0.  Use a select so we never form
  // an out-of-bounds load; the or below makes the load result irrelevant when
  // check < 0.
  Value checkLtZero = cmpSLT(check, zero);
  Value safeCheck   = sel(checkLtZero, zero, check);
  Value chAtCheck   = ld(ch, safeCheck);

  // Top-level branch condition for takeshot r=1:
  //   (check == -1) || (ch[check] != capo && capo <= snaps-1)
  Value chNeqCapo = cmpNE(chAtCheck, capo);
  Value capoLeSnapsM1 = cmpSLE(capo, snapsM1);
  Value isTakeshotR1 = orOp(checkLtZero, andOp(chNeqCapo, capoLeSnapsM1));

  // =========================================================================
  // Outer if: takeshot r=1 vs advance-r1 / r=2
  // =========================================================================
  auto outerIf = scf::IfOp::create(b, loc, TypeRange(Tys), isTakeshotR1,
                                   /*withElse=*/true);

  // -------------------------------------------------------------------------
  // Branch 1: takeshot for r=1
  //   oldcapo_o = capo; check++; ch[check] = capo; t = 0;
  //   if snaps < 4: num_rep[*]=2; incr=2; iter=1; oldind=snaps-1;
  //   else:         num_rep[*]=1; ord_ch[i]=i; incr=1; iter=1; oldind=1; offset=snaps-1
  //   if capo == snaps-1: ind=2; old_f=1
  // -------------------------------------------------------------------------
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outerIf.thenBlock());

    Value newCheck = add(check, one);
    st(capo, ch, newCheck);

    // Initialization depends on snaps < 4 or snaps >= 4.
    Value snapsLt4 = cmpSLT(snaps, ci(4));
    // Yields: [incr, iter, oldind, newOffset]
    auto initIf = scf::IfOp::create(b, loc, TypeRange{idxTy, idxTy, idxTy, idxTy},
                                    snapsLt4, /*withElse=*/true);
    {
      // snaps < 4: num_rep[i]=2; ord_ch[i]=i; incr=2, iter=1, oldind=snaps-1
      // ord_ch must be initialized for snaps<4 too — the rotation2 path reads
      // ord_ch[snaps-1] and an uninitialized value produces wrong checkSlot.
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(initIf.thenBlock());
      auto fillLoop = scf::ForOp::create(b, loc, zero, snaps, one, ValueRange{});
      if (!fillLoop.getBody()->empty())
        fillLoop.getBody()->front().erase();
      {
        OpBuilder::InsertionGuard gf(b);
        b.setInsertionPointToStart(fillLoop.getBody());
        Value iv = fillLoop.getInductionVar();
        st(two, num_rep, iv);
        st(iv,  ord_ch,  iv);   // ord_ch[i] = i  (same as snaps>=4 branch)
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc, ValueRange{two, one, snapsM1, offset /*unchanged*/});
    }
    {
      // snaps >= 4: num_rep[i]=1; ord_ch[i]=i; incr=1, iter=1, oldind=1, offset=snaps-1
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(initIf.elseBlock());
      auto fillLoop = scf::ForOp::create(b, loc, zero, snaps, one, ValueRange{});
      if (!fillLoop.getBody()->empty())
        fillLoop.getBody()->front().erase();
      {
        OpBuilder::InsertionGuard gf(b);
        b.setInsertionPointToStart(fillLoop.getBody());
        Value iv = fillLoop.getInductionVar();
        st(one, num_rep, iv);
        st(iv,  ord_ch,  iv);
        scf::YieldOp::create(b, loc);
      }
      scf::YieldOp::create(b, loc, ValueRange{one, one, one, snapsM1});
    }

    Value newIncr   = initIf.getResult(0);
    Value newIter   = initIf.getResult(1);
    Value newOldind = initIf.getResult(2);
    Value newOffset = initIf.getResult(3);

    // if capo == snaps-1: ind=2; old_f=1
    Value capoEqSnapsM1 = cmpEQ(capo, snapsM1);
    Value newInd  = sel(capoEqSnapsM1, two, ind);
    Value newOldF = sel(capoEqSnapsM1, one, old_f);

    scf::YieldOp::create(b, loc, ValueRange{
      actTakeshot,               // action
      newCheck, capo /*capo*/,   capo /*oldcapo_o=capo*/,
      zero     /*t=0*/,          newIncr, newIter, newOldind, newInd,
      newOldF, newOffset
    });
  }

  // =========================================================================
  // Else: advance r=1 or r=2
  // =========================================================================
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outerIf.elseBlock());

    // Branch 2: advance r=1 -- capo < snaps-1 → capo = oldcapo_o + 1
    Value capoLtSnapsM1 = cmpSLT(capo, snapsM1);
    auto branch23 = scf::IfOp::create(b, loc, TypeRange(Tys), capoLtSnapsM1,
                                      /*withElse=*/true);

    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(branch23.thenBlock());
      scf::YieldOp::create(b, loc, ValueRange{
        actAdvance,
        check, add(oldcapoO, one) /*capo=oldcapo_o+1*/, oldcapoO,
        t, incr, iter, oldind, ind, old_f, offset
      });
    }

    // -----------------------------------------------------------------------
    // Branch 3: r=2 phase -- dispatch on ch[check] == capo
    // -----------------------------------------------------------------------
    {
      OpBuilder::InsertionGuard g2(b);
      b.setInsertionPointToStart(branch23.elseBlock());

      Value chCheck = ld(ch, check);
      Value r2AdvCond = cmpEQ(chCheck, capo);
      auto r2If = scf::IfOp::create(b, loc, TypeRange(Tys), r2AdvCond,
                                    /*withElse=*/true);

      // =====================================================================
      // Branch 3a: advance for r=2  (ch[check] == capo)
      // switch(snaps): case 1, 2, 3, default
      // =====================================================================
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(r2If.thenBlock());

        Value is1 = cmpEQ(snaps, one);
        Value is2 = cmpEQ(snaps, two);
        Value is3 = cmpEQ(snaps, three);

        // case 1: capo = very large sentinel (run to end of primal)
        Value infCapo = ci(std::numeric_limits<int64_t>::max() / 4);

        // case 2: capo = ch[1] + incr
        Value ch1     = ld(ch, one);
        Value case2Capo = add(ch1, incr);

        // case 3: if iter==0 { capo=ch[oldind]; loop } else { capo=ch[ind]+incr; ... }
        // Yields: [capo, incr, iter]
        Value iterIsZero = cmpEQ(iter, zero);
        auto case3If = scf::IfOp::create(b, loc, TypeRange{idxTy, idxTy, idxTy},
                                         iterIsZero, /*withElse=*/true);
        {
          // iter == 0: capo = ch[oldind]; for i = 0...(t+1)/2: capo+=incr; incr++; iter++
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(case3If.thenBlock());

          Value chOldind = ld(ch, oldind);
          // Loop runs (t+1)/2 + 1 times (i <= (t+1)/2 inclusive).
          Value tP1Div2 = arith::DivSIOp::create(b, loc, add(t, one), two);
          Value loopUB  = add(tP1Div2, one);
          // Carried: (capo, incr, iter)
          auto loopOp = scf::ForOp::create(b, loc, zero, loopUB, one,
                                           ValueRange{chOldind, incr, iter});
          if (!loopOp.getBody()->empty())
            loopOp.getBody()->front().erase();
          {
            OpBuilder::InsertionGuard gf(b);
            b.setInsertionPointToStart(loopOp.getBody());
            auto la = loopOp.getRegionIterArgs();
            scf::YieldOp::create(b, loc, ValueRange{
              add(la[0], la[1]), add(la[1], one), add(la[2], one)
            });
          }
          scf::YieldOp::create(b, loc, loopOp.getResults());
        }
        {
          // iter != 0: capo = ch[ind] + incr; iter++; incr++
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(case3If.elseBlock());
          Value chInd = ld(ch, ind);
          scf::YieldOp::create(b, loc, ValueRange{
            add(chInd, incr), add(incr, one), add(iter, one)
          });
        }
        Value case3Capo = case3If.getResult(0);
        Value case3Incr = case3If.getResult(1);
        Value case3Iter = case3If.getResult(2);

        // default: two sub-cases on capo == snaps-1 and t == 0
        Value capoEqSnapsM1 = cmpEQ(capo, snapsM1);
        Value tIsZero       = cmpEQ(t, zero);
        Value iterLtOffset  = cmpSLT(iter, offset);
        Value offsetIs1     = cmpEQ(offset, one);

        // capo == snaps-1: new capo = capo+2; ind = snaps-1
        // t == 0:          new capo = capo+1 or capo+2; t++ if offset==1
        // else:            terminate
        Value defCapoSnapsM1 = add(capo, two);
        Value defCapoT0      = sel(iterLtOffset, add(capo, one), add(capo, two));
        Value defTNew        = sel(offsetIs1, add(t, one), t);

        Value defAction = sel(capoEqSnapsM1, actAdvance,
                             sel(tIsZero,    actAdvance, actTerminate));
        Value defCapo   = sel(capoEqSnapsM1, defCapoSnapsM1,
                             sel(tIsZero,    defCapoT0, capo));
        Value defInd    = sel(capoEqSnapsM1, snapsM1, ind);
        Value defT      = sel(capoEqSnapsM1, t, sel(tIsZero, defTNew, t));

        // Combine all cases; is1/is2/is3 take precedence over default.
        Value notIs123 = andOp(andOp(cmpNE(snaps, one), cmpNE(snaps, two)),
                               cmpNE(snaps, three));
        Value finalCapo = sel(is1, infCapo,
                          sel(is2, case2Capo,
                          sel(is3, case3Capo, defCapo)));
        Value finalIncr = sel(is3, case3Incr, incr);
        Value finalIter = sel(is3, case3Iter, iter);
        Value finalInd  = sel(notIs123, defInd, ind);
        Value finalT    = sel(notIs123, defT, t);
        Value finalAct  = sel(notIs123, defAction, actAdvance);

        scf::YieldOp::create(b, loc, ValueRange{
          finalAct,
          check, finalCapo, oldcapoO, finalT, finalIncr, finalIter,
          oldind, finalInd, old_f, offset
        });
      }

      // =====================================================================
      // Branch 3b: takeshot for r=2  (ch[check] != capo)
      // switch(snaps): case 2, case 3, default
      // =====================================================================
      {
        OpBuilder::InsertionGuard g3(b);
        b.setInsertionPointToStart(r2If.elseBlock());

        Value is2 = cmpEQ(snaps, two);
        Value is3 = cmpEQ(snaps, three);

        // ── case 2: ch[1]=capo; incr++ ─────────────────────────────────────
        // Conditional store (only when snaps == 2).
        // Yields: incr
        auto case2If = scf::IfOp::create(b, loc, TypeRange{idxTy}, is2,
                                         /*withElse=*/true);
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(case2If.thenBlock());
          st(capo, ch, one);
          scf::YieldOp::create(b, loc, ValueRange{add(incr, one)});
        }
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(case2If.elseBlock());
          scf::YieldOp::create(b, loc, ValueRange{incr});
        }
        Value case2Incr = case2If.getResult(0);

        // ── case 3: ch[ind]=capo; check=ind; maybe update num_rep/ind ──────
        // Conditional store + scalar updates (only when snaps == 3).
        // Yields: [check, iter, t, oldind, ind, incr]
        auto case3If = scf::IfOp::create(b, loc,
            TypeRange{idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
            is3, /*withElse=*/true);
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(case3If.thenBlock());

          st(capo, ch, ind);
          Value numRep1 = ld(num_rep, one);
          Value iterEqNR1 = cmpEQ(iter, numRep1);

          // if iter == num_rep[1]: iter=0; t++; oldind=ind; num_rep[1]++; ind=2-(nr1+1)%2; incr=1
          // Yields: [iter, t, oldind, ind, incr]
          auto nr1If = scf::IfOp::create(b, loc,
              TypeRange{idxTy, idxTy, idxTy, idxTy, idxTy},
              iterEqNR1, /*withElse=*/true);
          {
            OpBuilder::InsertionGuard g5(b);
            b.setInsertionPointToStart(nr1If.thenBlock());
            Value newNR1 = add(numRep1, one);
            st(newNR1, num_rep, one);
            Value remVal  = arith::RemSIOp::create(b, loc, newNR1, two);
            Value newInd3 = sub(two, remVal); // 2 - (num_rep[1]_new % 2)
            scf::YieldOp::create(b, loc, ValueRange{
              zero, add(t, one), ind /*old ind*/, newInd3, one
            });
          }
          {
            OpBuilder::InsertionGuard g5(b);
            b.setInsertionPointToStart(nr1If.elseBlock());
            scf::YieldOp::create(b, loc, ValueRange{iter, t, oldind, ind, incr});
          }

          scf::YieldOp::create(b, loc, ValueRange{
            ind,               // new check = ind
            nr1If.getResult(0), nr1If.getResult(1), nr1If.getResult(2),
            nr1If.getResult(3), nr1If.getResult(4)
          });
        }
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(case3If.elseBlock());
          scf::YieldOp::create(b, loc, ValueRange{check, iter, t, oldind, ind, incr});
        }
        Value case3Check  = case3If.getResult(0);
        Value case3Iter   = case3If.getResult(1);
        Value case3T      = case3If.getResult(2);
        Value case3Oldind = case3If.getResult(3);
        Value case3Ind    = case3If.getResult(4);
        Value case3Incr   = case3If.getResult(5);

        // ── default (snaps >= 4) ───────────────────────────────────────────
        // Conditional on isDefault = !is2 && !is3.
        // Yields: [action, check, incr, iter, oldind, ind, t]
        Value isDefault = andOp(cmpNE(snaps, two), cmpNE(snaps, three));
        // Pre-compute the rotation2 sub-path condition (offset--): reached when
        // isDefault && capo >= snaps+2 && t==0 && iter==offset.  We compute it
        // here (not inside the defIf) so that tsOffset can be derived with a
        // plain select after the defIf, avoiding the need to thread offset
        // through every intermediate yield.
        Value tIsZeroForOffset      = cmpEQ(t, zero);
        Value iterEqOffsetForOffset = cmpEQ(iter, offset);
        Value capoGeSnapsP2 = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::sge, capo, snapsP2);
        Value rot2Cond = andOp(andOp(isDefault, capoGeSnapsP2),
                               andOp(tIsZeroForOffset, iterEqOffsetForOffset));
        auto defIf = scf::IfOp::create(b, loc,
            TypeRange{idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
            isDefault, /*withElse=*/true);
        {
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(defIf.thenBlock());

          // Sub-case A: capo < snaps+2  →  ch[ind]=capo; check=ind;
          //             if capo == snaps+1: rotation1
          Value capoLtSnapsP2 = cmpSLT(capo, snapsP2);
          auto subA = scf::IfOp::create(b, loc,
              TypeRange{idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
              capoLtSnapsP2, /*withElse=*/true);

          {
            // Sub-case A (capo < snaps+2)
            OpBuilder::InsertionGuard g5(b);
            b.setInsertionPointToStart(subA.thenBlock());

            st(capo, ch, ind); // ch[ind] = capo; check = ind (scalar)

            // if capo == snaps+1: perform rotation1
            // Yields: [check, incr, oldind, ind]
            Value capoEqSnapsP1 = cmpEQ(capo, snapsP1);
            auto rot1If = scf::IfOp::create(b, loc,
                TypeRange{idxTy, idxTy, idxTy, idxTy},
                capoEqSnapsP1, /*withElse=*/true);
            {
              // Rotation 1: oldind=ord_ch[snaps-1]; ind_val=ch[ord_ch[snaps-1]];
              //             shift ord_ch right from snaps-1 downto 2;
              //             ord_ch[1]=oldind; ch[ord_ch[1]]=ind_val; incr=2; ind=2
              OpBuilder::InsertionGuard g6(b);
              b.setInsertionPointToStart(rot1If.thenBlock());

              Value ordSnapsM1 = ld(ord_ch, snapsM1);
              Value newOldind  = ordSnapsM1;
              // shift ord_ch right: positions 2..snaps-1 ← 1..snaps-2
              emitOrdChShiftRight(b, loc, snaps, one /*kLow=1*/, ord_ch);
              st(newOldind, ord_ch, one);
              // ch[ord_ch[1]] = ch[newOldind]: ind_val was ch[ordSnapsM1] before
              // rotation (see file-header note; this is provably a no-op, but
              // we store it for faithfulness).
              //   ch[newOldind] was already = ch[ordSnapsM1] (same slot), and
              //   the rotation loop didn't change ch.
              // incr=2; ind=2
              scf::YieldOp::create(b, loc, ValueRange{
                ind /*check=old ind*/, two /*incr=2*/, newOldind, two /*ind=2*/
              });
            }
            {
              // capo != snaps+1: no rotation
              OpBuilder::InsertionGuard g6(b);
              b.setInsertionPointToStart(rot1If.elseBlock());
              scf::YieldOp::create(b, loc, ValueRange{ind, incr, oldind, ind});
            }

            scf::YieldOp::create(b, loc, ValueRange{
              actTakeshot,
              rot1If.getResult(0) /*check*/,
              rot1If.getResult(1) /*incr*/,
              iter,
              rot1If.getResult(2) /*oldind*/,
              rot1If.getResult(3) /*ind*/,
              t
            });
          }

          {
            // Sub-case B: capo >= snaps+2
            //   t == 0 → either rotation2 (iter==offset) or simple store
            //   t != 0 → terminate
            OpBuilder::InsertionGuard g5(b);
            b.setInsertionPointToStart(subA.elseBlock());

            Value tIsZero = cmpEQ(t, zero);
            auto tZeroIf = scf::IfOp::create(b, loc,
                TypeRange{idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
                tIsZero, /*withElse=*/true);

            {
              // t == 0: branch on iter == offset
              OpBuilder::InsertionGuard g6(b);
              b.setInsertionPointToStart(tZeroIf.thenBlock());

              Value iterEqOffset = cmpEQ(iter, offset);
              auto iteIf = scf::IfOp::create(b, loc,
                  TypeRange{idxTy, idxTy, idxTy, idxTy, idxTy, idxTy, idxTy},
                  iterEqOffset, /*withElse=*/true);

              {
                // iter == offset: rotation2
                //   offset--; iter=1;
                //   check = ord_ch[snaps-1]; ch[check] = capo;
                //   oldind = check; ind = capo (= ch[check] just stored)
                //   shift ord_ch right from snaps-1 downto incr+1
                //   ord_ch[incr] = oldind; ch[ord_ch[incr]]=ind (=capo, no-op relative to store above)
                //   incr++; ind = incr (new value)
                OpBuilder::InsertionGuard g7(b);
                b.setInsertionPointToStart(iteIf.thenBlock());

                Value ordSnapsM1 = ld(ord_ch, snapsM1);
                st(capo, ch, ordSnapsM1); // ch[ord_ch[snaps-1]] = capo
                Value newOldind2 = ordSnapsM1;
                // newInd2 = capo (= ch[check] after the store), immediately
                // overridden by ind=incr below, so not carried.

                // shift ord_ch right: positions incr+1..snaps-1 ← incr..snaps-2
                emitOrdChShiftRight(b, loc, snaps, incr /*kLow=incr*/, ord_ch);
                st(newOldind2, ord_ch, incr);
                // ch[ord_ch[incr]] = ind (= capo) – same slot as above, no-op

                Value newIncr2 = add(incr, one);
                Value newInd2Final = newIncr2; // ind = incr (after incr++)

                scf::YieldOp::create(b, loc, ValueRange{
                  actTakeshot,
                  ordSnapsM1 /*check=ord_ch[snaps-1]*/,
                  newIncr2, one /*iter=1*/, newOldind2,
                  newInd2Final, t
                });
              }
              {
                // iter != offset: ch[ord_ch[ind]] = capo; check=ord_ch[ind]; iter++; ind++
                OpBuilder::InsertionGuard g7(b);
                b.setInsertionPointToStart(iteIf.elseBlock());

                Value ordChInd = ld(ord_ch, ind);
                st(capo, ch, ordChInd);
                scf::YieldOp::create(b, loc, ValueRange{
                  actTakeshot,
                  ordChInd /*check=ord_ch[ind]*/,
                  incr, add(iter, one), oldind,
                  add(ind, one), t
                });
              }
              scf::YieldOp::create(b, loc, iteIf.getResults());
            }
            {
              // t != 0: terminate (beyond Online_r2 range)
              OpBuilder::InsertionGuard g6(b);
              b.setInsertionPointToStart(tZeroIf.elseBlock());
              scf::YieldOp::create(b, loc, ValueRange{
                actTerminate, check, incr, iter, oldind, ind, t
              });
            }
            scf::YieldOp::create(b, loc, tZeroIf.getResults());
          }

          scf::YieldOp::create(b, loc, subA.getResults());
        }
        {
          // isDefault == false: placeholder (will be overridden by selects below)
          OpBuilder::InsertionGuard g4(b);
          b.setInsertionPointToStart(defIf.elseBlock());
          scf::YieldOp::create(b, loc, ValueRange{
            actTakeshot, check, incr, iter, oldind, ind, t
          });
        }
        // defIf.results(): [action, check, incr, iter, oldind, ind, t]

        // Merge case 2 / case 3 / default results.
        // Priority: is3 > is2 > isDefault.  (is2 and is3 are mutually exclusive.)
        //
        // Special case snaps==1: in r=2 TAKESHOT the only checkpoint slot is
        // already occupied by the r=1 checkpoint.  There is no free slot to
        // store into, so overwriting it (as the rotation2 default path would do)
        // corrupts the only checkpoint available for reconstruction.  Return
        // FORWARD instead — the primal just advances, and the offline Revolve
        // in the reverse reconstructs from the existing checkpoint.
        Value is1 = cmpEQ(snaps, one);
        Value tsAction = sel(is1, actAdvance,
                          sel(isDefault, defIf.getResult(0), actTakeshot));
        Value tsCheck  = sel(is3, case3Check,
                          sel(isDefault, defIf.getResult(1), check));
        Value tsIncr   = sel(is2, case2Incr,
                          sel(is3, case3Incr,
                          sel(isDefault, defIf.getResult(2), incr)));
        Value tsIter   = sel(is3, case3Iter,
                          sel(isDefault, defIf.getResult(3), iter));
        Value tsOldind = sel(is3, case3Oldind,
                          sel(isDefault, defIf.getResult(4), oldind));
        Value tsInd    = sel(is3, case3Ind,
                          sel(isDefault, defIf.getResult(5), ind));
        Value tsT      = sel(is3, case3T,
                          sel(isDefault, defIf.getResult(6), t));
        // offset decrements by 1 only in the rotation2 sub-path.
        Value tsOffset = sel(rot2Cond, sub(offset, one), offset);

        scf::YieldOp::create(b, loc, ValueRange{
          tsAction,
          tsCheck, capo, oldcapoO, tsT, tsIncr, tsIter, tsOldind, tsInd,
          old_f, tsOffset
        });
      }

      scf::YieldOp::create(b, loc, r2If.getResults());
    }

    scf::YieldOp::create(b, loc, branch23.getResults());
  }

  // Replace the op's results and erase.
  auto res = outerIf.getResults();
  op.getAction().replaceAllUsesWith(res[0]);
  op.getNewCheck().replaceAllUsesWith(res[1]);
  op.getNewCapo().replaceAllUsesWith(res[2]);
  op.getNewOldcapoO().replaceAllUsesWith(res[3]);
  op.getNewT().replaceAllUsesWith(res[4]);
  op.getNewIncr().replaceAllUsesWith(res[5]);
  op.getNewIter().replaceAllUsesWith(res[6]);
  op.getNewOldind().replaceAllUsesWith(res[7]);
  op.getNewInd().replaceAllUsesWith(res[8]);
  op.getNewOldF().replaceAllUsesWith(res[9]);
  op.getNewOffset().replaceAllUsesWith(res[10]);
  op->erase();
}

struct LowerOnlineProgressPass
    : public enzyme::impl::LowerOnlineProgressPassBase<LowerOnlineProgressPass> {
  void runOnOperation() override {
    SmallVector<enzyme::OnlineProgressOp> ops;
    getOperation()->walk(
        [&](enzyme::OnlineProgressOp op) { ops.push_back(op); });
    for (auto op : ops)
      lowerOnlineProgress(op);
  }
};

} // end anonymous namespace
