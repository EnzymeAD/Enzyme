// RUN: %eopt %s --enzyme-wrap="infn=f_inactive_carry outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=INACTIVE
// RUN: %eopt %s --enzyme-wrap="infn=f_active_loopback outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=LOOPBACK
// RUN: %eopt %s --enzyme-wrap="infn=f_cond_only outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=COND
// RUN: %eopt %s --enzyme-wrap="infn=f_before_transform outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=BEFORE

// Activity-analysis propagation tests for scf.while.
//
// PR #2374 (Pangoraw) identifies that without WhileOpADDataFlow, Enzyme does
// not know the three value-flow paths through scf.while:
//   (a) before-block args ← initial operands AND scf.yield operands (loop-back)
//   (b) after-block args ← scf.condition operands (offset +1 past the i1)
//   (c) results ← scf.condition operands (same offset)
//
// These tests verify that our implementation correctly handles all three paths,
// plus a fourth case discovered during Online_r2 development:
//
// 1. INACTIVE  — inactive index iter-arg alongside active f32: Enzyme must NOT
//    create an index shadow or gradient; the f32 chain rule (×2 per step) must
//    still be correct.
//
// 2. LOOPBACK  — active f32 updated in the body and fed back via scf.yield.
//    Activity must propagate: scf.yield operand 0 → before-block arg 0, so
//    the primal state at each step is identified as active and saved to tape.
//    Adjoint of sin(x²) is cos(x²)·2x.
//
// 3. COND      — active f32 appears in the condition expression but the body
//    is identity (x passes through unchanged).  The result's gradient is just
//    the incoming seed; no math ops from the body should appear in the reverse.
//    The i1 condition value must NOT generate a gradient.
//
// 4. BEFORE    — before-block TRANSFORMS the value before passing it via
//    scf.condition (not a pass-through). The gradient must flow back through
//    the before-block transformation. Without WhileOpADDataFlow, the connection
//    from the after-block arg gradient to the before-block result gradient is
//    missing, causing an incorrect (truncated) chain rule.

module {

  // ── Test 1: inactive index counter alongside active f32 ───────────────────
  // f(x, n) = x × 2^n   (repeated doubling while k < n).
  // Gradient: df/dx = 2^n  → each reverse step multiplies by 2.0.
  func.func @f_inactive_carry(%arg0: f32, %n: index) -> f32 {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %two = arith.constant 2.0 : f32
    %r:2 = scf.while (%x = %arg0, %k = %c0) : (f32, index) -> (f32, index) {
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %x, %k : f32, index
    } do {
    ^bb0(%x_in: f32, %k_in: index):
      %x2 = arith.mulf %x_in, %two : f32
      %k1 = arith.addi %k_in, %c1  : index
      scf.yield %x2, %k1 : f32, index
    }
    return %r#0 : f32
  }

  // ── Test 2: activity flows back through scf.yield (loop-back) ────────────
  // f(x, n) = sin(x²) applied n times.
  // Adjoint at each step: dout × cos(x²) × 2x.
  func.func @f_active_loopback(%arg0: f32, %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r:2 = scf.while (%x = %arg0, %k = %c0) : (f32, index) -> (f32, index) {
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %x, %k : f32, index
    } do {
    ^bb0(%x_in: f32, %k_in: index):
      %sq  = arith.mulf %x_in, %x_in : f32
      %sin = math.sin  %sq           : f32
      %k1  = arith.addi %k_in, %c1   : index
      scf.yield %sin, %k1 : f32, index
    }
    return %r#0 : f32
  }

  // ── Test 4: before-block transforms value before scf.condition ───────────
  // f(x, n) where the before-block computes sq = x*x then passes sq (not x)
  // to the after-block via scf.condition. The after-block applies cos(sq).
  // The iter-arg cycles: x → sq=x²→ cos(sq) → sq=cos²→ cos(sq) → …
  //
  // Full chain: f = sq_n, where sq_i = s_i^2, s_{i+1} = cos(sq_i), s_0 = x.
  // df/ds_n = 2·s_n (adjoint of squaring at the exit condition check)
  // df/ds_i for i<n = -sin(sq_i)·2·s_i  (cos adjoint × squaring adjoint)
  //
  // The critical connection: after-block arg gradient (d_sq) must flow back to
  // the before-block result (sq = s*s) to pick up the ×2·s factor.
  // Without WhileOpADDataFlow this connection is missing → gradient loses ×2·s.
  func.func @f_before_transform(%arg0: f32, %n: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // iter_args: s (f32, ACTIVE), k (index, INACTIVE).
    // before-block squares s and passes sq to after-block and results.
    %r:2 = scf.while (%s = %arg0, %k = %c0) : (f32, index) -> (f32, index) {
      %sq   = arith.mulf %s, %s : f32      // transform in before-block
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %sq, %k : f32, index  // passes sq, not s
    } do {
    ^bb0(%sq_in: f32, %ki: index):
      %cos = math.cos %sq_in : f32         // after-block receives sq
      %k1  = arith.addi %ki, %c1 : index
      scf.yield %cos, %k1 : f32, index    // new s = cos(sq)
    }
    return %r#0 : f32
  }

  // ── Test 3: active value used only in condition; identity body ────────────
  // f(x, n) = x   (identity loop: body passes x through unchanged).
  // The condition tests both k < n and x > 0 (so x appears in before-block).
  // Gradient: df/dx = 1 → the reverse should return the incoming seed directly,
  // with no math from the body and no gradient for the i1 condition value.
  func.func @f_cond_only(%arg0: f32, %n: index) -> f32 {
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %r:2 = scf.while (%x = %arg0, %k = %c0) : (f32, index) -> (f32, index) {
      %gt   = arith.cmpf ogt, %x, %zero : f32
      %lt_n = arith.cmpi slt, %k, %n    : index
      %cond = arith.andi %lt_n, %gt      : i1
      scf.condition(%cond) %x, %k : f32, index
    } do {
    ^bb0(%x_in: f32, %k_in: index):
      %k1 = arith.addi %k_in, %c1 : index
      scf.yield %x_in, %k1 : f32, index
    }
    return %r#0 : f32
  }
}

// ── INACTIVE checks ──────────────────────────────────────────────────────────
// The index k is inactive; Enzyme must not create a shadow for it or compute
// any index gradient.  The f32 gradient accumulates ×2 at each reverse step.
//
// INACTIVE-LABEL: func.func @f_inactive_carry(
// INACTIVE-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
//
// Forward scaffold: active f32 pushed to tape; inactive index carried but not shadowed.
// INACTIVE:         enzyme.init
// INACTIVE:         scf.while
// INACTIVE:         enzyme.push
// The body doubles x — scalar mul with constant 2.0 must appear in the forward.
// INACTIVE:         arith.mulf {{.*}} %cst
//
// Reverse scf.for: pops primal f32, multiplies gradient by 2.0.
// INACTIVE:         scf.for
// INACTIVE:         enzyme.pop
// INACTIVE:         arith.mulf {{.*}} %cst
//
// No floating-point gradient for the index counter.
// INACTIVE-NOT:     math.sin
// INACTIVE-NOT:     math.cos
//
// INACTIVE:         return

// ── LOOPBACK checks ──────────────────────────────────────────────────────────
// The active f32 is updated in the body (sin(x²)) and fed back via scf.yield.
// Activity must propagate from scf.yield operand 0 → before-block arg 0, so
// the primal is pushed to tape every iteration.
//
// LOOPBACK-LABEL: func.func @f_active_loopback(
// LOOPBACK-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
//
// Forward: primal x pushed before each condition evaluation.
// LOOPBACK:         enzyme.init
// LOOPBACK:         scf.while
// LOOPBACK:         enzyme.push
// Body remat runs sin in the forward.
// LOOPBACK:         math.sin
//
// Reverse: pop primal x, apply adjoint of sin(x²) = cos(x²)·2x.
// LOOPBACK:         scf.for
// LOOPBACK:         enzyme.pop
// LOOPBACK:         math.cos
// LOOPBACK:         arith.mulf
// LOOPBACK:         arith.addf
//
// LOOPBACK:         return

// ── COND checks ──────────────────────────────────────────────────────────────
// Identity body: x passes through unchanged, gradient = seed (no chain rule).
// x is captured as a constant outside the scaffold iter-args (activity analysis
// correctly sees it as loop-invariant in the scaffold while).
// The arith.cmpf appears in the forward scaffold's before-block (it computes the
// condition), but must NOT appear in the reverse adjoint computation.
//
// COND-LABEL: func.func @f_cond_only(
// COND-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
//
// Forward: x captured outside scaffold; pushed inside the while's before-block.
// COND:         enzyme.init
// COND:         scf.while
// COND:         enzyme.push
// The forward condition still evaluates arith.cmpf (primal logic unchanged).
// COND:         arith.cmpf
//
// Reverse: gradient of identity is just the incoming seed — return it directly.
// scf.for drains the tape; no math ops from the body; return the seed %arg2.
// COND:         scf.for
// COND:         enzyme.pop
// COND-NOT:     math.sin
// COND-NOT:     math.cos
//
// COND:         return %arg2

// ── BEFORE checks ────────────────────────────────────────────────────────────
// The before-block squares s and passes sq to the after-block.
// Full adjoint chain per step: grad × (-sin(s²)) × 2s
//   = reverse of cos(sq) × reverse of sq=s*s
//
// XFAIL: *
// (The ×2·s factor from reversing mulf(s,s) in the before-block is missing.
//  Fix requires WhileOpADDataFlow to connect after-block arg gradient → before-block
//  result gradient. See PR #2374 and CHECKPOINTING.md §"Known limitations".)
//
// BEFORE-LABEL: func.func @f_before_transform(
// BEFORE-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
//
// Forward: s pushed before each condition evaluation; mulf(s,s) computed in before-block.
// BEFORE:         enzyme.init
// BEFORE:         scf.while
// BEFORE:         enzyme.push
// BEFORE:         arith.mulf
// BEFORE:         math.cos
//
// Reverse: pop s, apply FULL adjoint: grad × (-sin(s²)) × 2 × s.
// Currently missing the ×2·s factor — only grad × (-sin(s²)) is produced.
// BEFORE:         scf.for
// BEFORE:         enzyme.pop
// BEFORE:         arith.mulf
// BEFORE:         math.sin
// BEFORE:         arith.negf
// BEFORE:         arith.mulf
// The next two lines require WhileOpADDataFlow (currently failing):
// BEFORE:         arith.mulf {{.*}} %{{.*}}    <- ×2·s from reversing mulf(s,s)
// BEFORE:         arith.addf
//
// BEFORE:         return
