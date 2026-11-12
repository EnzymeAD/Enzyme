// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Full-tape (no checkpointing) differentiation of an scf.while loop whose
// trip count is unknown at compile time.  Enzyme must discover the trip count
// during the forward pass by carrying a counter, push each iteration's
// loop-carried f32 state to a dynamic tape, and then replay the adjoint in a
// reverse scf.for loop that pops from the tape.
//
// f(x, n) = cos(x*x) applied n times (n is enzyme_const, only x is active).
// df/dx involves -sin(x*x)*2x chained across all n iterations.

module {
  func.func @main(%arg0: f32, %n: index) -> f32 {
    %step = arith.constant 1 : index
    %lb   = arith.constant 0 : index
    %result:2 = scf.while (%s = %arg0, %k = %lb) : (f32, index) -> (f32, index) {
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %s, %k : f32, index
    } do {
    ^bb0(%s_in: f32, %k_in: index):
      %sq  = arith.mulf %s_in, %s_in : f32
      %cos = math.cos  %sq           : f32
      %k1  = arith.addi %k_in, %step : index
      scf.yield %cos, %k1 : f32, index
    }
    return %result#0 : f32
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32

// ── Forward pass ─────────────────────────────────────────────────────────────
// Dynamic tape: the before-block f32 state is pushed to a Cache at each step.
// CHECK:         !enzyme.Cache<f32>

// Scaffold scf.while carries an extra numIters counter (index).
// The primal body (mulf + cos) runs inside the after block.
// CHECK:         scf.while
// CHECK:         arith.mulf
// CHECK:         math.cos

// ── Reverse pass ─────────────────────────────────────────────────────────────
// Reverse scf.for iterates numIters times, popping primal state at each step.
// CHECK:         scf.for

// Each reverse iteration pops one primal f32 from the tape.
// CHECK:         enzyme.pop

// Adjoint of cos(x*x): -sin(x*x)*2x — chained across all iterations.
// CHECK:         math.sin
// CHECK:         arith.negf

// CHECK:         return
