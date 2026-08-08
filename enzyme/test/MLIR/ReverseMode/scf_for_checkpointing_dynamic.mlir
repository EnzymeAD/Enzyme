// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Periodic checkpointing with a dynamic (runtime) upper bound. The period has to
// be stated -- there is no compile-time trip count to take the square root of --
// and the segment count is then ceil(numIters / period), computed at runtime.
// Each segment's length is clamped to min(period, numIters - segmentBase), since
// whether the last one is short cannot be decided here.

module {
  func.func @main(%arg0: f32, %ub: index) -> (f32) {
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %sum = scf.for %iv = %lb to %ub step %step iter_args(%s = %arg0) -> (f32) {
      %sq = arith.mulf %s, %s : f32
      %c = math.cos %sq : f32
      scf.yield %c : f32
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64}
    return %sum : f32
  }
}

// CHECK-LABEL: func.func @main(

// Trip count and segment count are computed at runtime, not folded.
// CHECK:         %[[NITERS:.+]] = arith.divui
// CHECK:         arith.divui

// Forward: one outer segment loop stepping by the period, with the inner
// recompute loop bounded by the runtime clamp.
// CHECK:         scf.for
// CHECK:           arith.minui
// CHECK:           scf.for

// Reverse: the segments are replayed back to front, again with the clamp, and
// the body's adjoint (sin) shows up inside.
// CHECK:         scf.for
// CHECK:           arith.minui
// CHECK:           scf.for
// CHECK:           scf.for
// CHECK:             math.sin
// CHECK:         return
