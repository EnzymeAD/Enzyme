// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Binomial checkpointing with a dynamic (runtime) upper bound. The trip count is
// computed at runtime as (ub - lb) / step; the checkpoint buffers stay statically
// sized by the checkpoint budget (3), independent of the trip count.

module {
  func.func @main(%arg0: f32, %ub: index) -> (f32) {
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %sum = scf.for %iv = %lb to %ub step %step iter_args(%s = %arg0) -> (f32) {
      %sq = arith.mulf %s, %s : f32
      %c = math.cos %sq : f32
      scf.yield %c : f32
    } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64}
    return %sum : f32
  }
}

// CHECK-LABEL: func.func @main(

// Statically-sized checkpoint buffers (budget = 3) even though the trip count is dynamic.
// CHECK-DAG:     %[[STATE:.+]] = memref.alloc() : memref<3xf32>
// CHECK-DAG:     %[[IDX:.+]] = memref.alloc() : memref<3xindex>

// Trip count is computed at runtime (not a compile-time constant).
// CHECK:         arith.divui

// The remat scf.while and the differentiated body (sin) appear in the reverse pass.
// CHECK:         scf.while
// CHECK:         math.sin

// All checkpoint buffers are freed.
// CHECK-DAG:     memref.dealloc %[[STATE]]
// CHECK-DAG:     memref.dealloc %[[IDX]]
// CHECK:         return
