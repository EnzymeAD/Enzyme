// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: f32) -> (f32) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %step = arith.constant 1 : index

    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %arg0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %sum_iter : f32
      %cos_next = math.cos %sum_next : f32
      scf.yield %cos_next : f32
    } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64}

    return %sum : f32
  }
}

// Binomial checkpointing allocates one memref buffer for the loop-carried state
// (budget = 3 snapshots) plus one memref for the recorded checkpoint steps, both
// deallocated after the reverse loop.

// CHECK-LABEL: func.func @main(
// CHECK-DAG:     %[[STATE:.+]] = memref.alloc() : memref<3xf32>
// CHECK-DAG:     %[[IDX:.+]] = memref.alloc() : memref<3xindex>

// Forward checkpoint-placement loop: budget (3) iterations, snapshotting state.
// CHECK:         scf.for %{{.*}} = %c0 to %c3 step %c1 iter_args(%[[STEP:.+]] = %c0, %[[S:.+]] = %arg0)
// CHECK:           memref.store %[[S]], %[[STATE]]
// CHECK:           memref.store %[[STEP]], %[[IDX]]
// CHECK:           scf.for {{.*}} iter_args
// CHECK:             arith.mulf
// CHECK:             math.cos

// Reverse loop over all 9 steps, with the remat scf.while reconstructing state.
// CHECK:         scf.for %{{.*}} = %c0 to %c9 step %c1
// CHECK:           memref.load %[[STATE]]
// CHECK:           memref.load %[[IDX]]
// CHECK:           scf.while
// CHECK:             memref.store {{.*}}, %[[STATE]]
// CHECK:             scf.for
// CHECK:           math.sin

// CHECK-DAG:     memref.dealloc %[[STATE]]
// CHECK-DAG:     memref.dealloc %[[IDX]]
// CHECK:         return
