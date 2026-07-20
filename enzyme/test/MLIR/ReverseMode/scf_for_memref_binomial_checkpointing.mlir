// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math | FileCheck %s

// Binomial checkpointing with a mutable (memref) value referenced from outside
// the loop: the reference is cloned once via the ClonableTypeInterface and freed
// after the reverse loop, alongside the checkpoint-state and step-index buffers.

module {

  func.func @main(%m: memref<f32>) -> f32 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %one = arith.constant 1 : index

    %init = arith.constant 0.0 : f32
    %0 = scf.for %i = %lb to %ub step %one iter_args(%it = %init) -> (f32) {
      %val = memref.load %m[] : memref<f32>
      %mul = arith.mulf %val, %it : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64}

    return %0 : f32
  }

}

// CHECK-LABEL: func.func @main(
// CHECK-DAG:     %[[STATE:.+]] = memref.alloc() : memref<3xf32>
// CHECK-DAG:     %[[IDX:.+]] = memref.alloc() : memref<3xindex>

// Forward checkpoint-placement loop (budget = 3).
// CHECK:         scf.for {{.*}} = %c0 to %c3 step %c1
// CHECK:           memref.store {{.*}}, %[[STATE]]
// CHECK:           memref.store {{.*}}, %[[IDX]]

// The mutable outside reference is snapshotted (cloned) for the reverse pass.
// CHECK:         %[[CLONE:.+]] = memref.alloc() : memref<f32>
// CHECK:         memref.copy %arg0, %[[CLONE]]

// Reverse loop over all 9 steps, with the remat scf.while.
// CHECK:         scf.for {{.*}} = %c0 to %c9 step %c1
// CHECK:           scf.while

// All allocations are freed.
// CHECK-DAG:     memref.dealloc %[[STATE]]
// CHECK-DAG:     memref.dealloc %[[IDX]]
// CHECK-DAG:     memref.dealloc %[[CLONE]]
// CHECK:         return
