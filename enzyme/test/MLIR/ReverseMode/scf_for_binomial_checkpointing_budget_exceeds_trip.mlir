// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s

// A budget larger than the trip count. With enzyme.use_safe_budgeting the
// *effective* budget is min(budget, numIters) = 3, so only slots [0, 3) are ever
// written, but the buffers are still sized by the static budget (8). Clone slots
// are allocated eagerly for all 8 and the teardown loop frees all 8 -- slots
// [3, 8) are allocated and freed without ever holding a snapshot.
//
// This pins the eager-allocation choice: a lazily-filled buffer would need a
// null sentinel per slot, which does not exist for a memref element type.

module {
  func.func @main(%m: memref<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %init = arith.constant 0.0 : f32

    %0 = scf.for %i = %c0 to %c3 step %c1 iter_args(%it = %init) -> (f32) {
      %val = memref.load %m[] : memref<f32>
      %mul = arith.mulf %val, %it : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 8 : i64,
       enzyme.use_safe_budgeting}

    return %0 : f32
  }
}

// CHECK-LABEL: func.func @main(
// Buffers are sized by the static budget, not the effective one.
// CHECK-DAG:     memref.alloc() : memref<8xf32>
// CHECK-DAG:     memref.alloc() : memref<8xindex>
// CHECK-DAG:     %[[SLOTS:.+]] = memref.alloc() : memref<8xmemref<f32>>

// All 8 slots get a clone, including the ones beyond the effective budget.
// CHECK-COUNT-8: memref.store %{{.+}}, %[[SLOTS]][%c{{[0-9]+}}] : memref<8xmemref<f32>>

// The forward placement loop is bounded by the effective budget (3), so only
// slots [0, 3) are ever snapshotted.
// CHECK:         scf.for %[[K:.+]] = %c0 to %c3 step %c1
// CHECK:           %[[FWDSLOT:.+]] = memref.load %[[SLOTS]][%[[K]]] : memref<8xmemref<f32>>
// CHECK-NEXT:      memref.copy %arg0, %[[FWDSLOT]]

// Reverse reads are still indexed by the stack pointer.
// CHECK:         scf.for %{{.+}} = %c0 to %c3 step %c1 iter_args(%[[SP:.+]] = %c3
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %c1 : index
// CHECK:           memref.load %[[SLOTS]][%[[CAPO]]] : memref<8xmemref<f32>>

// Teardown covers the whole buffer, so the unused slots are freed too.
// CHECK:         scf.for %[[J:.+]] = %c0 to %c8 step %c1 {
// CHECK-NEXT:      %[[FREESLOT:.+]] = memref.load %[[SLOTS]][%[[J]]] : memref<8xmemref<f32>>
// CHECK-NEXT:      memref.dealloc %[[FREESLOT]] : memref<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[SLOTS]] : memref<8xmemref<f32>>
