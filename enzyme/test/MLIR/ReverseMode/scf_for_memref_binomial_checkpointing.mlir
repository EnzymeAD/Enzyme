// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math | FileCheck %s

// Binomial checkpointing with a mutable (memref) value referenced from outside
// the loop. The reference gets a budget-sized buffer of clone handles, filled up
// front so a checkpoint is a plain copy into an existing allocation, plus one
// working clone the reverse pass replays into. Slot j pairs with the state and
// step-index buffers' slot j, so all three are indexed by the checkpoint stack
// pointer -- never by the reverse induction variable, which is what regressed
// here (the reverse read used to be `numIters-1 - iv` into a budget-sized
// buffer, i.e. out of bounds for every iteration past the budget).

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
// CHECK-DAG:     %[[SLOTS:.+]] = memref.alloc() : memref<3xmemref<f32>>

// One clone per slot, allocated up front and stored into the handle buffer.
// CHECK:         %[[C0:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %arg0, %[[C0]]
// CHECK-NEXT:    memref.store %[[C0]], %[[SLOTS]][%c0]
// CHECK:         %[[C1:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %arg0, %[[C1]]
// CHECK-NEXT:    memref.store %[[C1]], %[[SLOTS]][%c1]
// CHECK:         %[[C2:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %arg0, %[[C2]]
// CHECK-NEXT:    memref.store %[[C2]], %[[SLOTS]][%c2]

// Forward checkpoint-placement loop (budget = 3): slot index is the loop IV.
// CHECK:         scf.for %[[K:.+]] = %c0 to %c3 step %c1
// CHECK:           memref.store {{.*}}, %[[STATE]][%[[K]]]
// CHECK:           memref.store {{.*}}, %[[IDX]][%[[K]]]
// CHECK:           %[[FWDSLOT:.+]] = memref.load %[[SLOTS]][%[[K]]] : memref<3xmemref<f32>>
// CHECK-NEXT:      memref.copy %arg0, %[[FWDSLOT]] : memref<f32> to memref<f32>

// The working clone the reverse pass replays into, outside the reverse loop.
// CHECK:         %[[WORK:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %arg0, %[[WORK]]

// Reverse loop over all 9 steps, carrying the stack pointer as an iter arg.
// CHECK:         scf.for %{{.+}} = %c0 to %c9 step %c1 iter_args(%[[SP:.+]] = %c3
// The slot index is the stack pointer minus one -- NOT a function of the
// reverse induction variable. That is the regression this test guards.
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %c1 : index
// CHECK:           memref.load %[[STATE]][%[[CAPO]]]
// CHECK:           memref.load %[[IDX]][%[[CAPO]]]
// CHECK:           %[[REVSLOT:.+]] = memref.load %[[SLOTS]][%[[CAPO]]] : memref<3xmemref<f32>>
// CHECK-NEXT:      memref.copy %[[REVSLOT]], %[[WORK]] : memref<f32> to memref<f32>

// The remat re-places a checkpoint at slot `acapo`; the snapshot moves with it.
// CHECK:           scf.while ({{.*}}%[[ACAPO:.+]] = %[[CAPO]]
// CHECK:             memref.store {{.*}}, %[[IDX]][%[[ACAPO]]]
// CHECK:             %[[ACSLOT:.+]] = memref.load %[[SLOTS]][%[[ACAPO]]] : memref<3xmemref<f32>>
// CHECK-NEXT:        memref.copy %[[WORK]], %[[ACSLOT]] : memref<f32> to memref<f32>
// The replay reads the working clone, never a checkpoint slot.
// CHECK:             memref.load %[[WORK]][] : memref<f32>

// Everything is freed: the state/index buffers, the working clone, then each
// slot's clone followed by the handle buffer.
// CHECK:         memref.dealloc %[[STATE]] : memref<3xf32>
// CHECK:         memref.dealloc %[[IDX]] : memref<3xindex>
// CHECK:         memref.dealloc %[[WORK]] : memref<f32>
// CHECK:         scf.for %[[J:.+]] = %c0 to %c3 step %c1 {
// CHECK-NEXT:      %[[FREESLOT:.+]] = memref.load %[[SLOTS]][%[[J]]] : memref<3xmemref<f32>>
// CHECK-NEXT:      memref.dealloc %[[FREESLOT]] : memref<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[SLOTS]] : memref<3xmemref<f32>>
// CHECK:         return
