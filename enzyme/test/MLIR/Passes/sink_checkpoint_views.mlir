// RUN: %eopt %s --sink-checkpoint-views | FileCheck %s

// The loop captures both %buf and a subview of it. Checkpointing would clone
// both -- two snapshots of the same memory -- so the view has to move inside.

func.func @sinks_view(%buf: memref<64xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  %view = memref.subview %buf[0] [32] [1] : memref<64xf32> to memref<32xf32>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %cst, %view[%i] : memref<32xf32>
    memref.store %cst, %buf[%i] : memref<64xf32>
  } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing}
  return
}

// CHECK-LABEL: func.func @sinks_view
// CHECK:         scf.for
// CHECK-NEXT:      %[[VIEW:.+]] = memref.subview
// CHECK:           memref.store %{{.+}}, %[[VIEW]]

// -----

// Without the checkpointing attribute nothing is snapshotted, so there is
// nothing to fix and the view stays where it is (sinking it would just be
// unrequested code motion).

func.func @leaves_plain_loop(%buf: memref<64xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  %view = memref.subview %buf[0] [32] [1] : memref<64xf32> to memref<32xf32>
  scf.for %i = %c0 to %n step %c1 {
    memref.store %cst, %view[%i] : memref<32xf32>
    memref.store %cst, %buf[%i] : memref<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @leaves_plain_loop
// CHECK:         %[[VIEW:.+]] = memref.subview
// CHECK:         scf.for
// CHECK-NEXT:      memref.store %{{.+}}, %[[VIEW]]
