// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// A loop with a runtime schedule that writes memory it reads: each snapshot
// also holds a clone of the memref, in a buffer with a row per slot, and the
// reverse sweep replays into a working clone of its own, so the memory the
// forward pass left is not overwritten.

module {
  func.func @main(%m: memref<10xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = arith.constant 0.0 : f64
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> (f64) {
      %v = memref.load %m[%i] : memref<10xf64>
      %a = arith.addf %acc, %v : f64
      memref.store %a, %m[%c0] : memref<10xf64>
      scf.yield %a : f64
    } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 4 : i64, enzyme.checkpoint_runtime,
       enzyme.disable_mincut = true}
    return %r : f64
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[M:.+]]: memref<10xf64>, %[[DM:.+]]: memref<10xf64>, %{{.+}}: f64)
// CHECK:         %[[H:.+]] = call @__enzyme_ckpt_schedule_begin(
// CHECK:         %[[ROWS:.+]] = arith.addi
// CHECK:         %[[BUF:.+]] = memref.alloc(%[[ROWS]]) : memref<?xf64>
// CHECK:         %[[MBUF:.+]] = memref.alloc(%[[ROWS]]) : memref<?x10xf64>

// The forward sweep snapshots and restores the memref itself.
// CHECK:         scf.while
// CHECK:           scf.index_switch
// CHECK:           case 1 {
// CHECK:             %[[ROW:.+]] = memref.subview %[[MBUF]]
// CHECK:             memref.copy %[[M]], %[[ROW]]
// CHECK:           case 2 {
// CHECK:             %[[ROW:.+]] = memref.subview %[[MBUF]]
// CHECK:             memref.copy %[[ROW]], %[[M]]
// CHECK:           case 3 {
// CHECK:             scf.for
// CHECK:               memref.store %{{.+}}, %[[M]][

// The reverse sweep starts its working clone from the extra row.
// CHECK:         %[[WORK:.+]] = memref.alloc() : memref<10xf64>
// CHECK:         memref.copy %[[M]], %[[WORK]]
// CHECK:         %[[ROW:.+]] = memref.subview %[[MBUF]]
// CHECK:         memref.copy %[[ROW]], %[[WORK]]
// CHECK:         scf.while
// CHECK:           scf.index_switch
// CHECK:           case 1 {
// CHECK:             %[[ROW:.+]] = memref.subview %[[MBUF]]
// CHECK:             memref.copy %[[WORK]], %[[ROW]]
// CHECK:           case 2 {
// CHECK:             %[[ROW:.+]] = memref.subview %[[MBUF]]
// CHECK:             memref.copy %[[ROW]], %[[WORK]]
// CHECK:           case 3 {
// CHECK:             scf.for
// CHECK:               memref.store %{{.+}}, %[[WORK]][
// CHECK:           case 5 {
// CHECK:             memref.store %{{.+}}, %[[WORK]][
// CHECK:             memref.load %[[DM]]
// CHECK:         memref.dealloc %[[BUF]]
// CHECK:         memref.dealloc %[[WORK]]
// CHECK:         memref.dealloc %[[MBUF]]
// CHECK:         call @__enzyme_ckpt_schedule_end(%[[H]])
