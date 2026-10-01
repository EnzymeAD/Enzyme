// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s

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
       enzyme.checkpoint_period = 8 : i64}

    return %0 : f32
  }
}

// CHECK-LABEL: func.func @main(
// CHECK:         memref.alloc() : memref<8xf32>
// CHECK-NEXT:    memref.alloc() : memref<8xindex>
// CHECK-NEXT:    %[[SLOTS:.+]] = memref.alloc() : memref<8xf32>

// CHECK:         scf.for %[[K:.+]] = %c0 to %c3 step %c1
// CHECK:           %[[FWDSLOT:.+]] = memref.subview %[[SLOTS]][%[[K]]] [1] [1] : memref<8xf32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %[[FWDSLOT]]

// CHECK:         scf.for %{{.+}} = %c0 to %c3 step %c1 iter_args(%[[SP:.+]] = %c3
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %c1 : index
// CHECK:           memref.subview %[[SLOTS]][%[[CAPO]]] [1] [1] : memref<8xf32> to memref<f32, strided<[], offset: ?>>

// CHECK:         memref.dealloc %[[SLOTS]] : memref<8xf32>
