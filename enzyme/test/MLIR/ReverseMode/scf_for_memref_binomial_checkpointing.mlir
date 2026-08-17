// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math | FileCheck %s

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

// CHECK:         %[[STATE:.+]] = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %[[IDX:.+]] = memref.alloc() : memref<3xindex>
// CHECK-NEXT:    %[[SLOTS:.+]] = memref.alloc() : memref<3xf32>

// CHECK:         scf.for %[[K:.+]] = %c0 to %c3 step %c1
// CHECK:           memref.store {{.*}}, %[[STATE]][%[[K]]]
// CHECK:           memref.store {{.*}}, %[[IDX]][%[[K]]]
// CHECK:           %[[FWDSLOT:.+]] = memref.subview %[[SLOTS]][%[[K]]] [1] [1] : memref<3xf32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %[[FWDSLOT]] : memref<f32> to memref<f32, strided<[], offset: ?>>

// CHECK:         %[[WORK:.+]] = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %arg0, %[[WORK]]

// CHECK:         scf.for %{{.+}} = %c0 to %c9 step %c1 iter_args(%[[SP:.+]] = %c3
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %c1 : index
// CHECK:           memref.load %[[STATE]][%[[CAPO]]]
// CHECK:           memref.load %[[IDX]][%[[CAPO]]]
// CHECK:           %[[REVSLOT:.+]] = memref.subview %[[SLOTS]][%[[CAPO]]] [1] [1] : memref<3xf32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:      memref.copy %[[REVSLOT]], %[[WORK]] : memref<f32, strided<[], offset: ?>> to memref<f32>

// CHECK:           scf.while ({{.*}}%[[ACAPO:.+]] = %[[CAPO]]
// CHECK:             memref.store {{.*}}, %[[IDX]][%[[ACAPO]]]
// CHECK:             %[[ACSLOT:.+]] = memref.subview %[[SLOTS]][%[[ACAPO]]] [1] [1] : memref<3xf32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:        memref.copy %[[WORK]], %[[ACSLOT]] : memref<f32> to memref<f32, strided<[], offset: ?>>
// The replay reads the working clone, never a checkpoint slot.
// CHECK:             memref.load %[[WORK]][] : memref<f32>

// CHECK:         memref.dealloc %[[STATE]] : memref<3xf32>
// CHECK-NEXT:    memref.dealloc %[[IDX]] : memref<3xindex>
// CHECK-NEXT:    memref.dealloc %[[WORK]] : memref<f32>
// CHECK-NEXT:    memref.dealloc %[[SLOTS]] : memref<3xf32>
// CHECK-NEXT:    return
