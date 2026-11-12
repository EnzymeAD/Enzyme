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
// CHECK-SAME:    %[[ARG0:.+]]: memref<f32>, %[[ARG1:.+]]: memref<f32>,

// Read-only %arg0 is not written in the loop or after it, so only the scalar
// state and step index need checkpoint buffers (no clone slots or memref.copy).
// CHECK:         %[[STATE:.+]] = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %[[IDX:.+]] = memref.alloc() : memref<3xindex>
// CHECK-NOT:     memref.alloc() : memref<3xf32>

// CHECK:         scf.for %[[K:.+]] = %[[C0:[^ ]+]] to %[[C3:[^ ]+]] step %[[C1:[^ ]+]] iter_args(
// CHECK:           memref.store {{.*}}, %[[STATE]][%[[K]]]
// CHECK:           memref.store {{.*}}, %[[IDX]][%[[K]]]
// CHECK-NOT:       memref.copy

// CHECK-NOT:     memref.alloc() : memref<f32>
// CHECK-NOT:     memref.copy

// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C9:.+]] step %[[C1]] iter_args(%[[SP:.+]] = %[[C3]]
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %[[C1]] : index
// CHECK:           memref.load %[[STATE]][%[[CAPO]]]
// CHECK:           memref.load %[[IDX]][%[[CAPO]]]
// CHECK-NOT:       memref.copy

// CHECK:           scf.while ({{.*}}%[[ACAPO:.+]] = %[[CAPO]]
// CHECK:             memref.store {{.*}}, %[[IDX]][%[[ACAPO]]]
// CHECK-NOT:         memref.copy
// The replay reads the original read-only memref directly.
// CHECK:             memref.load %[[ARG0]][] : memref<f32>

// CHECK:         memref.load %[[ARG1]][] : memref<f32>
// CHECK:         memref.store {{.*}}, %[[ARG1]][] : memref<f32>

// CHECK:         memref.dealloc %[[STATE]] : memref<3xf32>
// CHECK-NEXT:    memref.dealloc %[[IDX]] : memref<3xindex>
// CHECK-NEXT:    return
