// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops)" | FileCheck %s
func.func @foo(%x: memref<?xf32>, %y: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%arg9) = (%c0) to (%c4) step (%c1) {
    %0 = memref.load %x[%arg9] : memref<?xf32>
    %1 = arith.mulf %0, %0 : f32
    memref.store %1, %y[%arg9] : memref<?xf32>
    scf.reduce
  }
  return
}

func.func @dfoo(%x: memref<?xf32>, %dx: memref<?xf32>, %y: memref<?xf32>, %dy: memref<?xf32>) {
  enzyme.autodiff @foo(%x, %dx, %y, %dy) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK: func.func private @diffefoo(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf32>
// CHECK-NEXT:    scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) {
// CHECK-NEXT:      %0 = memref.load %arg0[%arg4] : memref<?xf32>
// CHECK-NEXT:      memref.store %0, %alloc[%arg4] : memref<4xf32>
// CHECK-NEXT:      %1 = arith.mulf %0, %0 : f32
// CHECK-NEXT:      memref.store %1, %arg2[%arg4] : memref<?xf32>
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.parallel (%arg4) = (%c0) to (%c4) step (%c1) {
// CHECK-NEXT:      %0 = memref.load %alloc[%arg4] : memref<4xf32>
// CHECK-NEXT:      %1 = memref.load %arg3[%arg4] : memref<?xf32>
// CHECK-NEXT:      %2 = arith.addf %1, %cst : f32
// CHECK-NEXT:      memref.store %cst, %arg3[%arg4] : memref<?xf32>
// CHECK-NEXT:      %3 = arith.mulf %2, %0 : f32
// CHECK-NEXT:      %4 = arith.addf %3, %cst : f32
// CHECK-NEXT:      %5 = arith.mulf %2, %0 : f32
// CHECK-NEXT:      %6 = arith.addf %4, %5 : f32
// CHECK-NEXT:      %7 = memref.atomic_rmw addf %6, %arg1[%arg4] : (f32, memref<?xf32>) -> f32
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
