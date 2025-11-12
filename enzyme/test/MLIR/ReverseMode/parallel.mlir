// RUN: %eopt %s --enzyme-wrap="infn=affine outfn= argTys=enzyme_active,enzyme_dup,enzyme_dup retTys= mode=ReverseModeCombined" --enzyme-wrap="infn=scf outfn= argTys=enzyme_active,enzyme_dup,enzyme_dup retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s

module {

  func.func @scf(%a: f32,
               %x: memref<?xf32>,
               %y: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%arg9) = (%c0) to (%c4) step (%c1) {
      %0 = memref.load %x[%arg9] : memref<?xf32>
      %1 = arith.mulf %0, %a : f32
      memref.store %1, %y[%arg9] : memref<?xf32>
    }
    return
  }

  func.func @affine(%a: f32,
                    %x: memref<?xf32>,
                    %y: memref<?xf32>) {
    affine.parallel (%arg9) = (0) to (4) {
      %0 = affine.load %x[%arg9] : memref<?xf32>
      %1 = arith.mulf %0, %a : f32
      affine.store %1, %y[%arg9] : memref<?xf32>
    }
    return
  }
}

// CHECK:  func.func @affine(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) -> f32 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf32>
// CHECK-NEXT:    affine.parallel (%arg5) = (0) to (4) {
// CHECK-NEXT:      %1 = affine.load %arg1[%arg5] : memref<?xf32>
// CHECK-NEXT:      memref.store %1, %alloc[%arg5] : memref<4xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %arg0 : f32
// CHECK-NEXT:      affine.store %2, %arg3[%arg5] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %0 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
// CHECK-NEXT:      %1 = memref.load %alloc[%arg5] : memref<4xf32>
// CHECK-NEXT:      %2 = memref.load %arg4[%arg5] : memref<?xf32>
// CHECK-NEXT:      memref.store %cst, %arg4[%arg5] : memref<?xf32>
// CHECK-NEXT:      %3 = arith.mulf %2, %arg0 : f32
// CHECK-NEXT:      %4 = arith.mulf %2, %1 : f32
// CHECK-NEXT:      %5 = memref.atomic_rmw addf %3, %arg2[%arg5] : (f32, memref<?xf32>) -> f32
// CHECK-NEXT:      affine.yield %4 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf32>
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }

// CHECK:  func.func @scf(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) -> f32 {
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf32>
// CHECK-NEXT:    scf.parallel (%arg5) = (%c0) to (%c4) step (%c1) {
// CHECK-NEXT:      %1 = memref.load %arg1[%arg5] : memref<?xf32>
// CHECK-NEXT:      memref.store %1, %alloc[%arg5] : memref<4xf32>
// CHECK-NEXT:      %2 = arith.mulf %1, %arg0 : f32
// CHECK-NEXT:      memref.store %2, %arg3[%arg5] : memref<?xf32>
// CHECK-NEXT:      scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    %0 = scf.parallel (%arg5) = (%c0) to (%c4) step (%c1) init (%cst) -> f32 {
// CHECK-NEXT:      %1 = memref.load %alloc[%arg5] : memref<4xf32>
// CHECK-NEXT:      %2 = memref.load %arg4[%arg5] : memref<?xf32>
// CHECK-NEXT:      memref.store %cst, %arg4[%arg5] : memref<?xf32>
// CHECK-NEXT:      %3 = arith.mulf %2, %arg0 : f32
// CHECK-NEXT:      %4 = arith.mulf %2, %1 : f32
// CHECK-NEXT:      %5 = memref.atomic_rmw addf %3, %arg2[%arg5] : (f32, memref<?xf32>) -> f32
// CHECK-NEXT:      scf.reduce(%4 : f32) {
// CHECK-NEXT:      ^bb0(%arg6: f32, %arg7: f32):
// CHECK-NEXT:        %6 = arith.addf %arg6, %arg7 : f32
// CHECK-NEXT:        scf.reduce.return %6 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf32>
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }

