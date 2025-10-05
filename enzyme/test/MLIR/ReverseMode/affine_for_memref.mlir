// RUN: %eopt %s --enzyme-wrap="infn=square_ip outfn= argTys=enzyme_dup,enzyme_const retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

func.func @square_ip(%arg0: memref<?xf32>, %ub: index) {
  affine.for %iv = 0 to %ub {
    %v = affine.load %arg0[2 * %iv] : memref<?xf32>
    %sq = arith.mulf %v, %v : f32
    affine.store %sq, %arg0[2 * %iv + 1] : memref<?xf32>
    affine.yield
  }
  return
}

// CHECK:  func.func @reduce(%arg0: f32, %arg1: index, %arg2: f32) -> f32 {
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%arg1) : memref<?x4xf32>
// CHECK-NEXT:    %[[v0:.+]] = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:      %subview = memref.subview %alloc[%arg3, 0] [1, 4] [1, 1] : memref<?x4xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v2:.+]] = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (f32) {
// CHECK-NEXT:        memref.store %arg6, %subview[%arg5] : memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v3:.+]] = arith.mulf %arg6, %arg0 : f32
// CHECK-NEXT:        scf.yield %[[v3]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v2]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v1:.+]]:3 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %arg2, %arg5 = %cst_0, %arg6 = %arg1) -> (f32, f32, index) {
// CHECK-NEXT:      %subview = memref.subview %alloc[%arg6, 0] [1, 4] [1, 1] : memref<?x4xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v2:.+]]:3 = scf.for %arg7 = %c0 to %c4 step %c1 iter_args(%arg8 = %arg4, %arg9 = %arg5, %arg10 = %c3) -> (f32, f32, index) {
// CHECK-NEXT:        %[[v4:.+]] = memref.load %subview[%arg10] : memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v5:.+]] = arith.mulf %arg8, %arg0 : f32
// CHECK-NEXT:        %[[v6:.+]] = arith.mulf %arg8, %[[v4]] : f32
// CHECK-NEXT:        %[[v7:.+]] = arith.addf %arg9, %[[v6]] : f32
// CHECK-NEXT:        %[[v8:.+]] = arith.subi %arg10, %c1 : index
// CHECK-NEXT:        scf.yield %[[v5]], %[[v7]], %[[v8]] : f32, f32, index
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[v3:.+]] = arith.subi %arg6, %c1 : index
// CHECK-NEXT:      scf.yield %[[v2]]#0, %[[v2]]#1, %[[v3]] : f32, f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?x4xf32>
// CHECK-NEXT:    return %[[v1]]#1 : f32
// CHECK-NEXT:  }
