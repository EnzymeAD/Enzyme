// RUN: %eopt %s --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

func.func @reduce(%x: f32, %ub: index) -> (f32) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %ub_inner = arith.constant 4 : index

  // Initial sum set to 0.
  %sum_0 = arith.constant 1.0 : f32
  // iter_args binds initial values to the loop's region arguments.
  %sum2 = scf.for %iv2 = %lb to %ub step %step
      iter_args(%sum_iter2 = %sum_0) -> (f32) {
    %sum = scf.for %iv = %lb to %ub_inner step %step
        iter_args(%sum_iter = %sum_iter2) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %x : f32
      // Yield current iteration sum to next iteration %sum_iter or to %sum
      // if final iteration.
      scf.yield %sum_next : f32
    }
    scf.yield %sum : f32
  }
  return %sum2 : f32
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
// CHECK-NEXT:      %[[v3:.+]] = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (f32) {
// CHECK-NEXT:        memref.store %arg6, %subview[%arg5] : memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v4:.+]] = arith.mulf %arg6, %arg0 : f32
// CHECK-NEXT:        scf.yield %[[v4]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v3]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v1:.+]] = arith.addf %arg2, %cst_0 : f32
// CHECK-NEXT:    %[[v2:.+]]:4 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %[[v1]], %arg5 = %cst_0, %arg6 = %arg1, %arg7 = %cst_0) -> (f32, f32, index, f32) {
// CHECK-NEXT:      %subview = memref.subview %alloc[%arg6, 0] [1, 4] [1, 1] : memref<?x4xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v3:.+]]:4 = scf.for %arg8 = %c0 to %c4 step %c1 iter_args(%arg9 = %arg4, %arg10 = %arg5, %arg11 = %c3, %arg12 = %arg5) -> (f32, f32, index, f32) {
// CHECK-NEXT:        %[[v6:.+]] = memref.load %subview[%arg11] : memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v7:.+]] = arith.mulf %arg9, %arg0 : f32
// CHECK-NEXT:        %[[v8:.+]] = arith.addf %[[v7]], %cst_0 : f32
// CHECK-NEXT:        %[[v9:.+]] = arith.mulf %arg9, %[[v6]] : f32
// CHECK-NEXT:        %[[v10:.+]] = arith.addf %arg10, %[[v9]] : f32
// CHECK-NEXT:        %[[v11:.+]] = arith.subi %arg11, %c1 : index
// CHECK-NEXT:        scf.yield %[[v8]], %[[v10]], %[[v11]], %[[v10]] : f32, f32, index, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[v4:.+]] = arith.addf %[[v3]]#0, %cst_0 : f32
// CHECK-NEXT:      %[[v5:.+]] = arith.subi %arg6, %c1 : index
// CHECK-NEXT:      scf.yield %[[v4]], %[[v3]]#1, %[[v5]], %[[v3]]#1 : f32, f32, index, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?x4xf32>
// CHECK-NEXT:    return %[[v2]]#1 : f32
// CHECK-NEXT:  }

