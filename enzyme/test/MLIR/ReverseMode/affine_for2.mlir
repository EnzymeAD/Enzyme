// RUN: %eopt %s  --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  func.func @reduce(%x: f32, %n : index) -> (f32) {
    %sum_0 = arith.constant 1.0 : f32

    %sum = affine.for %iv = 0 to %n
        iter_args(%sum_iter = %sum_0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %x : f32
      affine.yield %sum_next : f32
    }

    return %sum : f32
  }
}

// CHECK:  func.func @reduce(%arg0: f32, %arg1: index, %arg2: f32) -> f32 {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%arg1) : memref<?xf32>
// CHECK-NEXT:    %0:2 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %cst, %arg5 = %c0) -> (f32, index) {
// CHECK-NEXT:      memref.store %arg4, %alloc[%arg5] : memref<?xf32>
// CHECK-NEXT:      %2 = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      %3 = arith.addi %arg5, %c1 : index
// CHECK-NEXT:      affine.yield %2, %3 : f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:3 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %arg2, %arg5 = %cst_0, %arg6 = %arg1) -> (f32, f32, index) {
// CHECK-NEXT:      %2 = memref.load %alloc[%arg6] : memref<?xf32>
// CHECK-NEXT:      %3 = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      %4 = arith.mulf %arg4, %2 : f32
// CHECK-NEXT:      %5 = arith.addf %arg5, %4 : f32
// CHECK-NEXT:      %6 = arith.subi %arg6, %c1 : index
// CHECK-NEXT:      affine.yield %3, %5, %6 : f32, f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?xf32>
// CHECK-NEXT:    return %1#1 : f32
// CHECK-NEXT:  }
