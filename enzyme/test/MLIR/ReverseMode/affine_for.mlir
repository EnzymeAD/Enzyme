// RUN: %eopt %s  --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  func.func @reduce(%x: f32) -> (f32) {
    %sum_0 = arith.constant 1.0 : f32

    %sum = affine.for %iv = 0 to 128
        iter_args(%sum_iter = %sum_0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %x : f32
      affine.yield %sum_next : f32
    }

    return %sum : f32
  }
}


// CHECK:  func.func @reduce(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c127 = arith.constant 127 : index
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<128xf32>
// CHECK-NEXT:    %0 = affine.for %arg2 = 0 to 128 iter_args(%arg3 = %cst) -> (f32) {
// CHECK-NEXT:      memref.store %arg3, %alloc[%arg2] : memref<128xf32>
// CHECK-NEXT:      %2 = arith.mulf %arg3, %arg0 : f32
// CHECK-NEXT:      affine.yield %2 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:3 = affine.for %arg2 = 0 to 128 iter_args(%arg3 = %arg1, %arg4 = %cst_0, %arg5 = %c127) -> (f32, f32, index) {
// CHECK-NEXT:      %2 = memref.load %alloc[%arg5] : memref<128xf32>
// CHECK-NEXT:      %3 = arith.mulf %arg3, %arg0 : f32
// CHECK-NEXT:      %4 = arith.mulf %arg3, %2 : f32
// CHECK-NEXT:      %5 = arith.addf %arg4, %4 : f32
// CHECK-NEXT:      %6 = arith.subi %arg5, %c1 : index
// CHECK-NEXT:      affine.yield %3, %5, %6 : f32, f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<128xf32>
// CHECK-NEXT:    return %1#1 : f32
// CHECK-NEXT:  }
