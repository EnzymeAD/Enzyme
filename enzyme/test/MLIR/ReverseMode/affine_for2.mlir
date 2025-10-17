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
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%arg1) : memref<?xf32>
// CHECK-NEXT:    %0 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:      memref.store %arg4, %alloc[%arg3] : memref<?xf32>
// CHECK-NEXT:      %2 = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      affine.yield %2 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:2 = affine.for %arg3 = 0 to %arg1 iter_args(%arg4 = %arg2, %arg5 = %cst_0) -> (f32, f32) {
// CHECK-NEXT:      %[[nm1:.+]] = arith.subi %arg1, %c1 : index
// CHECK-NEXT:      %[[ridx:.+]] = arith.subi %[[nm1]], %arg3 : index
// CHECK-NEXT:      %[[a2:.+]] = memref.load %alloc[%[[ridx]]] : memref<?xf32>
// CHECK-NEXT:      %[[a3:.+]] = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      %[[a4:.+]] = arith.mulf %arg4, %[[a2]] : f32
// CHECK-NEXT:      %[[a5:.+]] = arith.addf %arg5, %[[a4]] : f32
// CHECK-NEXT:      affine.yield %[[a3]], %[[a5]] : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?xf32>
// CHECK-NEXT:    return %1#1 : f32
// CHECK-NEXT:  }
