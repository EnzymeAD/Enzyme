// RUN: %eopt %s --enzyme-wrap="infn=sum_tail outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

#front = affine_set<(d0) : (-d0 + 2 >= 0)>

func.func @sum_tail(%x: memref<4xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %sum = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> f64 {
    %selected = affine.if #front(%i) -> f64 {
      %value = affine.load %x[%i] : memref<4xf64>

      %cos_value = math.cos %value : f64

      affine.yield %cos_value : f64
    } else {
      affine.yield %zero : f64
    }
    %next = arith.addf %acc, %selected : f64
    affine.yield %next : f64
  }
  return %sum : f64
}

// CHECK:  func.func @sum_tail(%arg0: memref<4xf64>, %arg1: memref<4xf64>, %arg2: f64) {
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[CACHE:.+]] = memref.alloc() : memref<4xf64>
// CHECK-NEXT:    %0 = affine.for %arg3 = 0 to 4 iter_args(%arg4 = %[[ZERO]]) -> (f64) {
// CHECK-NEXT:      %2 = affine.if #set(%arg3) -> f64 {
// CHECK-NEXT:        %4 = affine.load %arg0[%arg3] : memref<4xf64>
// CHECK-NEXT:        memref.store %4, %[[CACHE]][%arg3] : memref<4xf64>
// CHECK-NEXT:        %5 = math.cos %4 : f64
// CHECK-NEXT:        affine.yield %5 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %[[ZERO]] : f64
// CHECK-NEXT:      } {preserve_cache}
// CHECK-NEXT:      %3 = arith.addf %arg4, %2 : f64
// CHECK-NEXT:      affine.yield %3 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = affine.for %arg3 = 0 to 4 iter_args(%arg4 = %arg2) -> (f64) {
// CHECK-NEXT:      %2 = arith.subi %c3, %arg3 : index
// CHECK-NEXT:      %3 = arith.cmpi sle, %2, %c2 : index
// CHECK-NEXT:      scf.if %3 {
// CHECK-NEXT:        %4 = memref.load %[[CACHE]][%2] : memref<4xf64>
// CHECK-NEXT:        %5 = math.sin %4 fastmath<fast> : f64
// CHECK-NEXT:        %6 = arith.negf %5 fastmath<fast> : f64
// CHECK-NEXT:        %7 = arith.mulf %arg4, %6 fastmath<fast> : f64
// CHECK-NEXT:        %8 = memref.load %arg1[%2] : memref<4xf64>
// CHECK-NEXT:        %9 = arith.addf %8, %7 fastmath<fast> : f64
// CHECK-NEXT:        memref.store %9, %arg1[%2] : memref<4xf64>
// CHECK-NEXT:      } else {
// CHECK-NEXT:      } {preserve_cache}
// CHECK-NEXT:      affine.yield %arg4 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[CACHE]] : memref<4xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
