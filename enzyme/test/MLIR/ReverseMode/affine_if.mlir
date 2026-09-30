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
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf64>
// CHECK-NEXT:    %0 = affine.for %arg3 = 0 to 4 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:      %2 = affine.if #set(%arg3) -> f64 {
// CHECK-NEXT:        %4 = affine.load %arg0[%arg3] : memref<4xf64>
// CHECK-NEXT:        memref.store %4, %alloc[%arg3] : memref<4xf64>
// CHECK-NEXT:        %5 = math.cos %4 : f64
// CHECK-NEXT:        affine.yield %5 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %cst : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %3 = arith.addf %arg4, %2 : f64
// CHECK-NEXT:      affine.yield %3 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = affine.for %arg3 = 0 to 4 iter_args(%arg4 = %arg2) -> (f64) {
// CHECK-NEXT:      %2 = affine.apply #{{.*}}(%arg3)
// CHECK-NEXT:      %3 = arith.subi %c2, %2 : index
// CHECK-NEXT:      %4 = arith.cmpi sge, %3, %c0 : index
// CHECK-NEXT:      scf.if %4 {
// CHECK-NEXT:        %5 = memref.load %alloc[%2] : memref<4xf64>
// CHECK-NEXT:        %6 = math.sin %5 fastmath<fast> : f64
// CHECK-NEXT:        %7 = arith.negf %6 fastmath<fast> : f64
// CHECK-NEXT:        %8 = arith.mulf %arg4, %7 fastmath<fast> : f64
// CHECK-NEXT:        %9 = memref.load %arg1[%2] : memref<4xf64>
// CHECK-NEXT:        %10 = arith.addf %9, %8 fastmath<fast> : f64
// CHECK-NEXT:        memref.store %10, %arg1[%2] : memref<4xf64>
// CHECK-NEXT:      } else {
// CHECK-NEXT:      } {preserve_cache}
// CHECK-NEXT:      affine.yield %arg4 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
