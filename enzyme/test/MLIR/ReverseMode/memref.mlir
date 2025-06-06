// RUN: %eopt %s --enzyme-wrap="infn=loadandret outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

func.func @loadandret(%x: memref<f64>) -> f64 {
  %res = memref.load %x[] : memref<f64>
  return %res : f64
}

// CHECK:  func.func @loadandret(%arg0: memref<f64>, %arg1: memref<f64>, %arg2: f64) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.addf %arg2, %cst : f64
// CHECK-NEXT:    %1 = memref.load %arg1[] : memref<f64>
// CHECK-NEXT:    %2 = arith.addf %1, %0 : f64
// CHECK-NEXT:    memref.store %2, %arg1[] : memref<f64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
