// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

module {
  func.func @main(%x: complex<f32>) -> (f32, f32) {
    %0 = complex.re %x : complex<f32>
    %1 = complex.im %x : complex<f32>
    return %0, %1 : f32, f32
  }
}

// CHECK:  func.func @main(%arg0: complex<f32>, %arg1: f32, %arg2: f32) -> complex<f32> {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %cst_0 = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:    %0 = arith.addf %arg1, %cst : f32
// CHECK-NEXT:    %1 = arith.addf %arg2, %cst : f32
// CHECK-NEXT:    %2 = arith.negf %1 : f32
// CHECK-NEXT:    %3 = complex.create %cst, %2 : complex<f32>
// CHECK-NEXT:    %4 = complex.conj %3 : complex<f32>
// CHECK-NEXT:    %5 = complex.add %cst_0, %4 : complex<f32>
// CHECK-NEXT:    %6 = complex.create %0, %cst : complex<f32>
// CHECK-NEXT:    %7 = complex.conj %6 : complex<f32>
// CHECK-NEXT:    %8 = complex.add %5, %7 : complex<f32>
// CHECK-NEXT:    return %8 : complex<f32>
// CHECK-NEXT:  }
