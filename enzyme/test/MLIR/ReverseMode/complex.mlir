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
// CHECK-NEXT:    %0 = arith.negf %arg2 fastmath<fast> : f32
// CHECK-NEXT:    %1 = complex.create %cst, %0 : complex<f32>
// CHECK-NEXT:    %2 = complex.conj %1 : complex<f32>
// CHECK-NEXT:    %3 = complex.add %2, %cst_0 fastmath<fast> : complex<f32>
// CHECK-NEXT:    %4 = complex.create %arg1, %cst : complex<f32>
// CHECK-NEXT:    %5 = complex.conj %4 : complex<f32>
// CHECK-NEXT:    %6 = complex.add %3, %5 fastmath<fast> : complex<f32>
// CHECK-NEXT:    return %6 : complex<f32>
// CHECK-NEXT:  }
