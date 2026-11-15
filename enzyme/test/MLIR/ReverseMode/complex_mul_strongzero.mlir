// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s

module {
  func.func @square(%x: complex<f64>) -> complex<f64> {
    %next = complex.mul %x, %x : complex<f64>
    return %next : complex<f64>
  }

  func.func @dsquare(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
    %r = enzyme.autodiff @square(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>], strong_zero=true }> : (complex<f64>, complex<f64>) -> complex<f64>
    return %r : complex<f64>
  }
}

// CHECK:  func.func private @diffesquare(%arg0: complex<f64>, %arg1: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %cst = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %0 = complex.conj %arg1 : complex<f64>
// CHECK-NEXT:    %1 = complex.eq %0, %cst : complex<f64>
// CHECK-NEXT:    %2 = complex.mul %0, %arg0 fastmath<fast> : complex<f64>
// CHECK-NEXT:    %3 = arith.select %1, %cst, %2 : complex<f64>
// CHECK-NEXT:    %4 = complex.conj %3 : complex<f64>
// CHECK-NEXT:    %5 = complex.add %4, %4 fastmath<fast> : complex<f64>
// CHECK-NEXT:    return %5 : complex<f64>
// CHECK-NEXT:  }
