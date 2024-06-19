// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s 

module {
  func.func @square(%x: complex<f64>) -> complex<f64> {
    %next = complex.mul %x, %x : complex<f64>
    return %next : complex<f64>
  }

  func.func @dsquare(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
    %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (complex<f64>, complex<f64>) -> complex<f64>
    return %r : complex<f64>
  }
}

// CHECK:  func.func private @diffesquare(%arg0: complex<f64>, %arg1: complex<f64>) -> complex<f64>
// CHECK-NEXT:    %0 = complex.conj %arg1 : complex<f64>
// CHECK-NEXT:    %1 = complex.mul %0, %arg0 : complex<f64>
// CHECK-NEXT:    %2 = complex.conj %1 : complex<f64>
// CHECK-NEXT:    %3 = complex.add %2, %2 : complex<f64>
// CHECK-NEXT:    return %3 : complex<f64>
// CHECK-NEXT:  }
