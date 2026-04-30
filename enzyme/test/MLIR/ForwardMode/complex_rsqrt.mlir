// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s 

module {
  func.func @rsqrt(%x: complex<f64>) -> complex<f64> {
    %next = complex.rsqrt %x : complex<f64>
    return %next : complex<f64>
  }

  func.func @drsqrt(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
    %r = enzyme.fwddiff @rsqrt(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (complex<f64>, complex<f64>) -> complex<f64>
    return %r : complex<f64>
  }
}

// CHECK:  func.func private @fwddiffersqrt(%arg0: complex<f64>, %arg1: complex<f64>) 
// CHECK-NEXT:    %0 = complex.constant [-1.5, 0.0] : complex<f64>
// CHECK-NEXT:    %1 = complex.rsqrt %arg1, %0 : complex<f64>
// CHECK-NEXT:    return %1 : complex<f64>
// CHECK-NEXT:  }
