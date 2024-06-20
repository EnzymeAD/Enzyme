// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s 

module {
  func.func @square(%x: complex<f64>) -> complex<f64> {
    %next = complex.mul %x, %x : complex<f64>
    return %next : complex<f64>
  }

  func.func @dsquare(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (complex<f64>, complex<f64>) -> complex<f64>
    return %r : complex<f64>
  }
}

// CHECK:  func.func private @fwddiffesquare(%arg0: complex<f64>, %arg1: complex<f64>) 
// CHECK-NEXT:    %0 = complex.mul %arg1, %arg0 : complex<f64>
// CHECK-NEXT:    %1 = complex.add %0, %0 : complex<f64>
// CHECK-NEXT:    return %1 : complex<f64>
// CHECK-NEXT:  }
