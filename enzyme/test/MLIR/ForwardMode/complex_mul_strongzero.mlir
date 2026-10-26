// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : complex<f64>) -> complex<f64> {
    %y = complex.mul %x, %x : complex<f64>
    return %y : complex<f64>
  }
  func.func @dsq(%x : complex<f64>, %dx : complex<f64>) -> complex<f64> {
    %r = enzyme.fwddiff @square(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], strong_zero=true }> : (complex<f64>, complex<f64>) -> (complex<f64>)
    return %r : complex<f64>
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: complex<f64>, %[[arg1:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:     %[[i0:.+]] = call @fwddiffesquare(%[[arg0]], %[[arg1]]) : (complex<f64>, complex<f64>) -> complex<f64>
// CHECK-NEXT:     return %[[i0]] : complex<f64>
// CHECK-NEXT:   }

// CHECK:  func.func private @fwddiffesquare(%arg0: complex<f64>, %arg1: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %cst = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %0 = complex.eq %arg1, %cst : complex<f64>
// CHECK-NEXT:    %1 = complex.mul %arg1, %arg0 fastmath<fast> : complex<f64>
// CHECK-NEXT:    %2 = arith.select %0, %cst, %1 : complex<f64>
// CHECK-NEXT:    %cst_0 = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %3 = complex.eq %arg1, %cst_0 : complex<f64>
// CHECK-NEXT:    %4 = complex.mul %arg1, %arg0 fastmath<fast> : complex<f64>
// CHECK-NEXT:    %5 = arith.select %3, %cst_0, %4 : complex<f64>
// CHECK-NEXT:    %6 = complex.add %2, %5 fastmath<fast> : complex<f64>
// CHECK-NEXT:    %7 = complex.mul %arg0, %arg0 : complex<f64>
// CHECK-NEXT:    return %6 : complex<f64>
// CHECK-NEXT:  }
