// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>], strong_zero=true } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-NEXT:     %[[i0:.+]] = call @fwddiffesquare(%[[arg0]], %[[arg1]]) : (f64, f64) -> f64
// CHECK-NEXT:     return %[[i0]] : f64
// CHECK-NEXT:   }

// CHECK:  func.func private @fwddiffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.cmpf oeq, %arg1, %cst : f64
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %2 = arith.select %0, %cst, %1 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %3 = arith.cmpf oeq, %arg1, %cst_0 : f64
// CHECK-NEXT:    %4 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %5 = arith.select %3, %cst_0, %4 : f64
// CHECK-NEXT:    %6 = arith.addf %2, %5 : f64
// CHECK-NEXT:    %7 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:    return %6 : f64
// CHECK-NEXT:  }
