// RUN: %eopt --enzyme %s | FileCheck %s
// ARUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s

module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>], strong_zero=true } : (f64, f64) -> f64
    return %r : f64
  }
}



// CHECK:  func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:    %1 = arith.addf %0, %0 : f64
// CHECK-NEXT:    return %1 : f64
// CHECK-NEXT:  }