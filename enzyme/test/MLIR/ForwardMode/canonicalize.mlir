// RUN: %eopt --canonicalize %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %p,%r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64,f64)
    return %r : f64
  }
}

// CHECK:   func.func @square(%arg0: f64) -> f64 {
// CHECK-NEXT:     %0 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:     return %0 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @dsq(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:     %0 = enzyme.fwddiff @square(%arg0, %arg1) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:     return %0 : f64
// CHECK-NEXT:   }
