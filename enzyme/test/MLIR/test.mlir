// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x) : (f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func @square(%arg0: f64) -> f64 {
// CHECK-NEXT:     %0 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:     return %0 : f64
// CHECK-NEXT:   }

// CHECK:   func.func @dsq(%arg0: f64) -> f64 {
// CHECK-NEXT:     %0 = enzyme.fwddiff @square(%arg0) : (f64) -> f64
// CHECK-NEXT:     return %0 : f64
// CHECK-NEXT:   }
