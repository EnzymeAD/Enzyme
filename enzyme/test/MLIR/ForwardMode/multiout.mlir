// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> (f64, f64) {
    %y = arith.mulf %x, %x : f64
    return %y, %x : f64, f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> (f64, f64) {
    %r:2 = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64, f64)
    return %r#0, %r#1 : f64, f64
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:     %[[i0:[0-9]+]]:2 = call @fwddiffesquare(%[[arg0]], %[[arg1]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:     return %[[i0]]#0, %[[i0]]#1 : f64, f64
// CHECK-NEXT:   }
// CHECK:   func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:     %[[i0:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i1:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i2:.+]] = arith.addf %[[i0]], %[[i1]] : f64
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:     return %[[i2]], %[[arg1]] : f64, f64
// CHECK-NEXT:   }
