// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @div(%x: f64, %y: f64) -> f64 {
    %r = arith.divf %x, %y : f64
    return %r : f64
  }

  func.func @ddiv(%x: f64, %dx: f64, %y: f64, %dy: f64) -> f64 {
    %dr = enzyme.fwddiff @div(%x, %dx, %y, %dy) {
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    } : (f64, f64, f64, f64) -> f64
    return %dr : f64
  }
}

// d(x / y) = (dx - dy * (x / y)) / y: the divisor is never squared.
// CHECK:  func.func private @fwddiffediv(%[[X:.+]]: f64, %[[DX:.+]]: f64, %[[Y:.+]]: f64, %[[DY:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[Q:.+]] = arith.divf %[[X]], %[[Y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[T:.+]] = arith.mulf %[[DY]], %[[Q]] fastmath<fast> : f64
// CHECK-NEXT:    %[[N:.+]] = arith.subf %[[DX]], %[[T]] fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.divf %[[N]], %[[Y]] fastmath<fast> : f64
// CHECK-NOT:     arith.mulf %[[Y]], %[[Y]]
// CHECK:         return %[[R]] : f64
