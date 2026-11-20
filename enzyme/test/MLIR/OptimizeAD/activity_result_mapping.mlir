// RUN: %eopt --enzyme-activity-opt %s | FileCheck %s
// RUN: %eopt --enzyme-activity-opt='dataflow=true' %s | FileCheck %s

// Refining a constant return must preserve a live zero derivative and must
// count a dupnoneed result as one output, even next to another changed result.
module {
  func.func @square_and_constant(%x: f64, %c: f64) -> (f64, f64) {
    %xx = arith.mulf %x, %x : f64
    return %xx, %c : f64, f64
  }
  func.func @live_zero(%x: f64, %dx: f64, %c: f64) -> (f64, f64) {
    %dxx, %dc = enzyme.fwddiff @square_and_constant(%x, %dx, %c) <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64) -> (f64, f64)
    return %dxx, %dc : f64, f64
  }
  func.func @live_primal_and_zero(%x: f64, %dx: f64, %c: f64) -> (f64, f64, f64) {
    %dxx, %p, %dc = enzyme.fwddiff @square_and_constant(%x, %dx, %c) <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dup>]}> : (f64, f64, f64) -> (f64, f64, f64)
    return %dxx, %p, %dc : f64, f64, f64
  }
}
// CHECK-LABEL: func.func @live_zero
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[D:.*]] = enzyme.fwddiff @square_and_constant
// CHECK-SAME: ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_constnoneed>]
// CHECK: return %[[D]], %[[ZERO]]
// CHECK-LABEL: func.func @live_primal_and_zero
// CHECK: %[[ZERO2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[D2:.*]]:2 = enzyme.fwddiff @square_and_constant
// CHECK-SAME: ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_const>]
// CHECK: return %[[D2]]#0, %[[D2]]#1, %[[ZERO2]]
