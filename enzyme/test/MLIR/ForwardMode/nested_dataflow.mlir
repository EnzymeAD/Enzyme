// RUN: %eopt --pass-pipeline="builtin.module(enzyme{dataflow=true},canonicalize)" %s | FileCheck %s

// Differentiating a function that itself contains an enzyme.fwddiff runs the
// dataflow activity analysis on the first-order function the inner op
// generated -- after an enclosing analysis has already stamped it with a
// pointer summary. The summary serves callers; the function's own per-value
// maps must still be computed, or every value inside looks inactive and the
// second derivative of anything is zero.

module {
  func.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @dsquare(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
  func.func @ddsquare(%x: f64, %dx: f64, %sx: f64, %sdx: f64) -> f64 {
    %r = enzyme.fwddiff @dsquare(%x, %dx, %sx, %sdx) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, f64, f64) -> f64
    return %r : f64
  }
}

// The second derivative of x*x in direction (sx, sdx) is 2 (sx dx + x sdx);
// with everything active the generated second-order function must read all
// four inputs, not collapse to a constant zero.
// CHECK-LABEL: func.func private @fwddiffefwddiffesquare
// CHECK-NOT: arith.constant 0.000000e+00
// CHECK: arith.mulf %arg3, %arg0
// CHECK: arith.mulf %arg1, %arg2
// CHECK: return
