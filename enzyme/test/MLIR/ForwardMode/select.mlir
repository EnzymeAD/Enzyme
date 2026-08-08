// RUN: %eopt --enzyme %s | FileCheck %s

// The condition carries no tangent; the result's tangent follows the same
// choice between the branch tangents, a constant branch contributing zero.

module {
  func.func @relu(%x: f64) -> f64 {
    %zero = arith.constant 0.000000e+00 : f64
    %c = arith.cmpf ugt, %x, %zero : f64
    %r = arith.select %c, %x, %zero : f64
    return %r : f64
  }
  func.func @drelu(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @relu(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddifferelu(%arg0: f64, %arg1: f64) -> f64
// CHECK: %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[c:.+]] = arith.cmpf ugt, %arg0, %[[cst]] : f64
// CHECK: %[[t:.+]] = arith.select %[[c]], %arg1, %{{.+}} : f64
// CHECK: return %[[t]] : f64
