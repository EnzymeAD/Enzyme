// RUN: %eopt --enzyme %s | FileCheck %s

// Test that a constant iter arg whose corresponding ForOp result is active
// (the "constant accumulator" pattern) is correctly differentiated.
// The iter arg %acc is initialized from a constant zero and is therefore
// marked constant by activity analysis, but the ForOp result is active
// because active values (%x) are accumulated into it through the body.
// The differentiated ForOp must have a shadow iter arg (also zero-initialized)
// that accumulates the tangent dx on each iteration.

module {
  func.func @square(%x : f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (f64) {
      %n = arith.addf %acc, %x : f64
      scf.yield %n : f64
    }
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// The differentiated ForOp must have TWO iter args: the primal accumulator
// (init = 0.0) and its shadow (init = 0.0, since the original init is a
// constant). On each iteration the shadow accumulates dx (= %arg1).

// CHECK:   func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-DAG:     %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[cst_0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c10:.+]] = arith.constant 10 : index
// CHECK-NEXT:     %[[r:.+]]:2 = scf.for %{{.+}} = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[acc:.+]] = %[[cst_0]], %[[sacc:.+]] = %[[cst]]) -> (f64, f64) {
// CHECK-NEXT:       %[[sn:.+]] = arith.addf %[[sacc]], %[[arg1]] : f64
// CHECK-NEXT:       %[[n:.+]] = arith.addf %[[acc]], %[[arg0]] : f64
// CHECK-NEXT:       scf.yield %[[n]], %[[sn]] : f64, f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[r]]#1 : f64
// CHECK-NEXT:   }
