// RUN: %eopt --canonicalize %s | FileCheck %s

// An enzyme.autodiff whose outputs do not line up with its activities: an
// enzyme_active return claims a primal among the results, but the op carries
// only the gradient (the shape a __enzyme_autodiff call raises to when the
// return is mis-said as active rather than activenoneed). There is no
// verifier saying the ranges must line up, so the canonicalizer used to walk
// off their ends; it must leave the op for the AD pass to diagnose instead.

module {
  llvm.func @square(%x: f64) -> f64 {
    %r = llvm.fmul %x, %x : f64
    llvm.return %r : f64
  }
  llvm.func @dsquare(%x: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = enzyme.autodiff @square(%x, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>]} : (f64, f64) -> f64
    llvm.return %0 : f64
  }
}

// CHECK-LABEL: llvm.func @dsquare
// CHECK: enzyme.autodiff @square
