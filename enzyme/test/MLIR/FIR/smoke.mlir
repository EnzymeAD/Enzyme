// The FIREnzyme plugin runs the `enzyme` pass inside a fir-opt that already
// carries the FIR/HLFIR dialects. A scalar function, to test only the wiring.
//
// REQUIRES: fir_enzyme_plugin
// RUN: %fir_enzyme --pass-pipeline='builtin.module(enzyme)' %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @dsquare(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) {
      activity=[#enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_dupnoneed>]
    } : (f64, f64) -> (f64)
    return %r : f64
  }
  // d/dx (x*x) = dx*x + dx*x
  // CHECK-LABEL: func.func private @fwddiffesquare(
  // CHECK-SAME:  %[[x:.+]]: f64, %[[dx:.+]]: f64)
  // CHECK:       %[[a:.+]] = arith.mulf %[[dx]], %[[x]] fastmath<fast> : f64
  // CHECK:       %[[b:.+]] = arith.mulf %[[dx]], %[[x]] fastmath<fast> : f64
  // CHECK:       arith.addf %[[a]], %[[b]] fastmath<fast> : f64
}
