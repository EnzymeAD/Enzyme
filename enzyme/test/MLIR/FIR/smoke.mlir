// Milestone 0: prove fir-enzyme-opt (fir-opt with Enzyme-MLIR linked in) runs
// the `enzyme` differentiation pass in a context that already carries the
// FIR/HLFIR dialects. Differentiating hlfir.* array intrinsics needs the Tier-1
// external models (next milestone); here we differentiate a scalar function to
// exercise only the FIR+Enzyme wiring.
//
// REQUIRES: fir_enzyme_opt
// RUN: %fireopt --enzyme %s | FileCheck %s

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
  // CHECK:       %[[a:.+]] = arith.mulf %[[dx]], %[[x]] : f64
  // CHECK:       %[[b:.+]] = arith.mulf %[[dx]], %[[x]] : f64
  // CHECK:       arith.addf %[[a]], %[[b]] : f64
}
