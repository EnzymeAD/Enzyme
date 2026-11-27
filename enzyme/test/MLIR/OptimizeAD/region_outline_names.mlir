// RUN: %eopt %s --outline-enzyme-regions | FileCheck %s
// RUN: %eopt %s --outline-enzyme-regions --inline-enzyme-regions --hoist-enzyme-regions --outline-enzyme-regions | FileCheck %s --check-prefix=ROUNDTRIP

// Existing symbols must survive, and generated requests must use fresh names.
module {
  func.func private @forward_to_fwddiff0(%x: f64) -> f64 {
    return %x : f64
  }
  func.func private @reverse_to_diff2(%x: f64) -> f64 {
    return %x : f64
  }

  // CHECK-LABEL: func.func @forward
  // CHECK: enzyme.fwddiff @forward_to_fwddiff1
  // ROUNDTRIP-LABEL: func.func @forward
  // ROUNDTRIP: enzyme.fwddiff @forward_to_fwddiff2
  func.func @forward(%x: f64, %dx: f64) -> f64 {
    %d = enzyme.fwddiff_region(%x, %dx) {
    ^bb0(%arg: f64):
      %square = arith.mulf %arg, %arg : f64
      enzyme.yield %square : f64
    } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    return %d : f64
  }

  // CHECK-LABEL: func.func @reverse
  // CHECK: enzyme.autodiff @reverse_to_diff3
  // ROUNDTRIP-LABEL: func.func @reverse
  // ROUNDTRIP: enzyme.autodiff @reverse_to_diff4
  func.func @reverse(%x: f64, %seed: f64) -> f64 {
    %d = enzyme.autodiff_region(%x, %seed) {
    ^bb0(%arg: f64):
      %square = arith.mulf %arg, %arg : f64
      enzyme.yield %square : f64
    } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
    return %d : f64
  }
}
