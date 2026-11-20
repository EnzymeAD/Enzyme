// RUN: %eopt --outline-enzyme-regions --split-input-file %s | FileCheck %s
// RUN: %eopt --pass-pipeline='builtin.module(outline-enzyme-regions,inline-enzyme-regions,symbol-dce,outline-enzyme-regions,symbol-dce)' --split-input-file %s | FileCheck %s

// The outer argument is active at the outer boundary, but its captured use is
// constant at the inner boundary. Outline inner regions before their parents,
// including when the two regions use different differentiation modes.
func.func @reverse_over_forward(%x: f64, %seed: f64) -> f64 {
  %d = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %one = arith.constant 1.0 : f64
    %inner = enzyme.fwddiff_region(%y, %one) {
    ^bb0(%z: f64):
      %product = arith.mulf %z, %y : f64
      enzyme.yield %product : f64
    } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    %result = arith.mulf %inner, %x : f64
    enzyme.yield %result : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @reverse_over_forward(
// CHECK-SAME: %[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK: enzyme.autodiff @reverse_over_forward_to_diff1(%[[X]], %[[X]], %[[SEED]]) <activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]
// CHECK-LABEL: func.func private @reverse_over_forward_to_diff1(
// CHECK-SAME: %[[Y:.*]]: f64, %[[CAPTURE:.*]]: f64)
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[INNER:.*]] = enzyme.fwddiff @reverse_over_forward_to_fwddiff0(%[[Y]], %[[ONE]], %[[Y]]) <activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>]
// CHECK: arith.mulf %[[INNER]], %[[CAPTURE]] : f64
// CHECK-LABEL: func.func private @reverse_over_forward_to_fwddiff0(
// CHECK-SAME: %[[Z:.*]]: f64, %[[INNER_CAPTURE:.*]]: f64)
// CHECK: arith.mulf %[[Z]], %[[INNER_CAPTURE]] : f64

// -----

func.func @forward_over_reverse(%x: f64, %seed: f64) -> f64 {
  %d = enzyme.fwddiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %one = arith.constant 1.0 : f64
    %inner = enzyme.autodiff_region(%y, %one) {
    ^bb0(%z: f64):
      %product = arith.mulf %z, %y : f64
      enzyme.yield %product : f64
    } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
    %result = arith.mulf %inner, %x : f64
    enzyme.yield %result : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @forward_over_reverse(
// CHECK-SAME: %[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK: enzyme.fwddiff @forward_over_reverse_to_fwddiff1(%[[X]], %[[SEED]], %[[X]]) <activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>]
// CHECK-LABEL: func.func private @forward_over_reverse_to_fwddiff1(
// CHECK-SAME: %[[Y:.*]]: f64, %[[CAPTURE:.*]]: f64)
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK: %[[INNER:.*]] = enzyme.autodiff @forward_over_reverse_to_diff0(%[[Y]], %[[Y]], %[[ONE]]) <activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]
// CHECK: arith.mulf %[[INNER]], %[[CAPTURE]] : f64
// CHECK-LABEL: func.func private @forward_over_reverse_to_diff0(
// CHECK-SAME: %[[Z:.*]]: f64, %[[INNER_CAPTURE:.*]]: f64)
// CHECK: arith.mulf %[[Z]], %[[INNER_CAPTURE]] : f64
