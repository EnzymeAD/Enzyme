// RUN: %eopt --outline-enzyme-regions --split-input-file %s | FileCheck %s
// RUN: %eopt --pass-pipeline='builtin.module(outline-enzyme-regions,inline-enzyme-regions,symbol-dce,outline-enzyme-regions,symbol-dce)' --split-input-file %s | FileCheck %s

// The incoming x is also captured, but only the block argument y is active.
// At y = x, d/dy (3*y*x) is 3*x, not 6*x.
func.func @capture_reverse(%x: f64, %seed: f64) -> f64 {
  %d = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %three = arith.constant 3.0 : f64
    %scaled = arith.mulf %y, %three : f64
    %product = arith.mulf %scaled, %x : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @capture_reverse(
// CHECK-SAME: %[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK: enzyme.autodiff @capture_reverse_to_diff0(%[[X]], %[[X]], %[[SEED]]) <activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]>
// CHECK-LABEL: func.func private @capture_reverse_to_diff0(
// CHECK-SAME: %[[Y:.*]]: f64, %[[CAPTURE:.*]]: f64)
// CHECK: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK: %[[SCALED:.*]] = arith.mulf %[[Y]], %[[THREE]] : f64
// CHECK: arith.mulf %[[SCALED]], %[[CAPTURE]] : f64

// -----

func.func @capture_forward(%x: f64, %dx: f64) -> f64 {
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    %three = arith.constant 3.0 : f64
    %scaled = arith.mulf %y, %three : f64
    %product = arith.mulf %scaled, %x : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @capture_forward(
// CHECK-SAME: %[[X:.*]]: f64, %[[DX:.*]]: f64)
// CHECK: enzyme.fwddiff @capture_forward_to_fwddiff0(%[[X]], %[[DX]], %[[X]]) <activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]>
// CHECK-LABEL: func.func private @capture_forward_to_fwddiff0(
// CHECK-SAME: %[[Y:.*]]: f64, %[[CAPTURE:.*]]: f64)
// CHECK: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK: %[[SCALED:.*]] = arith.mulf %[[Y]], %[[THREE]] : f64
// CHECK: arith.mulf %[[SCALED]], %[[CAPTURE]] : f64

// -----

// Repeated incoming values have independent activities. The capture must not
// become either the first active argument or the second constant argument.
func.func @repeated_reverse(%x: f64, %seed: f64) -> f64 {
  %d = enzyme.autodiff_region(%x, %x, %seed) {
  ^bb0(%y: f64, %z: f64):
    %sum = arith.addf %y, %z : f64
    %product = arith.mulf %sum, %x : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @repeated_reverse(
// CHECK-SAME: %[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK: enzyme.autodiff @repeated_reverse_to_diff0(%[[X]], %[[X]], %[[X]], %[[SEED]]) <activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>]
// CHECK-LABEL: func.func private @repeated_reverse_to_diff0(
// CHECK-SAME: %[[Y:.*]]: f64, %[[Z:.*]]: f64, %[[CAPTURE:.*]]: f64)
// CHECK: %[[SUM:.*]] = arith.addf %[[Y]], %[[Z]] : f64
// CHECK: arith.mulf %[[SUM]], %[[CAPTURE]] : f64

// -----

// Repeat the same primal with two distinct forward seeds.
func.func @repeated_forward(%x: f64, %dx: f64, %dz: f64) -> f64 {
  %d = enzyme.fwddiff_region(%x, %dx, %x, %dz) {
  ^bb0(%y: f64, %z: f64):
    %sum = arith.addf %y, %z : f64
    %product = arith.mulf %sum, %x : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @repeated_forward(
// CHECK-SAME: %[[X:.*]]: f64, %[[DX:.*]]: f64, %[[DZ:.*]]: f64)
// CHECK: enzyme.fwddiff @repeated_forward_to_fwddiff0(%[[X]], %[[DX]], %[[X]], %[[DZ]], %[[X]]) <activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>]
// CHECK-LABEL: func.func private @repeated_forward_to_fwddiff0(
// CHECK-SAME: %[[Y:.*]]: f64, %[[Z:.*]]: f64, %[[CAPTURE:.*]]: f64)
// CHECK: %[[SUM:.*]] = arith.addf %[[Y]], %[[Z]] : f64
// CHECK: arith.mulf %[[SUM]], %[[CAPTURE]] : f64

// -----

// Captures inside a loop retain the same boundary activity as direct uses.
func.func @loop_capture(%x: f64, %seed: f64, %n: index) -> f64 {
  %d = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sum = scf.for %i = %zero to %n step %one iter_args(%v = %y) -> f64 {
      %product = arith.mulf %v, %x : f64
      scf.yield %product : f64
    }
    enzyme.yield %sum : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %d : f64
}

// CHECK-LABEL: func.func @loop_capture(
// CHECK-SAME: %[[X:.*]]: f64, %[[SEED:.*]]: f64, %[[N:.*]]: index)
// CHECK: enzyme.autodiff @loop_capture_to_diff0(%[[X]], %[[X]], %[[N]], %[[SEED]]) <activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>]
// CHECK-LABEL: func.func private @loop_capture_to_diff0(
// CHECK-SAME: %[[Y:.*]]: f64, %[[CAPTURE:.*]]: f64, %[[N:.*]]: index)
// CHECK: scf.for {{.*}} to %[[N]] {{.*}} iter_args(%[[V:.*]] = %[[Y]])
// CHECK: arith.mulf %[[V]], %[[CAPTURE]] : f64
