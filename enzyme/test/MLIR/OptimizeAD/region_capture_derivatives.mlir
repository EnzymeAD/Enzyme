// RUN: %eopt %s --pass-pipeline='builtin.module(outline-enzyme-regions,enzyme{postpasses="canonicalize,remove-unnecessary-enzyme-ops,canonicalize"},inline,canonicalize,symbol-dce)' | FileCheck %s
// RUN: %eopt %s --pass-pipeline='builtin.module(outline-enzyme-regions,inline-enzyme-regions,symbol-dce,outline-enzyme-regions,enzyme{postpasses="canonicalize,remove-unnecessary-enzyme-ops,canonicalize"},inline,canonicalize,symbol-dce)' | FileCheck %s

// Captures are constant at the region's differentiation boundary, even when
// they have the same incoming SSA value as an explicit active argument.
// Check the generated derivatives in both AD modes, directly and after an
// inline/outline round trip. Inlining exposes the derivative formulas
// without requiring a separate execution toolchain.

// CHECK-LABEL: func.func @capture_reverse
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[DX:.*]] = arith.mulf %[[SEED]], %[[X]]
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[DX]], %[[THREE]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @capture_reverse(%x: f64, %seed: f64) -> f64 {
  %gradient = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %three = arith.constant 3.0 : f64
    %a = arith.mulf %three, %y : f64
    %b = arith.mulf %a, %x : f64
    enzyme.yield %b : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %gradient : f64
}

// CHECK-LABEL: func.func @capture_forward
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[D3:.*]] = arith.mulf %[[SEED]], %[[THREE]]
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[D3]], %[[X]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @capture_forward(%x: f64, %seed: f64) -> f64 {
  %tangent = enzyme.fwddiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %three = arith.constant 3.0 : f64
    %a = arith.mulf %three, %y : f64
    %b = arith.mulf %a, %x : f64
    enzyme.yield %b : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %tangent : f64
}

// The same value occurs in two active slots, an explicit constant slot and a
// capture. The two active slots must keep their independent differentiation
// roles, and neither constant occurrence contributes a derivative.
// CHECK-LABEL: func.func @repeated_reverse
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[DZ:.*]] = arith.mulf %[[SEED]], %[[X]]
// CHECK-NEXT: %[[DY:.*]] = arith.mulf %[[SEED]], %[[X]]
// CHECK-NEXT: return %[[DY]], %[[DZ]] : f64, f64
func.func @repeated_reverse(%x: f64, %seed: f64) -> (f64, f64) {
  %gradients:2 = enzyme.autodiff_region(%x, %x, %x, %seed) {
  ^bb0(%y: f64, %c: f64, %z: f64):
    %a = arith.mulf %y, %c : f64
    %b = arith.mulf %z, %x : f64
    %sum = arith.addf %a, %b : f64
    enzyme.yield %sum : f64
  } <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64, f64, f64) -> (f64, f64)
  return %gradients#0, %gradients#1 : f64, f64
}

// CHECK-LABEL: func.func @repeated_forward
// CHECK-SAME: (%[[X:.*]]: f64, %[[DY:.*]]: f64, %[[DZ:.*]]: f64)
// CHECK-NEXT: %[[A:.*]] = arith.mulf %[[DY]], %[[X]]
// CHECK-NEXT: %[[B:.*]] = arith.mulf %[[DZ]], %[[X]]
// CHECK-NEXT: %[[RESULT:.*]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @repeated_forward(%x: f64, %dy: f64, %dz: f64) -> f64 {
  %tangent = enzyme.fwddiff_region(%x, %dy, %x, %x, %dz) {
  ^bb0(%y: f64, %c: f64, %z: f64):
    %a = arith.mulf %y, %c : f64
    %b = arith.mulf %z, %x : f64
    %sum = arith.addf %a, %b : f64
    enzyme.yield %sum : f64
  } <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64, f64, f64) -> f64
  return %tangent : f64
}

// Scalar SCF equivalent of Enzyme-JAX #3204: start at .2, repeatedly compute
// d/dy(3*y*x) at y=x while x<1. Correct steps are .2 -> .6 -> 1.8; conflating
// the capture with the active argument produces .2 -> 1.2 instead.
// CHECK-LABEL: func.func @loop_reverse()
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[INITIAL:.*]] = arith.constant 2.000000e-01 : f64
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT: %[[RESULT:.*]] = scf.while (%[[X:.*]] = %[[INITIAL]])
// CHECK-NEXT: %[[CONTINUE:.*]] = arith.cmpf olt, %[[X]], %[[ONE]] : f64
// CHECK-NEXT: scf.condition(%[[CONTINUE]]) %[[X]] : f64
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f64):
// CHECK-NEXT: %[[NEXT:.*]] = arith.mulf %[[X]], %[[THREE]]
// CHECK-NEXT: scf.yield %[[NEXT]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @loop_reverse() -> f64 {
  %initial = arith.constant 0.2 : f64
  %one = arith.constant 1.0 : f64
  %result = scf.while (%x = %initial) : (f64) -> f64 {
    %continue = arith.cmpf olt, %x, %one : f64
    scf.condition(%continue) %x : f64
  } do {
  ^bb0(%x: f64):
    %gradient = enzyme.autodiff_region(%x, %one) {
    ^bb0(%y: f64):
      %three = arith.constant 3.0 : f64
      %a = arith.mulf %three, %y : f64
      %b = arith.mulf %a, %x : f64
      enzyme.yield %b : f64
    } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
    scf.yield %gradient : f64
  }
  return %result : f64
}

// CHECK-LABEL: func.func @loop_forward()
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[INITIAL:.*]] = arith.constant 2.000000e-01 : f64
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT: %[[RESULT:.*]] = scf.while (%[[X:.*]] = %[[INITIAL]])
// CHECK-NEXT: %[[CONTINUE:.*]] = arith.cmpf olt, %[[X]], %[[ONE]] : f64
// CHECK-NEXT: scf.condition(%[[CONTINUE]]) %[[X]] : f64
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f64):
// CHECK-NEXT: %[[NEXT:.*]] = arith.mulf %[[X]], %[[THREE]]
// CHECK-NEXT: scf.yield %[[NEXT]] : f64
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @loop_forward() -> f64 {
  %initial = arith.constant 0.2 : f64
  %one = arith.constant 1.0 : f64
  %result = scf.while (%x = %initial) : (f64) -> f64 {
    %continue = arith.cmpf olt, %x, %one : f64
    scf.condition(%continue) %x : f64
  } do {
  ^bb0(%x: f64):
    %tangent = enzyme.fwddiff_region(%x, %one) {
    ^bb0(%y: f64):
      %three = arith.constant 3.0 : f64
      %a = arith.mulf %three, %y : f64
      %b = arith.mulf %a, %x : f64
      enzyme.yield %b : f64
    } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    scf.yield %tangent : f64
  }
  return %result : f64
}

// Computations on captured or explicitly inactive values stay inactive for
// this derivative, even though the caller's x is also the incoming value of
// active y.
// CHECK-LABEL: func.func @inactive_coefficient_reverse
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[SQUARE:.*]] = arith.mulf %[[X]], %[[X]]
// CHECK-NEXT: %[[COEFFICIENT:.*]] = arith.addf %[[SQUARE]], %[[THREE]]
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[SEED]], %[[COEFFICIENT]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @inactive_coefficient_reverse(%x: f64, %seed: f64) -> f64 {
  %gradient = enzyme.autodiff_region(%x, %x, %seed) {
  ^bb0(%y: f64, %c: f64):
    %three = arith.constant 3.0 : f64
    %square = arith.mulf %c, %x : f64
    %coefficient = arith.addf %square, %three : f64
    %product = arith.mulf %y, %coefficient : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64, f64) -> f64
  return %gradient : f64
}

// CHECK-LABEL: func.func @inactive_coefficient_forward
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[SQUARE:.*]] = arith.mulf %[[X]], %[[X]]
// CHECK-NEXT: %[[COEFFICIENT:.*]] = arith.addf %[[SQUARE]], %[[THREE]]
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[SEED]], %[[COEFFICIENT]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @inactive_coefficient_forward(%x: f64, %seed: f64) -> f64 {
  %tangent = enzyme.fwddiff_region(%x, %seed, %x) {
  ^bb0(%y: f64, %c: f64):
    %three = arith.constant 3.0 : f64
    %square = arith.mulf %c, %x : f64
    %coefficient = arith.addf %square, %three : f64
    %product = arith.mulf %y, %coefficient : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64) -> f64
  return %tangent : f64
}

// Derivative postpasses remove temporary reverse caches before differentiating
// the generated derivative again. Constant at an inner boundary does not mean
// constant at an enclosing boundary. The inner derivative is 3*x and its outer
// derivative is 3, since the inner capture is active in the outer region.
// CHECK-LABEL: func.func @nested_forward
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[SEED]], %[[THREE]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @nested_forward(%x: f64, %seed: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %outer = enzyme.fwddiff_region(%x, %seed) {
  ^bb0(%outer_x: f64):
    %inner = enzyme.fwddiff_region(%outer_x, %one) {
    ^bb0(%inner_y: f64):
      %three = arith.constant 3.0 : f64
      %a = arith.mulf %three, %inner_y : f64
      %b = arith.mulf %a, %outer_x : f64
      enzyme.yield %b : f64
    } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    enzyme.yield %inner : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %outer : f64
}

// CHECK-LABEL: func.func @nested_reverse
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[SEED]], %[[THREE]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @nested_reverse(%x: f64, %seed: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %outer = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%outer_x: f64):
    %inner = enzyme.autodiff_region(%outer_x, %one) {
    ^bb0(%inner_y: f64):
      %three = arith.constant 3.0 : f64
      %a = arith.mulf %three, %inner_y : f64
      %b = arith.mulf %a, %outer_x : f64
      enzyme.yield %b : f64
    } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
    enzyme.yield %inner : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %outer : f64
}

// CHECK-LABEL: func.func @nested_forward_over_reverse
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[SEED]], %[[THREE]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @nested_forward_over_reverse(%x: f64, %seed: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %outer = enzyme.fwddiff_region(%x, %seed) {
  ^bb0(%outer_x: f64):
    %inner = enzyme.autodiff_region(%outer_x, %one) {
    ^bb0(%inner_y: f64):
      %three = arith.constant 3.0 : f64
      %a = arith.mulf %three, %inner_y : f64
      %b = arith.mulf %a, %outer_x : f64
      enzyme.yield %b : f64
    } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
    enzyme.yield %inner : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %outer : f64
}

// CHECK-LABEL: func.func @nested_reverse_over_forward
// CHECK-SAME: (%[[X:.*]]: f64, %[[SEED:.*]]: f64)
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT: %[[RESULT:.*]] = arith.mulf %[[SEED]], %[[THREE]]
// CHECK-NEXT: return %[[RESULT]] : f64
func.func @nested_reverse_over_forward(%x: f64, %seed: f64) -> f64 {
  %one = arith.constant 1.0 : f64
  %outer = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%outer_x: f64):
    %inner = enzyme.fwddiff_region(%outer_x, %one) {
    ^bb0(%inner_y: f64):
      %three = arith.constant 3.0 : f64
      %a = arith.mulf %three, %inner_y : f64
      %b = arith.mulf %a, %outer_x : f64
      enzyme.yield %b : f64
    } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    enzyme.yield %inner : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %outer : f64
}
