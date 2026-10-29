// RUN: %eopt --enzyme-test-forward-call-handler=cube --enzyme-test-forward-call-handler=scale --enzyme-test-forward-call-handler=ext --enzyme-test-forward-call-handler=wcube --enzyme-test-forward-call-handler=twice:tangent --enzyme %s | FileCheck %s

// A call whose callee has a registered forward call handler is differentiated
// by the handler, not through the callee's body. The test handlers call
// @<callee>_jvp (or @<callee>_tangent) instead; their tangents are deliberately
// not the derivatives of the bodies, so the output shows which one was used.

module {
  func.func private @cube(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func private @cube_jvp(%x: f64, %dx: f64) -> (f64, f64) {
    %seven = arith.constant 7.0 : f64
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    %dy = arith.mulf %seven, %dx : f64
    return %x3, %dy : f64, f64
  }

  // The handler runs for an active call; a call with only constant operands
  // and results is left as it is.
  func.func @f(%x: f64, %c: f64) -> f64 {
    %y = func.call @cube(%x) : (f64) -> f64
    %k = func.call @cube(%c) : (f64) -> f64
    %r = arith.addf %y, %k : f64
    return %r : f64
  }
  func.func @df(%x: f64, %dx: f64, %c: f64) -> f64 {
    %r = enzyme.fwddiff @f(%x, %dx, %c) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, f64) -> f64
    return %r : f64
  }

  // A constant operand of an active call gets a zero shadow.
  func.func private @scale(%x: f64, %s: f64) -> f64 {
    %y = arith.mulf %x, %s : f64
    return %y : f64
  }
  func.func private @scale_jvp(%x: f64, %dx: f64, %s: f64, %ds: f64) -> (f64, f64) {
    %y = arith.mulf %x, %s : f64
    %t = arith.addf %dx, %ds : f64
    return %y, %t : f64, f64
  }
  func.func @g(%x: f64, %s: f64) -> f64 {
    %y = func.call @scale(%x, %s) : (f64, f64) -> f64
    return %y : f64
  }
  func.func @dg(%x: f64, %dx: f64, %s: f64) -> f64 {
    %r = enzyme.fwddiff @g(%x, %dx, %s) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, f64) -> f64
    return %r : f64
  }

  // A handler stands in for a missing body.
  func.func private @ext(f64) -> f64
  func.func private @ext_jvp(%x: f64, %dx: f64) -> (f64, f64) {
    %two = arith.constant 2.0 : f64
    %y = arith.mulf %x, %x : f64
    %t = arith.mulf %two, %x : f64
    %dy = arith.mulf %t, %dx : f64
    return %y, %dy : f64, f64
  }
  func.func @h(%x: f64) -> f64 {
    %y = func.call @ext(%x) : (f64) -> f64
    return %y : f64
  }
  func.func @dh(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @h(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }

  // At width 2 the handler sees shadows with the leading width dimension.
  func.func private @wcube(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func private @wcube_jvp(%x: f64, %dx: tensor<2xf64>) -> (f64, tensor<2xf64>) {
    %seven = arith.constant dense<7.0> : tensor<2xf64>
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    %dy = arith.mulf %seven, %dx : tensor<2xf64>
    return %x3, %dy : f64, tensor<2xf64>
  }
  func.func @w(%x: f64) -> f64 {
    %y = func.call @wcube(%x) : (f64) -> f64
    return %y : f64
  }
  func.func @dw(%x: f64, %dx: tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @w(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], width=2 }> : (f64, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }

  // A handler may leave the call's clone to compute the primal result.
  func.func private @twice(%x: f64) -> f64 {
    %y = arith.addf %x, %x : f64
    return %y : f64
  }
  func.func private @twice_tangent(%x: f64, %dx: f64) -> f64 {
    %three = arith.constant 3.0 : f64
    %dy = arith.mulf %three, %dx : f64
    return %dy : f64
  }
  func.func @t(%x: f64) -> f64 {
    %y = func.call @twice(%x) : (f64) -> f64
    %z = arith.mulf %y, %y : f64
    return %z : f64
  }
  func.func @dt(%x: f64, %dx: f64) -> (f64, f64) {
    %r:2 = enzyme.fwddiff @t(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dup>] }> : (f64, f64) -> (f64, f64)
    return %r#0, %r#1 : f64, f64
  }

  // Handlers are keyed on calls: differentiating @cube itself goes through
  // its body.
  func.func @dcube(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @cube(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffef(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[c:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[y:.+]]:2 = call @cube_jvp(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[k:.+]] = call @cube(%[[c]]) : (f64) -> f64
// CHECK-NEXT:    %{{.+}} = arith.addf %[[y]]#0, %[[k]] : f64
// CHECK-NEXT:    return %[[y]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffeg(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[s:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[y:.+]]:2 = call @scale_jvp(%[[x]], %[[dx]], %[[s]], %[[zero]]) : (f64, f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[y]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffeh(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[y:.+]]:2 = call @ext_jvp(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[y]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffe2w(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %[[y:.+]]:2 = call @wcube_jvp(%[[x]], %[[dx]]) : (f64, tensor<2xf64>) -> (f64, tensor<2xf64>)
// CHECK-NEXT:    return %[[y]]#1 : tensor<2xf64>

// CHECK-LABEL: func.func private @fwddiffet(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[dy:.+]] = call @twice_tangent(%[[x]], %[[dx]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[y:.+]] = call @twice(%[[x]]) : (f64) -> f64
// CHECK-NEXT:    %[[a:.+]] = arith.mulf %[[dy]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[b:.+]] = arith.mulf %[[dy]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dz:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[z:.+]] = arith.mulf %[[y]], %[[y]] : f64
// CHECK-NEXT:    return %[[z]], %[[dz]] : f64, f64

// CHECK-LABEL: func.func private @fwddiffecube(
// CHECK-NOT:     call
// CHECK:         return
