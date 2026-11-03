// RUN: %eopt --test-register-forward-call-handlers="handlers=cube,scale,ext,wcube,twice:fwd,sel,sq:primal-first,shared" --enzyme %s | FileCheck %s

// The rules' tangents differ from the callees' derivatives, so the output shows
// whether the handler or the body was used.

module {
  func.func private @cube(%x: f64) -> f64 attributes {enzyme_math = "cube"} {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func private @cube_fwd_dup(%x: f64, %dx: f64) -> (f64, f64) {
    %seven = arith.constant 7.0 : f64
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    %dy = arith.mulf %seven, %dx : f64
    return %x3, %dy : f64, f64
  }

  // A call with only constant operands and results is left as it is.
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
  func.func private @scale(%x: f64, %s: f64) -> f64 attributes {enzyme_math = "scale"} {
    %y = arith.mulf %x, %s : f64
    return %y : f64
  }
  func.func private @scale_fwd_dup(%x: f64, %dx: f64, %s: f64, %ds: f64) -> (f64, f64) {
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
  func.func private @ext(f64) -> f64 attributes {enzyme_math = "ext"}
  func.func private @ext_fwd_dup(%x: f64, %dx: f64) -> (f64, f64) {
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
  func.func private @wcube(%x: f64) -> f64 attributes {enzyme_math = "wcube"} {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func private @wcube_fwd_dup(%x: f64, %dx: tensor<2xf64>) -> (f64, tensor<2xf64>) {
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
  func.func private @twice(%x: f64) -> f64 attributes {enzyme_math = "twice"} {
    %y = arith.addf %x, %x : f64
    return %y : f64
  }
  func.func private @twice_fwd(%x: f64, %dx: f64) -> f64 {
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

  // Differentiating @cube itself goes through its body.
  func.func @dcube(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @cube(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }

  // The derivative of @cube does not inherit its key.
  func.func @ddcube(%x: f64, %ddx: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @dcube(%x, %ddx, %dx) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, f64) -> f64
    return %r : f64
  }

  // Zero shadow of a constant result of an already replaced call.
  func.func private @split(%x: f64) -> (f64, i1) {
    %y = arith.mulf %x, %x : f64
    %zero = arith.constant 0.0 : f64
    %b = arith.cmpf ogt, %x, %zero : f64
    return %y, %b : f64, i1
  }
  func.func private @sel(%y: f64, %b: i1) -> f64 attributes {enzyme_math = "sel"} {
    %zero = arith.constant 0.0 : f64
    %r = arith.select %b, %y, %zero : f64
    return %r : f64
  }
  func.func private @sel_fwd_dup(%y: f64, %dy: f64, %b: i1, %db: i1) -> (f64, f64) {
    %zero = arith.constant 0.0 : f64
    %r = arith.select %b, %y, %zero : f64
    %dr = arith.select %b, %dy, %zero : f64
    return %r, %dr : f64, f64
  }
  func.func @s(%x: f64) -> f64 {
    %y, %b = func.call @split(%x) : (f64) -> (f64, i1)
    %r = func.call @sel(%y, %b) : (f64, i1) -> f64
    return %r : f64
  }
  func.func @ds(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @s(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }

  // A handler may replace the primal results first and go on inserting ops.
  func.func private @sq(%x: f64) -> f64 attributes {enzyme_math = "sq"} {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func private @sq_primal(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func private @sq_fwd(%x: f64, %dx: f64) -> f64 {
    %five = arith.constant 5.0 : f64
    %dy = arith.mulf %five, %dx : f64
    return %dy : f64
  }
  func.func @q(%x: f64) -> f64 {
    %y = func.call @sq(%x) : (f64) -> f64
    %z = arith.addf %y, %x : f64
    return %z : f64
  }
  func.func @dq(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @q(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }

  // Callees with the same key share its handler; the call's key wins.
  func.func private @alpha(%x: f64) -> f64 attributes {enzyme_math = "shared"} {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func private @beta(%x: f64) -> f64 attributes {enzyme_math = "shared"} {
    %y = arith.addf %x, %x : f64
    return %y : f64
  }
  func.func private @gamma(%x: f64) -> f64 attributes {enzyme_math = "cube"} {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
  func.func private @shared_fwd_dup(%x: f64, %dx: f64) -> (f64, f64) {
    %eleven = arith.constant 11.0 : f64
    %dy = arith.mulf %eleven, %dx : f64
    return %x, %dy : f64, f64
  }
  func.func @k(%x: f64) -> f64 {
    %a = func.call @alpha(%x) : (f64) -> f64
    %b = func.call @beta(%a) : (f64) -> f64
    %c = func.call @gamma(%b) : (f64) -> f64
    %d = func.call @gamma(%c) {enzyme_math = "shared"} : (f64) -> f64
    return %d : f64
  }
  func.func @dk(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @k(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }

  // A key on an indirect call selects its handler.
  llvm.func @ind(%fp: !llvm.ptr, %x: f64) -> f64 {
    %y = llvm.call %fp(%x) {enzyme_math = "cube"} : !llvm.ptr, (f64) -> f64
    llvm.return %y : f64
  }
  llvm.func @dind(%fp: !llvm.ptr, %x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @ind(%fp, %x, %dx) <{ activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!llvm.ptr, f64, f64) -> f64
    llvm.return %r : f64
  }

  // Without `enzyme_math`, the callee's name is the key. @delta's key has no
  // handler, so its name is not used.
  func.func private @shared(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func private @delta(%x: f64) -> f64 attributes {enzyme_math = "unregistered"} {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @e(%x: f64) -> f64 {
    %y = func.call @shared(%x) : (f64) -> f64
    %z = func.call @delta(%y) : (f64) -> f64
    return %z : f64
  }
  func.func @de(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @e(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffef(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[c:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[y:.+]]:2 = call @cube_fwd_dup(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[k:.+]] = call @cube(%[[c]]) : (f64) -> f64
// CHECK-NEXT:    %{{.+}} = arith.addf %[[y]]#0, %[[k]] : f64
// CHECK-NEXT:    return %[[y]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffeg(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[s:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[y:.+]]:2 = call @scale_fwd_dup(%[[x]], %[[dx]], %[[s]], %[[zero]]) : (f64, f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[y]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffeh(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[y:.+]]:2 = call @ext_fwd_dup(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[y]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffe2w(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %[[y:.+]]:2 = call @wcube_fwd_dup(%[[x]], %[[dx]]) : (f64, tensor<2xf64>) -> (f64, tensor<2xf64>)
// CHECK-NEXT:    return %[[y]]#1 : tensor<2xf64>

// CHECK-LABEL: func.func private @fwddiffet(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[dy:.+]] = call @twice_fwd(%[[x]], %[[dx]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[y:.+]] = call @twice(%[[x]]) : (f64) -> f64
// CHECK-NEXT:    %[[a:.+]] = arith.mulf %[[dy]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[b:.+]] = arith.mulf %[[dy]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dz:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[z:.+]] = arith.mulf %[[y]], %[[y]] : f64
// CHECK-NEXT:    return %[[z]], %[[dz]] : f64, f64

// CHECK-LABEL: func.func private @fwddiffecube(
// CHECK-SAME:      %{{.+}}: f64, %{{.+}}: f64) -> f64 {
// CHECK-NOT:     call
// CHECK:         return

// CHECK-LABEL: func.func private @fwddiffedcube(
// CHECK-SAME:      %[[x:.+]]: f64, %[[ddx:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[r:.+]]:2 = call @fwddiffefwddiffecube(%[[x]], %[[ddx]], %[[dx]]) : (f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[r]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffes(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[false:.+]] = arith.constant false
// CHECK-NEXT:    %[[y:.+]]:3 = call @fwddiffesplit(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64, i1)
// CHECK-NEXT:    %[[r:.+]]:2 = call @sel_fwd_dup(%[[y]]#0, %[[y]]#1, %[[y]]#2, %[[false]]) : (f64, f64, i1, i1) -> (f64, f64)
// CHECK-NEXT:    return %[[r]]#1 : f64

// CHECK-LABEL: func.func private @fwddiffeq(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[y:.+]] = call @sq_primal(%[[x]]) : (f64) -> f64
// CHECK-NEXT:    %[[dy:.+]] = call @sq_fwd(%[[x]], %[[dx]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[dz:.+]] = arith.addf %[[dy]], %[[dx]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = arith.addf %[[y]], %[[x]] : f64
// CHECK-NEXT:    return %[[dz]] : f64

// CHECK-LABEL: func.func private @fwddiffek(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[a:.+]]:2 = call @shared_fwd_dup(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[b:.+]]:2 = call @shared_fwd_dup(%[[a]]#0, %[[a]]#1) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[c:.+]]:2 = call @cube_fwd_dup(%[[b]]#0, %[[b]]#1) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[d:.+]]:2 = call @shared_fwd_dup(%[[c]]#0, %[[c]]#1) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[d]]#1 : f64

// CHECK-LABEL: llvm.func @fwddiffeind(
// CHECK-SAME:      %{{.+}}: !llvm.ptr, %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64
// CHECK-NEXT:    %{{.+}}:2 = func.call @cube_fwd_dup(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)

// CHECK-LABEL: func.func private @fwddiffee(
// CHECK-SAME:      %[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[y:.+]]:2 = call @shared_fwd_dup(%[[x]], %[[dx]]) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %[[z:.+]]:2 = call @fwddiffedelta(%[[y]]#0, %[[y]]#1) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %[[z]]#1 : f64
