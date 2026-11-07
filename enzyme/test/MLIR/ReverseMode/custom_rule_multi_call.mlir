// RUN: %eopt %s --split-input-file --enzyme --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// The same callee called twice in one differentiated function, with the same
// activity. No custom rule is authored: split mode derives one.

module {
  func.func private @sq(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  func.func @outer(%a: f64, %b: f64) -> f64 {
    %ya = func.call @sq(%a) : (f64) -> f64
    %yb = func.call @sq(%b) : (f64) -> f64
    %z = arith.addf %ya, %yb : f64
    return %z : f64
  }

  func.func @main(%a: f64, %b: f64, %dz: f64) -> (f64, f64) {
    %da, %db = enzyme.autodiff @outer(%a, %b, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64, f64) -> (f64, f64)
    return %da, %db : f64, f64
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %0:2 = call @sq_reverse_rule_primal(%arg0) : (f64) -> (f64, f64)
// CHECK-NEXT:    %1:2 = call @sq_reverse_rule_primal(%arg1) : (f64) -> (f64, f64)
// CHECK-NEXT:    %2 = call @sq_reverse_rule_reverse(%arg2, %1#1) : (f64, f64) -> f64
// CHECK-NEXT:    %3 = call @sq_reverse_rule_reverse(%arg2, %0#1) : (f64, f64) -> f64
// CHECK-NEXT:    return %3, %2 : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @sq_reverse_rule_reverse(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    %2 = arith.addf %0, %1 fastmath<fast> : f64
// CHECK-NEXT:    return %2 : f64
// CHECK-NEXT:  }

// -----

// The same callee called twice with DIFFERENT activity: the second call's
// operand is a constant.

module {
  func.func private @mul(%x: f64, %c: f64) -> f64 {
    %y = arith.mulf %x, %c : f64
    return %y : f64
  }

  func.func @outer(%a: f64, %b: f64) -> f64 {
    %k = arith.constant 3.0 : f64
    %ya = func.call @mul(%a, %b) : (f64, f64) -> f64
    %yb = func.call @mul(%a, %k) : (f64, f64) -> f64
    %z = arith.addf %ya, %yb : f64
    return %z : f64
  }

  func.func @main(%a: f64, %b: f64, %dz: f64) -> (f64, f64) {
    %da, %db = enzyme.autodiff @outer(%a, %b, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64, f64) -> (f64, f64)
    return %da, %db : f64, f64
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %cst = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %0:3 = call @mul_reverse_rule_0_primal(%arg0, %arg1) : (f64, f64) -> (f64, f64, f64)
// CHECK-NEXT:    %1:2 = call @mul_reverse_rule_primal(%arg0, %cst) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %2 = call @mul_reverse_rule_reverse(%arg2, %1#1) : (f64, f64) -> f64
// CHECK-NEXT:    %3:2 = call @mul_reverse_rule_0_reverse(%arg2, %0#1, %0#2) : (f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    %4 = arith.addf %2, %3#0 fastmath<fast> : f64
// CHECK-NEXT:    return %4, %3#1 : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_reverse_rule_reverse(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_reverse_rule_0_reverse(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 fastmath<fast> : f64
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 fastmath<fast> : f64
// CHECK-NEXT:    return %0, %1 : f64, f64
// CHECK-NEXT:  }

// -----

// An authored custom rule called twice.

module {
  func.func private @sq(%x: f64) -> f64 attributes {enzyme.custom_rule = @sq_rule} {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  enzyme.custom_reverse_rule @sq_rule {
    %cx = "enzyme.init"() : () -> !enzyme.Cache<f64>
    enzyme.custom_reverse_rule.augmented_primal (%x: f64) -> f64 {
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<f64>, f64) -> ()
      %y = arith.mulf %x, %x : f64
      enzyme.yield %y : f64
    }
    enzyme.custom_reverse_rule.reverse (%ybar: f64) -> f64 {
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<f64>) -> f64
      %three = arith.constant 3.0 : f64
      %t = arith.mulf %three, %x : f64
      %xbar = arith.mulf %ybar, %t : f64
      enzyme.yield %xbar : f64
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f64) -> f64
  }

  func.func @outer(%a: f64, %b: f64) -> f64 {
    %ya = func.call @sq(%a) : (f64) -> f64
    %yb = func.call @sq(%b) : (f64) -> f64
    %z = arith.addf %ya, %yb : f64
    return %z : f64
  }

  func.func @main(%a: f64, %b: f64, %dz: f64) -> (f64, f64) {
    %da, %db = enzyme.autodiff @outer(%a, %b, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64, f64) -> (f64, f64)
    return %da, %db : f64, f64
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %0:2 = call @sq_rule_primal(%arg0) : (f64) -> (f64, f64)
// CHECK-NEXT:    %1:2 = call @sq_rule_primal(%arg1) : (f64) -> (f64, f64)
// CHECK-NEXT:    %2 = call @sq_rule_reverse(%arg2, %1#1) : (f64, f64) -> f64
// CHECK-NEXT:    %3 = call @sq_rule_reverse(%arg2, %0#1) : (f64, f64) -> f64
// CHECK-NEXT:    return %3, %2 : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @sq_rule_reverse(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg1, %cst : f64
// CHECK-NEXT:    %1 = arith.mulf %arg0, %0 : f64
// CHECK-NEXT:    return %1 : f64
// CHECK-NEXT:  }
