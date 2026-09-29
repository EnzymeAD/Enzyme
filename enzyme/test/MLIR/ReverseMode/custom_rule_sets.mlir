// RUN: %eopt %s --split-input-file --enzyme --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math --verify-diagnostics | FileCheck %s

// A rule set: one authored rule per activity pattern. Each call uses the rule
// that serves its activities while differentiating the fewest arguments. The
// rules mark their cotangents (factors 2 and 5) so the output shows which ran.

module {
  func.func private @mul(%a: f64, %x: f64) -> f64 attributes {enzyme.custom_rule = [@mul_both, @mul_x]} {
    %y = arith.mulf %a, %x : f64
    return %y : f64
  }

  enzyme.custom_reverse_rule @mul_both {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<f64>
    %cx = "enzyme.init"() : () -> !enzyme.Cache<f64>
    enzyme.custom_reverse_rule.augmented_primal (%a: f64, %x: f64) -> f64 {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<f64>, f64) -> ()
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<f64>, f64) -> ()
      %y = arith.mulf %a, %x : f64
      enzyme.yield %y : f64
    }
    enzyme.custom_reverse_rule.reverse (%ybar: f64) -> (f64, f64) {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<f64>) -> f64
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<f64>) -> f64
      %two = arith.constant 2.0 : f64
      %abar0 = arith.mulf %ybar, %x : f64
      %abar = arith.mulf %two, %abar0 : f64
      %xbar0 = arith.mulf %ybar, %a : f64
      %xbar = arith.mulf %two, %xbar0 : f64
      enzyme.yield %abar, %xbar : f64, f64
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f64, f64) -> f64
  }

  enzyme.custom_reverse_rule @mul_x {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<f64>
    enzyme.custom_reverse_rule.augmented_primal (%a: f64, %x: f64) -> f64 {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<f64>, f64) -> ()
      %y = arith.mulf %a, %x : f64
      enzyme.yield %y : f64
    }
    enzyme.custom_reverse_rule.reverse (%ybar: f64) -> f64 {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<f64>) -> f64
      %five = arith.constant 5.0 : f64
      %xbar0 = arith.mulf %ybar, %a : f64
      %xbar = arith.mulf %five, %xbar0 : f64
      enzyme.yield %xbar : f64
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f64, f64) -> f64
  }

  // Both operands active, then only x active (the first operand is a constant).
  func.func @outer(%a: f64, %x: f64) -> f64 {
    %k = arith.constant 3.0 : f64
    %y1 = func.call @mul(%a, %x) : (f64, f64) -> f64
    %y2 = func.call @mul(%k, %x) : (f64, f64) -> f64
    %z = arith.addf %y1, %y2 : f64
    return %z : f64
  }

  func.func @main(%a: f64, %x: f64, %dz: f64) -> (f64, f64) {
    %da, %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64, f64) -> (f64, f64)
    return %da, %dx : f64, f64
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %cst = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %0:3 = call @mul_both_primal(%arg0, %arg1) : (f64, f64) -> (f64, f64, f64)
// CHECK-NEXT:    %1:2 = call @mul_x_primal(%cst, %arg1) : (f64, f64) -> (f64, f64)
// CHECK-NEXT:    %2 = call @mul_x_reverse(%arg2, %1#1) : (f64, f64) -> f64
// CHECK-NEXT:    %3:2 = call @mul_both_reverse(%arg2, %0#1, %0#2) : (f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    %4 = arith.addf %2, %3#1 fastmath<fast> : f64
// CHECK-NEXT:    return %3#0, %4 : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_both_reverse(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 : f64
// CHECK-NEXT:    %1 = arith.mulf %0, %cst : f64
// CHECK-NEXT:    %2 = arith.mulf %arg0, %arg1 : f64
// CHECK-NEXT:    %3 = arith.mulf %2, %cst : f64
// CHECK-NEXT:    return %1, %3 : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_x_reverse(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg1 : f64
// CHECK-NEXT:    %1 = arith.mulf %0, %cst : f64
// CHECK-NEXT:    return %1 : f64
// CHECK-NEXT:  }

// -----

// A single all-active rule serves a call whose first operand is a constant:
// the cotangent it computes for that operand is dropped.

module {
  func.func private @mul(%a: f64, %x: f64) -> f64 attributes {enzyme.custom_rule = @mul_both} {
    %y = arith.mulf %a, %x : f64
    return %y : f64
  }

  enzyme.custom_reverse_rule @mul_both {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<f64>
    %cx = "enzyme.init"() : () -> !enzyme.Cache<f64>
    enzyme.custom_reverse_rule.augmented_primal (%a: f64, %x: f64) -> f64 {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<f64>, f64) -> ()
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<f64>, f64) -> ()
      %y = arith.mulf %a, %x : f64
      enzyme.yield %y : f64
    }
    enzyme.custom_reverse_rule.reverse (%ybar: f64) -> (f64, f64) {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<f64>) -> f64
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<f64>) -> f64
      %two = arith.constant 2.0 : f64
      %abar0 = arith.mulf %ybar, %x : f64
      %abar = arith.mulf %two, %abar0 : f64
      %xbar0 = arith.mulf %ybar, %a : f64
      %xbar = arith.mulf %two, %xbar0 : f64
      enzyme.yield %abar, %xbar : f64, f64
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f64, f64) -> f64
  }

  func.func @outer(%x: f64) -> f64 {
    %k = arith.constant 3.0 : f64
    %y = func.call @mul(%k, %x) : (f64, f64) -> f64
    return %y : f64
  }

  func.func @main(%x: f64, %dz: f64) -> f64 {
    %dx = enzyme.autodiff @outer(%x, %dz) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %0:3 = call @mul_both_primal(%cst, %arg0) : (f64, f64) -> (f64, f64, f64)
// CHECK-NEXT:    %1:2 = call @mul_both_reverse(%arg1, %0#1, %0#2) : (f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    return %1#1 : f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_both_reverse(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 : f64
// CHECK-NEXT:    %1 = arith.mulf %0, %cst : f64
// CHECK-NEXT:    %2 = arith.mulf %arg0, %arg1 : f64
// CHECK-NEXT:    %3 = arith.mulf %2, %cst : f64
// CHECK-NEXT:    return %1, %3 : f64, f64
// CHECK-NEXT:  }

// -----

// No rule in the set serves a call whose first operand is active: both rules
// declare it constant.

module {
  func.func private @mul(%a: f64, %x: f64) -> f64 attributes {enzyme.custom_rule = [@mul_x]} {
    %y = arith.mulf %a, %x : f64
    return %y : f64
  }

  enzyme.custom_reverse_rule @mul_x {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<f64>
    enzyme.custom_reverse_rule.augmented_primal (%a: f64, %x: f64) -> f64 {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<f64>, f64) -> ()
      %y = arith.mulf %a, %x : f64
      enzyme.yield %y : f64
    }
    enzyme.custom_reverse_rule.reverse (%ybar: f64) -> f64 {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<f64>) -> f64
      %xbar = arith.mulf %ybar, %a : f64
      enzyme.yield %xbar : f64
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f64, f64) -> f64
  }

  func.func @outer(%a: f64, %x: f64) -> f64 {
    // expected-error @below {{could not find a rule with the right activity (rule activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_active>])}}
    %y = func.call @mul(%a, %x) : (f64, f64) -> f64
    return %y : f64
  }

  func.func @main(%a: f64, %x: f64, %dz: f64) -> (f64, f64) {
    %da, %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64, f64) -> (f64, f64)
    return %da, %dx : f64, f64
  }
}
