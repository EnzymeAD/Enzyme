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

// CHECK-LABEL: func.func @main

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

// CHECK-LABEL: func.func @main

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

// CHECK-LABEL: func.func @main
