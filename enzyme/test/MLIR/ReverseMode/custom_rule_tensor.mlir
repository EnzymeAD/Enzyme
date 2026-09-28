// RUN: %eopt %s --split-input-file --enzyme --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// A tensor-typed custom reverse rule shaped like one a frontend generates from
// an authored derivative graph: the augmented primal stages its residuals into
// caches, and the reverse reads them back. The x cotangent is deliberately
// doubled, so the output proves the rule was used rather than derived.

module {
  func.func private @mul(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> attributes {enzyme.custom_rule = @mul_rule} {
    %y = arith.mulf %a, %x : tensor<4xf64>
    return %y : tensor<4xf64>
  }

  enzyme.custom_reverse_rule @mul_rule {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<tensor<4xf64>>
    %cx = "enzyme.init"() : () -> !enzyme.Cache<tensor<4xf64>>
    enzyme.custom_reverse_rule.augmented_primal (%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<tensor<4xf64>>, tensor<4xf64>) -> ()
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<tensor<4xf64>>, tensor<4xf64>) -> ()
      %y = arith.mulf %a, %x : tensor<4xf64>
      enzyme.yield %y : tensor<4xf64>
    }
    enzyme.custom_reverse_rule.reverse (%ybar: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<tensor<4xf64>>) -> tensor<4xf64>
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<tensor<4xf64>>) -> tensor<4xf64>
      %abar = arith.mulf %ybar, %x : tensor<4xf64>
      %xraw = arith.mulf %ybar, %a : tensor<4xf64>
      %two = arith.constant dense<2.0> : tensor<4xf64>
      %xbar = arith.mulf %two, %xraw : tensor<4xf64>
      enzyme.yield %abar, %xbar : tensor<4xf64>, tensor<4xf64>
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %y = func.call @mul(%a, %x) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %z = arith.mulf %y, %y : tensor<4xf64>
    return %z : tensor<4xf64>
  }

  func.func @main(%a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %da, %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %da, %dx : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL: func.func private @mul_rule_reverse
// CHECK: arith.constant dense<2.000000e+00>

// -----

// The same all-active rule called with a CONSTANT first operand (data). A
// frontend emitting the rule at trace time cannot know this activity.

module {
  func.func private @mul(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> attributes {enzyme.custom_rule = @mul_rule} {
    %y = arith.mulf %a, %x : tensor<4xf64>
    return %y : tensor<4xf64>
  }

  enzyme.custom_reverse_rule @mul_rule {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<tensor<4xf64>>
    %cx = "enzyme.init"() : () -> !enzyme.Cache<tensor<4xf64>>
    enzyme.custom_reverse_rule.augmented_primal (%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<tensor<4xf64>>, tensor<4xf64>) -> ()
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<tensor<4xf64>>, tensor<4xf64>) -> ()
      %y = arith.mulf %a, %x : tensor<4xf64>
      enzyme.yield %y : tensor<4xf64>
    }
    enzyme.custom_reverse_rule.reverse (%ybar: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<tensor<4xf64>>) -> tensor<4xf64>
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<tensor<4xf64>>) -> tensor<4xf64>
      %abar = arith.mulf %ybar, %x : tensor<4xf64>
      %xraw = arith.mulf %ybar, %a : tensor<4xf64>
      %two = arith.constant dense<2.0> : tensor<4xf64>
      %xbar = arith.mulf %two, %xraw : tensor<4xf64>
      enzyme.yield %abar, %xbar : tensor<4xf64>, tensor<4xf64>
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %y = func.call @mul(%a, %x) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %z = arith.mulf %y, %y : tensor<4xf64>
    return %z : tensor<4xf64>
  }

  func.func @main(%a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> tensor<4xf64> {
    %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %dx : tensor<4xf64>
  }
}

// CHECK-LABEL: func.func private @mul_rule_reverse
// CHECK: arith.constant dense<2.000000e+00>

// -----

// The rule called once per iteration of a loop: the tape of every call has to
// survive until the reverse sweep.

module {
  func.func private @mul(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> attributes {enzyme.custom_rule = @mul_rule} {
    %y = arith.mulf %a, %x : tensor<4xf64>
    return %y : tensor<4xf64>
  }

  enzyme.custom_reverse_rule @mul_rule {
    %ca = "enzyme.init"() : () -> !enzyme.Cache<tensor<4xf64>>
    %cx = "enzyme.init"() : () -> !enzyme.Cache<tensor<4xf64>>
    enzyme.custom_reverse_rule.augmented_primal (%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
      "enzyme.push"(%ca, %a) : (!enzyme.Cache<tensor<4xf64>>, tensor<4xf64>) -> ()
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<tensor<4xf64>>, tensor<4xf64>) -> ()
      %y = arith.mulf %a, %x : tensor<4xf64>
      enzyme.yield %y : tensor<4xf64>
    }
    enzyme.custom_reverse_rule.reverse (%ybar: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
      %a = "enzyme.pop"(%ca) : (!enzyme.Cache<tensor<4xf64>>) -> tensor<4xf64>
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<tensor<4xf64>>) -> tensor<4xf64>
      %abar = arith.mulf %ybar, %x : tensor<4xf64>
      %xraw = arith.mulf %ybar, %a : tensor<4xf64>
      %two = arith.constant dense<2.0> : tensor<4xf64>
      %xbar = arith.mulf %two, %xraw : tensor<4xf64>
      enzyme.yield %abar, %xbar : tensor<4xf64>, tensor<4xf64>
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %r = scf.for %iv = %lb to %ub step %step iter_args(%xi = %x) -> (tensor<4xf64>) {
      %y = func.call @mul(%a, %xi) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
      scf.yield %y : tensor<4xf64>
    }
    return %r : tensor<4xf64>
  }

  func.func @main(%a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %da, %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %da, %dx : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL: func.func private @mul_rule_reverse
// CHECK: arith.constant dense<2.000000e+00>
