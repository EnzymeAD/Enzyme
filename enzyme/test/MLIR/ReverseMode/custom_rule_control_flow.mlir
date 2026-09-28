// RUN: %eopt %s --split-input-file --enzyme="postpasses=canonicalize,remove-unnecessary-enzyme-ops" --lower-enzyme-custom-rules-to-func --canonicalize | FileCheck %s

// Custom-rule calls inside structured control flow, in the pass order Reactant
// uses: the cache cleanup runs inside the enzyme pass, before the rules are
// lowered to functions. The augmented primal hands the rule's caches to the
// caller as typed values, so a loop stacks them per iteration and a branch
// yields them like any other value. The x cotangent is doubled by the rule,
// so the output shows the rule was used.

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
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %x) -> (tensor<4xf64>) {
      %y = func.call @mul(%a, %acc) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
      scf.yield %y : tensor<4xf64>
    }
    return %r : tensor<4xf64>
  }

  func.func @main(%a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %da, %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %da, %dx : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL: func.func private @diffeouter
// CHECK-NOT: enzyme.
// CHECK: scf.for
// CHECK: call @mul_rule_primal
// CHECK: scf.for
// CHECK: call @mul_rule_reverse
// CHECK-LABEL: func.func private @mul_rule_reverse
// CHECK: arith.constant dense<2.000000e+00>

// -----

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
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%p: i1, %a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %r = scf.if %p -> (tensor<4xf64>) {
      %y = func.call @mul(%a, %x) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
      scf.yield %y : tensor<4xf64>
    } else {
      scf.yield %x : tensor<4xf64>
    }
    return %r : tensor<4xf64>
  }

  func.func @main(%p: i1, %a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %da, %dx = enzyme.autodiff @outer(%p, %a, %x, %dz) {
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (i1, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %da, %dx : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL: func.func private @diffeouter
// CHECK-NOT: enzyme.
// The primal and the reverse branch test the same condition and are merged.
// CHECK: scf.if
// CHECK: call @mul_rule_primal
// CHECK: call @mul_rule_reverse
// CHECK-LABEL: func.func private @mul_rule_reverse
// CHECK: arith.constant dense<2.000000e+00>
