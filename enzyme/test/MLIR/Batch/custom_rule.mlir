// RUN: %eopt --enzyme-batch %s | FileCheck %s --check-prefix=BATCH
// RUN: %eopt --enzyme-batch --enzyme --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s --check-prefix=AD

// A scalar function batched over a broadcast calls a callee that carries a
// custom rule (as a ReactiveKernels scalar derivative rule under Reactant
// does). The batched callee must name a batched copy of the rule: batched
// signatures, and caches of batched values. The rule marks its cotangent
// (factor 3) so the gradient shows the batched rule was used.

module {
  func.func private @sq(%x: tensor<f64>) -> tensor<f64> attributes {enzyme.custom_rule = [@sq_rule]} {
    %y = arith.mulf %x, %x : tensor<f64>
    return %y : tensor<f64>
  }

  enzyme.custom_reverse_rule @sq_rule {
    %cx = "enzyme.init"() : () -> !enzyme.Cache<tensor<f64>>
    enzyme.custom_reverse_rule.augmented_primal (%x: tensor<f64>) -> tensor<f64> {
      "enzyme.push"(%cx, %x) : (!enzyme.Cache<tensor<f64>>, tensor<f64>) -> ()
      %y = arith.mulf %x, %x : tensor<f64>
      enzyme.yield %y : tensor<f64>
    }
    enzyme.custom_reverse_rule.reverse (%ybar: tensor<f64>) -> tensor<f64> {
      %x = "enzyme.pop"(%cx) : (!enzyme.Cache<tensor<f64>>) -> tensor<f64>
      %three = arith.constant dense<3.0> : tensor<f64>
      %t = arith.mulf %three, %x : tensor<f64>
      %xbar = arith.mulf %ybar, %t : tensor<f64>
      enzyme.yield %xbar : tensor<f64>
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (tensor<f64>) -> tensor<f64>
  }

  func.func private @elem(%x: tensor<f64>) -> tensor<f64> {
    %y = func.call @sq(%x) : (tensor<f64>) -> tensor<f64>
    return %y : tensor<f64>
  }

  func.func @outer(%x: tensor<3xf64>) -> tensor<3xf64> {
    %y = enzyme.batch @elem(%x) {batch_shape = array<i64: 3>} : (tensor<3xf64>) -> tensor<3xf64>
    return %y : tensor<3xf64>
  }

  func.func @main(%x: tensor<3xf64>, %dy: tensor<3xf64>) -> tensor<3xf64> {
    %dx = enzyme.autodiff @outer(%x, %dy) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %dx : tensor<3xf64>
  }
}

// BATCH-DAG: func.func private @batched_sq(%arg0: tensor<3xf64>) -> tensor<3xf64> attributes {enzyme.custom_rule = [@batched_sq_rule]}
// BATCH-DAG: enzyme.custom_reverse_rule @batched_sq_rule
// BATCH-DAG: "enzyme.init"() : () -> !enzyme.Cache<tensor<3xf64>>
// BATCH-DAG: enzyme.custom_reverse_rule.augmented_primal (%{{.+}}: tensor<3xf64>) -> tensor<3xf64>
// BATCH-DAG: enzyme.custom_reverse_rule.reverse (%{{.+}}: tensor<3xf64>) -> tensor<3xf64>
// BATCH-DAG: function_type = (tensor<3xf64>) -> tensor<3xf64>

// AD-LABEL:  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// AD-NEXT:    %0 = call @diffeouter(%arg0, %arg1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
// AD-NEXT:    return %0 : tensor<3xf64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @diffeouter(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// AD-NEXT:    %0:2 = call @batched_elem_reverse_rule_primal(%arg0) : (tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
// AD-NEXT:    %1 = call @batched_elem_reverse_rule_reverse(%arg1, %0#1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
// AD-NEXT:    return %1 : tensor<3xf64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @batched_sq_rule_primal(%arg0: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
// AD-NEXT:    %0 = arith.mulf %arg0, %arg0 : tensor<3xf64>
// AD-NEXT:    return %0, %arg0 : tensor<3xf64>, tensor<3xf64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @batched_sq_rule_reverse(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// AD-NEXT:    %cst = arith.constant dense<3.000000e+00> : tensor<3xf64>
// AD-NEXT:    %0 = arith.mulf %arg1, %cst : tensor<3xf64>
// AD-NEXT:    %1 = arith.mulf %arg0, %0 : tensor<3xf64>
// AD-NEXT:    return %1 : tensor<3xf64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @sq_rule_primal(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// AD-NEXT:    %0 = arith.mulf %arg0, %arg0 : tensor<f64>
// AD-NEXT:    return %0, %arg0 : tensor<f64>, tensor<f64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @sq_rule_reverse(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
// AD-NEXT:    %cst = arith.constant dense<3.000000e+00> : tensor<f64>
// AD-NEXT:    %0 = arith.mulf %arg1, %cst : tensor<f64>
// AD-NEXT:    %1 = arith.mulf %arg0, %0 : tensor<f64>
// AD-NEXT:    return %1 : tensor<f64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @batched_elem_reverse_rule_primal(%arg0: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
// AD-NEXT:    %0:2 = call @batched_sq_rule_primal(%arg0) : (tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
// AD-NEXT:    return %0#0, %0#1 : tensor<3xf64>, tensor<3xf64>
// AD-NEXT:  }

// AD-LABEL:  func.func private @batched_elem_reverse_rule_reverse(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// AD-NEXT:    %0 = call @batched_sq_rule_reverse(%arg0, %arg1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
// AD-NEXT:    return %0 : tensor<3xf64>
// AD-NEXT:  }

// AD-NOT: enzyme.
