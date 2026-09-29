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
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %y = func.call @mul(%a, %x) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %z = arith.mulf %y, %y : tensor<4xf64>
    return %z : tensor<4xf64>
  }

  func.func @main(%a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %da, %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %da, %dx : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %0:3 = call @mul_rule_primal(%arg0, %arg1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:    %1 = arith.mulf %arg2, %0#0 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:    %2 = arith.mulf %arg2, %0#0 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:    %3 = arith.addf %1, %2 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:    %4:2 = call @mul_rule_reverse(%3, %0#1, %0#2) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:    return %4#0, %4#1 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_rule_reverse(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<2.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 : tensor<4xf64>
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:    %2 = arith.mulf %1, %cst : tensor<4xf64>
// CHECK-NEXT:    return %0, %2 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }

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
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  }

  func.func @outer(%a: tensor<4xf64>, %x: tensor<4xf64>) -> tensor<4xf64> {
    %y = func.call @mul(%a, %x) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %z = arith.mulf %y, %y : tensor<4xf64>
    return %z : tensor<4xf64>
  }

  func.func @main(%a: tensor<4xf64>, %x: tensor<4xf64>, %dz: tensor<4xf64>) -> tensor<4xf64> {
    %dx = enzyme.autodiff @outer(%a, %x, %dz) {
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %dx : tensor<4xf64>
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %0:3 = call @mul_rule_primal(%arg0, %arg1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:    %1 = arith.mulf %arg2, %0#0 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:    %2 = arith.mulf %arg2, %0#0 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:    %3 = arith.addf %1, %2 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:    %4:2 = call @mul_rule_reverse(%3, %0#1, %0#2) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:    return %4#1 : tensor<4xf64>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_rule_reverse(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<2.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 : tensor<4xf64>
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:    %2 = arith.mulf %1, %cst : tensor<4xf64>
// CHECK-NEXT:    return %0, %2 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }

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
    activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
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
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %da, %dx : tensor<4xf64>, tensor<4xf64>
  }
}

// CHECK-LABEL:  func.func private @diffeouter(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3xtensor<4xf64>>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<3xtensor<4xf64>>
// CHECK-NEXT:    %0 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg1) -> (tensor<4xf64>) {
// CHECK-NEXT:      %2:3 = func.call @mul_rule_primal(%arg0, %arg4) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:      memref.store %2#2, %alloc_0[%arg3] : memref<3xtensor<4xf64>>
// CHECK-NEXT:      memref.store %2#1, %alloc[%arg3] : memref<3xtensor<4xf64>>
// CHECK-NEXT:      scf.yield %2#0 : tensor<4xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2, %arg5 = %cst) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:      %2 = arith.subi %c2, %arg3 : index
// CHECK-NEXT:      %3 = memref.load %alloc[%2] : memref<3xtensor<4xf64>>
// CHECK-NEXT:      %4 = memref.load %alloc_0[%2] : memref<3xtensor<4xf64>>
// CHECK-NEXT:      %5:2 = func.call @mul_rule_reverse(%arg4, %3, %4) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:      %6 = arith.addf %arg5, %5#0 fastmath<fast> : tensor<4xf64>
// CHECK-NEXT:      scf.yield %5#1, %6 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<3xtensor<4xf64>>
// CHECK-NEXT:    memref.dealloc %alloc : memref<3xtensor<4xf64>>
// CHECK-NEXT:    return %1#1, %1#0 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_rule_reverse(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<2.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 : tensor<4xf64>
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:    %2 = arith.mulf %1, %cst : tensor<4xf64>
// CHECK-NEXT:    return %0, %2 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }
