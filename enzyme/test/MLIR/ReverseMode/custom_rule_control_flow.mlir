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

// CHECK-LABEL:  func.func private @diffeouter(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
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

// The primal and the reverse branch test the same condition and are merged.

// CHECK-LABEL:  func.func private @diffeouter(%arg0: i1, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %0:2 = scf.if %arg0 -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:      %1:3 = func.call @mul_rule_primal(%arg1, %arg2) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:      %2:2 = func.call @mul_rule_reverse(%arg3, %1#1, %1#2) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
// CHECK-NEXT:      scf.yield %2#0, %2#1 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %cst, %arg3 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0, %0#1 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @mul_rule_reverse(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<2.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg2 : tensor<4xf64>
// CHECK-NEXT:    %1 = arith.mulf %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:    %2 = arith.mulf %1, %cst : tensor<4xf64>
// CHECK-NEXT:    return %0, %2 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }
