// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-custom-rules-to-func --enzyme-simplify-math %s | FileCheck %s

module {
  func.func private @helper(%x: f64) -> f64 {
    %c = arith.constant 1.5 : f64
    %r = arith.mulf %x, %c : f64
    return %r : f64
  }

  func.func private @inner_1arg_4ret(%arg0: f64) -> (f64, f64, f64, f64) {
    %cst2 = arith.constant 2.0 : f64
    %cst3 = arith.constant 3.0 : f64
    %cst4 = arith.constant 4.0 : f64
    %cst5 = arith.constant 5.0 : f64
    %h1 = func.call @helper(%arg0) : (f64) -> f64
    %a = arith.mulf %h1, %cst2 : f64
    %h2 = func.call @helper(%a) : (f64) -> f64
    %b = arith.mulf %h2, %cst3 : f64
    %h3 = func.call @helper(%b) : (f64) -> f64
    %c = arith.addf %h3, %cst4 : f64
    %h4 = func.call @helper(%c) : (f64) -> f64
    %d = arith.mulf %h4, %cst5 : f64
    return %a, %b, %c, %d : f64, f64, f64, f64
  }

  func.func private @helper2(%x: f64, %y: f64) -> (f64, f64) {
    %sum = arith.addf %x, %y : f64
    %prod = arith.mulf %x, %y : f64
    return %sum, %prod : f64, f64
  }

  func.func @outer_to_diff(%arg0: f64) -> f64 {
    %prep = func.call @helper(%arg0) : (f64) -> f64
    %results:4 = func.call @inner_1arg_4ret(%prep) : (f64) -> (f64, f64, f64, f64)
    %h:2 = func.call @helper2(%results#0, %results#1) : (f64, f64) -> (f64, f64)
    %sum1 = arith.addf %h#0, %h#1 : f64
    %sum2 = arith.addf %sum1, %results#2 : f64
    %sum3 = arith.addf %sum2, %results#3 : f64
    return %sum3 : f64
  }

  func.func @test(%arg0: f64, %seed: f64) -> f64 {
    %r:2 = enzyme.autodiff @outer_to_diff(%arg0, %seed) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (f64, f64) -> (f64, f64)
    return %r#1 : f64
  }
}

// CHECK-LABEL:  func.func private @diffeouter_to_diff(%arg0: f64, %arg1: f64) -> (f64, f64) {
// CHECK-NEXT:    %0 = call @helper_reverse_rule_primal(%arg0) : (f64) -> f64
// CHECK-NEXT:    %1:4 = call @inner_1arg_4ret_reverse_rule_primal(%0) : (f64) -> (f64, f64, f64, f64)
// CHECK-NEXT:    %2:4 = call @helper2_reverse_rule_primal(%1#0, %1#1) : (f64, f64) -> (f64, f64, f64, f64)
// CHECK-NEXT:    %3 = arith.addf %2#0, %2#1 : f64
// CHECK-NEXT:    %4 = arith.addf %3, %1#2 : f64
// CHECK-NEXT:    %5 = arith.addf %4, %1#3 : f64
// CHECK-NEXT:    %6:2 = call @helper2_reverse_rule_reverse(%arg1, %arg1, %2#2, %2#3) : (f64, f64, f64, f64) -> (f64, f64)
// CHECK-NEXT:    %7 = call @inner_1arg_4ret_reverse_rule_reverse(%6#0, %6#1, %arg1, %arg1) : (f64, f64, f64, f64) -> f64
// CHECK-NEXT:    %8 = call @helper_reverse_rule_reverse(%7) : (f64) -> f64
// CHECK-NEXT:    return %5, %8 : f64, f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @helper_reverse_rule_reverse(%arg0: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 1.500000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %cst fastmath<fast> : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @inner_1arg_4ret_reverse_rule_reverse(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg3, %cst_1 fastmath<fast> : f64
// CHECK-NEXT:    %1 = call @helper_reverse_rule_reverse(%0) : (f64) -> f64
// CHECK-NEXT:    %2 = arith.addf %arg2, %1 fastmath<fast> : f64
// CHECK-NEXT:    %3 = call @helper_reverse_rule_reverse(%2) : (f64) -> f64
// CHECK-NEXT:    %4 = arith.addf %arg1, %3 fastmath<fast> : f64
// CHECK-NEXT:    %5 = arith.mulf %4, %cst_0 fastmath<fast> : f64
// CHECK-NEXT:    %6 = call @helper_reverse_rule_reverse(%5) : (f64) -> f64
// CHECK-NEXT:    %7 = arith.addf %arg0, %6 fastmath<fast> : f64
// CHECK-NEXT:    %8 = arith.mulf %7, %cst fastmath<fast> : f64
// CHECK-NEXT:    %9 = call @helper_reverse_rule_reverse(%8) : (f64) -> f64
// CHECK-NEXT:    return %9 : f64
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func private @helper2_reverse_rule_reverse(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) -> (f64, f64) {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg3 fastmath<fast> : f64
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg2 fastmath<fast> : f64
// CHECK-NEXT:    %2 = arith.addf %0, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %3 = arith.addf %1, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    return %2, %3 : f64, f64
// CHECK-NEXT:  }
