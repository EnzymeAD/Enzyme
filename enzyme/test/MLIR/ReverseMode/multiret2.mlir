// RUN: %eopt --enzyme %s | FileCheck %s

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
    %r:2 = enzyme.autodiff @outer_to_diff(%arg0, %seed) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (f64, f64) -> (f64, f64)
    return %r#1 : f64
  }
}

// CHECK: func.func @test
// CHECK: call @diffeouter_to_diff
// CHECK: func.func private @diffeouter_to_diff
// CHECK: func.func private @diffeinner_1arg_4ret({{.+}}: f64, {{.+}}: f64, {{.+}}: f64, {{.+}}: f64, {{.+}}: f64) -> f64
