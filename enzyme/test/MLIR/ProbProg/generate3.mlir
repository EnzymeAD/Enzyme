// RUN: %eopt --probprog %s -allow-unregistered-dialect| FileCheck %s

module {
  func.func private @fake(%arg0: tensor<2xi64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>)
  func.func private @fake_logpdf(%arg0: tensor<10xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<10xf64>, tensor<f64>, tensor<f64>)
  func.func private @fake_1(%arg0: tensor<2xi64>, %arg1: tensor<10xf64>, %arg2: tensor<f64>) -> (tensor<10xf64>, tensor<2xi64>, tensor<10xf64>, tensor<f64>)
  func.func private @fake_logpdf_1(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>, %arg2: tensor<f64>) -> (tensor<f64>, tensor<10xf64>, tensor<10xf64>, tensor<f64>)
  func.func private @model(%arg0: tensor<2xi64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>) {
    %0:4 = enzyme.sample @fake(%arg0, %arg1, %arg2) {logpdf = @fake_logpdf, name = "s", symbol = 1 : ui64, traced_input_indices = array<i64: 1, 2>, traced_output_indices = array<i64: 0>, alias_map = array<i64: 1, 0, 2, 1, 3, 2>} : (tensor<2xi64>, tensor<f64>, tensor<f64>) -> (tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>)
    %1:4 = enzyme.sample @fake_1(%0#1, %0#0, %0#3) {logpdf = @fake_logpdf_1, name = "t", symbol = 2 : ui64, traced_input_indices = array<i64: 1, 2>, traced_output_indices = array<i64: 0>, alias_map = array<i64: 1, 0, 2, 1, 3, 2>} : (tensor<2xi64>, tensor<10xf64>, tensor<f64>) -> (tensor<10xf64>, tensor<2xi64>, tensor<10xf64>, tensor<f64>)
    return %1#0, %arg0, %0#2, %1#3 : tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>
  }
  func.func @main(%arg0: tensor<2xi64> {tf.aliasing_output = 2 : i32}, %arg1: tensor<f64> {tf.aliasing_output = 3 : i32}, %arg2: tensor<f64> {tf.aliasing_output = 4 : i32}) -> (tensor<f64>, tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>) {
    %0:5 = enzyme.generate @model(%arg0, %arg1, %arg2) {constraints = [#enzyme.constraint<symbol = 1, values = [dense<5.000000e+00> : tensor<10xf64>]>], trace = 42 : ui64} : (tensor<2xi64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<f64>, tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>
  }
}

// CHECK: func.func private @model.generate(%[[rng:.+]]: tensor<2xi64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (tensor<f64>, tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT: %[[zero:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[s_const:.+]] = arith.constant dense<5.000000e+00> : tensor<10xf64>
// CHECK-NEXT: %[[s_logpdf:.+]]:4 = call @fake_logpdf(%[[s_const]], %[[mean]], %[[stddev]]) : (tensor<10xf64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<10xf64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %[[weight1:.+]] = arith.addf %[[zero]], %[[s_logpdf]]#0 : tensor<f64>
// CHECK-NEXT: enzyme.addSampleToTrace %[[s_const]] {name = "s", symbol = 1 : ui64, trace = 42 : ui64} : (tensor<10xf64>) -> ()
// CHECK-NEXT: %[[t_call:.+]]:4 = call @fake_1(%[[rng]], %[[s_const]], %[[stddev]]) : (tensor<2xi64>, tensor<10xf64>, tensor<f64>) -> (tensor<10xf64>, tensor<2xi64>, tensor<10xf64>, tensor<f64>)
// CHECK-NEXT: %[[t_logpdf:.+]]:4 = call @fake_logpdf_1(%[[t_call]]#0, %[[s_const]], %[[stddev]]) : (tensor<10xf64>, tensor<10xf64>, tensor<f64>) -> (tensor<f64>, tensor<10xf64>, tensor<10xf64>, tensor<f64>)
// CHECK-NEXT: %[[weight2:.+]] = arith.addf %[[weight1]], %[[t_logpdf]]#0 : tensor<f64>
// CHECK-NEXT: enzyme.addSampleToTrace %[[t_call]]#0 {name = "t", symbol = 2 : ui64, trace = 42 : ui64} : (tensor<10xf64>) -> ()
// CHECK-NEXT: return %[[weight2]], %[[t_call]]#0, %[[rng]], %[[mean]], %[[t_call]]#3 : tensor<f64>, tensor<10xf64>, tensor<2xi64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }