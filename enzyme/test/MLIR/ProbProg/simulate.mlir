// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:4 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %t:4 = enzyme.sample @normal(%s#1, %s#0, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t#0, %t#1, %t#2, %t#3 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @simulate(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %res:6 = enzyme.simulate @test(%rng, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#2, %res#3, %res#4, %res#5 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CHECK: func.func @test.simulate(
// CHECK-NEXT:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %[[TRACE:.*]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %[[N1:.*]]:4 = call @normal(
// CHECK-NEXT:     enzyme.addSampleToTrace(%[[N1]]#0 : tensor<f64>) into %[[TRACE]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:     %[[LP1:.*]] = call @logpdf(%[[N1]]#0,
// CHECK-NEXT:     %[[SUM1:.*]] = arith.addf %[[LP1]], %[[CST]] : tensor<f64>
// CHECK-NEXT:     %[[N2:.*]]:4 = call @normal(
// CHECK-NEXT:     enzyme.addSampleToTrace(%[[N2]]#0 : tensor<f64>) into %[[TRACE]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:     %[[LP2:.*]] = call @logpdf(%[[N2]]#0,
// CHECK-NEXT:     %[[SUM2:.*]] = arith.addf %[[SUM1]], %[[LP2]] : tensor<f64>
// CHECK-NEXT:     return %[[TRACE]], %[[SUM2]], %[[N2]]#0, %[[N2]]#1, %[[N2]]#2, %[[N2]]#3 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
