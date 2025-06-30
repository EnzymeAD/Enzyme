// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @two_normals(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:4 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<3>, name="s", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %t:4 = enzyme.sample @normal(%s#1, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<4>, name="t", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %s#0, %t#0, %t#1, %t#2, %t#3 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:4 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %t:5 = enzyme.sample @two_normals(%s#1, %s#0, %stddev) { symbol = #enzyme.symbol<2>, name="t", traced_output_indices = array<i64: 0, 1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t#0, %t#1, %t#2, %t#3, %t#4 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @simulate(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %res:7 = enzyme.simulate @test(%rng, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#2, %res#3, %res#4, %res#5, %res#6 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CHECK: func.func @test.simulate(%[[RNG:.*]]: tensor<2xui64>, %[[MEAN:.*]]: tensor<f64>, %[[STD:.*]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %[[TRACE:.*]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %[[N1:.*]]:4 = call @normal(%[[RNG]], %[[MEAN]], %[[STD]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     enzyme.addSampleToTrace(%[[N1]]#0 : tensor<f64>) into %[[TRACE]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:     %[[LP1:.*]] = call @logpdf(%[[N1]]#0, %[[MEAN]], %[[STD]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %[[SUM1:.*]] = arith.addf %[[LP1]], %[[CST]] : tensor<f64>
// CHECK-NEXT:     %[[TN:.*]]:5 = call @two_normals(%[[N1]]#1, %[[N1]]#0, %[[STD]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     enzyme.addSampleToTrace(%[[TN]]#0, %[[TN]]#1 : tensor<f64>, tensor<f64>) into %[[TRACE]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:     %[[SUB:.*]]:7 = call @two_normals.simulate(%[[N1]]#1, %[[N1]]#0, %[[STD]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     enzyme.addSubtrace %[[SUB]]#0 into %[[TRACE]]
// CHECK-NEXT:     %[[SUM2:.*]] = arith.addf %[[SUM1]], %[[SUB]]#1 : tensor<f64>
// CHECK-NEXT:     return %[[TRACE]], %[[SUM2]], %[[TN]]#0, %[[TN]]#1, %[[TN]]#2, %[[TN]]#3, %[[TN]]#4 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }

// CHECK: func.func @two_normals.simulate(%[[RNG2:.*]]: tensor<2xui64>, %[[MEAN2:.*]]: tensor<f64>, %[[STD2:.*]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %[[CST2:.*]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %[[TRACE2:.*]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %[[N1TN:.*]]:4 = call @normal(%[[RNG2]], %[[MEAN2]], %[[STD2]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     enzyme.addSampleToTrace(%[[N1TN]]#0 : tensor<f64>) into %[[TRACE2]] {symbol = #enzyme.symbol<3>}
// CHECK-NEXT:     %[[LP1TN:.*]] = call @logpdf(%[[N1TN]]#0, %[[MEAN2]], %[[STD2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %[[SUM1TN:.*]] = arith.addf %[[LP1TN]], %[[CST2]] : tensor<f64>
// CHECK-NEXT:     %[[N2TN:.*]]:4 = call @normal(%[[N1TN]]#1, %[[MEAN2]], %[[STD2]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     enzyme.addSampleToTrace(%[[N2TN]]#0 : tensor<f64>) into %[[TRACE2]] {symbol = #enzyme.symbol<4>}
// CHECK-NEXT:     %[[LP2TN:.*]] = call @logpdf(%[[N2TN]]#0, %[[MEAN2]], %[[STD2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %[[SUM2TN:.*]] = arith.addf %[[SUM1TN]], %[[LP2TN]] : tensor<f64>
// CHECK-NEXT:     return %[[TRACE2]], %[[SUM2TN]], %[[N1TN]]#0, %[[N2TN]]#0, %[[N2TN]]#1, %[[N2TN]]#2, %[[N2TN]]#3 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
