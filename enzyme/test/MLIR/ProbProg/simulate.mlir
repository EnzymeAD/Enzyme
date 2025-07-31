// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @simulate(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    // CHECK: %[[call_res:.+]]:4 = call @test.simulate(%[[arg0:.+]], %[[arg1:.+]], %[[arg2:.+]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %res:4 = enzyme.simulate @test(%rng, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK:  func.func @test.simulate(%[[arg0:.+]]: tensor<2xui64>, %[[arg1:.+]]: tensor<f64>, %[[arg2:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[normal_call1:.+]]:2 = call @normal(%[[arg0]], %[[arg1]], %[[arg2]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[logpdf_call1:.+]] = call @logpdf(%[[normal_call1]]#1, %[[arg1]], %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[addf1:.+]] = arith.addf %[[logpdf_call1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:    %[[trace1:.+]] = enzyme.addSampleToTrace(%[[normal_call1]]#1 : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %[[normal_call2:.+]]:2 = call @normal(%[[normal_call1]]#0, %[[normal_call1]]#1, %[[arg2]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[logpdf_call2:.+]] = call @logpdf(%[[normal_call2]]#1, %[[normal_call1]]#1, %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[addf2:.+]] = arith.addf %[[addf1]], %[[logpdf_call2]] : tensor<f64>
// CHECK-NEXT:    %[[trace2:.+]] = enzyme.addSampleToTrace(%[[normal_call2]]#1 : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %[[trace3:.+]] = enzyme.addWeightToTrace(%[[addf2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:    %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[normal_call2]]#1 : tensor<f64>) into %[[trace3]]
// CHECK-NEXT:    return %[[final_trace]], %[[addf2]], %[[normal_call2]]#0, %[[normal_call2]]#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>
// CHECK-NEXT:  }
