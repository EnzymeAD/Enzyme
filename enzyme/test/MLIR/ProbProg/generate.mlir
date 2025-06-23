// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // Result is (output_rng_state, sample)
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { 
      symbol = 1 : ui64, 
      name="s",
      logpdf = @normal_logpdf,
      traced_input_indices = array<i64: 1, 2>,
      traced_output_indices = array<i64: 1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { 
      symbol = 2 : ui64, 
      name="t",
      logpdf = @normal_logpdf,
      traced_input_indices = array<i64: 1, 2>,
      traced_output_indices = array<i64: 1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @foo(%rng : tensor<2xui64>, %x : tensor<f64>, %y : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:3 = enzyme.generate @test(%rng, %x, %y) { 
      trace=42 : ui64, 
      constraints=100 : ui64,
      name="res" 
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2 : tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK: func.func @test.generate(%[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>) {
// CHECK-NEXT: %[[zero:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[s:.+]]:2 = enzyme.sampleOrExtractConstraint @normal(%[[rng]], %[[mean]], %[[stddev]]) {constraints = 100 : ui64, symbol = 1 : ui64, traced_output_indices = array<i64: 1>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[s_logpdf:.+]] = call @normal_logpdf(%[[s]]#1, %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[weight1:.+]] = arith.addf %[[zero]], %[[s_logpdf]] : tensor<f64>
// CHECK-NEXT: enzyme.addSampleToTrace %[[s]]#1 {name = "s", symbol = 1 : ui64, trace = 42 : ui64} : (tensor<f64>) -> ()
// CHECK-NEXT: %[[t:.+]]:2 = enzyme.sampleOrExtractConstraint @normal(%[[s]]#0, %[[s]]#1, %[[stddev]]) {constraints = 100 : ui64, symbol = 2 : ui64, traced_output_indices = array<i64: 1>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[t_logpdf:.+]] = call @normal_logpdf(%[[t]]#1, %[[s]]#1, %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[weight2:.+]] = arith.addf %[[weight1]], %[[t_logpdf]] : tensor<f64>
// CHECK-NEXT: enzyme.addSampleToTrace %[[t]]#1 {name = "t", symbol = 2 : ui64, trace = 42 : ui64} : (tensor<f64>) -> ()
// CHECK-NEXT: return %[[weight2]], %[[t]]#0, %[[t]]#1 : tensor<f64>, tensor<2xui64>, tensor<f64>
// CHECK-NEXT: }
