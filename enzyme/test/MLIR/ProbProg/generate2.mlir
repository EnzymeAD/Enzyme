// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @joint(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
  func.func private @joint_logpdf(%x1 : tensor<f64>, %x2 : tensor<2xf64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %s:3 = enzyme.sample @joint(%rng, %mean, %stddev) {
      symbol = #enzyme.symbol<5>,
      name = "s",
      logpdf = @joint_logpdf
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %s#0, %s#1, %s#2 : tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }

  func.func @foo(%rng : tensor<2xui64>, %x : tensor<f64>, %y : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %cst = arith.constant dense<42> : tensor<ui64>
    %0 = builtin.unrealized_conversion_cast %cst : tensor<ui64> to !enzyme.Constraint
    // CHECK: %[[call_res:.+]]:5 = call @test.generate(%[[constraint:.+]], %[[arg0:.+]], %[[arg1:.+]], %[[arg2:.+]]) : (!enzyme.Constraint, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    %res:5 = enzyme.generate @test(%rng, %x, %y) given %0 {
      constrained_addresses = [[#enzyme.symbol<2>], [#enzyme.symbol<5>]]
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %res#1, %res#2, %res#3, %res#4 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }
}

// CHECK:  func.func @test.generate(%[[arg0:.+]]: !enzyme.Constraint, %[[arg1:.+]]: tensor<2xui64>, %[[arg2:.+]]: tensor<f64>, %[[arg3:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[sample_from_constraint:.+]]:2 = enzyme.getSampleFromConstraint %[[arg0]] {symbol = #enzyme.symbol<5>} : tensor<f64>, tensor<2xf64>
// CHECK-NEXT:    %[[logpdf_call:.+]] = call @joint_logpdf(%[[sample_from_constraint]]#0, %[[sample_from_constraint]]#1, %[[arg2]], %[[arg3]]) : (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[addf:.+]] = arith.addf %[[logpdf_call]], %[[cst]] : tensor<f64>
// CHECK-NEXT:    %[[trace1:.+]] = enzyme.addSampleToTrace(%[[sample_from_constraint]]#0, %[[sample_from_constraint]]#1 : tensor<f64>, tensor<2xf64>) into %[[trace_init]] {symbol = #enzyme.symbol<5>}
// CHECK-NEXT:    %[[trace2:.+]] = enzyme.addWeightToTrace(%[[addf]] : tensor<f64>) into %[[trace1]]
// CHECK-NEXT:    %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[sample_from_constraint]]#0, %[[sample_from_constraint]]#1 : tensor<f64>, tensor<2xf64>) into %[[trace2]]
// CHECK-NEXT:    return %[[final_trace]], %[[addf]], %[[arg1]], %[[sample_from_constraint]]#0, %[[sample_from_constraint]]#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
// CHECK-NEXT:  }
