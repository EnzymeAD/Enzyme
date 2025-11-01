// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @two_normals(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<3>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<4>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %s#1, %t#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:3 = enzyme.sample @two_normals(%s#0, %s#1, %stddev) { symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>)
    %u:3 = enzyme.sample @two_normals(%t#0, %t#1, %stddev) { symbol = #enzyme.symbol<6>, name="u" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %u#0, %u#1, %u#2 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @generate(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %cst = arith.constant dense<42> : tensor<ui64>
    %0 = builtin.unrealized_conversion_cast %cst : tensor<ui64> to !enzyme.Constraint
    
    %res:5 = enzyme.generate @test(%rng, %mean, %stddev) given %0 { 
      name = "test_generate", 
      constrained_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>, #enzyme.symbol<3>], [#enzyme.symbol<6>, #enzyme.symbol<4>]]
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    
    return %res#0, %res#1, %res#2, %res#3, %res#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CHECK: func.func @test.generate(%[[constraint:.+]]: !enzyme.Constraint, %[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %[[trace_init:.+]] = enzyme.initTrace {source_fn = @test.generate} : !enzyme.Trace
// CHECK-NEXT:   %[[sample_from_constraint1:.+]] = enzyme.getSampleFromConstraint %[[constraint]] {symbol = #enzyme.symbol<1>} : tensor<f64>
// CHECK-NEXT:   %[[logpdf1:.+]] = call @logpdf(%[[sample_from_constraint1]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:   %[[trace1:.+]] = enzyme.addSampleToTrace(%[[sample_from_constraint1]] : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:   %[[subconstraint1:.+]] = enzyme.getSubconstraint %[[constraint]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:   %[[gen_res1:.+]]:5 = call @two_normals.generate_0(%[[subconstraint1]], %[[rng]], %[[sample_from_constraint1]], %[[stddev]]) : (!enzyme.Constraint, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:   %[[trace2:.+]] = enzyme.addSubtrace %[[gen_res1]]#0 into %[[trace1]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:   %[[addf2:.+]] = arith.addf %[[addf1]], %[[gen_res1]]#1 : tensor<f64>
// CHECK-NEXT:   %[[trace3:.+]] = enzyme.addSampleToTrace(%[[gen_res1]]#3, %[[gen_res1]]#4 : tensor<f64>, tensor<f64>) into %[[trace2]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:   %[[subconstraint2:.+]] = enzyme.getSubconstraint %[[constraint]] {symbol = #enzyme.symbol<6>}
// CHECK-NEXT:   %[[gen_res2:.+]]:5 = call @two_normals.generate(%[[subconstraint2]], %[[gen_res1]]#2, %[[gen_res1]]#3, %[[stddev]]) : (!enzyme.Constraint, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:   %[[trace4:.+]] = enzyme.addSubtrace %[[gen_res2]]#0 into %[[trace3]] {symbol = #enzyme.symbol<6>}
// CHECK-NEXT:   %[[addf3:.+]] = arith.addf %[[addf2]], %[[gen_res2]]#1 : tensor<f64>
// CHECK-NEXT:   %[[trace5:.+]] = enzyme.addSampleToTrace(%[[gen_res2]]#3, %[[gen_res2]]#4 : tensor<f64>, tensor<f64>) into %[[trace4]] {symbol = #enzyme.symbol<6>}
// CHECK-NEXT:   %[[trace6:.+]] = enzyme.addWeightToTrace(%[[addf3]] : tensor<f64>) into %[[trace5]]
// CHECK-NEXT:   %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[gen_res2]]#3, %[[gen_res2]]#4 : tensor<f64>, tensor<f64>) into %[[trace6]]
// CHECK-NEXT:   return %[[final_trace]], %[[addf3]], %[[gen_res2]]#2, %[[gen_res2]]#3, %[[gen_res2]]#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func @two_normals.generate(%[[constraint:.+]]: !enzyme.Constraint, %[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %[[trace_init:.+]] = enzyme.initTrace {source_fn = @two_normals.generate} : !enzyme.Trace
// CHECK-NEXT:   %[[normal_call:.+]]:2 = call @normal(%[[rng]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:   %[[logpdf1:.+]] = call @logpdf(%[[normal_call]]#1, %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:   %[[trace1:.+]] = enzyme.addSampleToTrace(%[[normal_call]]#1 : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<3>}
// CHECK-NEXT:   %[[sample_from_constraint:.+]] = enzyme.getSampleFromConstraint %[[constraint]] {symbol = #enzyme.symbol<4>} : tensor<f64>
// CHECK-NEXT:   %[[logpdf2:.+]] = call @logpdf(%[[sample_from_constraint]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf2:.+]] = arith.addf %[[addf1]], %[[logpdf2]] : tensor<f64>
// CHECK-NEXT:   %[[trace2:.+]] = enzyme.addSampleToTrace(%[[sample_from_constraint]] : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<4>}
// CHECK-NEXT:   %[[trace3:.+]] = enzyme.addWeightToTrace(%[[addf2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:   %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[normal_call]]#1, %[[sample_from_constraint]] : tensor<f64>, tensor<f64>) into %[[trace3]]
// CHECK-NEXT:   return %[[final_trace]], %[[addf2]], %[[normal_call]]#0, %[[normal_call]]#1, %[[sample_from_constraint]] : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func @two_normals.generate_0(%[[constraint:.+]]: !enzyme.Constraint, %[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %[[trace_init:.+]] = enzyme.initTrace {source_fn = @two_normals.generate_0} : !enzyme.Trace
// CHECK-NEXT:   %[[sample_from_constraint1:.+]] = enzyme.getSampleFromConstraint %[[constraint]] {symbol = #enzyme.symbol<3>} : tensor<f64>
// CHECK-NEXT:   %[[logpdf1:.+]] = call @logpdf(%[[sample_from_constraint1]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:   %[[trace1:.+]] = enzyme.addSampleToTrace(%[[sample_from_constraint1]] : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<3>}
// CHECK-NEXT:   %[[normal_call:.+]]:2 = call @normal(%[[rng]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:   %[[logpdf2:.+]] = call @logpdf(%[[normal_call]]#1, %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf2:.+]] = arith.addf %[[addf1]], %[[logpdf2]] : tensor<f64>
// CHECK-NEXT:   %[[trace2:.+]] = enzyme.addSampleToTrace(%[[normal_call]]#1 : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<4>}
// CHECK-NEXT:   %[[trace3:.+]] = enzyme.addWeightToTrace(%[[addf2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:   %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[sample_from_constraint1]], %[[normal_call]]#1 : tensor<f64>, tensor<f64>) into %[[trace3]]
// CHECK-NEXT:   return %[[final_trace]], %[[addf2]], %[[normal_call]]#0, %[[sample_from_constraint1]], %[[normal_call]]#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }