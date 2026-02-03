// RUN: %eopt --probprog %s | FileCheck %s
// RUN: %eopt --probprog --inline --cse --canonicalize %s | FileCheck %s --check-prefix=INLINE

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
    %unused = enzyme.initTrace : !enzyme.Trace
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
// CHECK-NEXT:   %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:   %[[sample_from_constraint1:.+]] = enzyme.getSampleFromConstraint %[[constraint]] {symbol = #enzyme.symbol<1>} : tensor<f64>
// CHECK-NEXT:   %[[logpdf1:.+]] = call @logpdf(%[[sample_from_constraint1]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:   %[[trace1:.+]] = enzyme.addSampleToTrace %[[sample_from_constraint1]] into %[[trace_init]] {symbol = #enzyme.symbol<1>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[subconstraint1:.+]] = enzyme.getSubconstraint %[[constraint]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:   %[[gen_res1:.+]]:5 = call @two_normals.generate_0(%[[subconstraint1]], %[[rng]], %[[sample_from_constraint1]], %[[stddev]]) : (!enzyme.Constraint, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:   %[[trace2:.+]] = enzyme.addSubtrace %[[gen_res1]]#0 into %[[trace1]] {symbol = #enzyme.symbol<2>} : (!enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// CHECK-NEXT:   %[[addf2:.+]] = arith.addf %[[addf1]], %[[gen_res1]]#1 : tensor<f64>
// CHECK-NEXT:   %[[trace3:.+]] = enzyme.addSampleToTrace %[[gen_res1]]#3, %[[gen_res1]]#4 into %[[trace2]] {symbol = #enzyme.symbol<2>} : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[subconstraint2:.+]] = enzyme.getSubconstraint %[[constraint]] {symbol = #enzyme.symbol<6>}
// CHECK-NEXT:   %[[gen_res2:.+]]:5 = call @two_normals.generate(%[[subconstraint2]], %[[gen_res1]]#2, %[[gen_res1]]#3, %[[stddev]]) : (!enzyme.Constraint, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:   %[[trace4:.+]] = enzyme.addSubtrace %[[gen_res2]]#0 into %[[trace3]] {symbol = #enzyme.symbol<6>} : (!enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// CHECK-NEXT:   %[[addf3:.+]] = arith.addf %[[addf2]], %[[gen_res2]]#1 : tensor<f64>
// CHECK-NEXT:   %[[trace5:.+]] = enzyme.addSampleToTrace %[[gen_res2]]#3, %[[gen_res2]]#4 into %[[trace4]] {symbol = #enzyme.symbol<6>} : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[trace6:.+]] = enzyme.addWeightToTrace %[[addf3]] into %[[trace5]] : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[final_trace:.+]] = enzyme.addRetvalToTrace %[[gen_res2]]#3, %[[gen_res2]]#4 into %[[trace6]] : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   return %[[final_trace]], %[[addf3]], %[[gen_res2]]#2, %[[gen_res2]]#3, %[[gen_res2]]#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func @two_normals.generate(%[[constraint:.+]]: !enzyme.Constraint, %[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:   %[[normal_call:.+]]:2 = call @normal(%[[rng]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:   %[[logpdf1:.+]] = call @logpdf(%[[normal_call]]#1, %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:   %[[trace1:.+]] = enzyme.addSampleToTrace %[[normal_call]]#1 into %[[trace_init]] {symbol = #enzyme.symbol<3>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[sample_from_constraint:.+]] = enzyme.getSampleFromConstraint %[[constraint]] {symbol = #enzyme.symbol<4>} : tensor<f64>
// CHECK-NEXT:   %[[logpdf2:.+]] = call @logpdf(%[[sample_from_constraint]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf2:.+]] = arith.addf %[[addf1]], %[[logpdf2]] : tensor<f64>
// CHECK-NEXT:   %[[trace2:.+]] = enzyme.addSampleToTrace %[[sample_from_constraint]] into %[[trace1]] {symbol = #enzyme.symbol<4>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[trace3:.+]] = enzyme.addWeightToTrace %[[addf2]] into %[[trace2]] : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[final_trace:.+]] = enzyme.addRetvalToTrace %[[normal_call]]#1, %[[sample_from_constraint]] into %[[trace3]] : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   return %[[final_trace]], %[[addf2]], %[[normal_call]]#0, %[[normal_call]]#1, %[[sample_from_constraint]] : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }

// CHECK: func.func @two_normals.generate_0(%[[constraint:.+]]: !enzyme.Constraint, %[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:   %[[sample_from_constraint1:.+]] = enzyme.getSampleFromConstraint %[[constraint]] {symbol = #enzyme.symbol<3>} : tensor<f64>
// CHECK-NEXT:   %[[logpdf1:.+]] = call @logpdf(%[[sample_from_constraint1]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:   %[[trace1:.+]] = enzyme.addSampleToTrace %[[sample_from_constraint1]] into %[[trace_init]] {symbol = #enzyme.symbol<3>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[normal_call:.+]]:2 = call @normal(%[[rng]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:   %[[logpdf2:.+]] = call @logpdf(%[[normal_call]]#1, %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:   %[[addf2:.+]] = arith.addf %[[addf1]], %[[logpdf2]] : tensor<f64>
// CHECK-NEXT:   %[[trace2:.+]] = enzyme.addSampleToTrace %[[normal_call]]#1 into %[[trace1]] {symbol = #enzyme.symbol<4>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[trace3:.+]] = enzyme.addWeightToTrace %[[addf2]] into %[[trace2]] : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   %[[final_trace:.+]] = enzyme.addRetvalToTrace %[[sample_from_constraint1]], %[[normal_call]]#1 into %[[trace3]] : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// CHECK-NEXT:   return %[[final_trace]], %[[addf2]], %[[normal_call]]#0, %[[sample_from_constraint1]], %[[normal_call]]#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }

// INLINE:  func.func @generate(%[[arg0:.+]]: tensor<2xui64>, %[[arg1:.+]]: tensor<f64>, %[[arg2:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// INLINE-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// INLINE-NEXT:    %[[cst_0:.+]] = arith.constant dense<42> : tensor<ui64>
// INLINE-NEXT:    %[[v0:.+]] = builtin.unrealized_conversion_cast %[[cst_0]] : tensor<ui64> to !enzyme.Constraint
// INLINE-NEXT:    %[[v1:.+]] = enzyme.initTrace : !enzyme.Trace
// INLINE-NEXT:    %[[v2:.+]] = enzyme.getSampleFromConstraint %[[v0]] {symbol = #enzyme.symbol<1>} : tensor<f64>
// INLINE-NEXT:    %[[v3:.+]] = call @logpdf(%[[v2]], %[[arg1]], %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// INLINE-NEXT:    %[[v4:.+]] = arith.addf %[[v3]], %[[cst]] : tensor<f64>
// INLINE-NEXT:    %[[v5:.+]] = enzyme.addSampleToTrace %[[v2]] into %[[v1]] {symbol = #enzyme.symbol<1>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v6:.+]] = enzyme.getSubconstraint %[[v0]] {symbol = #enzyme.symbol<2>}
// INLINE-NEXT:    %[[v7:.+]] = enzyme.initTrace : !enzyme.Trace
// INLINE-NEXT:    %[[v8:.+]] = enzyme.getSampleFromConstraint %[[v6]] {symbol = #enzyme.symbol<3>} : tensor<f64>
// INLINE-NEXT:    %[[v9:.+]] = call @logpdf(%[[v8]], %[[v2]], %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// INLINE-NEXT:    %[[v10:.+]] = arith.addf %[[v9]], %[[cst]] : tensor<f64>
// INLINE-NEXT:    %[[v11:.+]] = enzyme.addSampleToTrace %[[v8]] into %[[v7]] {symbol = #enzyme.symbol<3>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v12:.+]]:2 = call @normal(%[[arg0]], %[[v2]], %[[arg2]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// INLINE-NEXT:    %[[v13:.+]] = call @logpdf(%[[v12]]#1, %[[v2]], %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// INLINE-NEXT:    %[[v14:.+]] = arith.addf %[[v10]], %[[v13]] : tensor<f64>
// INLINE-NEXT:    %[[v15:.+]] = enzyme.addSampleToTrace %[[v12]]#1 into %[[v11]] {symbol = #enzyme.symbol<4>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v16:.+]] = enzyme.addWeightToTrace %[[v14]] into %[[v15]] : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v17:.+]] = enzyme.addRetvalToTrace %[[v8]], %[[v12]]#1 into %[[v16]] : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v18:.+]] = enzyme.addSubtrace %[[v17]] into %[[v5]] {symbol = #enzyme.symbol<2>} : (!enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// INLINE-NEXT:    %[[v19:.+]] = arith.addf %[[v4]], %[[v14]] : tensor<f64>
// INLINE-NEXT:    %[[v20:.+]] = enzyme.addSampleToTrace %[[v8]], %[[v12]]#1 into %[[v18]] {symbol = #enzyme.symbol<2>} : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v21:.+]] = enzyme.getSubconstraint %[[v0]] {symbol = #enzyme.symbol<6>}
// INLINE-NEXT:    %[[v22:.+]] = enzyme.initTrace : !enzyme.Trace
// INLINE-NEXT:    %[[v23:.+]]:2 = call @normal(%[[v12]]#0, %[[v8]], %[[arg2]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// INLINE-NEXT:    %[[v24:.+]] = call @logpdf(%[[v23]]#1, %[[v8]], %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// INLINE-NEXT:    %[[v25:.+]] = arith.addf %[[v24]], %[[cst]] : tensor<f64>
// INLINE-NEXT:    %[[v26:.+]] = enzyme.addSampleToTrace %[[v23]]#1 into %[[v22]] {symbol = #enzyme.symbol<3>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v27:.+]] = enzyme.getSampleFromConstraint %[[v21]] {symbol = #enzyme.symbol<4>} : tensor<f64>
// INLINE-NEXT:    %[[v28:.+]] = call @logpdf(%[[v27]], %[[v8]], %[[arg2]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// INLINE-NEXT:    %[[v29:.+]] = arith.addf %[[v25]], %[[v28]] : tensor<f64>
// INLINE-NEXT:    %[[v30:.+]] = enzyme.addSampleToTrace %[[v27]] into %[[v26]] {symbol = #enzyme.symbol<4>} : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v31:.+]] = enzyme.addWeightToTrace %[[v29]] into %[[v30]] : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v32:.+]] = enzyme.addRetvalToTrace %[[v23]]#1, %[[v27]] into %[[v31]] : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v33:.+]] = enzyme.addSubtrace %[[v32]] into %[[v20]] {symbol = #enzyme.symbol<6>} : (!enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// INLINE-NEXT:    %[[v34:.+]] = arith.addf %[[v19]], %[[v29]] : tensor<f64>
// INLINE-NEXT:    %[[v35:.+]] = enzyme.addSampleToTrace %[[v23]]#1, %[[v27]] into %[[v33]] {symbol = #enzyme.symbol<6>} : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v36:.+]] = enzyme.addWeightToTrace %[[v34]] into %[[v35]] : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    %[[v37:.+]] = enzyme.addRetvalToTrace %[[v23]]#1, %[[v27]] into %[[v36]] : (!enzyme.Trace, tensor<f64>, tensor<f64>) -> !enzyme.Trace
// INLINE-NEXT:    return %[[v37]], %[[v34]], %[[v23]]#0, %[[v23]]#1, %[[v27]] : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// INLINE-NEXT:  }
