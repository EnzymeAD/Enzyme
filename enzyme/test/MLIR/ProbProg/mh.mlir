// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>) {
    %cst = arith.constant dense<42> : tensor<ui64>
    %0 = builtin.unrealized_conversion_cast %cst : tensor<ui64> to !enzyme.Trace
    %false_t = arith.constant dense<false> : tensor<i1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %res:2 = scf.for %i = %c0 to %c1000 step %c1 iter_args(%trace = %0, %rng1 = %rng) -> (!enzyme.Trace, tensor<2xui64>) {
      %step1:3 = enzyme.mh @test(%rng1, %mean, %stddev) given %trace { name = "mh_1", selection = [[#enzyme.symbol<1>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
      %step2:3 = enzyme.mh @test(%step1#2, %mean, %stddev) given %step1#0 { name = "mh_2", selection = [[#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
      scf.yield %step2#0, %step2#2 : !enzyme.Trace, tensor<2xui64>
    }
    return %res#0, %res#1 : !enzyme.Trace, tensor<2xui64>
  }
}

// CHECK:  func.func @mh(%[[arg0:.+]]: tensor<2xui64>, %[[arg1:.+]]: tensor<f64>, %[[arg2:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[cst_0:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[c1000:.+]] = arith.constant 1000 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[cst_1:.+]] = arith.constant dense<42> : tensor<ui64>
// CHECK-NEXT:    %[[v0:.+]] = builtin.unrealized_conversion_cast %[[cst_1]] : tensor<ui64> to !enzyme.Trace
// CHECK-NEXT:    %[[v1:.+]]:2 = scf.for %[[arg3:.+]] = %[[c0]] to %[[c1000]] step %[[c1]] iter_args(%[[arg4:.+]] = %[[v0]], %[[arg5:.+]] = %[[arg0]]) -> (!enzyme.Trace, tensor<2xui64>) {
// CHECK-NEXT:      %[[v2:.+]]:3 = func.call @test.regenerate_0(%[[arg4]], %[[arg5]], %[[arg1]], %[[arg2]]) : (!enzyme.Trace, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %[[v3:.+]] = enzyme.getWeightFromTrace %[[arg4]] : tensor<f64>
// CHECK-NEXT:      %[[v4:.+]] = arith.subf %[[v2]]#1, %[[v3]] : tensor<f64>
// CHECK-NEXT:      %[[output_rng_state:.+]], %[[result:.+]] = enzyme.random %[[v2]]#2, %[[cst_0]], %[[cst]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:      %[[v5:.+]] = math.log %[[result]] : tensor<f64>
// CHECK-NEXT:      %[[v6:.+]] = arith.cmpf olt, %[[v5]], %[[v4]] : tensor<f64>
// CHECK-NEXT:      %[[extracted:.+]] = tensor.extract %[[v6]][] : tensor<i1>
// CHECK-NEXT:      %[[v7:.+]] = arith.select %[[extracted]], %[[v2]]#0, %[[arg4]] : !enzyme.Trace
// CHECK-NEXT:      %[[v8:.+]]:3 = func.call @test.regenerate(%[[v7]], %[[output_rng_state]], %[[arg1]], %[[arg2]]) : (!enzyme.Trace, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %[[v9:.+]] = enzyme.getWeightFromTrace %[[v7]] : tensor<f64>
// CHECK-NEXT:      %[[v10:.+]] = arith.subf %[[v8]]#1, %[[v9]] : tensor<f64>
// CHECK-NEXT:      %[[output_rng_state_2:.+]], %[[result_3:.+]] = enzyme.random %[[v8]]#2, %[[cst_0]], %[[cst]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:      %[[v11:.+]] = math.log %[[result_3]] : tensor<f64>
// CHECK-NEXT:      %[[v12:.+]] = arith.cmpf olt, %[[v11]], %[[v10]] : tensor<f64>
// CHECK-NEXT:      %[[extracted_4:.+]] = tensor.extract %[[v12]][] : tensor<i1>
// CHECK-NEXT:      %[[v13:.+]] = arith.select %[[extracted_4]], %[[v8]]#0, %[[v7]] : !enzyme.Trace
// CHECK-NEXT:      scf.yield %[[v13]], %[[output_rng_state_2]] : !enzyme.Trace, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[v1]]#0, %[[v1]]#1 : !enzyme.Trace, tensor<2xui64>
// CHECK-NEXT:  }


// CHECK:  func.func @test.regenerate(%[[arg0:.+]]: !enzyme.Trace, %[[arg1:.+]]: tensor<2xui64>, %[[arg2:.+]]: tensor<f64>, %[[arg3:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[sample1:.+]] = enzyme.getSampleFromTrace %[[arg0]] {symbol = #enzyme.symbol<1>} : tensor<f64>
// CHECK-NEXT:    %[[logpdf1:.+]] = call @logpdf(%[[sample1]], %[[arg2]], %[[arg3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[weight1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:    %[[trace1:.+]] = enzyme.addSampleToTrace(%[[sample1]] : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %[[normal_call:.+]]:2 = call @normal(%[[arg1]], %[[sample1]], %[[arg3]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[logpdf2:.+]] = call @logpdf(%[[normal_call]]#1, %[[sample1]], %[[arg3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[weight2:.+]] = arith.addf %[[weight1]], %[[logpdf2]] : tensor<f64>
// CHECK-NEXT:    %[[trace2:.+]] = enzyme.addSampleToTrace(%[[normal_call]]#1 : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %[[trace3:.+]] = enzyme.addWeightToTrace(%[[weight2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:    %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[normal_call]]#1 : tensor<f64>) into %[[trace3]]
// CHECK-NEXT:    return %[[final_trace]], %[[weight2]], %[[normal_call]]#0 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:  }

// CHECK:  func.func @test.regenerate_0(%[[arg0:.+]]: !enzyme.Trace, %[[arg1:.+]]: tensor<2xui64>, %[[arg2:.+]]: tensor<f64>, %[[arg3:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[normal_call:.+]]:2 = call @normal(%[[arg1]], %[[arg2]], %[[arg3]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[logpdf1:.+]] = call @logpdf(%[[normal_call]]#1, %[[arg2]], %[[arg3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[weight1:.+]] = arith.addf %[[logpdf1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:    %[[trace1:.+]] = enzyme.addSampleToTrace(%[[normal_call]]#1 : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %[[sample2:.+]] = enzyme.getSampleFromTrace %[[arg0]] {symbol = #enzyme.symbol<2>} : tensor<f64>
// CHECK-NEXT:    %[[logpdf2:.+]] = call @logpdf(%[[sample2]], %[[normal_call]]#1, %[[arg3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[weight2:.+]] = arith.addf %[[weight1]], %[[logpdf2]] : tensor<f64>
// CHECK-NEXT:    %[[trace2:.+]] = enzyme.addSampleToTrace(%[[sample2]] : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %[[trace3:.+]] = enzyme.addWeightToTrace(%[[weight2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:    %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[sample2]] : tensor<f64>) into %[[trace3]]
// CHECK-NEXT:    return %[[final_trace]], %[[weight2]], %[[normal_call]]#0 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:  }
