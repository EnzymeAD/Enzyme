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


// CHECK:  func.func @mh(%[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>) {
// CHECK-NEXT:    %[[c1000:.+]] = arith.constant 1000 : index
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<42> : tensor<ui64>
// CHECK-NEXT:    %[[trace_init:.+]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<ui64> to !enzyme.Trace
// CHECK-NEXT:    %[[for_res:.+]]:2 = scf.for %[[i:.+]] = %[[c0]] to %[[c1000]] step %[[c1]] iter_args(%[[trace_in:.+]] = %[[trace_init]], %[[rng_in:.+]] = %[[rng]]) -> (!enzyme.Trace, tensor<2xui64>) {
// CHECK-NEXT:      %[[regen1:.+]]:3 = func.call @test.regenerate_0(%[[rng_in]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %[[w_prev:.+]] = enzyme.getWeightFromTrace %[[trace_in]] : tensor<f64>
// CHECK-NEXT:      %[[delta1:.+]] = arith.subf %[[regen1]]#1, %[[w_prev]] : tensor<f64>
// CHECK-NEXT:      %[[rng_out1:.+]], %[[value1:.+]] = enzyme.rand %[[regen1]]#2 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:      %[[log1:.+]] = math.log %[[value1]] : tensor<f64>
// CHECK-NEXT:      %[[cmp1:.+]] = arith.cmpf olt, %[[log1]], %[[delta1]] : tensor<f64>
// CHECK-NEXT:      %[[acc1:.+]] = tensor.extract %[[cmp1]][] : tensor<i1>
// CHECK-NEXT:      %[[trace_mid:.+]] = arith.select %[[acc1]], %[[regen1]]#0, %[[trace_in]] : !enzyme.Trace
// CHECK-NEXT:      %[[regen2:.+]]:3 = func.call @test.regenerate(%[[rng_out1]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %[[w_curr:.+]] = enzyme.getWeightFromTrace %[[trace_mid]] : tensor<f64>
// CHECK-NEXT:      %[[delta2:.+]] = arith.subf %[[regen2]]#1, %[[w_curr]] : tensor<f64>
// CHECK-NEXT:      %[[rng_out2:.+]], %[[value2:.+]] = enzyme.rand %[[regen2]]#2 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:      %[[log2:.+]] = math.log %[[value2]] : tensor<f64>
// CHECK-NEXT:      %[[cmp2:.+]] = arith.cmpf olt, %[[log2]], %[[delta2]] : tensor<f64>
// CHECK-NEXT:      %[[acc2:.+]] = tensor.extract %[[cmp2]][] : tensor<i1>
// CHECK-NEXT:      %[[trace_out:.+]] = arith.select %[[acc2]], %[[regen2]]#0, %[[trace_mid]] : !enzyme.Trace
// CHECK-NEXT:      scf.yield %[[trace_out]], %[[rng_out2]] : !enzyme.Trace, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[for_res]]#0, %[[for_res]]#1 : !enzyme.Trace, tensor<2xui64>
// CHECK-NEXT:  }

// CHECK:  func.func @test.regenerate(%[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[s1:.+]] = enzyme.getSampleFromTrace %[[trace_init]] {symbol = #enzyme.symbol<1>} : tensor<f64>
// CHECK-NEXT:    %[[log1:.+]] = call @logpdf(%[[s1]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[w1:.+]] = arith.addf %[[log1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:    %[[trace1:.+]] = enzyme.addSampleToTrace(%[[s1]] : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %[[n1:.+]]:2 = call @normal(%[[rng]], %[[s1]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[log2:.+]] = call @logpdf(%[[n1]]#1, %[[s1]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[w2:.+]] = arith.addf %[[w1]], %[[log2]] : tensor<f64>
// CHECK-NEXT:    %[[trace2:.+]] = enzyme.addSampleToTrace(%[[n1]]#1 : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %[[trace3:.+]] = enzyme.addWeightToTrace(%[[w2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:    %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[n1]]#1 : tensor<f64>) into %[[trace3]]
// CHECK-NEXT:    return %[[final_trace]], %[[w2]], %[[n1]]#0 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:  }

// CHECK:  func.func @test.regenerate_0(%[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[trace_init:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[n1:.+]]:2 = call @normal(%[[rng]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[log1:.+]] = call @logpdf(%[[n1]]#1, %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[w1:.+]] = arith.addf %[[log1]], %[[cst]] : tensor<f64>
// CHECK-NEXT:    %[[trace1:.+]] = enzyme.addSampleToTrace(%[[n1]]#1 : tensor<f64>) into %[[trace_init]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %[[s2:.+]] = enzyme.getSampleFromTrace %[[trace1]] {symbol = #enzyme.symbol<2>} : tensor<f64>
// CHECK-NEXT:    %[[log2:.+]] = call @logpdf(%[[s2]], %[[n1]]#1, %[[stddev]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[w2:.+]] = arith.addf %[[w1]], %[[log2]] : tensor<f64>
// CHECK-NEXT:    %[[trace2:.+]] = enzyme.addSampleToTrace(%[[s2]] : tensor<f64>) into %[[trace1]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %[[trace3:.+]] = enzyme.addWeightToTrace(%[[w2]] : tensor<f64>) into %[[trace2]]
// CHECK-NEXT:    %[[final_trace:.+]] = enzyme.addRetvalToTrace(%[[s2]] : tensor<f64>) into %[[trace3]]
// CHECK-NEXT:    return %[[final_trace]], %[[w2]], %[[n1]]#0 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:  }
