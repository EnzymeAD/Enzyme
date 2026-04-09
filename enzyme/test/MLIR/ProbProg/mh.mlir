// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.5, 1.0]]> : tensor<1x2xf64>
    %init_weight = arith.constant dense<-2.0> : tensor<f64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %res:3 = scf.for %i = %c0 to %c1000 step %c1 iter_args(%trace = %init_trace, %weight = %init_weight, %rng1 = %rng) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>) {
      %step1:4 = enzyme.mh @test(%rng1, %mean, %stddev) given %trace weight %weight
          { name = "mh_1", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], regenerate_addresses = [[#enzyme.symbol<2>]] }
          : (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
      %step2:4 = enzyme.mh @test(%step1#3, %mean, %stddev) given %step1#0 weight %step1#1
          { name = "mh_2", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], regenerate_addresses = [[#enzyme.symbol<1>]] }
          : (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
      scf.yield %step2#0, %step2#1, %step2#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>
    }
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>)
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<{{.*}}> : tensor<1x2xf64>
// CHECK-DAG: %[[INIT_WEIGHT:.+]] = arith.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C1000:.+]] = arith.constant 1000 : index
// CHECK: %[[LOOP:.+]]:3 = scf.for %[[IV:.+]] = %[[C0]] to %[[C1000]] step %[[C1]] iter_args(%[[TR:.+]] = %[[INIT_TRACE]], %[[WT:.+]] = %[[INIT_WEIGHT]], %[[RNG0:.+]] = %[[RNG]]) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT: %[[REGEN1:.+]]:4 = func.call @test.regenerate_0(%[[TR]], %[[RNG0]], %[[MEAN]], %[[STDDEV]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[WDIFF1:.+]] = arith.subf %[[REGEN1]]#1, %[[WT]] : tensor<f64>
// CHECK-NEXT: %[[RNG1:.+]], %[[U1:.+]] = enzyme.random %[[REGEN1]]#2, %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[LOG1:.+]] = math.log %[[U1]] : tensor<f64>
// CHECK-NEXT: %[[ACC1:.+]] = arith.cmpf olt, %[[LOG1]], %[[WDIFF1]] : tensor<f64>
// CHECK-NEXT: %[[SEL_TR1:.+]] = enzyme.select %[[ACC1]], %[[REGEN1]]#0, %[[TR]] : (tensor<i1>, tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SEL_WT1:.+]] = arith.select %[[ACC1]], %[[REGEN1]]#1, %[[WT]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: %[[REGEN2:.+]]:4 = func.call @test.regenerate(%[[SEL_TR1]], %[[RNG1]], %[[MEAN]], %[[STDDEV]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[WDIFF2:.+]] = arith.subf %[[REGEN2]]#1, %[[SEL_WT1]] : tensor<f64>
// CHECK-NEXT: %[[RNG2:.+]], %[[U2:.+]] = enzyme.random %[[REGEN2]]#2, %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[LOG2:.+]] = math.log %[[U2]] : tensor<f64>
// CHECK-NEXT: %[[ACC2:.+]] = arith.cmpf olt, %[[LOG2]], %[[WDIFF2]] : tensor<f64>
// CHECK-NEXT: %[[SEL_TR2:.+]] = enzyme.select %[[ACC2]], %[[REGEN2]]#0, %[[SEL_TR1]] : (tensor<i1>, tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SEL_WT2:.+]] = arith.select %[[ACC2]], %[[REGEN2]]#1, %[[SEL_WT1]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: scf.yield %[[SEL_TR2]], %[[SEL_WT2]], %[[RNG2]] : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LOOP]]#0, %[[LOOP]]#1, %[[LOOP]]#2 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>

// CHECK-LABEL: func.func @test.regenerate
// CHECK-SAME: (%[[R_ARG0:.+]]: tensor<1x2xf64>, %[[R_ARG1:.+]]: tensor<2xui64>, %[[R_ARG2:.+]]: tensor<f64>, %[[R_ARG3:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[R_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[R_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[R_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[R_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK: %[[R_REGEN:.+]]:2 = call @normal(%[[R_ARG1]], %[[R_ARG2]], %[[R_ARG3]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[R_LP1:.+]] = call @logpdf(%[[R_REGEN]]#1, %[[R_ARG2]], %[[R_ARG3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[R_W1:.+]] = arith.addf %[[R_LP1]], %[[R_ZERO]] : tensor<f64>
// CHECK-NEXT: %[[R_RS1:.+]] = enzyme.reshape %[[R_REGEN]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[R_TR1:.+]] = enzyme.dynamic_update_slice %[[R_TRACE_INIT]], %[[R_RS1]], %[[R_C0]], %[[R_C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[R_KEPT_SLICED:.+]] = enzyme.slice %[[R_ARG0]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[R_KEPT:.+]] = enzyme.reshape %[[R_KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[R_LP2:.+]] = call @logpdf(%[[R_KEPT]], %[[R_REGEN]]#1, %[[R_ARG3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[R_W2:.+]] = arith.addf %[[R_W1]], %[[R_LP2]] : tensor<f64>
// CHECK-NEXT: %[[R_RS2:.+]] = enzyme.reshape %[[R_KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[R_TR2:.+]] = enzyme.dynamic_update_slice %[[R_TR1]], %[[R_RS2]], %[[R_C0]], %[[R_C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: return %[[R_TR2]], %[[R_W2]], %[[R_REGEN]]#0, %[[R_KEPT]] : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @test.regenerate_0
// CHECK-SAME: (%[[R0_ARG0:.+]]: tensor<1x2xf64>, %[[R0_ARG1:.+]]: tensor<2xui64>, %[[R0_ARG2:.+]]: tensor<f64>, %[[R0_ARG3:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[R0_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[R0_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[R0_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[R0_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK: %[[R0_KEPT_SLICED:.+]] = enzyme.slice %[[R0_ARG0]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[R0_KEPT:.+]] = enzyme.reshape %[[R0_KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[R0_LP1:.+]] = call @logpdf(%[[R0_KEPT]], %[[R0_ARG2]], %[[R0_ARG3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[R0_W1:.+]] = arith.addf %[[R0_LP1]], %[[R0_ZERO]] : tensor<f64>
// CHECK-NEXT: %[[R0_RS1:.+]] = enzyme.reshape %[[R0_KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[R0_TR1:.+]] = enzyme.dynamic_update_slice %[[R0_TRACE_INIT]], %[[R0_RS1]], %[[R0_C0]], %[[R0_C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[R0_REGEN:.+]]:2 = call @normal(%[[R0_ARG1]], %[[R0_KEPT]], %[[R0_ARG3]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[R0_LP2:.+]] = call @logpdf(%[[R0_REGEN]]#1, %[[R0_KEPT]], %[[R0_ARG3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[R0_W2:.+]] = arith.addf %[[R0_W1]], %[[R0_LP2]] : tensor<f64>
// CHECK-NEXT: %[[R0_RS2:.+]] = enzyme.reshape %[[R0_REGEN]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[R0_TR2:.+]] = enzyme.dynamic_update_slice %[[R0_TR1]], %[[R0_RS2]], %[[R0_C0]], %[[R0_C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: return %[[R0_TR2]], %[[R0_W2]], %[[R0_REGEN]]#0, %[[R0_REGEN]]#1 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
