// RUN: %eopt --probprog --split-input-file %s | FileCheck %s --check-prefix=BASE
// RUN: %eopt --probprog --split-input-file %s | FileCheck %s --check-prefix=HIER

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test_base(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %prev_trace = arith.constant dense<[[0.5, 1.0]]> : tensor<1x2xf64>
    %res:4 = enzyme.regenerate @model(%rng, %mean, %stddev) given %prev_trace
        { selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], regenerate_addresses = [[#enzyme.symbol<2>]] }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// BASE-LABEL: func.func @test_base
// BASE-SAME: (%[[BASE_RNG:.+]]: tensor<2xui64>, %[[BASE_MEAN:.+]]: tensor<f64>, %[[BASE_STDDEV:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// BASE-NEXT: %[[BASE_CST:.+]] = arith.constant dense<{{.*}}> : tensor<1x2xf64>
// BASE-NEXT: %[[BASE_CALL:.+]]:4 = call @model.regenerate(%[[BASE_CST]], %[[BASE_RNG]], %[[BASE_MEAN]], %[[BASE_STDDEV]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// BASE-NEXT: return %[[BASE_CALL]]#0, %[[BASE_CALL]]#1, %[[BASE_CALL]]#2, %[[BASE_CALL]]#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>

// BASE-LABEL: func.func @model.regenerate
// BASE-SAME: (%[[ARG0:.+]]: tensor<1x2xf64>, %[[ARG1:.+]]: tensor<2xui64>, %[[ARG2:.+]]: tensor<f64>, %[[ARG3:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// BASE-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// BASE-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// BASE-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// BASE-DAG: %[[TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// BASE: %[[KEPT_SLICED:.+]] = enzyme.slice %[[ARG0]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// BASE-NEXT: %[[KEPT:.+]] = enzyme.reshape %[[KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// BASE-NEXT: %[[LP1:.+]] = call @logpdf(%[[KEPT]], %[[ARG2]], %[[ARG3]])
// BASE-NEXT: %[[W1:.+]] = arith.addf %[[LP1]], %[[ZERO]] : tensor<f64>
// BASE-NEXT: %[[RS1:.+]] = enzyme.reshape %[[KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// BASE-NEXT: %[[TR1:.+]] = enzyme.dynamic_update_slice %[[TRACE_INIT]], %[[RS1]], %[[C0]], %[[C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// BASE-NEXT: %[[REGEN:.+]]:2 = call @normal(%[[ARG1]], %[[KEPT]], %[[ARG3]])
// BASE-NEXT: %[[LP2:.+]] = call @logpdf(%[[REGEN]]#1, %[[KEPT]], %[[ARG3]])
// BASE-NEXT: %[[W2:.+]] = arith.addf %[[W1]], %[[LP2]] : tensor<f64>
// BASE-NEXT: %[[RS2:.+]] = enzyme.reshape %[[REGEN]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// BASE-NEXT: %[[TR2:.+]] = enzyme.dynamic_update_slice %[[TR1]], %[[RS2]], %[[C0]], %[[C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// BASE-NEXT: return %[[TR2]], %[[W2]], %[[REGEN]]#0, %[[REGEN]]#1

// -----

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @inner(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<3> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<4> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %s#1, %t#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @outer(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:3 = enzyme.sample @inner(%s#0, %s#1, %stddev) { symbol = #enzyme.symbol<2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t#0, %t#1, %t#2 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test_hier(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %prev_trace = arith.constant dense<[[0.5, 1.0, 2.0]]> : tensor<1x3xf64>
    %res:5 = enzyme.regenerate @outer(%rng, %mean, %stddev) given %prev_trace
        { selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>, #enzyme.symbol<3>], [#enzyme.symbol<2>, #enzyme.symbol<4>]],
          regenerate_addresses = [[#enzyme.symbol<2>, #enzyme.symbol<4>]] }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3, %res#4 : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// HIER-LABEL: func.func @test_hier
// HIER-SAME: (%[[HIER_RNG:.+]]: tensor<2xui64>, %[[HIER_MEAN:.+]]: tensor<f64>, %[[HIER_STDDEV:.+]]: tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-NEXT: %[[HIER_CST:.+]] = arith.constant dense<{{.*}}> : tensor<1x3xf64>
// HIER-NEXT: %[[HIER_CALL:.+]]:5 = call @outer.regenerate(%[[HIER_CST]], %[[HIER_RNG]], %[[HIER_MEAN]], %[[HIER_STDDEV]]) : (tensor<1x3xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-NEXT: return %[[HIER_CALL]]#0, %[[HIER_CALL]]#1, %[[HIER_CALL]]#2, %[[HIER_CALL]]#3, %[[HIER_CALL]]#4 : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>

// HIER-LABEL: func.func @outer.regenerate
// HIER-SAME: (%[[O_ARG0:.+]]: tensor<1x3xf64>, %[[O_ARG1:.+]]: tensor<2xui64>, %[[O_ARG2:.+]]: tensor<f64>, %[[O_ARG3:.+]]: tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-DAG: %[[O_C1:.+]] = arith.constant dense<1> : tensor<i64>
// HIER-DAG: %[[O_C0:.+]] = arith.constant dense<0> : tensor<i64>
// HIER-DAG: %[[O_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// HIER-DAG: %[[O_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x3xf64>
// HIER: %[[O_KEPT_SLICED:.+]] = enzyme.slice %[[O_ARG0]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[O_KEPT:.+]] = enzyme.reshape %[[O_KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// HIER-NEXT: %[[O_LP1:.+]] = call @logpdf(%[[O_KEPT]], %[[O_ARG2]], %[[O_ARG3]])
// HIER-NEXT: %[[O_W1:.+]] = arith.addf %[[O_LP1]], %[[O_ZERO]] : tensor<f64>
// HIER-NEXT: %[[O_RS1:.+]] = enzyme.reshape %[[O_KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[O_TR1:.+]] = enzyme.dynamic_update_slice %[[O_TRACE_INIT]], %[[O_RS1]], %[[O_C0]], %[[O_C0]] : (tensor<1x3xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// HIER-NEXT: %[[O_SUB_TRACE:.+]] = enzyme.slice %[[O_ARG0]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x2xf64>
// HIER-NEXT: %[[O_NESTED:.+]]:5 = call @inner.regenerate(%[[O_SUB_TRACE]], %[[O_ARG1]], %[[O_KEPT]], %[[O_ARG3]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-NEXT: %[[O_W2:.+]] = arith.addf %[[O_W1]], %[[O_NESTED]]#1 : tensor<f64>
// HIER-NEXT: %[[O_TR2:.+]] = enzyme.dynamic_update_slice %[[O_TR1]], %[[O_NESTED]]#0, %[[O_C0]], %[[O_C1]] : (tensor<1x3xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// HIER-NEXT: return %[[O_TR2]], %[[O_W2]], %[[O_NESTED]]#2, %[[O_NESTED]]#3, %[[O_NESTED]]#4

// HIER-LABEL: func.func @inner.regenerate
// HIER-SAME: (%[[I_ARG0:.+]]: tensor<1x2xf64>, %[[I_ARG1:.+]]: tensor<2xui64>, %[[I_ARG2:.+]]: tensor<f64>, %[[I_ARG3:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-DAG: %[[I_C1:.+]] = arith.constant dense<1> : tensor<i64>
// HIER-DAG: %[[I_C0:.+]] = arith.constant dense<0> : tensor<i64>
// HIER-DAG: %[[I_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// HIER-DAG: %[[I_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// HIER: %[[I_KEPT_SLICED:.+]] = enzyme.slice %[[I_ARG0]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[I_KEPT:.+]] = enzyme.reshape %[[I_KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// HIER-NEXT: %[[I_LP1:.+]] = call @logpdf(%[[I_KEPT]], %[[I_ARG2]], %[[I_ARG3]])
// HIER-NEXT: %[[I_W1:.+]] = arith.addf %[[I_LP1]], %[[I_ZERO]] : tensor<f64>
// HIER-NEXT: %[[I_RS1:.+]] = enzyme.reshape %[[I_KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[I_TR1:.+]] = enzyme.dynamic_update_slice %[[I_TRACE_INIT]], %[[I_RS1]], %[[I_C0]], %[[I_C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// HIER-NEXT: %[[I_REGEN:.+]]:2 = call @normal(%[[I_ARG1]], %[[I_ARG2]], %[[I_ARG3]])
// HIER-NEXT: %[[I_LP2:.+]] = call @logpdf(%[[I_REGEN]]#1, %[[I_ARG2]], %[[I_ARG3]])
// HIER-NEXT: %[[I_W2:.+]] = arith.addf %[[I_W1]], %[[I_LP2]] : tensor<f64>
// HIER-NEXT: %[[I_RS2:.+]] = enzyme.reshape %[[I_REGEN]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[I_TR2:.+]] = enzyme.dynamic_update_slice %[[I_TR1]], %[[I_RS2]], %[[I_C0]], %[[I_C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// HIER-NEXT: return %[[I_TR2]], %[[I_W2]], %[[I_REGEN]]#0, %[[I_KEPT]], %[[I_REGEN]]#1
