// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %pred : tensor<i1>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %if:2 = impulse.if (%pred) ({
      %e:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %e#0, %e#1 : tensor<2xui64>, tensor<f64>
    }, {
      %l:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %l#0, %l#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %pred : tensor<i1>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:4 = impulse.regenerate @model(%rng, %pred, %mean, %stddev) given %trace
        { selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], regenerate_addresses = [[#impulse.symbol<1>]] }
        : (tensor<2xui64>, tensor<i1>, tensor<f64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x2xf64>, %[[ARG1:.+]]: tensor<2xui64>, %[[ARG2:.+]]: tensor<i1>, %[[ARG3:.+]]: tensor<f64>, %[[ARG4:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK: %[[IF:.+]]:4 = impulse.if(%[[ARG2]]) ({
// CHECK-NEXT: %[[TS:.+]]:2 = func.call @normal(%[[ARG1]], %[[ARG3]], %[[ARG4]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[TLP:.+]] = func.call @logpdf(%[[TS]]#1, %[[ARG3]], %[[ARG4]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[TW:.+]] = arith.addf %[[TLP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[TRS:.+]] = impulse.reshape %[[TS]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[TTR:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[TRS]], %[[C0]], %[[C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: impulse.yield %[[TS]]#0, %[[TS]]#1, %[[TTR]], %[[TW]] : tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[SLICE:.+]] = impulse.slice %[[ARG0]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[FVAL:.+]] = impulse.reshape %[[SLICE]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[FLP:.+]] = func.call @logpdf(%[[FVAL]], %[[ARG3]], %[[ARG4]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[FW:.+]] = arith.addf %[[FLP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[FRS:.+]] = impulse.reshape %[[FVAL]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[FTR:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[FRS]], %[[C0]], %[[C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: impulse.yield %[[ARG1]], %[[FVAL]], %[[FTR]], %[[FW]] : tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
// CHECK-NEXT: return %[[IF]]#2, %[[IF]]#3, %[[IF]]#0, %[[IF]]#1 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
