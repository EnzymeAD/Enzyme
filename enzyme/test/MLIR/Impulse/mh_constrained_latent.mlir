// RUN: %eopt --expand-impulse %s | FileCheck %s

// Constrained top-level l reads the old trace: the constraint would freeze it.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name = "s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %l:2 = impulse.sample @normal(%s#0, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<2>, name = "l" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %pred = arith.cmpf ogt, %s#1, %mean : tensor<f64>
    %if:2 = impulse.if (%pred) ({
      %e:2 = impulse.sample @normal(%l#0, %l#1, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<3>, name = "ye" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %e#0, %e#1 : tensor<2xui64>, tensor<f64>
    }, {
      %f:2 = impulse.sample @normal(%l#0, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<4>, name = "yl" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %f#0, %f#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x4xf64>, %weight : tensor<f64>, %obs : tensor<1x3xf64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>) {
    %step:4 = impulse.mh @model(%rng, %mean, %stddev) given %trace weight %weight constrained_by %obs
        { name = "mh", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          regenerate_addresses = [[#impulse.symbol<1>]],
          constrained_addresses = [[#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]] }
        : (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x3xf64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
    return %step#0, %step#1, %step#2, %step#3 : tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @model.regenerate(
// CHECK-SAME: %[[TR:[^:]+]]: tensor<1x4xf64>, %[[OBS:[^:]+]]: tensor<1x3xf64>, %[[R:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>)
// CHECK: %[[LSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[L:.+]] = impulse.reshape %[[LSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[LPL:.+]] = call @logpdf(%[[L]], %[[M]], %[[SD]])
// CHECK: impulse.if(
// CHECK-NEXT: %[[YESL:.+]] = impulse.slice %[[OBS]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[YE:.+]] = impulse.reshape %[[YESL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %{{.+}} = func.call @logpdf(%[[YE]], %[[L]], %[[SD]])
// CHECK: }, {
// CHECK-NEXT: %[[YLSL:.+]] = impulse.slice %[[OBS]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x3xf64>) -> tensor<1x1xf64>
// CHECK-NOT: impulse.slice %[[OBS]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>
