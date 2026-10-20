// RUN: %eopt --expand-impulse %s | FileCheck %s

// Sampler mu + U^T z (U upper Cholesky); logpdf solves U^T w = x - mu once.

module {
  func.func private @mvn_s(%r: tensor<2xui64>, %mu: tensor<2xf64>, %sig: tensor<2x2xf64>) -> (tensor<2xui64>, tensor<2xf64>) {
    return %r, %mu : tensor<2xui64>, tensor<2xf64>
  }
  func.func @mvn_model(%r: tensor<2xui64>, %mu: tensor<2xf64>, %sig: tensor<2x2xf64>) -> (tensor<2xui64>, tensor<2xf64>) {
    %o:2 = impulse.sample @mvn_s(%r, %mu, %sig) {symbol = #impulse.symbol<1>, dist = #impulse.dist<MVN>}
      : (tensor<2xui64>, tensor<2xf64>, tensor<2x2xf64>) -> (tensor<2xui64>, tensor<2xf64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<2xf64>
  }
  func.func @mvn_top(%r: tensor<2xui64>, %mu: tensor<2xf64>, %sig: tensor<2x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<2xf64>) {
    %t, %w, %rng, %v = impulse.simulate @mvn_model(%r, %mu, %sig) {selection = [[#impulse.symbol<1>]]}
      : (tensor<2xui64>, tensor<2xf64>, tensor<2x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<2xf64>)
    return %t, %w, %rng, %v : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<2xf64>
  }
}

// CHECK-LABEL: func.func @mvn_model.simulate
// CHECK: %{{.*}}, %[[Z:[a-zA-Z0-9_]+]] = impulse.random %{{.*}} {rng_distribution = #impulse.rng_distribution<NORMAL>}
// CHECK: %[[U:.+]] = impulse.cholesky %arg2 {lower = false}
// CHECK: %[[UTZ:.+]] = impulse.dot %[[U]], %[[Z]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>}
// CHECK: %[[DRAW:.+]] = arith.addf %arg1, %[[UTZ]]
// CHECK: %[[DIFF:.+]] = arith.subf %[[DRAW]], %arg1 : tensor<2xf64>
// CHECK-NEXT: %[[U2:.+]] = impulse.cholesky %arg2 {lower = false} : (tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT: %[[DCOL:.+]] = impulse.reshape %[[DIFF]] : (tensor<2xf64>) -> tensor<2x1xf64>
// CHECK-NEXT: %[[W:.+]] = impulse.triangular_solve %[[U2]], %[[DCOL]] {lower = false, transpose_a = #impulse.transpose<TRANSPOSE>} : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CHECK-NEXT: %[[WV:.+]] = impulse.reshape %[[W]] : (tensor<2x1xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[WSQ:.+]] = arith.mulf %[[WV]], %[[WV]] : tensor<2xf64>
// CHECK-NEXT: %{{.+}} = impulse.dot %[[WSQ]], %{{.+}} {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NOT: impulse.triangular_solve
// CHECK: return
