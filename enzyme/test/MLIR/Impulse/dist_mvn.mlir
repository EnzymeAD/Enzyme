// RUN: %eopt --expand-impulse %s | FileCheck %s

// Gaussian-quadratic-form logpdf via two triangular solves against U.
// test pins the cholesky uplo, the dot contraction dims, and the two triangular
// solves' transpose flags.

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
// CHECK: %{{.*}}, %[[Z:[a-zA-Z0-9_]+]] = impulse.random %{{.*}} {rng_distribution = #impulse<rng_distribution NORMAL>}
// CHECK: %[[U:.+]] = impulse.cholesky %arg2 {lower = false}
// CHECK: %[[UTZ:.+]] = impulse.dot %[[U]], %[[Z]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>}
// CHECK: %[[DRAW:.+]] = arith.addf %arg1, %[[UTZ]]
// logpdf: alpha = Sigma^-1 diff via solve U^T w = diff then U alpha = w.
// CHECK: impulse.cholesky %arg2 {lower = false}
// CHECK: impulse.triangular_solve %{{.*}}, %{{.*}} {lower = false, transpose_a = #impulse<transpose TRANSPOSE>}
// CHECK: impulse.triangular_solve %{{.*}}, %{{.*}} {lower = false}
