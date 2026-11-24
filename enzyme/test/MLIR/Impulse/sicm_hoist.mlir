// RUN: %eopt --sicm %s | FileCheck %s

// With no rule set loaded the pass hoists what does not depend on a sample and
// builds the logpdf region; the factorization of the scaled matrix stays.

// CHECK-LABEL: func.func @gp(
// CHECK-SAME: %{{.+}}: tensor<2xui64>, %[[X:.+]]: tensor<3x3xf64>,
// CHECK: %[[R:.+]] = impulse.dot %[[X]], %[[X]] <lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: {{.*}} = impulse.infer_region @model(
// CHECK: %[[TAU:.+]]:2 = impulse.sample_region
// CHECK: %[[TAUSQ:.+]] = arith.mulf %[[TAU]]#1, %[[TAU]]#1 : tensor<f64>
// CHECK-NEXT: %[[K:.+]] = impulse.dot %[[TAUSQ]], %[[R]] <lhs_contracting_dimensions = [], rhs_contracting_dimensions = []> : (tensor<f64>, tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: %[[L:.+]] = impulse.cholesky %[[K]] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT: impulse.yield %[[TAU]]#0, %[[L]] : tensor<2xui64>, tensor<3x3xf64>
// CHECK-NEXT: } logpdf {
// CHECK-NEXT: ^bb0(%[[POS:.+]]: tensor<f64>):
// CHECK-NEXT: %[[LP:.+]] = arith.negf %[[POS]] : tensor<f64>
// CHECK-NEXT: impulse.yield %[[LP]], %[[LP]] : tensor<f64>, tensor<f64>

module {
  func.func private @halfnormal_sampler(%rng : tensor<2xui64>, %std : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %sample = math.absf %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }
  func.func private @halfnormal_logpdf(%x : tensor<f64>, %std : tensor<f64>) -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }
  func.func @gp(%rng : tensor<2xui64>, %X : tensor<3x3xf64>, %trace : tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %std = arith.constant dense<1.0> : tensor<f64>
    %res:8 = impulse.infer_region @model(%rng, %X) given %trace init_step_size = %step_size sampling {
    ^bb0(%arg_rng: tensor<2xui64>, %arg_X: tensor<3x3xf64>):
      %tau:2 = impulse.sample_region(%arg_rng, %std) sampling {
      ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
        %v = math.absf %s : tensor<f64>
        impulse.yield %r, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      ^bb0(%x: tensor<f64>, %s: tensor<f64>):
        %lp = arith.negf %x : tensor<f64>
        impulse.yield %lp : tensor<f64>
      } <{address = [#impulse.symbol<1>], fn = @halfnormal_sampler, logpdf_fn = @halfnormal_logpdf, support = #impulse.support<POSITIVE>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %R = impulse.dot %arg_X, %arg_X <{lhs_contracting_dimensions = array<i64: 1>, rhs_contracting_dimensions = array<i64: 1>}> : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
      %tausq = arith.mulf %tau#1, %tau#1 : tensor<f64>
      %K = impulse.dot %tausq, %R <{lhs_contracting_dimensions = array<i64>, rhs_contracting_dimensions = array<i64>}> : (tensor<f64>, tensor<3x3xf64>) -> tensor<3x3xf64>
      %L = impulse.cholesky %K <{lower = true}> : (tensor<3x3xf64>) -> tensor<3x3xf64>
      impulse.yield %tau#0, %L : tensor<2xui64>, tensor<3x3xf64>
    } logpdf {
    } <{all_addresses = [[#impulse.symbol<1>]], hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<3x3xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
