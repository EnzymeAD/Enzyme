// RUN: %eopt %s --inline-infer-regions | FileCheck %s

module {
  func.func private @model_logpdf(%position : tensor<8xf64>,
                                  %B : tensor<8xf64>) -> tensor<f64> {
    %scaled = arith.mulf %position, %B : tensor<8xf64>
    %sq = impulse.dot %scaled, %scaled {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK: impulse.infer_region(%[[RNG:[0-9a-z_]+]], %[[B:[0-9a-z_]+]])
  // CHECK-SAME: logpdf_fn = @model_logpdf
  // CHECK-SAME: sampler {
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[POS:.+]]: tensor<8xf64>):
  // CHECK-NEXT: %[[SCALED:.+]] = arith.mulf %[[POS]], %[[B]] : tensor<8xf64>
  // CHECK-NEXT: %[[SQ:.+]] = impulse.dot %[[SCALED]], %[[SCALED]]
  // CHECK: impulse.yield %{{.+}} : tensor<f64>
  // CHECK: num_position_args = 1 : i64
  // CHECK-SAME: position_size = 8 : i64
  // CHECK-NOT: impulse.infer(
  func.func @main(%rng : tensor<2xui64>,
                  %B : tensor<8xf64>)
      -> (tensor<1x8xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<0.0> : tensor<1x8xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %imm = arith.constant dense<1.0> : tensor<8xf64>
    %res:3 = "impulse.infer"(%rng, %B, %imm, %step_size, %init_pos) {
      logpdf_fn = @model_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operandSegmentSizes = array<i32: 2, 0, 0, 1, 1, 1, 0>
    } : (tensor<2xui64>, tensor<8xf64>, tensor<8xf64>, tensor<f64>, tensor<1x8xf64>)
        -> (tensor<1x8xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x8xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
