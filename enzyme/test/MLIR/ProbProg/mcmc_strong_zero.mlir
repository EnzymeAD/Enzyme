// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %sum_sq = enzyme.dot %x, %x {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_strong_zero
  // CHECK: enzyme.autodiff_region
  // CHECK: strong_zero = true
  // CHECK: enzyme.for_loop
  // CHECK: enzyme.autodiff_region
  // CHECK: strong_zero = true
  func.func @nuts_strong_zero(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "enzyme.mcmc"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #enzyme.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      strong_zero = true,
      name = "nuts_strong_zero",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @nuts_default
  // CHECK: enzyme.autodiff_region
  // CHECK-NOT: strong_zero
  // CHECK: enzyme.for_loop
  // CHECK: enzyme.autodiff_region
  // CHECK-NOT: strong_zero
  // CHECK: enzyme.yield
  func.func @nuts_default(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "enzyme.mcmc"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #enzyme.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_default",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_strong_zero
  // CHECK: enzyme.autodiff_region
  // CHECK: strong_zero = true
  // CHECK: enzyme.for_loop
  // CHECK: enzyme.autodiff_region
  // CHECK: strong_zero = true
  func.func @hmc_strong_zero(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "enzyme.mcmc"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      strong_zero = true,
      name = "hmc_strong_zero",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
