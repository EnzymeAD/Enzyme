// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func @logpdf(%x : tensor<1x2xf64>) -> tensor<f64> {
    %sum_sq = enzyme.dot %x, %x {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0, 1>, rhs_contracting_dimensions = array<i64: 0, 1>} : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_logpdf
  // CHECK: call @logpdf
  // CHECK-NEXT: %[[U0:.+]] = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %[[NEG:.+]] = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: enzyme.for_loop
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @nuts_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc (%rng)
      step_size = %step_size
      logpdf_fn = @logpdf
      initial_position = %init_pos
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "nuts_logpdf", selection = [], all_addresses = [], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_logpdf
  // CHECK: call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: enzyme.for_loop
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @hmc_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc (%rng)
      step_size = %step_size
      logpdf_fn = @logpdf
      initial_position = %init_pos
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "hmc_logpdf", selection = [], all_addresses = [], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
