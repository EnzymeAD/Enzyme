// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %sum_sq = impulse.dot %x, %x {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
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
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @nuts_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)

    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_logpdf
  // CHECK: call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @hmc_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @shifted_logpdf(%x : tensor<2xf64>, %mu : tensor<2xf64>) -> tensor<f64> {
    %diff = arith.subf %x, %mu : tensor<2xf64>
    %sum_sq = impulse.dot %diff, %diff {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_shifted_logpdf
  // CHECK: call @shifted_logpdf
  // CHECK-NEXT: %[[U0:.+]] = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @shifted_logpdf
  // CHECK-NEXT: %[[NEG:.+]] = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @shifted_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @nuts_shifted_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %mu, %step_size, %init_pos) {
      logpdf_fn = @shifted_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_shifted_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 2, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_shifted_logpdf
  // CHECK: call @shifted_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @shifted_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @shifted_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @hmc_shifted_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %mu, %step_size, %init_pos) {
      logpdf_fn = @shifted_logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_shifted_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 2, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @anisotropic_logpdf(%x : tensor<2xf64>, %mu : tensor<2xf64>, %precision : tensor<2xf64>) -> tensor<f64> {
    %diff = arith.subf %x, %mu : tensor<2xf64>
    %diff_sq = arith.mulf %diff, %diff : tensor<2xf64>
    %weighted = arith.mulf %precision, %diff_sq : tensor<2xf64>
    %ones = arith.constant dense<1.0> : tensor<2xf64>
    %sum = impulse.dot %ones, %weighted {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_anisotropic_logpdf
  // CHECK: call @anisotropic_logpdf
  // CHECK-NEXT: %[[U0:.+]] = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @anisotropic_logpdf
  // CHECK-NEXT: %[[NEG:.+]] = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @anisotropic_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @nuts_anisotropic_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>, %precision : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %mu, %precision, %step_size, %init_pos) {
      logpdf_fn = @anisotropic_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_anisotropic_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 3, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_anisotropic_logpdf
  // CHECK: call @anisotropic_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @anisotropic_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @anisotropic_logpdf
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK-NEXT: enzyme.yield
  func.func @hmc_anisotropic_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>, %precision : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %mu, %precision, %step_size, %init_pos) {
      logpdf_fn = @anisotropic_logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_anisotropic_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 3, 0, 0, 1, 1, 0, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
