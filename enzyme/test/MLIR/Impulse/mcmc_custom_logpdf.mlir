// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %sum_sq = impulse.dot %x, %x <{lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_logpdf
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK: %{{.+}} = call @logpdf(
  // CHECK-NEXT: %{{.+}} = arith.negf
  // Cloned, not called: a func.call's reverse rule reruns the forward density.
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[SS0:.+]] = impulse.dot %[[Q0]], %[[Q0]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP0:.+]] = arith.mulf %[[SS0]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U0]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q1:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[SS1:.+]] = impulse.dot %[[Q1]], %[[Q1]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP1:.+]] = arith.mulf %[[SS1]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U1]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK-NOT: call @logpdf
  func.func @nuts_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)

    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_logpdf
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK: %{{.+}} = call @logpdf(
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[SS0:.+]] = impulse.dot %[[Q0]], %[[Q0]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP0:.+]] = arith.mulf %[[SS0]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U0]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q1:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[SS1:.+]] = impulse.dot %[[Q1]], %[[Q1]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP1:.+]] = arith.mulf %[[SS1]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U1]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK-NOT: call @logpdf
  func.func @hmc_logpdf(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  func.func @shifted_logpdf(%x : tensor<2xf64>, %mu : tensor<2xf64>) -> tensor<f64> {
    %diff = arith.subf %x, %mu : tensor<2xf64>
    %sum_sq = impulse.dot %diff, %diff <{lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_shifted_logpdf
  // CHECK-SAME: (%{{.+}}: tensor<2xui64>, %[[MU:[^:]+]]: tensor<2xf64>)
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK: %{{.+}} = call @shifted_logpdf(
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D0:.+]] = arith.subf %[[Q0]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS0:.+]] = impulse.dot %[[D0]], %[[D0]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP0:.+]] = arith.mulf %[[SS0]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U0]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q1:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D1:.+]] = arith.subf %[[Q1]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS1:.+]] = impulse.dot %[[D1]], %[[D1]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP1:.+]] = arith.mulf %[[SS1]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U1]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK-NOT: call @shifted_logpdf
  func.func @nuts_shifted_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %mu, %step_size, %init_pos) {
      logpdf_fn = @shifted_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_shifted_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 2, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_shifted_logpdf
  // CHECK-SAME: (%{{.+}}: tensor<2xui64>, %[[MU:[^:]+]]: tensor<2xf64>)
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK: %{{.+}} = call @shifted_logpdf(
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D0:.+]] = arith.subf %[[Q0]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS0:.+]] = impulse.dot %[[D0]], %[[D0]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP0:.+]] = arith.mulf %[[SS0]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U0]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q1:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D1:.+]] = arith.subf %[[Q1]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS1:.+]] = impulse.dot %[[D1]], %[[D1]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP1:.+]] = arith.mulf %[[SS1]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U1]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK-NOT: call @shifted_logpdf
  func.func @hmc_shifted_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %mu, %step_size, %init_pos) {
      logpdf_fn = @shifted_logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_shifted_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 2, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  func.func @anisotropic_logpdf(%x : tensor<2xf64>, %mu : tensor<2xf64>, %precision : tensor<2xf64>) -> tensor<f64> {
    %diff = arith.subf %x, %mu : tensor<2xf64>
    %diff_sq = arith.mulf %diff, %diff : tensor<2xf64>
    %weighted = arith.mulf %precision, %diff_sq : tensor<2xf64>
    %ones = arith.constant dense<1.0> : tensor<2xf64>
    %sum = impulse.dot %ones, %weighted <{lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_anisotropic_logpdf
  // CHECK-SAME: (%{{.+}}: tensor<2xui64>, %[[MU:[^:]+]]: tensor<2xf64>, %[[PREC:[^:]+]]: tensor<2xf64>)
  // CHECK-DAG: %[[ONES:.+]] = arith.constant dense<1.000000e+00> : tensor<2xf64>
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK: %{{.+}} = call @anisotropic_logpdf(
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D0:.+]] = arith.subf %[[Q0]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[DSQ0:.+]] = arith.mulf %[[D0]], %[[D0]] : tensor<2xf64>
  // CHECK-NEXT: %[[W0:.+]] = arith.mulf %[[PREC]], %[[DSQ0]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS0:.+]] = impulse.dot %[[ONES]], %[[W0]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP0:.+]] = arith.mulf %[[SS0]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U0]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q1:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D1:.+]] = arith.subf %[[Q1]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[DSQ1:.+]] = arith.mulf %[[D1]], %[[D1]] : tensor<2xf64>
  // CHECK-NEXT: %[[W1:.+]] = arith.mulf %[[PREC]], %[[DSQ1]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS1:.+]] = impulse.dot %[[ONES]], %[[W1]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP1:.+]] = arith.mulf %[[SS1]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U1]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK-NOT: call @anisotropic_logpdf
  func.func @nuts_anisotropic_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>, %precision : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %mu, %precision, %step_size, %init_pos) {
      logpdf_fn = @anisotropic_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_anisotropic_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 3, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @hmc_anisotropic_logpdf
  // CHECK-SAME: (%{{.+}}: tensor<2xui64>, %[[MU:[^:]+]]: tensor<2xf64>, %[[PREC:[^:]+]]: tensor<2xf64>)
  // CHECK-DAG: %[[ONES:.+]] = arith.constant dense<1.000000e+00> : tensor<2xf64>
  // CHECK-DAG: %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
  // CHECK: %{{.+}} = call @anisotropic_logpdf(
  // CHECK-NEXT: %{{.+}} = arith.negf
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D0:.+]] = arith.subf %[[Q0]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[DSQ0:.+]] = arith.mulf %[[D0]], %[[D0]] : tensor<2xf64>
  // CHECK-NEXT: %[[W0:.+]] = arith.mulf %[[PREC]], %[[DSQ0]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS0:.+]] = impulse.dot %[[ONES]], %[[W0]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP0:.+]] = arith.mulf %[[SS0]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U0:.+]] = arith.negf %[[LP0]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U0]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK: impulse.for
  // CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%[[Q1:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[D1:.+]] = arith.subf %[[Q1]], %[[MU]] : tensor<2xf64>
  // CHECK-NEXT: %[[DSQ1:.+]] = arith.mulf %[[D1]], %[[D1]] : tensor<2xf64>
  // CHECK-NEXT: %[[W1:.+]] = arith.mulf %[[PREC]], %[[DSQ1]] : tensor<2xf64>
  // CHECK-NEXT: %[[SS1:.+]] = impulse.dot %[[ONES]], %[[W1]] {{.+}} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: %[[LP1:.+]] = arith.mulf %[[SS1]], %[[NEGHALF]] : tensor<f64>
  // CHECK-NEXT: %[[U1:.+]] = arith.negf %[[LP1]] : tensor<f64>
  // CHECK-NEXT: enzyme.yield %[[U1]], %{{.+}} : tensor<f64>, tensor<2xui64>
  // CHECK-NOT: call @anisotropic_logpdf
  func.func @hmc_anisotropic_logpdf(%rng : tensor<2xui64>, %mu : tensor<2xf64>, %precision : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %mu, %precision, %step_size, %init_pos) {
      logpdf_fn = @anisotropic_logpdf,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_anisotropic_logpdf",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 3, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
