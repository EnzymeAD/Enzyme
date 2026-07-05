// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %sum_sq = impulse.dot %x, %x {lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_multichain
  // CHECK: %[[TBUF:.+]] = arith.constant dense<0.000000e+00> : tensor<2x2x2xf64>
  // CHECK: %[[KEYS:.+]]:2 = impulse.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
  // CHECK: %[[FOR0:.+]]:6 = impulse.for
  // CHECK: %[[FOR1:.+]]:6 = impulse.for
  // CHECK: %[[T0R:.+]] = impulse.reshape %[[FOR0]]#4 : (tensor<2x2xf64>) -> tensor<1x2x2xf64>
  // CHECK-NEXT: %[[T0U:.+]] = impulse.dynamic_update_slice %[[TBUF]], %[[T0R]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2x2xf64>, tensor<1x2x2xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x2xf64>
  // CHECK-NEXT: %[[T1R:.+]] = impulse.reshape %[[FOR1]]#4 : (tensor<2x2xf64>) -> tensor<1x2x2xf64>
  // CHECK-NEXT: %[[T1U:.+]] = impulse.dynamic_update_slice %[[T0U]], %[[T1R]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2x2xf64>, tensor<1x2x2xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x2xf64>
  // CHECK-NEXT: %[[D0R:.+]] = impulse.reshape %[[FOR0]]#5 : (tensor<2xi1>) -> tensor<1x2xi1>
  // CHECK-NEXT: %[[D0U:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[D0R]], %{{.+}}, %{{.+}} : (tensor<2x2xi1>, tensor<1x2xi1>, tensor<i64>, tensor<i64>) -> tensor<2x2xi1>
  // CHECK-NEXT: %[[D1R:.+]] = impulse.reshape %[[FOR1]]#5 : (tensor<2xi1>) -> tensor<1x2xi1>
  // CHECK-NEXT: %[[D1U:.+]] = impulse.dynamic_update_slice %[[D0U]], %[[D1R]], %{{.+}}, %{{.+}} : (tensor<2x2xi1>, tensor<1x2xi1>, tensor<i64>, tensor<i64>) -> tensor<2x2xi1>
  // CHECK-NEXT: return %[[T1U]], %[[D1U]], %[[FOR1]]#3 : tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>
  func.func @nuts_multichain(%rng : tensor<2xui64>) -> (tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 2, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_multichain",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 2,
      num_chains = 2,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>)

    return %res#0, %res#1, %res#2 : tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>
  }

  // dimension. position/gradient/inverse_mass are 2x1x2, the scalars are 2.
  // CHECK-LABEL: func.func @nuts_multichain_state
  // CHECK: %[[SKEYS:.+]]:2 = impulse.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
  // CHECK: return %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>
  func.func @nuts_multichain_state(%rng : tensor<2xui64>) -> (tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:8 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 2, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_multichain_state",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 2,
      num_chains = 2,
      state_schema = ["position", "gradient", "potential_energy", "step_size", "inverse_mass_matrix"],
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>)

    return %res#0, %res#1, %res#2, %res#3, %res#4, %res#5, %res#6, %res#7 : tensor<2x2x2xf64>, tensor<2x2xi1>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>
  }
}
