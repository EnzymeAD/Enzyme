// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %sum_sq = impulse.dot %x, %x <{lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @nuts_multichain
  // CHECK: %[[TBUF:.+]] = arith.constant dense<0.000000e+00> : tensor<2x2x2xf64>
  // CHECK: %[[KEYS:.+]]:2 = impulse.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
  // CHECK: %[[FOR0:.+]]:7 = impulse.for
  // CHECK: %[[FOR1:.+]]:7 = impulse.for
  // CHECK: %[[T0R:.+]] = impulse.reshape %[[FOR0]]#4 : (tensor<2x2xf64>) -> tensor<1x2x2xf64>
  // CHECK-NEXT: %[[T0U:.+]] = impulse.dynamic_update_slice %[[TBUF]], %[[T0R]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2x2xf64>, tensor<1x2x2xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x2xf64>
  // CHECK-NEXT: %[[T1R:.+]] = impulse.reshape %[[FOR1]]#4 : (tensor<2x2xf64>) -> tensor<1x2x2xf64>
  // CHECK-NEXT: %[[T1U:.+]] = impulse.dynamic_update_slice %[[T0U]], %[[T1R]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2x2xf64>, tensor<1x2x2xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x2xf64>
  // CHECK-NEXT: %[[D0R:.+]] = impulse.reshape %[[FOR0]]#5 : (tensor<2x2xi1>) -> tensor<1x2x2xi1>
  // CHECK-NEXT: %[[D0U:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[D0R]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2x2xi1>, tensor<1x2x2xi1>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x2xi1>
  // CHECK-NEXT: %[[D1R:.+]] = impulse.reshape %[[FOR1]]#5 : (tensor<2x2xi1>) -> tensor<1x2x2xi1>
  // CHECK-NEXT: %[[D1U:.+]] = impulse.dynamic_update_slice %[[D0U]], %[[D1R]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<2x2x2xi1>, tensor<1x2x2xi1>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x2x2xi1>
  // CHECK-NEXT: %[[L0R:.+]] = impulse.reshape %[[FOR0]]#6 : (tensor<2xf64>) -> tensor<1x2xf64>
  // CHECK-NEXT: %[[L0U:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[L0R]], %{{.+}}, %{{.+}} : (tensor<2x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
  // CHECK-NEXT: %[[L1R:.+]] = impulse.reshape %[[FOR1]]#6 : (tensor<2xf64>) -> tensor<1x2xf64>
  // CHECK-NEXT: %[[L1U:.+]] = impulse.dynamic_update_slice %[[L0U]], %[[L1R]], %{{.+}}, %{{.+}} : (tensor<2x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
  // CHECK-NEXT: return %[[T1U]], %[[D1U]], %[[L1U]], %[[FOR1]]#3 : tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>
  func.func @nuts_multichain(%rng : tensor<2xui64>) -> (tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[[1.0, -1.0]], [[1.0, -1.0]]]> : tensor<2x1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 2, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_multichain",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 2,
      num_chains = 2,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<2x1x2xf64>) -> (tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>)

    return %res#0, %res#1, %res#2, %res#3 : tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>
  }

  // ui64 has no arith.constant, so the rng slot buffer is a broadcast of a key.
  // CHECK-LABEL: func.func @nuts_multichain_state
  // CHECK: %[[SKEYS:.+]]:2 = impulse.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
  // CHECK: %[[RBUF:.+]] = "enzyme.broadcast"(%[[SFOR0:.+]]#3) <{shape = array<i64: 2>}> : (tensor<2xui64>) -> tensor<2x2xui64>
  // CHECK: %[[R0R:.+]] = impulse.reshape %[[SFOR0]]#3 : (tensor<2xui64>) -> tensor<1x2xui64>
  // CHECK-NEXT: %[[R0U:.+]] = impulse.dynamic_update_slice %[[RBUF]], %[[R0R]], %{{.+}}, %{{.+}} : (tensor<2x2xui64>, tensor<1x2xui64>, tensor<i64>, tensor<i64>) -> tensor<2x2xui64>
  // CHECK-NEXT: %[[R1R:.+]] = impulse.reshape %[[SFOR1:.+]]#3 : (tensor<2xui64>) -> tensor<1x2xui64>
  // CHECK-NEXT: %[[R1U:.+]] = impulse.dynamic_update_slice %[[R0U]], %[[R1R]], %{{.+}}, %{{.+}} : (tensor<2x2xui64>, tensor<1x2xui64>, tensor<i64>, tensor<i64>) -> tensor<2x2xui64>
  // CHECK: return %{{.+}}, %{{.+}}, %[[SFOR1]]#3, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[R1U]] : tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>, tensor<2x2xui64>
  func.func @nuts_multichain_state(%rng : tensor<2xui64>) -> (tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>, tensor<2x2xui64>) {
    %init_pos = arith.constant dense<[[[1.0, -1.0]], [[1.0, -1.0]]]> : tensor<2x1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:10 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 2, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "nuts_multichain_state",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 2,
      num_chains = 2,
      state_schema = ["position", "gradient", "potential_energy", "step_size", "inverse_mass_matrix", "rng"],
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 0, 1, 1, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<2x1x2xf64>) -> (tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>, tensor<2x2xui64>)

    return %res#0, %res#1, %res#2, %res#3, %res#4, %res#5, %res#6, %res#7, %res#8, %res#9 : tensor<2x2x2xf64>, tensor<2x2x2xi1>, tensor<2x2xf64>, tensor<2xui64>, tensor<2x1x2xf64>, tensor<2x1x2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2x1x2xf64>, tensor<2x2xui64>
  }
}
