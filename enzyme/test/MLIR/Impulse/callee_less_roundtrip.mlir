// RUN: %eopt %s | %eopt | FileCheck %s

// Without `fn` the pretty form starts with `(`, which the callee parser skips.

module {
  // CHECK-LABEL: func.func @emitted_site
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[SCALE:.+]]: tensor<f64>)
  // CHECK-NEXT: %[[X:.+]]:2 = impulse.sample(%[[RNG]], %[[SCALE]]) <symbol = <1>, dist = <HALF_NORMAL>, name = "x"> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT: return %[[X]]#0, %[[X]]#1 : tensor<2xui64>, tensor<f64>
  func.func @emitted_site(%rng : tensor<2xui64>, %scale : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %x:2 = impulse.sample(%rng, %scale) <{dist = #impulse.dist<HALF_NORMAL>, symbol = #impulse.symbol<1>, name = "x"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @traced_site
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[SCALE:.+]]: tensor<f64>)
  // CHECK-NEXT: %[[X:.+]]:2 = impulse.sample @emitted_site(%[[RNG]], %[[SCALE]]) <symbol = <1>, name = "x"> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func @traced_site(%rng : tensor<2xui64>, %scale : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %x:2 = impulse.sample @emitted_site(%rng, %scale) <{symbol = #impulse.symbol<1>, name = "x"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %c = arith.constant dense<0.0> : tensor<f64>
    return %c : tensor<f64>
  }

  // CHECK-LABEL: func.func @custom_logpdf_infer
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>)
  // CHECK-NEXT: %[[POS:.+]] = arith.constant dense<{{\[}}[1.000000e+00, -1.000000e+00]]> : tensor<1x2xf64>
  // CHECK-NEXT: %[[STEP:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = impulse.infer(%[[RNG]]) init_step_size = %[[STEP]] logpdf_fn = @logpdf init_position = %[[POS]] <selection = [], all_addresses = [], nuts_config = <max_tree_depth = 3, max_delta_energy = 1.000000e+03 : f64, adapt_step_size = false, adapt_mass_matrix = false>> : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  func.func @custom_logpdf_infer(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer(%rng) init_step_size = %step_size logpdf_fn = @logpdf init_position = %init_pos <{
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [], all_addresses = [], num_samples = 1
    }> : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @custom_logpdf_infer_region
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[B:.+]]: tensor<2xf64>)
  // CHECK-NEXT: %[[POS:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
  // CHECK-NEXT: %[[STEP:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = impulse.infer_region(%[[RNG]], %[[B]]) init_step_size = %[[STEP]] logpdf_fn = @logpdf init_position = %[[POS]] sampler {
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[Q:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[QB:.+]] = arith.mulf %[[Q]], %[[B]] : tensor<2xf64>
  // CHECK-NEXT: %[[D:.+]] = impulse.dot %[[QB]], %[[QB]] <{{.+}}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: impulse.yield %[[D]] : tensor<f64>
  // CHECK-NEXT: } <selection = [], all_addresses = [], nuts_config = <max_tree_depth = 3>, num_position_args = 1, position_size = 2> : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  func.func @custom_logpdf_infer_region(%rng : tensor<2xui64>, %B : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<0.0> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer_region(%rng, %B) init_step_size = %step_size logpdf_fn = @logpdf init_position = %init_pos sampler {
    } logpdf {
    ^bb0(%q : tensor<2xf64>):
      %qb = arith.mulf %q, %B : tensor<2xf64>
      %d = impulse.dot %qb, %qb <{lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
      impulse.yield %d : tensor<f64>
    } <{
      nuts_config = #impulse.nuts_config<max_tree_depth = 3>,
      selection = [], all_addresses = [], num_samples = 1, num_position_args = 1, position_size = 2
    }> : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}
