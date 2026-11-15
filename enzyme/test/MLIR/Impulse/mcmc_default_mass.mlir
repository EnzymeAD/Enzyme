// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func @logpdf_1d(%x : tensor<1xf64>) -> tensor<f64> {
    %sq = arith.mulf %x, %x : tensor<1xf64>
    %s = impulse.reshape %sq : (tensor<1xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %s : tensor<f64>
    return %result : tensor<f64>
  }

  func.func @logpdf_2d(%x : tensor<2xf64>) -> tensor<f64> {
    %sum_sq = impulse.dot %x, %x <{lhs_batching_dimensions = array<i64>, rhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sum_sq : tensor<f64>
    return %result : tensor<f64>
  }

  func.func @hmc_adapt_default_mass(%rng : tensor<2xui64>) -> (tensor<1x1xf64>, tensor<1x1xf64>) {
    %init_pos = arith.constant dense<[[0.5]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:9 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf_1d,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = true>,
      name = "hmc_adapt_default_mass",
      selection = [],
      all_addresses = [],
      num_warmup = 1,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 1, 1, 0, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)
    return %res#0, %res#8 : tensor<1x1xf64>, tensor<1x1xf64>
  }

  func.func @hmc_default_mass_2d(%rng : tensor<2xui64>) -> tensor<1x2xf64> {
    %init_pos = arith.constant dense<[[0.5, -0.5]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:9 = "impulse.infer"(%rng, %step_size, %init_pos) {
      logpdf_fn = @logpdf_2d,
      hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      name = "hmc_default_mass_2d",
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 1, 1, 0, 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)
    return %res#0 : tensor<1x2xf64>
  }
}

// ============================================================
// hmc_adapt_default_mass: no inverse_mass_matrix, adapt_mass_matrix = true.
// The finalized Welford diagonal is reshaped to the carried tensor<1x1xf64>.
// ============================================================
// CHECK-LABEL: func.func @hmc_adapt_default_mass
// CHECK-SAME: (%{{.+}}: tensor<2xui64>) -> (tensor<1x1xf64>, tensor<1x1xf64>)
// CHECK-DAG: %[[ONE_1D:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[ZERO_1D:.+]] = arith.constant dense<0.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[ONES:.+]] = arith.constant dense<1.000000e+00> : tensor<1x1xf64>
//
// --- Warmup loop carries the default inverse mass and its sqrt ---
// CHECK: %[[WARMUP:.+]]:16 = impulse.for(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ONES]], %[[ONES]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ZERO_1D]], %[[ZERO_1D]], %{{.+}}, %{{.+}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK-NEXT: ^bb0(%{{.+}}: tensor<i64>, %{{.+}}: tensor<1x1xf64>, %{{.+}}: tensor<1x1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %[[INV:.+]]: tensor<1x1xf64>, %[[SQRT:.+]]: tensor<1x1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<i64>):
//
// --- Window end: both branches yield the carried types ---
// CHECK: %{{.+}}:10 = impulse.if(%{{.+}}) ({
// CHECK: %[[NEW_INV:.+]] = arith.addf %{{.+}}, %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[NEW_STD:.+]] = math.sqrt %[[NEW_INV]] : tensor<1xf64>
// CHECK-NEXT: %[[NEW_SQRT:.+]] = arith.divf %[[ONE_1D]], %[[NEW_STD]] : tensor<1xf64>
// CHECK-NEXT: %[[NEW_INV_2D:.+]] = impulse.reshape %[[NEW_INV]] : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[NEW_SQRT_2D:.+]] = impulse.reshape %[[NEW_SQRT]] : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: impulse.yield %[[NEW_INV_2D]], %[[NEW_SQRT_2D]], %[[ZERO_1D]], %[[ZERO_1D]], {{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK-NEXT: impulse.yield %[[INV]], %[[SQRT]], {{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<1x1xf64>, tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>)
//
// --- final_inverse_mass_matrix is the adapted value from warmup ---
// CHECK: %[[SAMPLING:.+]]:7 = impulse.for
// CHECK: return %[[SAMPLING]]#4, %[[WARMUP]]#5 : tensor<1x1xf64>, tensor<1x1xf64>

// ============================================================
// hmc_default_mass_2d: no inverse_mass_matrix, position size 2.
// The default tensor<1x2xf64> inverse mass takes the diagonal path.
// ============================================================
// CHECK-LABEL: func.func @hmc_default_mass_2d
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ONES2:.+]] = arith.constant dense<1.000000e+00> : tensor<1x2xf64>
//
// --- Momentum and kinetic energy ---
// CHECK: %{{.+}}, %[[EPS2:.+]] = impulse.random {{.*}} <rng_distribution = NORMAL> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x2xf64>)
// CHECK-NEXT: %[[SQRT_2D:.+]] = impulse.reshape %[[ONES2]] : (tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[P:.+]] = arith.mulf %[[SQRT_2D]], %[[EPS2]] : tensor<1x2xf64>
// CHECK-NEXT: %[[INV_2D:.+]] = impulse.reshape %[[ONES2]] : (tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[V:.+]] = arith.mulf %[[INV_2D]], %[[P]] : tensor<1x2xf64>
// CHECK-NEXT: %[[PV:.+]] = impulse.dot %[[P]], %[[V]] <{{.*}}lhs_contracting_dimensions = [0, 1]{{.*}}> : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.mulf %[[PV]], %[[HALF]] : tensor<f64>
//
// --- Leapfrog position update: q + eps * (M^-1 * p_half) ---
// CHECK: %[[P_HALF:.+]] = arith.subf %{{.+}}, %{{.+}} : tensor<1x2xf64>
// CHECK-NEXT: %[[LF_INV_2D:.+]] = impulse.reshape %[[ONES2]] : (tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[LF_V:.+]] = arith.mulf %[[LF_INV_2D]], %[[P_HALF]] : tensor<1x2xf64>
// CHECK-NEXT: %[[LF_STEP:.+]] = arith.mulf %{{.+}}, %[[LF_V]] : tensor<1x2xf64>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[LF_STEP]] : tensor<1x2xf64>
