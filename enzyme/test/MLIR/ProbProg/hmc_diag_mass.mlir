// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %s#1, %t#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @hmc_diag_mass(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0, 0.0]]> : tensor<1x2xf64>
    %inv_mass = arith.constant dense<[2.0, 3.0]> : tensor<2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      inverse_mass_matrix = %inv_mass
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
        name = "hmc_diag", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @nuts_diag_mass(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0, 0.0]]> : tensor<1x2xf64>
    %inv_mass = arith.constant dense<[2.0, 3.0]> : tensor<2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      inverse_mass_matrix = %inv_mass
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "nuts_diag", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @hmc_diag_mass
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
// CHECK-DAG: %[[INV_MASS:.+]] = arith.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf64>
// CHECK-DAG: %[[MASS_SQRT:.+]] = arith.constant dense<[0.70710678118654746, 0.57735026918962584]> : tensor<2xf64>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK: enzyme.for_loop
// CHECK: ^bb0(
// CHECK: %[[EPS_RNG:.+]], %[[EPS:.+]] = enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x2xf64>)
// CHECK-NEXT: %[[SQRT_2D:.+]] = enzyme.reshape %[[MASS_SQRT]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[P:.+]] = arith.mulf %[[SQRT_2D]], %[[EPS]] : tensor<1x2xf64>
// CHECK-NEXT: %[[INV_2D:.+]] = enzyme.reshape %[[INV_MASS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[V:.+]] = arith.mulf %[[INV_2D]], %[[P]] : tensor<1x2xf64>
// CHECK-NEXT: %[[KE_DOT:.+]] = enzyme.dot %[[P]], %[[V]] {{{.*}}lhs_contracting_dimensions = array<i64: 0, 1>{{.*}}} : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
// CHECK-NEXT: %[[KE:.+]] = arith.mulf %[[KE_DOT]], %[[HALF]] : tensor<f64>
// CHECK: enzyme.for_loop
// CHECK: ^bb0(%[[LF_I:.+]]: tensor<i64>, %[[LF_Q:.+]]: tensor<1x2xf64>, %[[LF_P:.+]]: tensor<1x2xf64>, %[[LF_G:.+]]: tensor<1x2xf64>, %[[LF_U:.+]]: tensor<f64>, %[[LF_RNG:.+]]: tensor<2xui64>):
// CHECK: %[[DIR:.+]] = enzyme.select {{.*}} : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[DIR_BC:.+]] = "enzyme.broadcast"(%[[DIR]]) <{shape = array<i64: 1, 2>}> : (tensor<f64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[HALF_STEP:.+]] = arith.mulf %[[DIR]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[HALF_STEP_BC:.+]] = "enzyme.broadcast"(%[[HALF_STEP]]) <{shape = array<i64: 1, 2>}> : (tensor<f64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[GRAD_SCALED:.+]] = arith.mulf %[[HALF_STEP_BC]], %[[LF_G]] : tensor<1x2xf64>
// CHECK-NEXT: %[[P_HALF:.+]] = arith.subf %[[LF_P]], %[[GRAD_SCALED]] : tensor<1x2xf64>
// CHECK-NEXT: %[[INV_2D_LF:.+]] = enzyme.reshape %[[INV_MASS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[V_LF:.+]] = arith.mulf %[[INV_2D_LF]], %[[P_HALF]] : tensor<1x2xf64>
// CHECK-NEXT: %[[DELTA_Q:.+]] = arith.mulf %[[DIR_BC]], %[[V_LF]] : tensor<1x2xf64>
// CHECK-NEXT: %[[Q_NEW:.+]] = arith.addf %[[LF_Q]], %[[DELTA_Q]] : tensor<1x2xf64>
// CHECK: enzyme.autodiff_region
// CHECK: func.call @test.generate
// CHECK: enzyme.yield
// CHECK: }
// CHECK: arith.mulf {{.*}} : tensor<1x2xf64>
// CHECK-NEXT: arith.subf {{.*}} : tensor<1x2xf64>
// CHECK: enzyme.yield {{.*}} : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<2xui64>
// CHECK-NEXT: }
// CHECK-NEXT: %[[INV_2D_FINAL:.+]] = enzyme.reshape %[[INV_MASS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[V_FINAL:.+]] = arith.mulf %[[INV_2D_FINAL]], {{.*}} : tensor<1x2xf64>
// CHECK-NEXT: %[[KE_FINAL_DOT:.+]] = enzyme.dot {{.*}}, %[[V_FINAL]] {{{.*}}lhs_contracting_dimensions = array<i64: 0, 1>{{.*}}} : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
// CHECK-NEXT: %[[KE_FINAL:.+]] = arith.mulf %[[KE_FINAL_DOT]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: arith.addf {{.*}} : tensor<f64>
// CHECK-NEXT: arith.subf {{.*}} : tensor<f64>
// CHECK-NEXT: math.exp {{.*}} : tensor<f64>
// CHECK-NEXT: arith.minimumf {{.*}} : tensor<f64>
// CHECK-NEXT: enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution UNIFORM>}
// CHECK-NEXT: arith.cmpf olt, {{.*}} : tensor<f64>
// CHECK-NEXT: enzyme.select {{.*}} : (tensor<i1>, tensor<1x2xf64>, tensor<1x2xf64>)
// CHECK-NEXT: enzyme.select {{.*}} : (tensor<i1>, tensor<1x2xf64>, tensor<1x2xf64>)
// CHECK-NEXT: enzyme.select {{.*}} : (tensor<i1>, tensor<f64>, tensor<f64>)

// CHECK-LABEL: func.func @nuts_diag_mass
// CHECK-SAME: (%{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
// CHECK-DAG: %[[N_INV_MASS:.+]] = arith.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf64>
// CHECK-DAG: %[[N_MASS_SQRT:.+]] = arith.constant dense<[0.70710678118654746, 0.57735026918962584]> : tensor<2xf64>
// CHECK: enzyme.for_loop
// CHECK: ^bb0(
// CHECK: enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x2xf64>)
// CHECK-NEXT: %[[N_SQRT_2D:.+]] = enzyme.reshape %[[N_MASS_SQRT]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[N_P:.+]] = arith.mulf %[[N_SQRT_2D]], {{.*}} : tensor<1x2xf64>
// CHECK-NEXT: %[[N_INV_2D:.+]] = enzyme.reshape %[[N_INV_MASS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: arith.mulf %[[N_INV_2D]], %[[N_P]] : tensor<1x2xf64>
// CHECK-NEXT: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
// CHECK: enzyme.while_loop
// CHECK: enzyme.while_loop
// CHECK: enzyme.reshape %[[N_INV_MASS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: arith.mulf {{.*}} : tensor<1x2xf64>
// CHECK: enzyme.reshape %[[N_INV_MASS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: arith.mulf {{.*}} : tensor<1x2xf64>
// CHECK-NEXT: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
