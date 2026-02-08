// RUN: %eopt --probprog --canonicalize %s | FileCheck %s

module {
  func.func private @exponential(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %rate : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @exponential(%rng, %rate) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s", support = #enzyme.support<POSITIVE> } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @exponential(%s#0, %rate) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t", support = #enzyme.support<POSITIVE> } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @hmc
  func.func @hmc(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[1.0, 1.0]]> : tensor<1x2xf64>

    %inverse_mass_matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %res:3 = enzyme.mcmc @test(%rng, %rate) given %init_trace
      inverse_mass_matrix = %inverse_mass_matrix
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.000000e+00 : f64>, name = "hmc", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<2x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}

// CHECK: %[[INIT_TRACE:.+]] = arith.constant dense<1.0{{.*}}> : tensor<1x2xf64>
// CHECK: enzyme.dynamic_slice %[[INIT_TRACE]]
// CHECK: enzyme.dynamic_update_slice
// CHECK: enzyme.dynamic_slice %[[INIT_TRACE]]
// CHECK: %[[EXTRACTED_POS:.+]] = enzyme.dynamic_update_slice
// CHECK: %[[FLATTENED:.+]] = enzyme.reshape %[[EXTRACTED_POS]] : (tensor<1x2xf64>) -> tensor<2xf64>

// CHECK: %[[SAMPLE1:.+]] = enzyme.slice %[[FLATTENED]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[LOG1:.+]] = math.log %[[SAMPLE1]] : tensor<1xf64>
// CHECK: %[[SAMPLE2:.+]] = enzyme.slice %[[FLATTENED]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[LOG2:.+]] = math.log %[[SAMPLE2]] : tensor<1xf64>

// CHECK: %[[ELEM1_SLICED:.+]] = enzyme.dynamic_slice %[[LOG1]], %{{.+}} {slice_sizes = array<i64: 1>} : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK: %[[ELEM1:.+]] = enzyme.reshape %[[ELEM1_SLICED]] : (tensor<1xf64>) -> tensor<f64>
// CHECK: %[[ELEM1_1D:.+]] = enzyme.reshape %[[ELEM1]] : (tensor<f64>) -> tensor<1xf64>
// CHECK: %[[POS1:.+]] = enzyme.dynamic_update_slice %{{.+}}, %[[ELEM1_1D]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK: %[[ELEM2_SLICED:.+]] = enzyme.dynamic_slice %[[LOG2]], %{{.+}} {slice_sizes = array<i64: 1>} : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK: %[[ELEM2:.+]] = enzyme.reshape %[[ELEM2_SLICED]] : (tensor<1xf64>) -> tensor<f64>
// CHECK: %[[ELEM2_1D:.+]] = enzyme.reshape %[[ELEM2]] : (tensor<f64>) -> tensor<1xf64>
// CHECK: %[[UNCONSTRAINED_POS:.+]] = enzyme.dynamic_update_slice %[[POS1]], %[[ELEM2_1D]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>

// CHECK: %[[UNCONSTRAINED_POS_2D:.+]] = enzyme.reshape %[[UNCONSTRAINED_POS]] : (tensor<2xf64>) -> tensor<1x2xf64>
//
// --- Constrain position for initial generate ---
// CHECK: enzyme.dynamic_slice
// CHECK: enzyme.dynamic_update_slice
// CHECK: enzyme.dynamic_slice
// CHECK: enzyme.dynamic_update_slice
// CHECK: %[[INIT_GEN:.+]]:4 = call @test.generate{{.*}}(%{{.+}}, %{{.+}}, %{{.+}})
// CHECK: %[[NEG_WEIGHT:.+]] = arith.negf %[[INIT_GEN]]#1 : tensor<f64>

// CHECK: %[[JAC_POS:.+]] = enzyme.reshape %[[UNCONSTRAINED_POS_2D]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[Z1:.+]] = enzyme.slice %[[JAC_POS]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[SUM1:.+]] = enzyme.dot %[[Z1]], %{{.+}} {{.*}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[PARTIAL_SUM:.+]] = arith.addf %[[SUM1]], %{{.+}} : tensor<f64>
// CHECK: %[[Z2:.+]] = enzyme.slice %[[JAC_POS]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[SUM2:.+]] = enzyme.dot %[[Z2]], %{{.+}} {{.*}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[TOTAL_JACOBIAN:.+]] = arith.addf %[[PARTIAL_SUM]], %[[SUM2]] : tensor<f64>

// CHECK: %[[U0:.+]] = arith.subf %[[NEG_WEIGHT]], %[[TOTAL_JACOBIAN]] : tensor<f64>

// CHECK: %[[AUTODIFF:.+]] = enzyme.autodiff_region(%[[UNCONSTRAINED_POS_2D]], %{{.+}}) {
// CHECK: ^bb0(%[[ARG:.+]]: tensor<1x2xf64>):
// CHECK: %[[ARG_1D:.+]] = enzyme.reshape %[[ARG]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[Z_SAMPLE1:.+]] = enzyme.slice %[[ARG_1D]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[EXP1:.+]] = math.exp %[[Z_SAMPLE1]] : tensor<1xf64>
// CHECK: %[[Z_SAMPLE2:.+]] = enzyme.slice %[[ARG_1D]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[EXP2:.+]] = math.exp %[[Z_SAMPLE2]] : tensor<1xf64>

// CHECK: %[[C_ELEM1_S:.+]] = enzyme.dynamic_slice %[[EXP1]], %{{.+}} {slice_sizes = array<i64: 1>} : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK: %[[C_ELEM1:.+]] = enzyme.reshape %[[C_ELEM1_S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK: %[[C_ELEM1_1D:.+]] = enzyme.reshape %[[C_ELEM1]] : (tensor<f64>) -> tensor<1xf64>
// CHECK: %[[C_POS1:.+]] = enzyme.dynamic_update_slice %{{.+}}, %[[C_ELEM1_1D]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK: %[[C_ELEM2_S:.+]] = enzyme.dynamic_slice %[[EXP2]], %{{.+}} {slice_sizes = array<i64: 1>} : (tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK: %[[C_ELEM2:.+]] = enzyme.reshape %[[C_ELEM2_S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK: %[[C_ELEM2_1D:.+]] = enzyme.reshape %[[C_ELEM2]] : (tensor<f64>) -> tensor<1xf64>
// CHECK: %[[CONSTRAINED_POS:.+]] = enzyme.dynamic_update_slice %[[C_POS1]], %[[C_ELEM2_1D]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>

// CHECK: enzyme.reshape %[[CONSTRAINED_POS]]
// CHECK: %[[GEN_RES:.+]]:4 = func.call @test.generate{{.*}}
// CHECK: %[[NEG_GEN_WEIGHT:.+]] = arith.negf %[[GEN_RES]]#1 : tensor<f64>

// CHECK: %[[ARG_1D_2:.+]] = enzyme.reshape %[[ARG]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[ARG_Z1:.+]] = enzyme.slice %[[ARG_1D_2]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[ARG_DOT1:.+]] = enzyme.dot %[[ARG_Z1]], %{{.+}}
// CHECK: %[[ARG_PARTIAL:.+]] = arith.addf %[[ARG_DOT1]], %{{.+}} : tensor<f64>
// CHECK: %[[ARG_Z2:.+]] = enzyme.slice %[[ARG_1D_2]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[ARG_DOT2:.+]] = enzyme.dot %[[ARG_Z2]], %{{.+}}
// CHECK: %[[JAC_SUM:.+]] = arith.addf %[[ARG_PARTIAL]], %[[ARG_DOT2]] : tensor<f64>
// CHECK: %[[ADJUSTED_U:.+]] = arith.subf %[[NEG_GEN_WEIGHT]], %[[JAC_SUM]] : tensor<f64>
// CHECK: enzyme.yield %[[ADJUSTED_U]], %{{.+}} : tensor<f64>, tensor<2xui64>
// CHECK: }

// CHECK: enzyme.for_loop

// CHECK: enzyme.autodiff_region
// CHECK: math.exp
// CHECK: math.exp
// CHECK: call @test.generate
// CHECK: enzyme.yield
// CHECK: }

// CHECK: math.exp
// CHECK: arith.minimumf
// CHECK: enzyme.random
// CHECK: arith.cmpf olt
// CHECK: %[[FINAL_SELECT:.+]] = enzyme.select
// CHECK: %[[FINAL_1D:.+]] = enzyme.reshape %[[FINAL_SELECT]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[FINAL_SAMPLE1:.+]] = enzyme.slice %[[FINAL_1D]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[FINAL_EXP1:.+]] = math.exp %[[FINAL_SAMPLE1]] : tensor<1xf64>
// CHECK: %[[FINAL_SAMPLE2:.+]] = enzyme.slice %[[FINAL_1D]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[FINAL_EXP2:.+]] = math.exp %[[FINAL_SAMPLE2]] : tensor<1xf64>

// CHECK: enzyme.dynamic_update_slice
// CHECK: return

// CHECK-LABEL: func.func @test.generate
// CHECK: enzyme.slice %{{.+}} {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}
// CHECK: enzyme.reshape
// CHECK: call @logpdf
// CHECK: enzyme.slice %{{.+}} {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>}
// CHECK: enzyme.reshape
// CHECK: call @logpdf
// CHECK: return
