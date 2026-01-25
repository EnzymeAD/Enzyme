// RUN: %eopt --probprog --canonicalize %s | FileCheck %s

module {
  func.func private @exponential(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %rate : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @exponential(%rng, %rate) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s", support = #enzyme.support<POSITIVE> } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @exponential(%s#0, %rate) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t", support = #enzyme.support<POSITIVE> } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @hmc(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace

    %inverse_mass_matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %res:3 = enzyme.mcmc @test(%rng, %rate) given %init_trace
      inverse_mass_matrix = %inverse_mass_matrix
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.000000e+00 : f64>, name = "hmc", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, !enzyme.Trace, tensor<2x2xf64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @hmc

// ============================================================================
// 1. UNCONSTRAIN TRANSFORM: math.log applied to trace samples
// For POSITIVE support, unconstrain(x) = log(x)
// ============================================================================

// Get flattened samples from trace, then recover and apply log to each sample
// CHECK: %[[FLATTENED:.+]] = enzyme.getFlattenedSamplesFromTrace %{{.+}} {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : (!enzyme.Trace) -> tensor<2xf64>
// CHECK: %[[SAMPLE1:.+]] = enzyme.recover_sample %[[FLATTENED]][0] : tensor<2xf64> -> tensor<1xf64>
// CHECK-NEXT: %[[LOG1:.+]] = math.log %[[SAMPLE1]] : tensor<1xf64>
// CHECK: %[[SAMPLE2:.+]] = enzyme.recover_sample %[[FLATTENED]][1] : tensor<2xf64> -> tensor<1xf64>
// CHECK-NEXT: %[[LOG2:.+]] = math.log %[[SAMPLE2]] : tensor<1xf64>

// Assemble the unconstrained position vector from the log-transformed samples
// CHECK: %[[EXTRACT1:.+]] = enzyme.dynamic_extract %[[LOG1]], %{{.+}} : (tensor<1xf64>, tensor<i64>) -> tensor<f64>
// CHECK: %[[POS1:.+]] = enzyme.dynamic_update %{{.+}}, %{{.+}}, %[[EXTRACT1]] : (tensor<2xf64>, tensor<i64>, tensor<f64>) -> tensor<2xf64>
// CHECK: %[[EXTRACT2:.+]] = enzyme.dynamic_extract %[[LOG2]], %{{.+}} : (tensor<1xf64>, tensor<i64>) -> tensor<f64>
// CHECK: %[[UNCONSTRAINED_POS:.+]] = enzyme.dynamic_update %[[POS1]], %{{.+}}, %[[EXTRACT2]] : (tensor<2xf64>, tensor<i64>, tensor<f64>) -> tensor<2xf64>

// ============================================================================
// 2. JACOBIAN CORRECTION: sum(z) subtracted from -weight
// For POSITIVE support, log|det(J)| = sum(z) where z is unconstrained position
// Potential energy U = -weight - sum(z)
// ============================================================================

// Get weight from trace and negate it: U_base = -weight
// CHECK: %[[WEIGHT:.+]] = enzyme.getWeightFromTrace %{{.+}} : (!enzyme.Trace) -> tensor<f64>
// CHECK-NEXT: %[[NEG_WEIGHT:.+]] = arith.negf %[[WEIGHT]] : tensor<f64>

// Compute sum(z) for Jacobian correction using dot product with ones vector
// CHECK: %[[Z1:.+]] = enzyme.recover_sample %[[UNCONSTRAINED_POS]][0] : tensor<2xf64> -> tensor<1xf64>
// CHECK-NEXT: %[[SUM1:.+]] = enzyme.dot %[[Z1]], %{{.+}} {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[Z2:.+]] = enzyme.recover_sample %[[UNCONSTRAINED_POS]][1] : tensor<2xf64> -> tensor<1xf64>
// CHECK-NEXT: %[[SUM2:.+]] = enzyme.dot %[[Z2]], %{{.+}} {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[TOTAL_JACOBIAN:.+]] = arith.addf %{{.+}}, %[[SUM2]] : tensor<f64>

// Subtract Jacobian correction from potential energy: U = -weight - sum(z)
// CHECK: %[[U0:.+]] = arith.subf %[[NEG_WEIGHT]], %[[TOTAL_JACOBIAN]] : tensor<f64>

// ============================================================================
// 3. CONSTRAIN TRANSFORM inside autodiff_region: math.exp before UpdateOp
// For POSITIVE support, constrain(z) = exp(z)
// The autodiff_region computes gradient of U(q) w.r.t. unconstrained position
// ============================================================================

// CHECK: %{{.+}} = enzyme.autodiff_region(%[[UNCONSTRAINED_POS]], %{{.+}}) {
// CHECK: ^bb0(%[[ARG:.+]]: tensor<2xf64>):

// Recover unconstrained samples and apply exp to constrain them
// CHECK: %[[Z_SAMPLE1:.+]] = enzyme.recover_sample %[[ARG]][0] : tensor<2xf64> -> tensor<1xf64>
// CHECK-NEXT: %[[EXP1:.+]] = math.exp %[[Z_SAMPLE1]] : tensor<1xf64>
// CHECK: %[[Z_SAMPLE2:.+]] = enzyme.recover_sample %[[ARG]][1] : tensor<2xf64> -> tensor<1xf64>
// CHECK-NEXT: %[[EXP2:.+]] = math.exp %[[Z_SAMPLE2]] : tensor<1xf64>

// Assemble constrained position vector from exp-transformed samples
// CHECK: %[[C_EXTRACT1:.+]] = enzyme.dynamic_extract %[[EXP1]], %{{.+}} : (tensor<1xf64>, tensor<i64>) -> tensor<f64>
// CHECK: %[[C_POS1:.+]] = enzyme.dynamic_update %{{.+}}, %{{.+}}, %[[C_EXTRACT1]] : (tensor<2xf64>, tensor<i64>, tensor<f64>) -> tensor<2xf64>
// CHECK: %[[C_EXTRACT2:.+]] = enzyme.dynamic_extract %[[EXP2]], %{{.+}} : (tensor<1xf64>, tensor<i64>) -> tensor<f64>
// CHECK: %[[CONSTRAINED_POS:.+]] = enzyme.dynamic_update %[[C_POS1]], %{{.+}}, %[[C_EXTRACT2]] : (tensor<2xf64>, tensor<i64>, tensor<f64>) -> tensor<2xf64>

// Call update function with constrained position
// CHECK: %[[UPDATE_RES:.+]]:3 = func.call @test.update{{.*}}(%{{.+}}, %[[CONSTRAINED_POS]], %{{.+}}, %{{.+}}) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)

// Compute adjusted potential energy: U = -weight - sum(z)
// CHECK: %[[NEG_UPDATE_WEIGHT:.+]] = arith.negf %[[UPDATE_RES]]#1 : tensor<f64>
// CHECK: enzyme.recover_sample %[[ARG]][0]
// CHECK-NEXT: enzyme.dot
// CHECK: enzyme.recover_sample %[[ARG]][1]
// CHECK-NEXT: enzyme.dot
// CHECK: %[[JAC_SUM:.+]] = arith.addf %{{.+}}, %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[ADJUSTED_U:.+]] = arith.subf %[[NEG_UPDATE_WEIGHT]], %[[JAC_SUM]] : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[ADJUSTED_U]], %{{.+}} : tensor<f64>, tensor<2xui64>
// CHECK: }
