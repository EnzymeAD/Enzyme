// RUN: %eopt --probprog %s | FileCheck %s

// Test HMC kernel internals with full coverage.
// Focus on: momentum sampling, leapfrog integration loop, MH accept/reject step.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // ============================================================================
  // HMC with single sample to isolate kernel internals
  // trajectory_length = 1.0 with step_size = 0.1 gives 10 leapfrog steps
  // ============================================================================
  func.func @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
        name = "hmc", selection = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// ============================================================================
// HMC Kernel Structure - Full Coverage CHECK patterns
// ============================================================================
// CHECK-LABEL: func.func @hmc
// CHECK-NEXT: %[[CST_NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-NEXT: %[[CST_1_I64:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[CST_0_I64:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[CST_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-NEXT: %[[CST_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT: %[[CST_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[CST_EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-NEXT: %[[CST_10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-NEXT: %[[CST_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>

// ============================================================================
// 1. Initial state extraction from trace
// ============================================================================
// CHECK-NEXT: %[[INIT_TRACE:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT: %[[RNG_SPLIT1:.+]]:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[RNG_SPLIT2:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[Q0:.+]] = enzyme.getFlattenedSamplesFromTrace %[[INIT_TRACE]] {selection = {{.*}}} : (!enzyme.Trace) -> tensor<1xf64>
// CHECK-NEXT: %[[WEIGHT:.+]] = enzyme.getWeightFromTrace %[[INIT_TRACE]] : (!enzyme.Trace) -> tensor<f64>
// CHECK-NEXT: %[[U0:.+]] = arith.negf %[[WEIGHT]] : tensor<f64>

// ============================================================================
// 2. Initial gradient computation via autodiff_region
// ============================================================================
// CHECK-NEXT: %[[INIT_GRAD:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[CST_ONE]]) {
// CHECK-NEXT: ^bb0(%[[AD_ARG:.+]]: tensor<1xf64>):
// CHECK-NEXT: %[[UPDATE_CALL:.+]]:3 = func.call @test.update{{.*}}(%[[INIT_TRACE]], %[[AD_ARG]], %[[RNG_SPLIT2]]#1, %arg1, %arg2) : (!enzyme.Trace, tensor<1xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT: %[[NEG_WEIGHT:.+]] = arith.negf %[[UPDATE_CALL]]#1 : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[NEG_WEIGHT]], %[[UPDATE_CALL]]#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<1xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<1xf64>)

// ============================================================================
// 3. Momentum sampling from N(0, I)
// ============================================================================
// CHECK-NEXT: %[[RNG_SPLIT3:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT2]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[RNG_SPLIT4:.+]] = enzyme.randomSplit %[[RNG_SPLIT3]]#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT: %[[RNG_OUT:.+]], %[[P0:.+]] = enzyme.random %[[RNG_SPLIT4]], %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

// ============================================================================
// 4. Initial kinetic energy: K0 = 0.5 * p^T * p, and Hamiltonian H0 = U0 + K0
// ============================================================================
// CHECK-NEXT: %[[P0_DOT:.+]] = enzyme.dot %[[P0]], %[[P0]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[K0:.+]] = arith.mulf %[[P0_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK-NEXT: %[[H0:.+]] = arith.addf %[[U0]], %[[K0]] : tensor<f64>

// ============================================================================
// 5. Leapfrog integration loop (Velocity Verlet)
// Loop iter_args: [q, p, grad, U, rng]
// ============================================================================
// CHECK-NEXT: %[[LEAPFROG:.+]]:5 = enzyme.for_loop(%[[CST_0_I64]] : tensor<i64>) to(%[[CST_10]] : tensor<i64>) step(%[[CST_1_I64]] : tensor<i64>) iter_args(%[[Q0]], %[[P0]], %[[INIT_GRAD]]#2, %[[U0]], %[[RNG_SPLIT3]]#2 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>) -> tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64> {
// CHECK-NEXT: ^bb0(%[[LOOP_IDX:.+]]: tensor<i64>, %[[Q_LOOP:.+]]: tensor<1xf64>, %[[P_LOOP:.+]]: tensor<1xf64>, %[[GRAD_LOOP:.+]]: tensor<1xf64>, %[[U_LOOP:.+]]: tensor<f64>, %[[RNG_LOOP:.+]]: tensor<2xui64>):

// Direction selection (always forward for HMC)
// CHECK-NEXT: %[[STEP_SIZE_SEL:.+]] = enzyme.select %[[CST_TRUE]], %[[CST_EPS]], %[[CST_NEG_EPS]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[STEP_BCAST:.+]] = "enzyme.broadcast"(%[[STEP_SIZE_SEL]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>

// Half step size for momentum updates: eps/2
// CHECK-NEXT: %[[HALF_STEP:.+]] = arith.mulf %[[STEP_SIZE_SEL]], %[[CST_HALF]] : tensor<f64>
// CHECK-NEXT: %[[HALF_STEP_BCAST:.+]] = "enzyme.broadcast"(%[[HALF_STEP]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>

// First half-step momentum update: p_half = p - (eps/2) * grad
// CHECK-NEXT: %[[GRAD_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[GRAD_LOOP]] : tensor<1xf64>
// CHECK-NEXT: %[[P_HALF:.+]] = arith.subf %[[P_LOOP]], %[[GRAD_SCALED]] : tensor<1xf64>

// Full step position update: q_new = q + eps * p_half (with M^-1 = I)
// CHECK-NEXT: %[[P_HALF_SCALED:.+]] = arith.mulf %[[STEP_BCAST]], %[[P_HALF]] : tensor<1xf64>
// CHECK-NEXT: %[[Q_NEW:.+]] = arith.addf %[[Q_LOOP]], %[[P_HALF_SCALED]] : tensor<1xf64>

// Gradient computation via autodiff_region
// CHECK-NEXT: %[[GRAD_RES:.+]]:3 = enzyme.autodiff_region(%[[Q_NEW]], %[[CST_ONE]]) {
// CHECK-NEXT: ^bb0(%[[AD_ARG2:.+]]: tensor<1xf64>):
// CHECK-NEXT: %[[UPDATE_CALL2:.+]]:3 = func.call @test.update{{.*}}(%[[INIT_TRACE]], %[[AD_ARG2]], %[[RNG_LOOP]], %arg1, %arg2) : (!enzyme.Trace, tensor<1xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT: %[[NEG_WEIGHT2:.+]] = arith.negf %[[UPDATE_CALL2]]#1 : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[NEG_WEIGHT2]], %[[UPDATE_CALL2]]#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<1xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<1xf64>)

// Second half-step momentum update: p_new = p_half - (eps/2) * grad_new
// CHECK-NEXT: %[[GRAD_NEW_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[GRAD_RES]]#2 : tensor<1xf64>
// CHECK-NEXT: %[[P_NEW:.+]] = arith.subf %[[P_HALF]], %[[GRAD_NEW_SCALED]] : tensor<1xf64>

// Yield updated state
// CHECK-NEXT: enzyme.yield %[[Q_NEW]], %[[P_NEW]], %[[GRAD_RES]]#2, %[[GRAD_RES]]#0, %[[GRAD_RES]]#1 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>
// CHECK-NEXT: }

// ============================================================================
// 6. Final kinetic energy and Hamiltonian
// ============================================================================
// K = 0.5 * p^T * p
// CHECK-NEXT: %[[P_FINAL_DOT:.+]] = enzyme.dot %[[LEAPFROG]]#1, %[[LEAPFROG]]#1 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[K_FINAL:.+]] = arith.mulf %[[P_FINAL_DOT]], %[[CST_HALF]] : tensor<f64>
// H = U + K
// CHECK-NEXT: %[[H_FINAL:.+]] = arith.addf %[[LEAPFROG]]#3, %[[K_FINAL]] : tensor<f64>

// ============================================================================
// 7. Metropolis-Hastings accept/reject
// ============================================================================
// Delta energy: H0 - H (positive delta means lower energy -> accept)
// CHECK-NEXT: %[[DELTA_H:.+]] = arith.subf %[[H0]], %[[H_FINAL]] : tensor<f64>
// Acceptance probability: min(1, exp(delta_H))
// CHECK-NEXT: %[[EXP_DELTA:.+]] = math.exp %[[DELTA_H]] : tensor<f64>
// CHECK-NEXT: %[[ACCEPT_PROB:.+]] = arith.minimumf %[[EXP_DELTA]], %[[CST_ONE]] : tensor<f64>
// Random uniform for MH step
// CHECK-NEXT: %[[RNG_MH:.+]], %[[U_RAND:.+]] = enzyme.random %[[LEAPFROG]]#4, %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// Compare: accept if u < alpha
// CHECK-NEXT: %[[ACCEPTED:.+]] = arith.cmpf olt, %[[U_RAND]], %[[ACCEPT_PROB]] : tensor<f64>
// Select new or old position based on acceptance
// CHECK-NEXT: %[[Q_FINAL:.+]] = enzyme.select %[[ACCEPTED]], %[[LEAPFROG]]#0, %[[Q0]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>

// ============================================================================
// 8. Create trace from accepted position
// ============================================================================
// CHECK-NEXT: %[[FINAL_TRACE:.+]]:3 = call @test.update(%[[INIT_TRACE]], %[[Q_FINAL]], %[[RNG_SPLIT3]]#0, %arg1, %arg2) : (!enzyme.Trace, tensor<1xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// Select trace based on acceptance
// CHECK-NEXT: %[[SEL_TRACE:.+]] = enzyme.select %[[ACCEPTED]], %[[FINAL_TRACE]]#0, %[[INIT_TRACE]] : (tensor<i1>, !enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// CHECK-NEXT: return %[[SEL_TRACE]], %[[ACCEPTED]], %[[FINAL_TRACE]]#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
