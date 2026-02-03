// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<10xi1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
        name = "hmc", selection = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<10xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<10xi1>, tensor<2xui64>
  }
}

// CHECK: %[[CST_NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK: %[[CST_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK: %[[CST_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK: %[[CST_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[CST_EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK: %[[CST_10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK: %[[CST_1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[CST_0:.+]] = arith.constant dense<0> : tensor<i64>

// CHECK: %[[ACCEPTED_BUF_INIT:.+]] = arith.constant dense<false> : tensor<10xi1>
// CHECK: %[[SAMPLES_BUF_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<10x1xf64>
// CHECK: %[[CST_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>

// CHECK: %[[INIT_TRACE:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK: %[[RNG_SPLIT1:.+]]:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[RNG_SPLIT2:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[Q0:.+]] = enzyme.getFlattenedSamplesFromTrace %[[INIT_TRACE]] {selection = {{.*}}} : (!enzyme.Trace) -> tensor<1xf64>
// CHECK: %[[WEIGHT:.+]] = enzyme.getWeightFromTrace %[[INIT_TRACE]] : (!enzyme.Trace) -> tensor<f64>
// CHECK: %[[U0:.+]] = arith.negf %[[WEIGHT]] : tensor<f64>

// Initial gradient
// CHECK: %[[INIT_GRAD:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[CST_ONE]]) {
// CHECK: ^bb0(%[[AD_ARG:.+]]: tensor<1xf64>):
// CHECK:   %[[UPDATE_CALL:.+]]:3 = func.call @test.update{{.*}}(%[[INIT_TRACE]], %[[AD_ARG]], %[[RNG_SPLIT2]]#1, %arg1, %arg2)
// CHECK:   %[[NEG_WEIGHT:.+]] = arith.negf %[[UPDATE_CALL]]#1 : tensor<f64>
// CHECK:   enzyme.yield %[[NEG_WEIGHT]], %[[UPDATE_CALL]]#2 : tensor<f64>, tensor<2xui64>
// CHECK: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}

// CHECK: %[[SAMPLE_LOOP:.+]]:6 = enzyme.for_loop(%[[CST_0]] : tensor<i64>) to(%[[CST_10]] : tensor<i64>) step(%[[CST_1]] : tensor<i64>) iter_args(%[[Q0]], %[[INIT_GRAD]]#2, %[[U0]], %[[RNG_SPLIT2]]#0, %[[SAMPLES_BUF_INIT]], %[[ACCEPTED_BUF_INIT]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK: ^bb0(%[[ITER:.+]]: tensor<i64>, %[[Q_LOOP:.+]]: tensor<1xf64>, %[[GRAD_LOOP:.+]]: tensor<1xf64>, %[[U_LOOP:.+]]: tensor<f64>, %[[RNG_LOOP:.+]]: tensor<2xui64>, %[[SAMPLES_BUF:.+]]: tensor<10x1xf64>, %[[ACCEPTED_BUF:.+]]: tensor<10xi1>):
// CHECK:   %[[MOM_SPLIT:.+]]:3 = enzyme.randomSplit %[[RNG_LOOP]] : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK:   %[[MOM_SPLIT2:.+]] = enzyme.randomSplit %[[MOM_SPLIT]]#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK:   %[[RNG_OUT:.+]], %[[P0:.+]] = enzyme.random %[[MOM_SPLIT2]], %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

// Initial kinetic energy: K0 = 0.5 * p^T * p
// CHECK:   %[[P0_DOT:.+]] = enzyme.dot %[[P0]], %[[P0]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:   %[[K0:.+]] = arith.mulf %[[P0_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK:   %[[H0:.+]] = arith.addf %[[U_LOOP]], %[[K0]] : tensor<f64>

// Leapfrog loop
// CHECK:   %[[LEAPFROG:.+]]:5 = enzyme.for_loop(%[[CST_0]] : tensor<i64>) to(%[[CST_10]] : tensor<i64>) step(%[[CST_1]] : tensor<i64>) iter_args(%[[Q_LOOP]], %[[P0]], %[[GRAD_LOOP]], %[[U_LOOP]], %[[MOM_SPLIT]]#2 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>) -> tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64> {
// CHECK:   ^bb0(%[[LF_IDX:.+]]: tensor<i64>, %[[LF_Q:.+]]: tensor<1xf64>, %[[LF_P:.+]]: tensor<1xf64>, %[[LF_GRAD:.+]]: tensor<1xf64>, %[[LF_U:.+]]: tensor<f64>, %[[LF_RNG:.+]]: tensor<2xui64>):

// Direction
// CHECK:     %[[STEP_SEL:.+]] = enzyme.select %[[CST_TRUE]], %[[CST_EPS]], %[[CST_NEG_EPS]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK:     %[[STEP_BCAST:.+]] = "enzyme.broadcast"(%[[STEP_SEL]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>

// Half step size
// CHECK:     %[[HALF_STEP:.+]] = arith.mulf %[[STEP_SEL]], %[[CST_HALF]] : tensor<f64>
// CHECK:     %[[HALF_STEP_BCAST:.+]] = "enzyme.broadcast"(%[[HALF_STEP]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>

// First half-step momentum update: p_half = p - (eps/2) * grad
// CHECK:     %[[GRAD_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[LF_GRAD]] : tensor<1xf64>
// CHECK:     %[[P_HALF:.+]] = arith.subf %[[LF_P]], %[[GRAD_SCALED]] : tensor<1xf64>

// Full step position update: q_new = q + eps * p_half (with M^-1 = I)
// CHECK:     %[[P_HALF_SCALED:.+]] = arith.mulf %[[STEP_BCAST]], %[[P_HALF]] : tensor<1xf64>
// CHECK:     %[[Q_NEW:.+]] = arith.addf %[[LF_Q]], %[[P_HALF_SCALED]] : tensor<1xf64>

// Gradient (inside loop)
// CHECK:     %[[GRAD_RES:.+]]:3 = enzyme.autodiff_region(%[[Q_NEW]], %[[CST_ONE]]) {
// CHECK:     ^bb0(%[[AD_ARG2:.+]]: tensor<1xf64>):
// CHECK:       %[[UPDATE_CALL2:.+]]:3 = func.call @test.update(%[[INIT_TRACE]], %[[AD_ARG2]], %[[LF_RNG]], %arg1, %arg2)
// CHECK:       %[[NEG_WEIGHT2:.+]] = arith.negf %[[UPDATE_CALL2]]#1 : tensor<f64>
// CHECK:       enzyme.yield %[[NEG_WEIGHT2]], %[[UPDATE_CALL2]]#2 : tensor<f64>, tensor<2xui64>
// CHECK:     } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}

// Second half-step momentum update: p_new = p_half - (eps/2) * grad_new
// CHECK:     %[[GRAD_NEW_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[GRAD_RES]]#2 : tensor<1xf64>
// CHECK:     %[[P_NEW:.+]] = arith.subf %[[P_HALF]], %[[GRAD_NEW_SCALED]] : tensor<1xf64>

// CHECK:     enzyme.yield %[[Q_NEW]], %[[P_NEW]], %[[GRAD_RES]]#2, %[[GRAD_RES]]#0, %[[GRAD_RES]]#1 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>
// CHECK:   }

// Metropolis-Hastings accept/reject
// Final kinetic energy: K = 0.5 * p^T * p
// CHECK:   %[[P_FINAL_DOT:.+]] = enzyme.dot %[[LEAPFROG]]#1, %[[LEAPFROG]]#1 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:   %[[K_FINAL:.+]] = arith.mulf %[[P_FINAL_DOT]], %[[CST_HALF]] : tensor<f64>

// Final Hamiltonian: H = U + K
// CHECK:   %[[H_FINAL:.+]] = arith.addf %[[LEAPFROG]]#3, %[[K_FINAL]] : tensor<f64>

// Delta energy: delta_H = H0 - H_final
// CHECK:   %[[DELTA_H:.+]] = arith.subf %[[H0]], %[[H_FINAL]] : tensor<f64>

// Acceptance probability: alpha = min(1, exp(delta_H))
// CHECK:   %[[EXP_DELTA:.+]] = math.exp %[[DELTA_H]] : tensor<f64>
// CHECK:   %[[ACCEPT_PROB:.+]] = arith.minimumf %[[EXP_DELTA]], %[[CST_ONE]] : tensor<f64>

// Random uniform for MH step
// CHECK:   %[[RNG_MH:.+]], %[[U_RAND:.+]] = enzyme.random %[[LEAPFROG]]#4, %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

// Accept if u < alpha
// CHECK:   %[[ACCEPTED:.+]] = arith.cmpf olt, %[[U_RAND]], %[[ACCEPT_PROB]] : tensor<f64>

// Select position
// CHECK:   %[[Q_SELECTED:.+]] = enzyme.select %[[ACCEPTED]], %[[LEAPFROG]]#0, %[[Q_LOOP]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[GRAD_SELECTED:.+]] = enzyme.select %[[ACCEPTED]], %[[LEAPFROG]]#2, %[[GRAD_LOOP]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[U_SELECTED:.+]] = enzyme.select %[[ACCEPTED]], %[[LEAPFROG]]#3, %[[U_LOOP]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>

// Storage condition
// CHECK:   %[[STORE_COND:.+]] = arith.cmpi sge, %[[ITER]], %[[CST_0]] : tensor<i64>

// Update samples buffer
// CHECK:   %[[SAMPLES_UPDATED:.+]] = enzyme.dynamic_update %[[SAMPLES_BUF]], %[[ITER]], %[[Q_SELECTED]] : (tensor<10x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<10x1xf64>
// CHECK:   %[[SAMPLES_FINAL:.+]] = enzyme.select %[[STORE_COND]], %[[SAMPLES_UPDATED]], %[[SAMPLES_BUF]] : (tensor<i1>, tensor<10x1xf64>, tensor<10x1xf64>) -> tensor<10x1xf64>

// Update accepted buffer
// CHECK:   %[[ACCEPTED_UPDATED:.+]] = enzyme.dynamic_update %[[ACCEPTED_BUF]], %[[ITER]], %[[ACCEPTED]] : (tensor<10xi1>, tensor<i64>, tensor<i1>) -> tensor<10xi1>
// CHECK:   %[[ACCEPTED_FINAL:.+]] = enzyme.select %[[STORE_COND]], %[[ACCEPTED_UPDATED]], %[[ACCEPTED_BUF]] : (tensor<i1>, tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>

// CHECK:   enzyme.yield %[[Q_SELECTED]], %[[GRAD_SELECTED]], %[[U_SELECTED]], %[[MOM_SPLIT]]#0, %[[SAMPLES_FINAL]], %[[ACCEPTED_FINAL]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK: }
