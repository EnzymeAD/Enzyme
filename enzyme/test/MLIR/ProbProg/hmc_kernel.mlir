// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
        name = "hmc", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @hmc
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
// CHECK-DAG: %[[NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-DAG: %[[C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant dense<false> : tensor<10xi1>
// CHECK-DAG: %[[SAMPLES_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<10x1xf64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
//
// --- RNG splits ---
// CHECK: %[[SPLIT1:.+]]:2 = enzyme.randomSplit %[[RNG]] : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[SPLIT2:.+]]:3 = enzyme.randomSplit %[[SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
//
// --- Extract initial position from trace tensor ---
// CHECK-NEXT: %[[Q0_SLICE:.+]] = enzyme.dynamic_slice %[[INIT_TRACE]], %[[C0]], %[[C0]] {slice_sizes = array<i64: 1, 1>} : (tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[Q0:.+]] = enzyme.dynamic_update_slice %[[INIT_TRACE]], %[[Q0_SLICE]], %[[C0]], %[[C0]] : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
//
// --- Constrain position and call generate for U0 ---
// CHECK-NEXT: %[[Q0_CONS_S:.+]] = enzyme.dynamic_slice %[[Q0]], %[[C0]], %[[C0]] {slice_sizes = array<i64: 1, 1>}
// CHECK-NEXT: %[[Q0_CONS:.+]] = enzyme.dynamic_update_slice %[[INIT_TRACE]], %[[Q0_CONS_S]], %[[C0]], %[[C0]]
// CHECK-NEXT: %[[GEN_INIT:.+]]:4 = call @test.generate{{.*}}(%[[Q0_CONS]], %[[SPLIT2]]#1, %[[MEAN]], %[[STDDEV]]) : (tensor<1x1xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[U0:.+]] = arith.negf %[[GEN_INIT]]#1 : tensor<f64>
//
// --- Initial gradient via autodiff ---
// CHECK-NEXT: %[[AD_INIT:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[ONE]]) {
// CHECK-NEXT: ^bb0(%[[AD_ARG:.+]]: tensor<1x1xf64>):
// CHECK-NEXT: %[[AD_SLICE:.+]] = enzyme.dynamic_slice %[[AD_ARG]], %[[C0]], %[[C0]] {slice_sizes = array<i64: 1, 1>}
// CHECK-NEXT: %[[AD_CONS:.+]] = enzyme.dynamic_update_slice %[[INIT_TRACE]], %[[AD_SLICE]], %[[C0]], %[[C0]]
// CHECK-NEXT: %[[AD_GEN:.+]]:4 = func.call @test.generate{{.*}}(%[[AD_CONS]], %[[SPLIT2]]#1, %[[MEAN]], %[[STDDEV]]) : (tensor<1x1xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[AD_NEG:.+]] = arith.negf %[[AD_GEN]]#1 : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[AD_NEG]], %[[AD_GEN]]#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}
//
// --- Main sampling loop ---
// CHECK: %[[LOOP:.+]]:6 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C10]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%[[Q0]], %[[AD_INIT]]#2, %[[U0]], %[[SPLIT2]]#0, %[[SAMPLES_INIT]], %[[ACC_INIT]] : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK-NEXT: ^bb0(%[[ITER:.+]]: tensor<i64>, %[[Q:.+]]: tensor<1x1xf64>, %[[GRAD:.+]]: tensor<1x1xf64>, %[[U:.+]]: tensor<f64>, %[[RNG_I:.+]]: tensor<2xui64>, %[[SAMP_BUF:.+]]: tensor<10x1xf64>, %[[ACC_BUF:.+]]: tensor<10xi1>):
//
// --- Sample momentum p ~ N(0, I) ---
// CHECK-NEXT: %[[RNG_S:.+]]:3 = enzyme.randomSplit %[[RNG_I]] : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[RNG_M:.+]]:2 = enzyme.randomSplit %[[RNG_S]]#1 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[RNG_P:.+]], %[[P:.+]] = enzyme.random %[[RNG_M]]#0, %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
//
// --- Initial kinetic energy K0 = 0.5 * p^T * p (contract over both dims for 2D) ---
// CHECK-NEXT: %[[KE0_DOT:.+]] = enzyme.dot %[[P]], %[[P]] {{{.*}}lhs_contracting_dimensions = array<i64: 0, 1>{{.*}}} : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[KE0:.+]] = arith.mulf %[[KE0_DOT]], %[[HALF]] : tensor<f64>
//
// --- Initial Hamiltonian H0 = U + K ---
// CHECK-NEXT: %[[H0:.+]] = arith.addf %[[U]], %[[KE0]] : tensor<f64>
//
// --- Leapfrog integration loop ---
// CHECK-NEXT: %[[LF:.+]]:5 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C10]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%[[Q]], %[[P]], %[[GRAD]], %[[U]], %[[RNG_S]]#2 : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>) -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64> {
// CHECK-NEXT: ^bb0(%[[LF_I:.+]]: tensor<i64>, %[[LF_Q:.+]]: tensor<1x1xf64>, %[[LF_P:.+]]: tensor<1x1xf64>, %[[LF_G:.+]]: tensor<1x1xf64>, %[[LF_U:.+]]: tensor<f64>, %[[LF_RNG:.+]]: tensor<2xui64>):
//
// --- Leapfrog: direction selection ---
// CHECK-NEXT: %[[DIR:.+]] = enzyme.select %[[TRUE]], %[[EPS]], %[[NEG_EPS]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[DIR_BC:.+]] = "enzyme.broadcast"(%[[DIR]]) <{shape = array<i64: 1, 1>}> : (tensor<f64>) -> tensor<1x1xf64>
//
// --- Leapfrog: half step momentum p_half = p - (eps/2) * grad ---
// CHECK-NEXT: %[[HALF_DIR:.+]] = arith.mulf %[[DIR]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[HALF_DIR_BC:.+]] = "enzyme.broadcast"(%[[HALF_DIR]]) <{shape = array<i64: 1, 1>}> : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[GRAD_SCALED:.+]] = arith.mulf %[[HALF_DIR_BC]], %[[LF_G]] : tensor<1x1xf64>
// CHECK-NEXT: %[[P_HALF:.+]] = arith.subf %[[LF_P]], %[[GRAD_SCALED]] : tensor<1x1xf64>
//
// --- Leapfrog: full step position q_new = q + eps * M^-1 * p_half ---
// CHECK-NEXT: %[[P_STEP:.+]] = arith.mulf %[[DIR_BC]], %[[P_HALF]] : tensor<1x1xf64>
// CHECK-NEXT: %[[Q_NEW:.+]] = arith.addf %[[LF_Q]], %[[P_STEP]] : tensor<1x1xf64>
//
// --- Leapfrog: gradient at new position ---
// CHECK-NEXT: %[[AD_LF:.+]]:3 = enzyme.autodiff_region(%[[Q_NEW]], %[[ONE]]) {
// CHECK-NEXT: ^bb0(%[[AD_LF_ARG:.+]]: tensor<1x1xf64>):
// CHECK-NEXT: %[[AD_LF_SLICE:.+]] = enzyme.dynamic_slice %[[AD_LF_ARG]], %[[C0]], %[[C0]] {slice_sizes = array<i64: 1, 1>}
// CHECK-NEXT: %[[AD_LF_CONS:.+]] = enzyme.dynamic_update_slice %[[INIT_TRACE]], %[[AD_LF_SLICE]], %[[C0]], %[[C0]]
// CHECK-NEXT: %[[AD_LF_GEN:.+]]:4 = func.call @test.generate(%[[AD_LF_CONS]], %[[LF_RNG]], %[[MEAN]], %[[STDDEV]]) : (tensor<1x1xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[AD_LF_NEG:.+]] = arith.negf %[[AD_LF_GEN]]#1 : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[AD_LF_NEG]], %[[AD_LF_GEN]]#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}
//
// --- Leapfrog: second half step momentum p_new = p_half - (eps/2) * grad_new ---
// CHECK-NEXT: %[[GRAD_NEW_SCALED:.+]] = arith.mulf %[[HALF_DIR_BC]], %[[AD_LF]]#2 : tensor<1x1xf64>
// CHECK-NEXT: %[[P_NEW:.+]] = arith.subf %[[P_HALF]], %[[GRAD_NEW_SCALED]] : tensor<1x1xf64>
//
// --- Leapfrog yield ---
// CHECK-NEXT: enzyme.yield %[[Q_NEW]], %[[P_NEW]], %[[AD_LF]]#2, %[[AD_LF]]#0, %[[AD_LF]]#1 : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>
// CHECK-NEXT: }
//
// --- Final kinetic energy K_new = 0.5 * p_new^T * p_new ---
// CHECK-NEXT: %[[KE_DOT:.+]] = enzyme.dot %[[LF]]#1, %[[LF]]#1 {{{.*}}lhs_contracting_dimensions = array<i64: 0, 1>{{.*}}} : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[KE:.+]] = arith.mulf %[[KE_DOT]], %[[HALF]] : tensor<f64>
//
// --- Final Hamiltonian H_new = U_new + K_new ---
// CHECK-NEXT: %[[H_NEW:.+]] = arith.addf %[[LF]]#3, %[[KE]] : tensor<f64>
//
// --- MH accept/reject: delta_H = H0 - H_new ---
// CHECK-NEXT: %[[DH:.+]] = arith.subf %[[H0]], %[[H_NEW]] : tensor<f64>
// CHECK-NEXT: %[[EXP_DH:.+]] = math.exp %[[DH]] : tensor<f64>
// CHECK-NEXT: %[[ACCEPT_PROB:.+]] = arith.minimumf %[[EXP_DH]], %[[ONE]] : tensor<f64>
//
// --- Draw uniform for MH ---
// CHECK-NEXT: %[[RNG_U:.+]], %[[UNIF:.+]] = enzyme.random %[[LF]]#4, %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
//
// --- Accept comparison ---
// CHECK-NEXT: %[[ACCEPTED:.+]] = arith.cmpf olt, %[[UNIF]], %[[ACCEPT_PROB]] : tensor<f64>
//
// --- Select q, grad, U based on acceptance ---
// CHECK-NEXT: %[[Q_SEL:.+]] = enzyme.select %[[ACCEPTED]], %[[LF]]#0, %[[Q]] : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[GRAD_SEL:.+]] = enzyme.select %[[ACCEPTED]], %[[LF]]#2, %[[GRAD]] : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[U_SEL:.+]] = enzyme.select %[[ACCEPTED]], %[[LF]]#3, %[[U]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
//
// --- Store samples: conditional on iteration index ---
// CHECK-NEXT: %[[STORE_COND:.+]] = arith.cmpi sge, %[[ITER]], %[[C0]] : tensor<i64>
// CHECK-NEXT: %[[SAMP_UPD:.+]] = enzyme.dynamic_update_slice %[[SAMP_BUF]], %[[Q_SEL]], %[[ITER]], %[[C0]] : (tensor<10x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<10x1xf64>
// CHECK-NEXT: %[[SAMP_SEL:.+]] = enzyme.select %[[STORE_COND]], %[[SAMP_UPD]], %[[SAMP_BUF]] : (tensor<i1>, tensor<10x1xf64>, tensor<10x1xf64>) -> tensor<10x1xf64>
// CHECK-NEXT: %[[ACC_1D:.+]] = enzyme.reshape %[[ACCEPTED]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %[[ACC_UPD:.+]] = enzyme.dynamic_update_slice %[[ACC_BUF]], %[[ACC_1D]], %[[ITER]] : (tensor<10xi1>, tensor<1xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK-NEXT: %[[ACC_SEL:.+]] = enzyme.select %[[STORE_COND]], %[[ACC_UPD]], %[[ACC_BUF]] : (tensor<i1>, tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
//
// --- Yield from sampling loop ---
// CHECK-NEXT: enzyme.yield %[[Q_SEL]], %[[GRAD_SEL]], %[[U_SEL]], %[[RNG_S]]#0, %[[SAMP_SEL]], %[[ACC_SEL]] : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LOOP]]#4, %[[LOOP]]#5, %[[LOOP]]#3 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
// CHECK-NEXT: }
//
// --- Generated function: test.generate ---
// CHECK-LABEL: func.func @test.generate
// CHECK-SAME: (%[[G_ARG0:.+]]: tensor<1x1xf64>, %[[G_ARG1:.+]]: tensor<2xui64>, %[[G_ARG2:.+]]: tensor<f64>, %[[G_ARG3:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[G_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[G_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[G_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
// CHECK: %[[G_SLICED:.+]] = enzyme.slice %[[G_ARG0]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[G_VAL:.+]] = enzyme.reshape %[[G_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[G_LP:.+]] = call @logpdf(%[[G_VAL]], %[[G_ARG2]], %[[G_ARG3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[G_W:.+]] = arith.addf %[[G_LP]], %[[G_ZERO]] : tensor<f64>
// CHECK-NEXT: %[[G_RS:.+]] = enzyme.reshape %[[G_VAL]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[G_TR:.+]] = enzyme.dynamic_update_slice %[[G_TRACE_INIT]], %[[G_RS]], %[[G_C0]], %[[G_C0]] : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK-NEXT: return %[[G_TR]], %[[G_W]], %[[G_ARG1]], %[[G_VAL]] : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
