// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // adapt_step_size = true, adapt_mass_matrix = true
  func.func @warmup_both(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5>,
        name = "warmup_both", selection = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }

  // adapt_step_size = true, adapt_mass_matrix = false
  func.func @warmup_step_only(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = true, adapt_mass_matrix = false>,
        name = "warmup_step_only", selection = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }

  // adapt_step_size = false, adapt_mass_matrix = true
  func.func @warmup_mass_only(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = true>,
        name = "warmup_mass_only", selection = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }

  // adapt_step_size = false, adapt_mass_matrix = false
  func.func @warmup_none(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "warmup_none", selection = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @warmup_both
// CHECK: %[[CST_LOG10:.+]] = arith.constant dense<2.3025850929940459> : tensor<f64>
// CHECK: %[[CST_SHRINK:.+]] = arith.constant dense<5.000000e-03> : tensor<f64>
// CHECK: %[[CST_5:.+]] = arith.constant dense<5.000000e+00> : tensor<f64>
// CHECK: %[[CST_FMAX:.+]] = arith.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK: %[[CST_FMIN:.+]] = arith.constant dense<4.940660e-324> : tensor<f64>
// CHECK: %[[CST_NEG_KAPPA:.+]] = arith.constant dense<-7.500000e-01> : tensor<f64>
// CHECK: %[[CST_GAMMA:.+]] = arith.constant dense<5.000000e-02> : tensor<f64>
// CHECK: %[[CST_TARGET:.+]] = arith.constant dense<8.000000e-01> : tensor<f64>
// CHECK: %[[CST_T0:.+]] = arith.constant dense<1.000000e+01> : tensor<f64>
// CHECK: %[[CST_NEG1:.+]] = arith.constant dense<-1> : tensor<i64>
// CHECK: %[[CST_INF:.+]] = arith.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK: %[[CST_CKPT_ZEROS:.+]] = arith.constant dense<0.000000e+00> : tensor<5x1xf64>
// CHECK: %[[CST_MAX_DEPTH:.+]] = arith.constant dense<5> : tensor<i64>
// CHECK: %[[CST_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK: %[[CST_FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK: %[[CST_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK: %[[CST_ZEROS_1:.+]] = arith.constant dense<0.000000e+00> : tensor<1xf64>
// CHECK: %[[CST_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[CST_PROX:.+]] = arith.constant dense<4.4408920985006262E-16> : tensor<f64>
// CHECK: %[[CST_ONES_1:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK: %[[CST_LAST_ITER:.+]] = arith.constant dense<9> : tensor<i64>
// CHECK: %[[CST_NUM_WARMUP:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK: %[[CST_1_I64:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[CST_0_I64:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK: %[[CST_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK: %[[CST_MAX_DELTA:.+]] = arith.constant dense<1.000000e+03> : tensor<f64>
// CHECK: %[[CST_EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>

// CHECK: %[[INIT_TRACE:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK: %[[RNG_SPLIT1:.+]]:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[RNG_SPLIT2:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[Q0:.+]] = enzyme.getFlattenedSamplesFromTrace %[[INIT_TRACE]] {selection = {{.*}}} : (!enzyme.Trace) -> tensor<1xf64>
// CHECK: %[[WEIGHT:.+]] = enzyme.getWeightFromTrace %[[INIT_TRACE]] : (!enzyme.Trace) -> tensor<f64>
// CHECK: %[[U0:.+]] = arith.negf %[[WEIGHT]] : tensor<f64>

// CHECK: %[[INIT_GRAD:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[CST_ONE]]) {
// CHECK: ^bb0(%[[AD_ARG:.+]]: tensor<1xf64>):
// CHECK: %[[UPD:.+]]:3 = func.call @test.update{{.*}}(%[[INIT_TRACE]], %[[AD_ARG]], %[[RNG_SPLIT2]]#1, %arg1, %arg2)
// CHECK: %[[NEG_W:.+]] = arith.negf %[[UPD]]#1 : tensor<f64>
// CHECK: enzyme.yield %[[NEG_W]], %[[UPD]]#2 : tensor<f64>, tensor<2xui64>
// CHECK: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}

// Warmup loop
// [q, grad, U, rng, stepSize, invMass, massMatrixSqrt, daState(5), welfordState(3), windowIdx]
// CHECK: %[[WARMUP:.+]]:16 = enzyme.for_loop(%[[CST_0_I64]] : tensor<i64>) to(%[[CST_NUM_WARMUP]] : tensor<i64>) step(%[[CST_1_I64]] : tensor<i64>) iter_args(%[[Q0]], %[[INIT_GRAD]]#2, %[[U0]], %[[RNG_SPLIT2]]#0, %[[CST_EPS]], %[[CST_ONES_1]], %[[CST_ONES_1]], %[[CST_ZERO]], %[[CST_ZERO]], %[[CST_ZERO]], %[[CST_0_I64]], %[[CST_PROX]], %[[CST_ZEROS_1]], %[[CST_ZEROS_1]], %[[CST_0_I64]], %[[CST_0_I64]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK: ^bb0(%[[ITER:.+]]: tensor<i64>, %[[Q_LOOP:.+]]: tensor<1xf64>, %[[GRAD_LOOP:.+]]: tensor<1xf64>, %[[U_LOOP:.+]]: tensor<f64>, %[[RNG_LOOP:.+]]: tensor<2xui64>, %[[STEP_LOOP:.+]]: tensor<f64>, %[[INVMASS_LOOP:.+]]: tensor<1xf64>, %[[MASSSQRT_LOOP:.+]]: tensor<1xf64>, %[[DA_LOG:.+]]: tensor<f64>, %[[DA_LOGAVG:.+]]: tensor<f64>, %[[DA_GRADAVG:.+]]: tensor<f64>, %[[DA_COUNT:.+]]: tensor<i64>, %[[DA_PROX:.+]]: tensor<f64>, %[[WELF_MEAN:.+]]: tensor<1xf64>, %[[WELF_M2:.+]]: tensor<1xf64>, %[[WELF_N:.+]]: tensor<i64>, %[[WIN_IDX:.+]]: tensor<i64>):

// Momentum sampling
// CHECK: %[[MOM_SPLIT:.+]]:3 = enzyme.randomSplit %[[RNG_LOOP]] : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[MOM_SPLIT2:.+]] = enzyme.randomSplit %[[MOM_SPLIT]]#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK: %[[MOM_RNG:.+]], %[[MOM_SAMP:.+]] = enzyme.random %[[MOM_SPLIT2]], %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

// CHECK: %[[P_SCALED:.+]] = arith.mulf %[[MASSSQRT_LOOP]], %[[MOM_SAMP]] : tensor<1xf64>

// Kinetic energy: K = 0.5 * p^T * M^-1 * p
// CHECK: %[[INV_P:.+]] = arith.mulf %[[INVMASS_LOOP]], %[[P_SCALED]] : tensor<1xf64>
// CHECK: %[[P_DOT:.+]] = enzyme.dot %[[P_SCALED]], %[[INV_P]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[K0:.+]] = arith.mulf %[[P_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK: %[[H0:.+]] = arith.addf %[[U_LOOP]], %[[K0]] : tensor<f64>

// NUTS tree building
// CHECK: %[[TREE:.+]]:18 = enzyme.while_loop({{.*}}) -> {{.*}} condition {
// CHECK: enzyme.yield
// CHECK: } body {
// CHECK: enzyme.yield
// CHECK: }

// Dual Averaging
// accept_prob = sum_accept_probs / max(num_proposals, 1)
// CHECK: %[[MAX_PROPS:.+]] = arith.maxsi %[[TREE]]#15, %[[CST_1_I64]] : tensor<i64>
// CHECK: %[[PROPS_F:.+]] = arith.sitofp %[[MAX_PROPS]] : tensor<i64> to tensor<f64>
// CHECK: %[[MEAN_ACCEPT:.+]] = arith.divf %[[TREE]]#14, %[[PROPS_F]] : tensor<f64>

// step_count = DA_COUNT + 1
// CHECK: %[[NEW_COUNT:.+]] = arith.addi %[[DA_COUNT]], %[[CST_1_I64]] : tensor<i64>
// CHECK: %[[COUNT_F:.+]] = arith.sitofp %[[NEW_COUNT]] : tensor<i64> to tensor<f64>

// g = target_accept_prob - accept_prob
// CHECK: %[[G:.+]] = arith.subf %[[CST_TARGET]], %[[MEAN_ACCEPT]] : tensor<f64>

// gradient_avg update: (1 - 1/(t+t0)) * g_avg + g/(t+t0)
// CHECK: %[[T_PLUS_T0:.+]] = arith.addf %[[COUNT_F]], %[[CST_T0]] : tensor<f64>
// CHECK: %[[INV_T:.+]] = arith.divf %[[CST_ONE]], %[[T_PLUS_T0]] : tensor<f64>
// CHECK: %[[ONE_MINUS:.+]] = arith.subf %[[CST_ONE]], %[[INV_T]] : tensor<f64>
// CHECK: %[[TERM1:.+]] = arith.mulf %[[ONE_MINUS]], %[[DA_GRADAVG]] : tensor<f64>
// CHECK: %[[TERM2:.+]] = arith.mulf %[[INV_T]], %[[G]] : tensor<f64>
// CHECK: %[[NEW_GRADAVG:.+]] = arith.addf %[[TERM1]], %[[TERM2]] : tensor<f64>

// log_step_size update: x_t = prox_center - sqrt(t)/gamma * g_avg
// CHECK: %[[SQRT_T:.+]] = math.sqrt %[[COUNT_F]] : tensor<f64>
// CHECK: %[[COEFF:.+]] = arith.divf %[[SQRT_T]], %[[CST_GAMMA]] : tensor<f64>
// CHECK: %[[SCALED_G:.+]] = arith.mulf %[[COEFF]], %[[NEW_GRADAVG]] : tensor<f64>
// CHECK: %[[NEW_LOG_SS:.+]] = arith.subf %[[DA_PROX]], %[[SCALED_G]] : tensor<f64>

// log_step_size_avg update: (1 - t^-kappa) * avg + t^-kappa * x_t
// CHECK: %[[T_POW:.+]] = math.powf %[[COUNT_F]], %[[CST_NEG_KAPPA]] : tensor<f64>
// CHECK: %[[ONE_MINUS_POW:.+]] = arith.subf %[[CST_ONE]], %[[T_POW]] : tensor<f64>
// CHECK: %[[AVG_TERM1:.+]] = arith.mulf %[[ONE_MINUS_POW]], %[[DA_LOGAVG]] : tensor<f64>
// CHECK: %[[AVG_TERM2:.+]] = arith.mulf %[[T_POW]], %[[NEW_LOG_SS]] : tensor<f64>
// CHECK: %[[NEW_LOGAVG:.+]] = arith.addf %[[AVG_TERM1]], %[[AVG_TERM2]] : tensor<f64>

// Get step size: exp(log_step_size) or exp(log_step_size_avg) at last iter
// CHECK: %[[EXP_LOG:.+]] = math.exp %[[NEW_LOG_SS]] : tensor<f64>
// CHECK: %[[EXP_AVG:.+]] = math.exp %[[NEW_LOGAVG]] : tensor<f64>
// CHECK: %[[IS_LAST:.+]] = arith.cmpi eq, %[[ITER]], %[[CST_LAST_ITER]] : tensor<i64>
// CHECK: %[[ADAPTED_SS:.+]] = enzyme.select %[[IS_LAST]], %[[EXP_AVG]], %[[EXP_LOG]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>

// Clamp step size to valid range
// CHECK: %[[SS_CLAMPED1:.+]] = arith.maximumf %[[ADAPTED_SS]], %[[CST_FMIN]] : tensor<f64>
// CHECK: %[[SS_CLAMPED2:.+]] = arith.minimumf %[[SS_CLAMPED1]], %[[CST_FMAX]] : tensor<f64>

// Window index check (is middle window: 0 < windowIdx < numWindows-1)
// CHECK: %[[WIN_GT_0:.+]] = arith.cmpi sgt, %[[WIN_IDX]], %[[CST_0_I64]] : tensor<i64>
// CHECK: %[[WIN_LT_LAST:.+]] = arith.cmpi slt, %[[WIN_IDX]], %[[CST_0_I64]] : tensor<i64>
// CHECK: %[[IS_MIDDLE:.+]] = arith.andi %[[WIN_GT_0]], %[[WIN_LT_LAST]] : tensor<i1>

// Welford Covariance Update
// n_new = n + 1
// CHECK: %[[WELF_N_NEW:.+]] = arith.addi %[[WELF_N]], %[[CST_1_I64]] : tensor<i64>
// CHECK: %[[WELF_N_F:.+]] = arith.sitofp %[[WELF_N_NEW]] : tensor<i64> to tensor<f64>
// CHECK: %[[WELF_N_BCAST:.+]] = "enzyme.broadcast"(%[[WELF_N_F]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>

// delta_pre = sample - mean
// CHECK: %[[DELTA_PRE:.+]] = arith.subf %[[TREE]]#6, %[[WELF_MEAN]] : tensor<1xf64>

// mean_new = mean + delta_pre / n
// CHECK: %[[DELTA_SCALED:.+]] = arith.divf %[[DELTA_PRE]], %[[WELF_N_BCAST]] : tensor<1xf64>
// CHECK: %[[MEAN_NEW:.+]] = arith.addf %[[WELF_MEAN]], %[[DELTA_SCALED]] : tensor<1xf64>

// delta_post = sample - mean_new
// CHECK: %[[DELTA_POST:.+]] = arith.subf %[[TREE]]#6, %[[MEAN_NEW]] : tensor<1xf64>

// m2_new = m2 + delta_pre * delta_post
// CHECK: %[[M2_UPDATE:.+]] = arith.mulf %[[DELTA_PRE]], %[[DELTA_POST]] : tensor<1xf64>
// CHECK: %[[M2_NEW:.+]] = arith.addf %[[WELF_M2]], %[[M2_UPDATE]] : tensor<1xf64>

// Conditional Welford update
// CHECK: %[[WELF_MEAN_SEL:.+]] = enzyme.select %[[IS_MIDDLE]], %[[MEAN_NEW]], %[[WELF_MEAN]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK: %[[WELF_M2_SEL:.+]] = enzyme.select %[[IS_MIDDLE]], %[[M2_NEW]], %[[WELF_M2]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK: %[[WELF_N_SEL:.+]] = enzyme.select %[[IS_MIDDLE]], %[[WELF_N_NEW]], %[[WELF_N]] : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>

// Window boundary detection and finalization
// CHECK: %[[WIN_END_IF:.+]]:10 = enzyme.if(%[[AT_MID_WIN_END:.+]]) ({

// True branch: finalize covariance and reinitialize for next window
// Compute sample variance: cov = m2 / (n - 1)
// CHECK:   %[[N_MINUS_1:.+]] = arith.subi %[[WELF_N_SEL]], %[[CST_1_I64]] : tensor<i64>
// CHECK:   %[[N_MINUS_1_F:.+]] = arith.sitofp %[[N_MINUS_1]] : tensor<i64> to tensor<f64>
// CHECK:   %[[N_MINUS_1_BCAST:.+]] = "enzyme.broadcast"(%[[N_MINUS_1_F]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:   %[[SAMPLE_VAR:.+]] = arith.divf %[[WELF_M2_SEL]], %[[N_MINUS_1_BCAST]] : tensor<1xf64>

// Regularization: scaled_cov = (n / (n + 5)) * cov + shrinkage
// CHECK:   %[[N_F:.+]] = arith.sitofp %[[WELF_N_SEL]] : tensor<i64> to tensor<f64>
// CHECK:   %[[N_PLUS_5:.+]] = arith.addf %[[N_F]], %[[CST_5:.+]] : tensor<f64>
// CHECK:   %[[SCALE:.+]] = arith.divf %[[N_F]], %[[N_PLUS_5]] : tensor<f64>
// CHECK:   %[[SCALE_BCAST:.+]] = "enzyme.broadcast"(%[[SCALE]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:   %[[SCALED_COV:.+]] = arith.mulf %[[SCALE_BCAST]], %[[SAMPLE_VAR]] : tensor<1xf64>
// CHECK:   %[[SHRINKAGE:.+]] = arith.divf %[[CST_SHRINK:.+]], %[[N_PLUS_5]] : tensor<f64>
// CHECK:   %[[SHRINKAGE_BCAST:.+]] = "enzyme.broadcast"(%[[SHRINKAGE]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:   %[[NEW_INV_MASS:.+]] = arith.addf %[[SCALED_COV]], %[[SHRINKAGE_BCAST]] : tensor<1xf64>

// Compute mass_matrix_sqrt_inv = 1.0 / sqrt(inv_mass)
// CHECK:   %[[NEW_MASS_SQRT:.+]] = math.sqrt %[[NEW_INV_MASS]] : tensor<1xf64>
// CHECK:   %[[NEW_MASS_SQRT_INV:.+]] = arith.divf %[[CST_ONES:.+]], %[[NEW_MASS_SQRT]] : tensor<1xf64>

// Reinitialize dual averaging with current adapted step size
// CHECK:   %[[NEW_LOG_SS_INIT:.+]] = math.log %[[SS_CLAMPED2]] : tensor<f64>
// CHECK:   %[[NEW_PROX:.+]] = arith.addf %[[NEW_LOG_SS_INIT]], %[[CST_LOG10]] : tensor<f64>

// CHECK:   enzyme.yield %[[NEW_INV_MASS]], %[[NEW_MASS_SQRT_INV]], %[[CST_ZEROS:.+]], %[[CST_ZEROS]], %[[CST_0_I64]], %[[CST_ZERO]], %[[CST_ZERO]], %[[CST_ZERO]], %[[CST_0_I64]], %[[NEW_PROX]] : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>
// CHECK: }, {

// False branch: pass through current values unchanged
// CHECK:   enzyme.yield %{{.+}}, %{{.+}}, %[[WELF_MEAN_SEL]], %[[WELF_M2_SEL]], %[[WELF_N_SEL]], %[[NEW_LOG_SS]], %[[NEW_LOGAVG]], %[[NEW_GRADAVG]], %[[NEW_COUNT:.+]], %[[DA_PROX]] : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>
// CHECK: }) : (tensor<i1>) -> (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>)

// Final yield of warmup loop
// Yield: (q, grad, U, rng, step_size, inv_mass, mass_sqrt, da_log, da_logavg, da_gradavg, da_count, da_prox, welf_mean, welf_m2, welf_n, win_idx)
// CHECK: enzyme.yield %[[TREE]]#6, %[[TREE]]#7, %[[TREE]]#8, %[[TREE_RNG:.+]]#0, %[[SS_CLAMPED2]], %[[WIN_END_IF]]#0, %[[WIN_END_IF]]#1, %[[WIN_END_IF]]#5, %[[WIN_END_IF]]#6, %[[WIN_END_IF]]#7, %[[WIN_END_IF]]#8, %[[WIN_END_IF]]#9, %[[WIN_END_IF]]#2, %[[WIN_END_IF]]#3, %[[WIN_END_IF]]#4, %[[NEW_WIN_IDX:.+]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK: }

// Post-warmup: sampling loop uses adapted step size and mass matrix
// CHECK: %[[POST_SPLIT:.+]]:3 = enzyme.randomSplit %[[WARMUP]]#3 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[POST_SPLIT2:.+]] = enzyme.randomSplit %[[POST_SPLIT]]#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK: %[[MOM_RNG:.+]], %[[MOM:.+]] = enzyme.random %[[POST_SPLIT2]], %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

// Apply mass matrix: scaled_mom = mass_sqrt * mom, then inv_mass * scaled_mom
// CHECK: %[[SCALED_MOM:.+]] = arith.mulf %[[WARMUP]]#6, %[[MOM]] : tensor<1xf64>
// CHECK: %[[V:.+]] = arith.mulf %[[WARMUP]]#5, %[[SCALED_MOM]] : tensor<1xf64>

// Kinetic energy: K = 0.5 * p^T * M^-1 * p
// CHECK: %[[P_DOT_V:.+]] = enzyme.dot %[[SCALED_MOM]], %[[V]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[K0:.+]] = arith.mulf %[[P_DOT_V]], %[[CST_HALF]] : tensor<f64>
// CHECK: %[[H0:.+]] = arith.addf %[[WARMUP]]#2, %[[K0]] : tensor<f64>

// Main NUTS tree building loop with adapted parameters
// CHECK: %[[NUTS_LOOP:.+]]:18 = enzyme.while_loop(%[[WARMUP]]#0, %[[SCALED_MOM]], %[[WARMUP]]#1, %[[WARMUP]]#0, %[[SCALED_MOM]], %[[WARMUP]]#1, %[[WARMUP]]#0, %[[WARMUP]]#1, %[[WARMUP]]#2, %[[H0]], %[[CST_0_I64]], %[[CST_ZERO]], %[[CST_FALSE]], %[[CST_FALSE]], %[[CST_ZERO]], %[[CST_0_I64]], %[[SCALED_MOM]], %[[POST_SPLIT]]#2 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>) -> tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64> condition {
// CHECK: ^bb0(%[[NL_ARG0:.+]]: tensor<1xf64>, %[[NL_ARG1:.+]]: tensor<1xf64>, %[[NL_ARG2:.+]]: tensor<1xf64>, %[[NL_ARG3:.+]]: tensor<1xf64>, %[[NL_ARG4:.+]]: tensor<1xf64>, %[[NL_ARG5:.+]]: tensor<1xf64>, %[[NL_ARG6:.+]]: tensor<1xf64>, %[[NL_ARG7:.+]]: tensor<1xf64>, %[[NL_ARG8:.+]]: tensor<f64>, %[[NL_ARG9:.+]]: tensor<f64>, %[[NL_ARG10:.+]]: tensor<i64>, %[[NL_ARG11:.+]]: tensor<f64>, %[[NL_ARG12:.+]]: tensor<i1>, %[[NL_ARG13:.+]]: tensor<i1>, %[[NL_ARG14:.+]]: tensor<f64>, %[[NL_ARG15:.+]]: tensor<i64>, %[[NL_ARG16:.+]]: tensor<1xf64>, %[[NL_ARG17:.+]]: tensor<2xui64>):
// CHECK:   %[[DEPTH_OK:.+]] = arith.cmpi slt, %[[NL_ARG10]], %[[CST_MAX_DEPTH:.+]] : tensor<i64>
// CHECK:   %[[NOT_TURN:.+]] = arith.xori %[[NL_ARG12]], %[[CST_TRUE]] : tensor<i1>
// CHECK:   %[[NOT_DIV:.+]] = arith.xori %[[NL_ARG13]], %[[CST_TRUE]] : tensor<i1>
// CHECK:   %[[COND1:.+]] = arith.andi %[[DEPTH_OK]], %[[NOT_TURN]] : tensor<i1>
// CHECK:   %[[COND2:.+]] = arith.andi %[[COND1]], %[[NOT_DIV]] : tensor<i1>
// CHECK:   enzyme.yield %[[COND2]] : tensor<i1>
// CHECK: } body {

// CHECK-LABEL: func.func @warmup_step_only
// CHECK: %[[WARMUP_STEP:.+]]:13 = enzyme.for_loop({{.*}}) iter_args({{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64> {
// CHECK: ^bb0(%[[ITER2:.+]]: tensor<i64>, %[[Q2:.+]]: tensor<1xf64>, %[[GRAD2:.+]]: tensor<1xf64>, %[[U2:.+]]: tensor<f64>, %[[RNG2:.+]]: tensor<2xui64>, %[[STEP2:.+]]: tensor<f64>, %[[INV2:.+]]: tensor<1xf64>, %[[SQRT2:.+]]: tensor<1xf64>, %[[DA_LOG2:.+]]: tensor<f64>, %[[DA_LOGAVG2:.+]]: tensor<f64>, %[[DA_GRADAVG2:.+]]: tensor<f64>, %[[DA_COUNT2:.+]]: tensor<i64>, %[[DA_PROX2:.+]]: tensor<f64>, %[[WIN2:.+]]: tensor<i64>):
// CHECK: enzyme.while_loop

// Dual averaging update
// CHECK: arith.addi {{.*}}, {{.*}} : tensor<i64>
// CHECK: arith.sitofp
// CHECK: arith.subf {{.*}}, {{.*}} : tensor<f64>
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: math.powf

// window boundary handling
// CHECK: enzyme.if

// CHECK: enzyme.yield
// CHECK: }

// CHECK-LABEL: func.func @warmup_mass_only
// CHECK: %[[WARMUP_MASS:.+]]:16 = enzyme.for_loop({{.*}}) iter_args({{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK: ^bb0({{.*}}: tensor<i64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<f64>, {{.*}}: tensor<2xui64>, {{.*}}: tensor<f64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<f64>, {{.*}}: tensor<f64>, {{.*}}: tensor<f64>, {{.*}}: tensor<i64>, {{.*}}: tensor<f64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<i64>, {{.*}}: tensor<i64>):

// NUTS tree building
// CHECK: enzyme.while_loop

// Welford update present (n+1, delta_pre, mean update, delta_post, m2 update)
// CHECK: arith.addi {{.*}}, {{.*}} : tensor<i64>
// CHECK: arith.sitofp
// CHECK: arith.subf {{.*}}, {{.*}} : tensor<1xf64>
// CHECK: arith.divf {{.*}}, {{.*}} : tensor<1xf64>
// CHECK: arith.addf {{.*}}, {{.*}} : tensor<1xf64>
// CHECK: arith.subf {{.*}}, {{.*}} : tensor<1xf64>
// CHECK: arith.mulf {{.*}}, {{.*}} : tensor<1xf64>

// Window boundary handling
// CHECK: enzyme.if

// CHECK: enzyme.yield
// CHECK: }

// CHECK-LABEL: func.func @warmup_none
// 13 iter_args: no Welford state, DA state still present but unused
// CHECK: %[[WARMUP_NONE:.+]]:13 = enzyme.for_loop({{.*}}) iter_args({{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64> {
// CHECK: ^bb0({{.*}}: tensor<i64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<f64>, {{.*}}: tensor<2xui64>, {{.*}}: tensor<f64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<1xf64>, {{.*}}: tensor<f64>, {{.*}}: tensor<f64>, {{.*}}: tensor<f64>, {{.*}}: tensor<i64>, {{.*}}: tensor<f64>, {{.*}}: tensor<i64>):

// NUTS tree building
// CHECK: enzyme.while_loop

// Window handling
// CHECK: enzyme.if

// CHECK: enzyme.yield
// CHECK: }
