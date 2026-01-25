// RUN: %eopt --probprog %s | FileCheck %s

// Test warmup adaptation loop structure for MCMC (HMC/NUTS).
// Tests the 4 combinations of adapt_step_size and adapt_mass_matrix boolean attributes.
// We focus on the warmup loop structure, not kernel internals.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // ============================================================================
  // Case 1: Both adaptations enabled (default)
  // adapt_step_size = true, adapt_mass_matrix = true
  // Warmup loop carries 16 iter_args: [q, grad, U, rng, stepSize, invMass, massMatrixSqrt, daState(5), welfordState(3), windowIdx]
  // ============================================================================
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

  // ============================================================================
  // Case 2: Step size adaptation only
  // adapt_step_size = true, adapt_mass_matrix = false
  // Warmup loop carries 13 iter_args: [q, grad, U, rng, stepSize, invMass, massMatrixSqrt, daState(5), windowIdx]
  // No Welford covariance state
  // ============================================================================
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

  // ============================================================================
  // Case 3: Mass matrix adaptation only
  // adapt_step_size = false, adapt_mass_matrix = true
  // Warmup loop carries 16 iter_args: includes Welford state but step size updates differently
  // ============================================================================
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

  // ============================================================================
  // Case 4: No adaptation
  // adapt_step_size = false, adapt_mass_matrix = false
  // Warmup loop carries 13 iter_args: no Welford state
  // ============================================================================
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

// ============================================================================
// CHECK patterns for Case 1: Both adaptations enabled
// ============================================================================
// CHECK-LABEL: func.func @warmup_both
// Warmup loop with 16 iter_args (includes Welford state for mass matrix adaptation)
// CHECK: %[[WARMUP_BOTH:.+]]:16 = enzyme.for_loop
// CHECK-SAME: iter_args(
// CHECK-SAME: ) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK: ^bb0(%[[ITER_IDX_1:.+]]: tensor<i64>, %[[Q_1:.+]]: tensor<1xf64>, %[[GRAD_1:.+]]: tensor<1xf64>, %[[U_1:.+]]: tensor<f64>, %[[RNG_1:.+]]: tensor<2xui64>, %[[STEP_1:.+]]: tensor<f64>, %[[INVMASS_1:.+]]: tensor<1xf64>, %[[MASSSQRT_1:.+]]: tensor<1xf64>, %[[DA_LOG_1:.+]]: tensor<f64>, %[[DA_LOGAVG_1:.+]]: tensor<f64>, %[[DA_GRADAVG_1:.+]]: tensor<f64>, %[[DA_COUNT_1:.+]]: tensor<i64>, %[[DA_PROX_1:.+]]: tensor<f64>, %[[WELF_MEAN_1:.+]]: tensor<1xf64>, %[[WELF_M2_1:.+]]: tensor<1xf64>, %[[WELF_N_1:.+]]: tensor<i64>, %[[WIN_IDX_1:.+]]: tensor<i64>):
// CHECK: enzyme.yield

// ============================================================================
// CHECK patterns for Case 2: Step size adaptation only
// ============================================================================
// CHECK-LABEL: func.func @warmup_step_only
// Warmup loop with 13 iter_args (no Welford state)
// CHECK: %[[WARMUP_STEP:.+]]:13 = enzyme.for_loop
// CHECK-SAME: iter_args(
// CHECK-SAME: ) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>
// CHECK: ^bb0(%[[ITER_IDX_2:.+]]: tensor<i64>, %[[Q_2:.+]]: tensor<1xf64>, %[[GRAD_2:.+]]: tensor<1xf64>, %[[U_2:.+]]: tensor<f64>, %[[RNG_2:.+]]: tensor<2xui64>, %[[STEP_2:.+]]: tensor<f64>, %[[INVMASS_2:.+]]: tensor<1xf64>, %[[MASSSQRT_2:.+]]: tensor<1xf64>, %[[DA_LOG_2:.+]]: tensor<f64>, %[[DA_LOGAVG_2:.+]]: tensor<f64>, %[[DA_GRADAVG_2:.+]]: tensor<f64>, %[[DA_COUNT_2:.+]]: tensor<i64>, %[[DA_PROX_2:.+]]: tensor<f64>, %[[WIN_IDX_2:.+]]: tensor<i64>):
// CHECK: enzyme.yield

// ============================================================================
// CHECK patterns for Case 3: Mass matrix adaptation only
// ============================================================================
// CHECK-LABEL: func.func @warmup_mass_only
// Warmup loop with 16 iter_args (includes Welford state)
// CHECK: %[[WARMUP_MASS:.+]]:16 = enzyme.for_loop
// CHECK-SAME: iter_args(
// CHECK-SAME: ) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK: ^bb0(%[[ITER_IDX_3:.+]]: tensor<i64>, %[[Q_3:.+]]: tensor<1xf64>, %[[GRAD_3:.+]]: tensor<1xf64>, %[[U_3:.+]]: tensor<f64>, %[[RNG_3:.+]]: tensor<2xui64>, %[[STEP_3:.+]]: tensor<f64>, %[[INVMASS_3:.+]]: tensor<1xf64>, %[[MASSSQRT_3:.+]]: tensor<1xf64>, %[[DA_LOG_3:.+]]: tensor<f64>, %[[DA_LOGAVG_3:.+]]: tensor<f64>, %[[DA_GRADAVG_3:.+]]: tensor<f64>, %[[DA_COUNT_3:.+]]: tensor<i64>, %[[DA_PROX_3:.+]]: tensor<f64>, %[[WELF_MEAN_3:.+]]: tensor<1xf64>, %[[WELF_M2_3:.+]]: tensor<1xf64>, %[[WELF_N_3:.+]]: tensor<i64>, %[[WIN_IDX_3:.+]]: tensor<i64>):
// CHECK: enzyme.yield

// ============================================================================
// CHECK patterns for Case 4: No adaptation
// ============================================================================
// CHECK-LABEL: func.func @warmup_none
// Warmup loop with 13 iter_args (no Welford state)
// CHECK: %[[WARMUP_NONE:.+]]:13 = enzyme.for_loop
// CHECK-SAME: iter_args(
// CHECK-SAME: ) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>
// CHECK: ^bb0(%[[ITER_IDX_4:.+]]: tensor<i64>, %[[Q_4:.+]]: tensor<1xf64>, %[[GRAD_4:.+]]: tensor<1xf64>, %[[U_4:.+]]: tensor<f64>, %[[RNG_4:.+]]: tensor<2xui64>, %[[STEP_4:.+]]: tensor<f64>, %[[INVMASS_4:.+]]: tensor<1xf64>, %[[MASSSQRT_4:.+]]: tensor<1xf64>, %[[DA_LOG_4:.+]]: tensor<f64>, %[[DA_LOGAVG_4:.+]]: tensor<f64>, %[[DA_GRADAVG_4:.+]]: tensor<f64>, %[[DA_COUNT_4:.+]]: tensor<i64>, %[[DA_PROX_4:.+]]: tensor<f64>, %[[WIN_IDX_4:.+]]: tensor<i64>):
// CHECK: enzyme.yield
