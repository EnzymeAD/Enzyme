// RUN: %eopt --inline-mcmc-regions --sample-invariant-code-motion %s | FileCheck %s --check-prefix=SICM

// ============================================================================
// Test 1: Simple hoisting - operation that doesn't depend on sample should be hoisted
// ============================================================================

// SICM-LABEL: func.func @test_simple_hoisting
// The constant and multiplication that only depends on prior_mean should be hoisted
// SICM: %[[CONST:.*]] = arith.constant dense<2.000000e+00> : tensor<f64>
// SICM: %[[TWICE:.*]] = arith.mulf %{{.*}}, %[[CONST]] : tensor<f64>
// SICM: enzyme.mcmc_region

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_simple_hoisting(%rng : tensor<2xui64>, %prior_mean : tensor<f64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_simple_hoist(%rng, %prior_mean) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model with a sample-invariant operation (doubling prior_mean)
  func.func @model_simple_hoist(%rng : tensor<2xui64>, %prior_mean : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    // This operation only depends on prior_mean (an input), so it should be hoisted
    %two = arith.constant dense<2.0> : tensor<f64>
    %twice_mean = arith.mulf %prior_mean, %two : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %twice_mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
  }
}

// ============================================================================
// Test 2: No hoisting of sample-dependent operations
// ============================================================================

// SICM-LABEL: func.func @test_no_hoist_sample_dep
// The multiplication depends on sample result, so it should stay inside mcmc_region
// Verify sample-dependent mulf does NOT appear before mcmc_region
// SICM-NOT: arith.mulf
// SICM: enzyme.mcmc_region
// SICM: enzyme.sample_region
// SICM: arith.mulf
// SICM: enzyme.yield

module @no_hoist_module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_no_hoist_sample_dep(%rng : tensor<2xui64>, %prior_mean : tensor<f64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_sample_dep(%rng, %prior_mean) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model where operations depend on sample results - should NOT be hoisted
  func.func @model_sample_dep(%rng : tensor<2xui64>, %prior_mean : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %prior_mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // This depends on sample result, so it should NOT be hoisted
    %two = arith.constant dense<2.0> : tensor<f64>
    %doubled = arith.mulf %x#1, %two : tensor<f64>

    return %x#0, %doubled : tensor<2xui64>, tensor<f64>
  }
}

// ============================================================================
// Test 3: Multiple operations - only invariant ones should be hoisted
// ============================================================================

// SICM-LABEL: func.func @test_mixed_hoisting
// Operations on prior_mean should be hoisted, operations on sample should not
// First the hoisted operations (multiply prior_mean by 2, then add offset)
// SICM: arith.mulf %{{.*}}
// SICM: arith.addf
// Verify no additional mulf (sample-dependent) appears before mcmc_region
// SICM-NOT: arith.mulf
// SICM: enzyme.mcmc_region

module @mixed_module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_mixed_hoisting(%rng : tensor<2xui64>, %prior_mean : tensor<f64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_mixed(%rng, %prior_mean) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_mixed(%rng : tensor<2xui64>, %prior_mean : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>

    // These operations only depend on prior_mean, should be hoisted
    %two = arith.constant dense<2.0> : tensor<f64>
    %twice_mean = arith.mulf %prior_mean, %two : tensor<f64>
    %offset = arith.constant dense<0.5> : tensor<f64>
    %adjusted_mean = arith.addf %twice_mean, %offset : tensor<f64>

    %x:2 = enzyme.sample @normal(%rng, %adjusted_mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // This depends on sample, should NOT be hoisted
    %result = arith.mulf %x#1, %two : tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}

// ============================================================================
// Test 4: Cholesky hoisting - cholesky on sample-invariant matrix should be hoisted
// ============================================================================

// SICM-LABEL: func.func @test_cholesky_hoisting
// The cholesky of base_cov should be hoisted since base_cov is an input
// SICM: %[[CHOL:.*]] = enzyme.cholesky %{{.*}}
// SICM: enzyme.mcmc_region
// No cholesky inside the mcmc_region
// SICM-NOT: enzyme.cholesky
// SICM: return

module @cholesky_module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_cholesky_hoisting(%rng : tensor<2xui64>, %base_cov : tensor<2x2xf64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_cholesky(%rng, %base_cov) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<1>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<2x2xf64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // Model where cholesky only depends on input base_cov - should be hoisted
  func.func @model_cholesky(%rng : tensor<2xui64>, %base_cov : tensor<2x2xf64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %mean = arith.constant dense<0.0> : tensor<f64>

    // Cholesky on sample-invariant input - should be hoisted
    %chol = enzyme.cholesky %base_cov : (tensor<2x2xf64>) -> tensor<2x2xf64>

    // Sample a value
    %x:2 = enzyme.sample @normal(%rng, %mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Use cholesky result with sample
    %zero_idx = arith.constant 0 : index
    %diag_elem = tensor.extract %chol[%zero_idx, %zero_idx] : tensor<2x2xf64>
    %diag_tensor = tensor.from_elements %diag_elem : tensor<f64>
    %result = arith.mulf %x#1, %diag_tensor : tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}

// ============================================================================
// Test 5: Hierarchical model - multiple sample sites with mixed dependence
// ============================================================================
//
// Hierarchical model structure:
//   twice_mean = prior_mean * 2.0          <- input-only, HOISTABLE
//   mu ~ Normal(twice_mean, 1.0)           <- sample site 1
//   adjusted_std = base_std * 0.5          <- input-only (between samples), HOISTABLE
//   shifted_mu = mu + prior_mean           <- sample-dependent, NOT hoistable
//   x ~ Normal(shifted_mu, adjusted_std)   <- sample site 2
//   result = x * 2.0                       <- sample-dependent, NOT hoistable
//
// Key property: adjusted_std is between two sample sites in program order,
// but only depends on the input base_std - SICM should hoist it.

// SICM-LABEL: func.func @test_hierarchical
// Hoisted constants (order may vary)
// SICM-DAG: %[[HALF:.*]] = arith.constant dense<5.000000e-01> : tensor<f64>
// SICM-DAG: %[[TWO:.*]] = arith.constant dense<2.000000e+00> : tensor<f64>
// SICM-DAG: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<f64>
// Hoisted input-only multiplications (consecutive)
// SICM: %[[TWICE:.*]] = arith.mulf %{{.*}}, %[[TWO]] : tensor<f64>
// SICM-NEXT: %[[ADJSTD:.*]] = arith.mulf %{{.*}}, %[[HALF]] : tensor<f64>
// No sample-dependent ops before mcmc_region
// SICM-NOT: arith.addf
// SICM-NOT: arith.mulf
// SICM: enzyme.mcmc_region
// Sample site 1 uses hoisted twice_mean and one
// SICM: enzyme.sample_region(%{{.*}}, %[[TWICE]], %[[ONE]])
// After sample_region 1 closes: sample-dependent shifted_mu stays inside
// SICM: symbol = #enzyme.symbol<1>
// SICM-NEXT: %[[SHIFT:.*]] = arith.addf
// Sample site 2 uses sample-dependent shifted_mu and hoisted adjusted_std
// SICM-NEXT: %{{.*}}:2 = enzyme.sample_region(%{{.*}}, %[[SHIFT]], %[[ADJSTD]])
// After sample_region 2 closes: sample-dependent result stays inside
// SICM: symbol = #enzyme.symbol<2>
// SICM-NEXT: %{{.*}} = arith.mulf %{{.*}}, %[[TWO]] : tensor<f64>
// SICM-NEXT: enzyme.yield

module @hierarchical_module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %neg = arith.negf %x : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_hierarchical(%rng : tensor<2xui64>, %prior_mean : tensor<f64>,
                                %base_std : tensor<f64>, %trace : tensor<1x2xf64>)
      -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_hierarchical(%rng, %prior_mean, %base_std) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>)

    return %result#0, %result#1, %result#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  func.func @model_hierarchical(%rng : tensor<2xui64>, %prior_mean : tensor<f64>,
                                 %base_std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %two = arith.constant dense<2.0> : tensor<f64>
    %half = arith.constant dense<0.5> : tensor<f64>

    // Input-only: should be hoisted
    %twice_mean = arith.mulf %prior_mean, %two : tensor<f64>

    // Sample site 1: group-level mean
    %mu:2 = enzyme.sample @normal(%rng, %twice_mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Input-only between samples: should be hoisted
    %adjusted_std = arith.mulf %base_std, %half : tensor<f64>

    // Sample-dependent (uses mu): should NOT be hoisted
    %shifted_mu = arith.addf %mu#1, %prior_mean : tensor<f64>

    // Sample site 2: observation-level
    %x:2 = enzyme.sample @normal(%mu#0, %shifted_mu, %adjusted_std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<2>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Sample-dependent (uses x): should NOT be hoisted
    %result = arith.mulf %x#1, %two : tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
