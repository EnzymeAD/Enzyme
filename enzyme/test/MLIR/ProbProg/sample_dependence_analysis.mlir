// RUN: %eopt --inline-mcmc-regions --test-print-sample-dependence %s | FileCheck %s

// ============================================================================
// Test: Sample dependence analysis classification
// ============================================================================

// CHECK: === SampleDependenceAnalysis for mcmc_region ===
// CHECK: Sample regions: 1

// Operations before sample - sample-invariant and hoistable
// CHECK: [INV] [HOIST] arith.mulf -> tensor<f64>

// Operations inside sampler region (nested in sample_region)
// CHECK: [INV] [HOIST] arith.addf -> tensor<f64>
// CHECK: [INV] [KEEP]  enzyme.yield

// Per-site logpdf bodies are merged into MCMCRegionOp's unified logpdf
// region by inline-mcmc-regions Phase 2, so arith.negf no longer appears here.

// The sample_region itself - sample-dependent
// CHECK: [DEP] [KEEP]  enzyme.sample_region -> tensor<2xui64>, tensor<f64>

// Operations after sample that depend on sample result
// CHECK: [DEP] [KEEP]  arith.mulf -> tensor<f64>

// Terminator - invariant but must stay
// CHECK: [INV] [KEEP]  enzyme.yield

// CHECK: === End SampleDependenceAnalysis ===

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

  func.func @test_basic(%rng : tensor<2xui64>, %prior_mean : tensor<f64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @model_basic(%rng, %prior_mean) given %trace
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

  func.func @model_basic(%rng : tensor<2xui64>, %prior_mean : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %two = arith.constant dense<2.0> : tensor<f64>

    // Sample-invariant: only depends on prior_mean (input)
    %twice_mean = arith.mulf %prior_mean, %two : tensor<f64>

    // Sample-dependent: sample_region itself
    %x:2 = enzyme.sample @normal(%rng, %twice_mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

    // Sample-dependent: depends on sample result
    %result = arith.mulf %x#1, %two : tensor<f64>

    return %x#0, %result : tensor<2xui64>, tensor<f64>
  }
}
