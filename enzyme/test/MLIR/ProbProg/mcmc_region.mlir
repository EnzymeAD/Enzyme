// RUN: %eopt --inline-mcmc-regions %s | FileCheck %s --check-prefix=INLINED
// RUN: %eopt --inline-mcmc-regions --outline-mcmc-regions %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %eopt %s | FileCheck %s --check-prefix=UNCHANGED

// Test that inlining works
// INLINED: enzyme.mcmc_region
// INLINED: enzyme.sample_region
// INLINED: sampler
// INLINED: logpdf
// INLINED-NOT: enzyme.mcmc @test_model

// Test that roundtrip works (outline after inline)
// ROUNDTRIP: enzyme.mcmc @
// ROUNDTRIP-NOT: enzyme.mcmc_region

// Test that without passes, mcmc is unchanged
// UNCHANGED: enzyme.mcmc @test_model
// UNCHANGED-NOT: enzyme.mcmc_region

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %sample = arith.addf %mean, %std : tensor<f64>
    return %rng, %sample : tensor<2xui64>, tensor<f64>
  }

  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %std : tensor<f64>)
      -> tensor<f64> {
    %two = arith.constant dense<2.0> : tensor<f64>
    %diff = arith.subf %x, %mean : tensor<f64>
    %scaled = arith.divf %diff, %std : tensor<f64>
    %sq = arith.mulf %scaled, %scaled : tensor<f64>
    %half_sq = arith.divf %sq, %two : tensor<f64>
    %neg = arith.negf %half_sq : tensor<f64>
    return %neg : tensor<f64>
  }

  func.func @test_model(%rng : tensor<2xui64>, %prior_mean : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>) {
    %std = arith.constant dense<1.0> : tensor<f64>
    %x:2 = enzyme.sample @normal(%rng, %prior_mean, %std) {
      logpdf = @normal_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @run_mcmc(%rng : tensor<2xui64>, %prior_mean : tensor<f64>, %trace : tensor<1x1xf64>)
      -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc @test_model(%rng, %prior_mean) given %trace
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
}
