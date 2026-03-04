// RUN: %eopt --inline-mcmc-regions --outline-mcmc-regions %s | FileCheck %s

module {
  // Scalar sampler: returns tensor<1xf64>
  func.func private @scalar_sampler(%rng : tensor<2xui64>, %scale : tensor<1xf64>)
      -> (tensor<2xui64>, tensor<1xf64>) {
    return %rng, %scale : tensor<2xui64>, tensor<1xf64>
  }

  func.func private @scalar_logpdf(%x : tensor<1xf64>, %scale : tensor<1xf64>)
      -> tensor<f64> {
    %zero = arith.constant dense<0.0> : tensor<f64>
    return %zero : tensor<f64>
  }

  // Vector sampler: returns tensor<4xf64>
  func.func private @vector_sampler(%rng : tensor<2xui64>, %mean : tensor<4xf64>)
      -> (tensor<2xui64>, tensor<4xf64>) {
    return %rng, %mean : tensor<2xui64>, tensor<4xf64>
  }

  func.func private @vector_logpdf(%x : tensor<4xf64>, %mean : tensor<4xf64>)
      -> tensor<f64> {
    %zero = arith.constant dense<0.0> : tensor<f64>
    return %zero : tensor<f64>
  }

  // Model with two sample sites of different shapes.
  // First sample: scalar (tensor<1xf64>), second sample: vector (tensor<4xf64>).
  // The model returns the vector sample output.
  func.func @multishape_model(%rng : tensor<2xui64>, %input : tensor<4xf64>)
      -> (tensor<2xui64>, tensor<4xf64>) {
    %scale = arith.constant dense<[2.0]> : tensor<1xf64>

    // alpha ~ scalar_sampler(scale) -> tensor<1xf64>
    %alpha:2 = enzyme.sample @scalar_sampler(%rng, %scale) {
      logpdf = @scalar_logpdf,
      symbol = #enzyme.symbol<1>
    } : (tensor<2xui64>, tensor<1xf64>) -> (tensor<2xui64>, tensor<1xf64>)

    // y ~ vector_sampler(input) -> tensor<4xf64>
    %y:2 = enzyme.sample @vector_sampler(%alpha#0, %input) {
      logpdf = @vector_logpdf,
      symbol = #enzyme.symbol<2>
    } : (tensor<2xui64>, tensor<4xf64>) -> (tensor<2xui64>, tensor<4xf64>)

    return %y#0, %y#1 : tensor<2xui64>, tensor<4xf64>
  }

  // CHECK-LABEL: func.func @run_mcmc_multishape
  func.func @run_mcmc_multishape(
      %rng : tensor<2xui64>, %input : tensor<4xf64>, %trace : tensor<1x5xf64>)
      -> (tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>) {
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:3 = enzyme.mcmc @multishape_model(%rng, %input) given %trace
        step_size = %step_size {
      selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      all_addresses = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]],
      hmc_config = #enzyme.hmc_config<trajectory_length = 1.0>,
      num_samples = 1 : i64,
      thinning = 1 : i64,
      num_warmup = 0 : i64
    } : (tensor<2xui64>, tensor<4xf64>, tensor<1x5xf64>, tensor<f64>)
        -> (tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>)

    return %result#0, %result#1, %result#2 : tensor<1x5xf64>, tensor<1xi1>, tensor<2xui64>
  }
}

// After inline+outline roundtrip, the mcmc op should reference the outlined
// function, and that function must have return type matching the vector output.

// The mcmc call appears before the outlined function definition in output
// CHECK: enzyme.mcmc @[[MODEL:[a-zA-Z0-9_]+]]

// The outlined function must return (tensor<2xui64>, tensor<4xf64>),
// NOT (tensor<2xui64>, tensor<1xf64>) from the scalar sample's yield.
// CHECK: func.func private @[[MODEL]](
// CHECK-SAME: ) -> (tensor<2xui64>, tensor<4xf64>)
