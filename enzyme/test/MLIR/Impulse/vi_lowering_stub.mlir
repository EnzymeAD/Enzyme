// RUN: %eopt --expand-impulse --verify-diagnostics --split-input-file %s

// No state bundle, so CUSTOM reaches the lowering guard, not the verifier.

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @svi_custom(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<5x1xf64>, tensor<5xf64>, tensor<2xui64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{guide_kind = CUSTOM is not yet supported}}
  %res:4 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    <{ vi_config = #impulse.vi_config<guide_kind = CUSTOM, optimizer_kind = SGD>,
      name = "svi_custom", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 5 }>
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<5x1xf64>, tensor<5xf64>, tensor<5xf64>, tensor<2xui64>)
  return %res#0, %res#1, %res#3 : tensor<5x1xf64>, tensor<5xf64>, tensor<2xui64>
}
