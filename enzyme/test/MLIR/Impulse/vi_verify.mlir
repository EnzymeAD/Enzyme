// RUN: %eopt --verify-diagnostics --split-input-file %s

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_a(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_hmc_and_vi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{exactly one of hmc_config, nuts_config, or vi_config must be specified}}
  %res:3 = impulse.infer @model_a(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    { hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
      name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_b(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_no_mode(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{exactly one of hmc_config, nuts_config, or vi_config must be specified}}
  %res:3 = impulse.infer @model_b(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    { name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_c(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_nuts_and_vi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{exactly one of hmc_config, nuts_config, or vi_config must be specified}}
  %res:3 = impulse.infer @model_c(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    { nuts_config = #impulse.nuts_config<>,
      vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
      name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_d(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_vi_multichain(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{`num_chains` > 1 requires a sampling config (hmc_config or nuts_config)}}
  %res:3 = impulse.infer @model_d(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    { vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
      num_chains = 2,
      name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}
