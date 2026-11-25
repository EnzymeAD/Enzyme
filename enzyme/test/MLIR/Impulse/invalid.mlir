// RUN: %eopt %s --split-input-file --verify-diagnostics

func.func @if_true_branch_type_mismatch(%pred : tensor<i1>, %a : tensor<2xf64>, %b : tensor<1x2xf64>) -> tensor<1x2xf64> {
  // expected-error @below {{'impulse.if' op true branch yields ('tensor<2xf64>') but results are ('tensor<1x2xf64>')}}
  %r = impulse.if(%pred) ({
    impulse.yield %a : tensor<2xf64>
  }, {
    impulse.yield %b : tensor<1x2xf64>
  }) : (tensor<i1>) -> tensor<1x2xf64>
  return %r : tensor<1x2xf64>
}

// -----

func.func @if_false_branch_type_mismatch(%pred : tensor<i1>, %a : tensor<2xf64>, %b : tensor<1x2xf64>) -> tensor<2xf64> {
  // expected-error @below {{'impulse.if' op false branch yields ('tensor<1x2xf64>') but results are ('tensor<2xf64>')}}
  %r = impulse.if(%pred) ({
    impulse.yield %a : tensor<2xf64>
  }, {
    impulse.yield %b : tensor<1x2xf64>
  }) : (tensor<i1>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}

// -----

func.func @if_yield_count_mismatch(%pred : tensor<i1>, %a : tensor<f64>) -> tensor<f64> {
  // expected-error @below {{'impulse.if' op true branch yields ('tensor<f64>', 'tensor<f64>') but results are ('tensor<f64>')}}
  %r = impulse.if(%pred) ({
    impulse.yield %a, %a : tensor<f64>, tensor<f64>
  }, {
    impulse.yield %a : tensor<f64>
  }) : (tensor<i1>) -> tensor<f64>
  return %r : tensor<f64>
}

// -----

func.func @simulate_missing_fn(%seed : tensor<2xui64>) -> tensor<1x1xf64> {
  // expected-error @below {{'impulse.simulate' op `fn` references '@missing', which is not a func.func}}
  %res:3 = impulse.simulate @missing(%seed) <{ all_addresses = [[#impulse.symbol<1>]] }> : (tensor<2xui64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>)
  return %res#0 : tensor<1x1xf64>
}

// -----

func.func private @sampler(%seed : tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)

func.func @sample_missing_logpdf(%seed : tensor<2xui64>) -> tensor<f64> {
  // expected-error @below {{'impulse.sample' op `logpdf_fn` references '@missing', which is not a func.func}}
  %r:2 = impulse.sample @sampler(%seed) <{ logpdf_fn = @missing, symbol = #impulse.symbol<1> }> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
  return %r#1 : tensor<f64>
}

// -----

func.func private @model(tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)

func.func @infer_region_without_config(%rng : tensor<2xui64>, %trace : tensor<1x1xf64>) {
  // expected-error @below {{'impulse.infer_region' op exactly one of hmc_config, nuts_config, vi_config, mh_config, or gibbs_config must be specified}}
  %res:4 = impulse.infer_region @model(%rng) given %trace sampling {
  } logpdf {
  } <{selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func @infer_region_missing_fn(%rng : tensor<2xui64>, %trace : tensor<1x1xf64>, %step : tensor<f64>) {
  // expected-error @below {{'impulse.infer_region' op `fn` references '@missing', which is not a func.func}}
  %res:4 = impulse.infer_region @missing(%rng) given %trace init_step_size = %step sampling {
  } logpdf {
  } <{hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func @select_type_mismatch(%cond : tensor<i1>, %a : tensor<2xf64>, %b : tensor<3xf64>) -> tensor<2xf64> {
  // expected-error @below {{'impulse.select' op failed to verify that all of {true_value, false_value, result} have same type}}
  %r = impulse.select %cond, %a, %b : (tensor<i1>, tensor<2xf64>, tensor<3xf64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}

// -----

func.func @dist_missing_parameter(%rng : tensor<2xui64>, %loc : tensor<f64>) -> tensor<f64> {
  // expected-error @below {{'impulse.sample' op `dist` reads 2 parameters, but `inputs` holds 1 after the RNG state}}
  %x:2 = impulse.sample(%rng, %loc) <{dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>, symbol = #impulse.symbol<1>}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %x#1 : tensor<f64>
}

// -----

func.func private @model(tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

func.func @vi_without_state(%rng : tensor<2xui64>, %mean : tensor<f64>, %trace : tensor<1x1xf64>, %lr : tensor<f64>) {
  // expected-error @below {{'impulse.infer' op vi_config requires the VI state in `state_out` and `state_schema`}}
  %res:4 = impulse.infer @model(%rng, %mean) given %trace init_step_size = %lr <{vi_config = #impulse.vi_config<guide_kind = AUTO_DELTA, optimizer_kind = SGD>, selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}
