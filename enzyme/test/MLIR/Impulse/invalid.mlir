// RUN: %eopt %s --split-input-file --verify-diagnostics

// Verifier diagnostics of the Impulse ops and attributes.

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

// -----

module {
  func.func private @s(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %m : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // expected-error @+1 {{explicit support contradicts the support derived from its dist attribute}}
    %o:2 = impulse.sample @s(%r, %m, %g) <{
      symbol = #impulse.symbol<1>,
      support = #impulse.support<REAL>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<EXP>]>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

// Accepted: STD_NORMAL + AFFINE derives REAL.
module {
  func.func private @s(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %m : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @s(%r, %m, %g) <{
      symbol = #impulse.symbol<1>,
      support = #impulse.support<REAL>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

module {
  func.func private @s(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %a : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // expected-error @+1 {{a transform chain is only supported on the STD_NORMAL and STD_UNIFORM bases}}
    %o:2 = impulse.sample @s(%r, %a, %b) <{
      symbol = #impulse.symbol<1>,
      dist = #impulse.dist<GAMMA, [#impulse.xform<AFFINE>]>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

// Accepted: STD_UNIFORM + AFFINE derives INTERVAL.
module {
  func.func private @s(%r: tensor<2xui64>, %lo: tensor<f64>, %sc: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %lo : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %lo: tensor<f64>, %sc: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @s(%r, %lo, %sc) <{
      symbol = #impulse.symbol<1>,
      support = #impulse.support<INTERVAL>,
      dist = #impulse.dist<STD_UNIFORM, [#impulse.xform<AFFINE>]>
    }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

// Accepted: STD_NORMAL + SIGMOID derives UNIT_INTERVAL, STD_NORMAL + SOFTPLUS
// derives POSITIVE.
module {
  func.func @f(%r: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
    %p:2 = impulse.sample(%r) <{
      symbol = #impulse.symbol<1>,
      support = #impulse.support<UNIT_INTERVAL>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<SIGMOID>]>
    }> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
    %s:2 = impulse.sample(%p#0) <{
      symbol = #impulse.symbol<2>,
      support = #impulse.support<POSITIVE>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<SOFTPLUS>]>
    }> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_a(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_hmc_and_vi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{exactly one of hmc_config, nuts_config, vi_config, mh_config, or gibbs_config must be specified}}
  %res:4 = impulse.infer @model_a(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    <{ hmc_config = #impulse.hmc_config<trajectory_length = 1.0>,
      vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
      name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_b(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_no_mode(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{exactly one of hmc_config, nuts_config, vi_config, mh_config, or gibbs_config must be specified}}
  %res:4 = impulse.infer @model_b(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    <{ name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_c(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_nuts_and_vi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{exactly one of hmc_config, nuts_config, vi_config, mh_config, or gibbs_config must be specified}}
  %res:4 = impulse.infer @model_c(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    <{ nuts_config = #impulse.nuts_config<>,
      vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
      name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}

// -----

func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

func.func @model_d(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
}

func.func @bad_vi_multichain(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) {
  %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
  %lr = arith.constant dense<0.05> : tensor<f64>
  // expected-error @below {{`num_chains` > 1 requires a sampling config (hmc_config or nuts_config)}}
  %res:4 = impulse.infer @model_d(%rng, %mean, %stddev) given %init_trace
    init_step_size = %lr
    <{ vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
      num_chains = 2,
      name = "bad", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
    : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<2xui64>)
  return
}
