// RUN: %eopt --inline-infer-regions --sicm --expand-impulse --verify-diagnostics --split-input-file %s

// Diagnostics of the lowering on inputs it does not support.

func.func @uniform(%rng : tensor<2xui64>, %low : tensor<f64>, %scale : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  %x:2 = impulse.sample(%rng, %low, %scale) <{dist = #impulse.dist<STD_UNIFORM, [#impulse.xform<AFFINE>]>, symbol = #impulse.symbol<1>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
}

func.func @derived_support_without_bounds(%rng : tensor<2xui64>, %low : tensor<f64>, %scale : tensor<f64>, %trace : tensor<1x1xf64>) -> tensor<1x1xf64> {
  %step = arith.constant dense<0.1> : tensor<f64>
  // expected-error @below {{Impulse: the INTERVAL support of the site at [#impulse.symbol<1>] has no `lower_bound` and `upper_bound`, which unconstraining the site needs}}
  %res:4 = impulse.infer @uniform(%rng, %low, %scale) given %trace init_step_size = %step <{hmc_config = #impulse.hmc_config<trajectory_length = 1.0>, selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  return %res#0 : tensor<1x1xf64>
}

// -----

func.func private @sampler(tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(tensor<f64>) -> tensor<f64>

func.func @shifted(%rng : tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
  %x:2 = impulse.sample @sampler(%rng) <{logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, support = #impulse.support<GREATER_THAN>}> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
  return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
}

func.func @explicit_support_without_bound(%rng : tensor<2xui64>, %trace : tensor<1x1xf64>) -> tensor<1x1xf64> {
  %step = arith.constant dense<0.1> : tensor<f64>
  // expected-error @below {{Impulse: the GREATER_THAN support of the site at [#impulse.symbol<1>] has no `lower_bound`, which unconstraining the site needs}}
  %res:4 = impulse.infer @shifted(%rng) given %trace init_step_size = %step <{nuts_config = #impulse.nuts_config<max_tree_depth = 2>, selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]]}> : (tensor<2xui64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  return %res#0 : tensor<1x1xf64>
}

// -----

func.func private @sampler(tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
func.func private @logpdf(tensor<f64>) -> tensor<f64>

func.func @inner(%rng : tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
  %x:2 = impulse.sample @sampler(%rng) <{logpdf_fn = @logpdf, symbol = #impulse.symbol<2>}> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
  return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
}

func.func @outer(%rng : tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
  %x:2 = impulse.sample @inner(%rng) <{symbol = #impulse.symbol<1>}> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
  return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
}

func.func @nested_selection(%rng : tensor<2xui64>, %trace : tensor<1x1xf64>) -> tensor<1x1xf64> {
  %step = arith.constant dense<0.1> : tensor<f64>
  // expected-error @below {{Impulse: HMC and NUTS on a model take only top-level addresses in `selection`, not [#impulse.symbol<1>, #impulse.symbol<2>]}}
  %res:4 = impulse.infer @outer(%rng) given %trace init_step_size = %step <{nuts_config = #impulse.nuts_config<max_tree_depth = 2>, selection = [[#impulse.symbol<1>, #impulse.symbol<2>]], all_addresses = [[#impulse.symbol<1>, #impulse.symbol<2>]]}> : (tensor<2xui64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  return %res#0 : tensor<1x1xf64>
}

// -----

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %c : tensor<f64>, %pred : i1) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @normal(%rng, %c, %c) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name = "a" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = scf.if %pred -> (tensor<2xui64>, tensor<f64>) {
      %s:2 = impulse.sample @normal(%a#0, %a#1, %c) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<2>, name = "b" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      scf.yield %s#0, %s#1 : tensor<2xui64>, tensor<f64>
    } else {
      scf.yield %a#0, %a#1 : tensor<2xui64>, tensor<f64>
    }
    return %b#0, %b#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %c : tensor<f64>, %pred : i1) -> (tensor<1x4xf64>, tensor<f64>) {
    %prev = arith.constant dense<[[0.1, 0.2, -1.1, -1.2]]> : tensor<1x4xf64>
    // expected-error @below {{Impulse: regenerate over a score-tail trace requires a straight-line body of top-level distribution sites}}
    %res:4 = impulse.regenerate @model(%rng, %c, %pred) given %prev
        <{ all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
          selection = [[#impulse.symbol<1>]] }>
        : (tensor<2xui64>, tensor<f64>, i1, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1 : tensor<1x4xf64>, tensor<f64>
  }
}
