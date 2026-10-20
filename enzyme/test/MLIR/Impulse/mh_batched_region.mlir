// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64>
  func.func private @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  func.func @mh_batched_independent(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x8xf64>, %w : tensor<f64>) -> (tensor<1x8xf64>, tensor<2xui64>, tensor<f64>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<f64>
    %one = arith.constant dense<1.000000e+00> : tensor<f64>
    %sl0_a = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_a = impulse.reshape %sl0_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z2 = impulse.slice %trace {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z2 = impulse.reshape %sl0_z2 : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z3 = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z3 = impulse.reshape %sl0_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr0, %acc0, %ld0, %rng0, %w0 = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z3:2 = impulse.sample_region(%z2#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin0_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %pin0_z2, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %l_z3 = arith.mulf %pin0_z3, %logit : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<2>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %sl1_a = impulse.slice %tr0 {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_a = impulse.reshape %sl1_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z1 = impulse.slice %tr0 {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z1 = impulse.reshape %sl1_z1 : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z3 = impulse.slice %tr0 {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z3 = impulse.reshape %sl1_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr1, %acc1, %ld1, %rng1, %w1 = impulse.infer_region @model(%rng0, %m, %logit) given %tr0 state = [%w0] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z3:2 = impulse.sample_region(%z2#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin1_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %pin1_z1, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %l_z3 = arith.mulf %pin1_z3, %logit : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr1, %rng1, %w1 : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>
  }

  func.func @mh_sequential_dependent(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x8xf64>, %w : tensor<f64>) -> (tensor<1x8xf64>, tensor<2xui64>, tensor<f64>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<f64>
    %one = arith.constant dense<1.000000e+00> : tensor<f64>
    %sl0_a = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_a = impulse.reshape %sl0_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z2 = impulse.slice %trace {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z2 = impulse.reshape %sl0_z2 : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z3 = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z3 = impulse.reshape %sl0_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr0, %acc0, %ld0, %rng0, %w0 = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg_z2 = arith.addf %lg, %z1#1 : tensor<f64>
      %z2:2 = impulse.sample_region(%z1#0, %lg_z2) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z3:2 = impulse.sample_region(%z2#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin0_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %lg_z2 = arith.addf %logit, %zv : tensor<f64>
      %l_z2 = arith.mulf %pin0_z2, %lg_z2 : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %l_z3 = arith.mulf %pin0_z3, %logit : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<2>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %sl1_a = impulse.slice %tr0 {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_a = impulse.reshape %sl1_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z1 = impulse.slice %tr0 {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z1 = impulse.reshape %sl1_z1 : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z3 = impulse.slice %tr0 {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z3 = impulse.reshape %sl1_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr1, %acc1, %ld1, %rng1, %w1 = impulse.infer_region @model(%rng0, %m, %logit) given %tr0 state = [%w0] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg_z2 = arith.addf %lg, %z1#1 : tensor<f64>
      %z2:2 = impulse.sample_region(%z1#0, %lg_z2) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z3:2 = impulse.sample_region(%z2#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin1_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %pin1_z1, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %lg_z2 = arith.addf %logit, %pin1_z1 : tensor<f64>
      %l_z2 = arith.mulf %zv, %lg_z2 : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %l_z3 = arith.mulf %pin1_z3, %logit : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr1, %rng1, %w1 : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>
  }

  func.func @mh_split_mixed(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x8xf64>, %w : tensor<f64>) -> (tensor<1x8xf64>, tensor<2xui64>, tensor<f64>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<f64>
    %one = arith.constant dense<1.000000e+00> : tensor<f64>
    %sl0_a = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_a = impulse.reshape %sl0_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z2 = impulse.slice %trace {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z2 = impulse.reshape %sl0_z2 : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z3 = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z3 = impulse.reshape %sl0_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr0, %acc0, %ld0, %rng0, %w0 = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg_z3 = arith.addf %lg, %z2#1 : tensor<f64>
      %z3:2 = impulse.sample_region(%z2#0, %lg_z3) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin0_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %pin0_z2, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %lg_z3 = arith.addf %logit, %pin0_z2 : tensor<f64>
      %l_z3 = arith.mulf %pin0_z3, %lg_z3 : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<2>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %sl1_a = impulse.slice %tr0 {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_a = impulse.reshape %sl1_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z1 = impulse.slice %tr0 {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z1 = impulse.reshape %sl1_z1 : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z3 = impulse.slice %tr0 {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z3 = impulse.reshape %sl1_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr1, %acc1, %ld1, %rng1, %w1 = impulse.infer_region @model(%rng0, %m, %logit) given %tr0 state = [%w0] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg_z3 = arith.addf %lg, %z2#1 : tensor<f64>
      %z3:2 = impulse.sample_region(%z2#0, %lg_z3) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin1_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %pin1_z1, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %lg_z3 = arith.addf %logit, %zv : tensor<f64>
      %l_z3 = arith.mulf %pin1_z3, %lg_z3 : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %sl2_a = impulse.slice %tr1 {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin2_a = impulse.reshape %sl2_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl2_z1 = impulse.slice %tr1 {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin2_z1 = impulse.reshape %sl2_z1 : (tensor<1x1xf64>) -> tensor<f64>
    %sl2_z2 = impulse.slice %tr1 {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin2_z2 = impulse.reshape %sl2_z2 : (tensor<1x1xf64>) -> tensor<f64>
    %tr2, %acc2, %ld2, %rng2, %w2 = impulse.infer_region @model(%rng1, %m, %logit) given %tr1 state = [%w1] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg_z3 = arith.addf %lg, %z2#1 : tensor<f64>
      %z3:2 = impulse.sample_region(%z2#0, %lg_z3) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin2_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %pin2_z1, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %pin2_z2, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %lg_z3 = arith.addf %logit, %pin2_z2 : tensor<f64>
      %l_z3 = arith.mulf %zv, %lg_z3 : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<4>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr2, %rng2, %w2 : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>
  }

  func.func @mh_sequential_trace_read(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x8xf64>, %w : tensor<f64>) -> (tensor<1x8xf64>, tensor<2xui64>, tensor<f64>, tensor<1x8xf64>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<f64>
    %one = arith.constant dense<1.000000e+00> : tensor<f64>
    %sl0_a = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_a = impulse.reshape %sl0_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z2 = impulse.slice %trace {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z2 = impulse.reshape %sl0_z2 : (tensor<1x1xf64>) -> tensor<f64>
    %sl0_z3 = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin0_z3 = impulse.reshape %sl0_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr0, %acc0, %ld0, %rng0, %w0 = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z3:2 = impulse.sample_region(%z2#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin0_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %pin0_z2, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %l_z3 = arith.mulf %pin0_z3, %logit : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<2>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    %sl1_a = impulse.slice %tr0 {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_a = impulse.reshape %sl1_a : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z1 = impulse.slice %tr0 {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z1 = impulse.reshape %sl1_z1 : (tensor<1x1xf64>) -> tensor<f64>
    %sl1_z3 = impulse.slice %tr0 {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
    %pin1_z3 = impulse.reshape %sl1_z3 : (tensor<1x1xf64>) -> tensor<f64>
    %tr1, %acc1, %ld1, %rng1, %w1 = impulse.infer_region @model(%rng0, %m, %logit) given %tr0 state = [%w0] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z3:2 = impulse.sample_region(%z2#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        %o, %u = impulse.random %s0, %zero, %one {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        %p = impulse.logistic %s1 : (tensor<f64>) -> tensor<f64>
        %t = arith.cmpf olt, %u, %p : tensor<f64>
        %v = arith.select %t, %one, %zero : tensor<i1>, tensor<f64>
        impulse.yield %o, %v : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], dist = #impulse.dist<BERNOULLI>, name = "z3"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z3#0, %z3#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%zv: tensor<f64>):
      %da = arith.subf %pin1_a, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l_z1 = arith.mulf %pin1_z1, %logit : tensor<f64>
      %lp_z1 = arith.subf %l_z1, %sp : tensor<f64>
      %l_z2 = arith.mulf %zv, %logit : tensor<f64>
      %lp_z2 = arith.subf %l_z2, %sp : tensor<f64>
      %l_z3 = arith.mulf %pin1_z3, %logit : tensor<f64>
      %lp_z3 = arith.subf %l_z3, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lp_z1 : tensor<f64>
      %j2 = arith.addf %j1, %lp_z2 : tensor<f64>
      %tot = arith.addf %j2, %lp_z3 : tensor<f64>
      impulse.yield %tot, %lpa, %lp_z1, %lp_z2, %lp_z3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr1, %rng1, %w1, %tr0 : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>, tensor<1x8xf64>
  }
}

// CHECK-LABEL: func.func @mh_batched_independent
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[LOGIT:[^:]+]]: tensor<f64>, %[[SP:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x8xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK: %[[PST:.+]], %[[PROP:.+]] = impulse.random %[[RNG]], %[[ZERO]], %[[ONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT: %[[AST:.+]], %[[ACCU:.+]] = impulse.random %[[PST]], %[[ZERO]], %[[ONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT: %[[LOGU:.+]] = math.log %[[ACCU]] : tensor<2xf64>
// CHECK-NEXT: %[[U1S:.+]] = impulse.slice %[[PROP]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[U1:.+]] = impulse.reshape %[[U1S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[P1:.+]] = impulse.logistic %[[LOGIT]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[T1:.+]] = arith.cmpf olt, %[[U1]], %[[P1]] : tensor<f64>
// CHECK-NEXT: %[[Z1:.+]] = arith.select %[[T1]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<f64>
// CHECK: %[[U2S:.+]] = impulse.slice %[[PROP]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[U2:.+]] = impulse.reshape %[[U2S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[P2:.+]] = impulse.logistic %[[LOGIT]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[T2:.+]] = arith.cmpf olt, %[[U2]], %[[P2]] : tensor<f64>
// CHECK-NEXT: %[[Z2:.+]] = arith.select %[[T2]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<f64>
// CHECK: %[[OLDV:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x2xf64>
// CHECK: %[[OLDS:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 7>, start_indices = array<i64: 0, 5>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[OLDSV:.+]] = impulse.reshape %[[OLDS]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[DELTA:.+]] = arith.subf %[[NEWS:.+]], %[[OLDSV]] : tensor<2xf64>
// CHECK-NEXT: %[[CORR:.+]] = arith.subf %[[NEWS]], %[[OLDSV]] : tensor<2xf64>
// CHECK-NEXT: %[[LA:.+]] = arith.subf %[[DELTA]], %[[CORR]] : tensor<2xf64>
// CHECK-NEXT: %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[LA]] : tensor<2xf64>
// CHECK-NEXT: %[[A1S:.+]] = impulse.slice %[[ACC]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi1>) -> tensor<1xi1>
// CHECK-NEXT: %[[A1:.+]] = impulse.reshape %[[A1S]] : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT: %[[D1S:.+]] = impulse.slice %[[DELTA]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[D1:.+]] = impulse.reshape %[[D1S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[WA1:.+]] = arith.addf %[[W]], %[[D1]] : tensor<f64>
// CHECK-NEXT: %[[W1:.+]] = arith.select %[[A1]], %[[WA1]], %[[W]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: %[[A2S:.+]] = impulse.slice %[[ACC]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xi1>) -> tensor<1xi1>
// CHECK-NEXT: %[[A2:.+]] = impulse.reshape %[[A2S]] : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT: %[[D2S:.+]] = impulse.slice %[[DELTA]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[D2:.+]] = impulse.reshape %[[D2S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[WA2:.+]] = arith.addf %[[W1]], %[[D2]] : tensor<f64>
// CHECK-NEXT: %[[W2:.+]] = arith.select %[[A2]], %[[WA2]], %[[W1]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: %[[SELV:.+]] = arith.select %[[ACC]], %{{.+}}, %{{.+}} : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT: %[[SELS:.+]] = arith.select %[[ACC]], %[[NEWS]], %[[OLDSV]] : tensor<2xi1>, tensor<2xf64>
// CHECK: %[[TV:.+]] = impulse.dynamic_update_slice %[[TR]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<1x8xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
// CHECK: %[[TS:.+]] = impulse.dynamic_update_slice %[[TV]], %{{.+}}, %{{.+}}, %{{.+}} : (tensor<1x8xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
// CHECK-NEXT: return %[[TS]], %[[AST]], %[[W2]] : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @mh_sequential_dependent
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S1:.+]], %{{.+}} = impulse.random %[[RNG]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S2:.+]], %[[U2:.+]] = impulse.random %[[S1]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = math.log %[[U2]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S3:.+]], %{{.+}} = impulse.random %[[S2]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S4:.+]], %[[U4:.+]] = impulse.random %[[S3]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = math.log %[[U4]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: return %{{.+}}, %[[S4]], %{{.+}} : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>

// z3's prior depends on z2: (z1, z2) is batched, z3 stays sequential.
// CHECK-LABEL: func.func @mh_split_mixed
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>
// CHECK: %[[PST:.+]], %{{.+}} = impulse.random %[[RNG]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT: %[[AST:.+]], %{{.+}} = impulse.random %[[PST]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK: %[[ACC:.+]] = arith.cmpf olt, %{{.+}}, %{{.+}} : tensor<2xf64>
// CHECK: %[[TB:.+]] = impulse.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<1x8xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[Z2S:.+]] = impulse.slice %[[TB]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x8xf64>) -> tensor<1x1xf64>
// CHECK: %[[S3:.+]], %{{.+}} = impulse.random %[[AST]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK: %[[S4:.+]], %{{.+}} = impulse.random %[[S3]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK: return %{{.+}}, %[[S4]], %{{.+}} : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @mh_sequential_trace_read
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S1:.+]], %{{.+}} = impulse.random %[[RNG]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S2:.+]], %{{.+}} = impulse.random %[[S1]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S3:.+]], %{{.+}} = impulse.random %[[S2]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[S4:.+]], %{{.+}} = impulse.random %[[S3]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NOT: tensor<2xf64>
// CHECK: return %[[T2:.+]], %[[S4]], %{{.+}}, %[[T1:.+]] : tensor<1x8xf64>, tensor<2xui64>, tensor<f64>, tensor<1x8xf64>
