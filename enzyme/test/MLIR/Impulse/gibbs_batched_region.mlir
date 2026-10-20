// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64>
  func.func private @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  // Batched: point 1's if passes the rng through, so point 2's terms skip z1.
  func.func @gibbs_batched(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %lout : tensor<f64>, %trace : tensor<1x7xf64>, %obs : tensor<1x4xf64>, %w : tensor<f64>) -> (tensor<1x7xf64>, tensor<2xui64>, tensor<f64>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<f64>
    %half = arith.constant dense<5.000000e-01> : tensor<f64>
    %asl = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x7xf64>) -> tensor<1x1xf64>
    %apin = impulse.reshape %asl : (tensor<1x1xf64>) -> tensor<f64>
    %y1sl = impulse.slice %obs {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %y1 = impulse.reshape %y1sl : (tensor<1x1xf64>) -> tensor<f64>
    %y2sl = impulse.slice %obs {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %y2 = impulse.reshape %y2sl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %diag, %ld, %orng, %wout = impulse.infer_region @model(%rng, %m, %logit) given %trace constrained_by %obs state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %p1 = arith.cmpf ogt, %z1#1, %half : tensor<f64>
      %if1:2 = impulse.if(%p1) ({
        %yo1:2 = impulse.sample_region(%z1#0, %mm) sampler {
        ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
          impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
        } logpdf {
        } {address = [#impulse.symbol<3>], fn = @draw, logpdf_fn = @lp, name = "yo1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %yo1#0, %yo1#1 : tensor<2xui64>, tensor<f64>
      }, {
        %yi1:2 = impulse.sample_region(%z1#0, %a#1) sampler {
        ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
          impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
        } logpdf {
        } {address = [#impulse.symbol<4>], fn = @draw, logpdf_fn = @lp, name = "yi1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %yi1#0, %yi1#1 : tensor<2xui64>, tensor<f64>
      }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%if1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<5>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %p2 = arith.cmpf ogt, %z2#1, %half : tensor<f64>
      %if2:2 = impulse.if(%p2) ({
        %yo2:2 = impulse.sample_region(%z2#0, %mm) sampler {
        ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
          impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
        } logpdf {
        } {address = [#impulse.symbol<6>], fn = @draw, logpdf_fn = @lp, name = "yo2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %yo2#0, %yo2#1 : tensor<2xui64>, tensor<f64>
      }, {
        %yi2:2 = impulse.sample_region(%z2#0, %a#1) sampler {
        ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
          impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
        } logpdf {
        } {address = [#impulse.symbol<7>], fn = @draw, logpdf_fn = @lp, name = "yi2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %yi2#0, %yi2#1 : tensor<2xui64>, tensor<f64>
      }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %if2#0, %if2#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%z1v: tensor<f64>, %z2v: tensor<f64>):
      %da = arith.subf %apin, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l1 = arith.mulf %z1v, %logit : tensor<f64>
      %lpz1 = arith.subf %l1, %sp : tensor<f64>
      %q1 = arith.cmpf ogt, %z1v, %half : tensor<f64>
      %b1:4 = impulse.if(%q1) ({
        impulse.yield %rng, %lout, %zero, %lout : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>
      }, {
        %d1 = arith.subf %y1, %apin : tensor<f64>
        %e1 = arith.mulf %d1, %d1 : tensor<f64>
        impulse.yield %rng, %zero, %e1, %e1 : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>
      }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>)
      %l2 = arith.mulf %z2v, %logit : tensor<f64>
      %lpz2 = arith.subf %l2, %sp : tensor<f64>
      %q2 = arith.cmpf ogt, %z2v, %half : tensor<f64>
      %b2:4 = impulse.if(%q2) ({
        impulse.yield %b1#0, %lout, %zero, %lout : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>
      }, {
        %d2 = arith.subf %y2, %apin : tensor<f64>
        %e2 = arith.mulf %d2, %d2 : tensor<f64>
        impulse.yield %b1#0, %zero, %e2, %e2 : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>
      }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>)
      %j1 = arith.addf %lpa, %lpz1 : tensor<f64>
      %j2 = arith.addf %j1, %b1#3 : tensor<f64>
      %j3 = arith.addf %j2, %lpz2 : tensor<f64>
      %tot = arith.addf %j3, %b2#3 : tensor<f64>
      impulse.yield %tot, %lpa, %lpz1, %b1#1, %b1#2, %lpz2, %b2#1, %b2#2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>], [#impulse.symbol<5>], [#impulse.symbol<6>], [#impulse.symbol<7>]], constrained_addresses = [[#impulse.symbol<3>], [#impulse.symbol<4>], [#impulse.symbol<6>], [#impulse.symbol<7>]], gibbs_config = #impulse.gibbs_config, num_position_args = 2 : i64, num_samples = 1 : i64, position_size = 2 : i64, selection = [[#impulse.symbol<2>], [#impulse.symbol<5>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x7xf64>, tensor<1x4xf64>) -> (tensor<1x7xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %orng, %wout : tensor<1x7xf64>, tensor<2xui64>, tensor<f64>
  }

  func.func @gibbs_batched_score_tail(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x6xf64>, %w : tensor<f64>) -> (tensor<1x6xf64>, tensor<2xui64>, tensor<f64>) {
    %asl = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x6xf64>) -> tensor<1x1xf64>
    %apin = impulse.reshape %asl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %diag, %ld, %orng, %wout = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %z2#0, %z2#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%z1v: tensor<f64>, %z2v: tensor<f64>):
      %da = arith.subf %apin, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l1 = arith.mulf %z1v, %logit : tensor<f64>
      %lpz1 = arith.subf %l1, %sp : tensor<f64>
      %l2 = arith.mulf %z2v, %logit : tensor<f64>
      %lpz2 = arith.subf %l2, %sp : tensor<f64>
      %j1 = arith.addf %lpa, %lpz1 : tensor<f64>
      %tot = arith.addf %j1, %lpz2 : tensor<f64>
      impulse.yield %tot, %lpa, %lpz1, %lpz2 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], gibbs_config = #impulse.gibbs_config, num_position_args = 2 : i64, num_samples = 1 : i64, position_size = 2 : i64, selection = [[#impulse.symbol<2>], [#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x6xf64>) -> (tensor<1x6xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %orng, %wout : tensor<1x6xf64>, tensor<2xui64>, tensor<f64>
  }

  // Sequential: the term of y depends on both flags.
  func.func @gibbs_shared_term(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>) -> (tensor<1x4xf64>, tensor<2xui64>, tensor<f64>) {
    %asl = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %apin = impulse.reshape %asl : (tensor<1x1xf64>) -> tensor<f64>
    %ysl = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %ypin = impulse.reshape %ysl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %diag, %ld, %orng, %wout = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z2:2 = impulse.sample_region(%z1#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %y:2 = impulse.sample_region(%z2#0, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], fn = @draw, logpdf_fn = @lp, name = "y"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %y#0, %y#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%z1v: tensor<f64>, %z2v: tensor<f64>):
      %da = arith.subf %apin, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l1 = arith.mulf %z1v, %logit : tensor<f64>
      %lpz1 = arith.subf %l1, %sp : tensor<f64>
      %l2 = arith.mulf %z2v, %logit : tensor<f64>
      %lpz2 = arith.subf %l2, %sp : tensor<f64>
      %zs = arith.addf %z1v, %z2v : tensor<f64>
      %dy = arith.subf %ypin, %zs : tensor<f64>
      %lpy = arith.mulf %dy, %dy : tensor<f64>
      %j1 = arith.addf %lpa, %lpz1 : tensor<f64>
      %j2 = arith.addf %j1, %lpz2 : tensor<f64>
      %tot = arith.addf %j2, %lpy : tensor<f64>
      impulse.yield %tot, %lpa, %lpz1, %lpz2, %lpy : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], gibbs_config = #impulse.gibbs_config, num_position_args = 2 : i64, num_samples = 1 : i64, position_size = 2 : i64, selection = [[#impulse.symbol<2>], [#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %orng, %wout : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>
  }

  // Sequential: z2 ~ Bernoulli(logit + z1).
  func.func @gibbs_dependent_prior(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>) -> (tensor<1x4xf64>, tensor<2xui64>, tensor<f64>) {
    %asl = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %apin = impulse.reshape %asl : (tensor<1x1xf64>) -> tensor<f64>
    %ysl = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %ypin = impulse.reshape %ysl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %diag, %ld, %orng, %wout = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg2 = arith.addf %lg, %z1#1 : tensor<f64>
      %z2:2 = impulse.sample_region(%z1#0, %lg2) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %y:2 = impulse.sample_region(%z2#0, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], fn = @draw, logpdf_fn = @lp, name = "y"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %y#0, %y#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%z1v: tensor<f64>, %z2v: tensor<f64>):
      %da = arith.subf %apin, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l1 = arith.mulf %z1v, %logit : tensor<f64>
      %lpz1 = arith.subf %l1, %sp : tensor<f64>
      %lg2 = arith.addf %logit, %z1v : tensor<f64>
      %l2 = arith.mulf %z2v, %lg2 : tensor<f64>
      %lpz2 = arith.subf %l2, %sp : tensor<f64>
      %dy = arith.subf %ypin, %apin : tensor<f64>
      %lpy = arith.mulf %dy, %dy : tensor<f64>
      %j1 = arith.addf %lpa, %lpz1 : tensor<f64>
      %j2 = arith.addf %j1, %lpz2 : tensor<f64>
      %tot = arith.addf %j2, %lpy : tensor<f64>
      impulse.yield %tot, %lpa, %lpz1, %lpz2, %lpy : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], gibbs_config = #impulse.gibbs_config, num_position_args = 2 : i64, num_samples = 1 : i64, position_size = 2 : i64, selection = [[#impulse.symbol<2>], [#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %orng, %wout : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>
  }

  // Sequential from the sampler's parameter dependence alone (logpdf hides it).
  func.func @gibbs_dependent_param_only(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %sp : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>) -> (tensor<1x4xf64>, tensor<2xui64>, tensor<f64>) {
    %asl = impulse.slice %trace {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %apin = impulse.reshape %asl : (tensor<1x1xf64>) -> tensor<f64>
    %ysl = impulse.slice %trace {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %ypin = impulse.reshape %ysl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %diag, %ld, %orng, %wout = impulse.infer_region @model(%rng, %m, %logit) given %trace state = [%w] sampler {
    ^bb0(%r: tensor<2xui64>, %mm: tensor<f64>, %lg: tensor<f64>):
      %a:2 = impulse.sample_region(%r, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %z1:2 = impulse.sample_region(%a#0, %lg) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z1"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %lg2 = arith.addf %lg, %z1#1 : tensor<f64>
      %z2:2 = impulse.sample_region(%z1#0, %lg2) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], dist = #impulse.dist<BERNOULLI>, name = "z2"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %y:2 = impulse.sample_region(%z2#0, %mm) sampler {
      ^bb0(%s0: tensor<2xui64>, %s1: tensor<f64>):
        impulse.yield %s0, %s1 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], fn = @draw, logpdf_fn = @lp, name = "y"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %y#0, %y#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%z1v: tensor<f64>, %z2v: tensor<f64>):
      %da = arith.subf %apin, %m : tensor<f64>
      %lpa = arith.mulf %da, %da : tensor<f64>
      %l1 = arith.mulf %z1v, %logit : tensor<f64>
      %lpz1 = arith.subf %l1, %sp : tensor<f64>
      %l2 = arith.mulf %z2v, %logit : tensor<f64>
      %lpz2 = arith.subf %l2, %sp : tensor<f64>
      %dy = arith.subf %ypin, %apin : tensor<f64>
      %lpy = arith.mulf %dy, %dy : tensor<f64>
      %j1 = arith.addf %lpa, %lpz1 : tensor<f64>
      %j2 = arith.addf %j1, %lpz2 : tensor<f64>
      %tot = arith.addf %j2, %lpy : tensor<f64>
      impulse.yield %tot, %lpa, %lpz1, %lpz2, %lpy : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]], gibbs_config = #impulse.gibbs_config, num_position_args = 2 : i64, num_samples = 1 : i64, position_size = 2 : i64, selection = [[#impulse.symbol<2>], [#impulse.symbol<3>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %orng, %wout : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @gibbs_batched
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[LOGIT:[^:]+]]: tensor<f64>, %[[SP:[^:]+]]: tensor<f64>, %[[LOUT:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x7xf64>, %[[OBS:[^:]+]]: tensor<1x4xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[C6:.+]] = arith.constant dense<6> : tensor<i64>
// CHECK-DAG: %[[C5:.+]] = arith.constant dense<5> : tensor<i64>
// CHECK-DAG: %[[C3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[C2:.+]] = arith.constant dense<2> : tensor<i64>
// CHECK-DAG: %[[C4:.+]] = arith.constant dense<4> : tensor<i64>
// CHECK-DAG: %[[VONE:.+]] = arith.constant dense<1.000000e+00> : tensor<2xf64>
// CHECK-DAG: %[[VHALF:.+]] = arith.constant dense<5.000000e-01> : tensor<2xf64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[VZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK: %[[CURSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 5>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 3>} : (tensor<1x7xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[CUR:.+]] = impulse.reshape %[[CURSL]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[P10M:.+]] = arith.mulf %[[LOGIT]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[P10:.+]] = arith.subf %[[P10M]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[Q10:.+]] = arith.cmpf ogt, %[[ZERO]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[B10:.+]]:4 = impulse.if(%[[Q10]]) ({
// CHECK-NEXT: impulse.yield %[[RNG]], %[[LOUT]], %[[ZERO]], %[[LOUT]] : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %[[P20M:.+]] = arith.mulf %[[LOGIT]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[P20:.+]] = arith.subf %[[P20M]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[Q20:.+]] = arith.cmpf ogt, %[[ZERO]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[B20:.+]]:4 = impulse.if(%[[Q20]]) ({
// CHECK-NEXT: impulse.yield %[[B10]]#0, %[[LOUT]], %[[ZERO]], %[[LOUT]] : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>
// CHECK: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %[[P11:.+]] = arith.subf %[[LOGIT]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[Q11:.+]] = arith.cmpf ogt, %[[ONE]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[B11:.+]]:4 = impulse.if(%[[Q11]]) ({
// CHECK: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %[[P21:.+]] = arith.subf %[[LOGIT]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[Q21:.+]] = arith.cmpf ogt, %[[ONE]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[B21:.+]]:4 = impulse.if(%[[Q21]]) ({
// CHECK: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %[[D1P:.+]] = arith.subf %[[P11]], %[[P10]] : tensor<f64>
// CHECK-NEXT: %[[D1O:.+]] = arith.subf %[[B11]]#1, %[[B10]]#1 : tensor<f64>
// CHECK-NEXT: %[[L1A:.+]] = arith.addf %[[D1P]], %[[D1O]] : tensor<f64>
// CHECK-NEXT: %[[D1I:.+]] = arith.subf %[[B11]]#2, %[[B10]]#2 : tensor<f64>
// CHECK-NEXT: %[[L1:.+]] = arith.addf %[[L1A]], %[[D1I]] : tensor<f64>
// CHECK-NEXT: %[[D2P:.+]] = arith.subf %[[P21]], %[[P20]] : tensor<f64>
// CHECK-NEXT: %[[D2O:.+]] = arith.subf %[[B21]]#1, %[[B20]]#1 : tensor<f64>
// CHECK-NEXT: %[[L2A:.+]] = arith.addf %[[D2P]], %[[D2O]] : tensor<f64>
// CHECK-NEXT: %[[D2I:.+]] = arith.subf %[[B21]]#2, %[[B20]]#2 : tensor<f64>
// CHECK-NEXT: %[[L2:.+]] = arith.addf %[[L2A]], %[[D2I]] : tensor<f64>
// CHECK-NEXT: %[[L1R:.+]] = impulse.reshape %[[L1]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[LV1:.+]] = impulse.dynamic_update_slice %[[VZERO]], %[[L1R]], %[[C0]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[L2R:.+]] = impulse.reshape %[[L2]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[LV:.+]] = impulse.dynamic_update_slice %[[LV1]], %[[L2R]], %[[C1]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[P:.+]] = impulse.logistic %[[LV]] : (tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[ORNG:.+]], %[[U:.+]] = impulse.random %[[RNG]], %[[ZERO]], %[[ONE]] {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT: %[[TAKE:.+]] = arith.cmpf olt, %[[U]], %[[P]] : tensor<2xf64>
// CHECK-NEXT: %[[CUR1:.+]] = arith.cmpf ogt, %[[CUR]], %[[VHALF]] : tensor<2xf64>
// CHECK-NEXT: %[[NEW:.+]] = arith.select %[[TAKE]], %[[VONE]], %[[VZERO]] : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT: %[[NEGL:.+]] = arith.negf %[[LV]] : tensor<2xf64>
// CHECK-NEXT: %[[UP:.+]] = arith.select %[[CUR1]], %[[VZERO]], %[[LV]] : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT: %[[DOWN:.+]] = arith.select %[[CUR1]], %[[NEGL]], %[[VZERO]] : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT: %[[DW:.+]] = arith.select %[[TAKE]], %[[UP]], %[[DOWN]] : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT: %[[DW1S:.+]] = impulse.slice %[[DW]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[DW1:.+]] = impulse.reshape %[[DW1S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[W1:.+]] = arith.addf %[[W]], %[[DW1]] : tensor<f64>
// CHECK-NEXT: %[[DW2S:.+]] = impulse.slice %[[DW]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[DW2:.+]] = impulse.reshape %[[DW2S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[W2:.+]] = arith.addf %[[W1]], %[[DW2]] : tensor<f64>
// CHECK-NEXT: %[[ROW:.+]] = impulse.reshape %[[NEW]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[N1:.+]] = impulse.slice %[[ROW]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T1:.+]] = impulse.dynamic_update_slice %[[TR]], %[[N1]], %[[C0]], %[[C1]] : (tensor<1x7xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x7xf64>
// CHECK-NEXT: %[[N2:.+]] = impulse.slice %[[ROW]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[N2]], %[[C0]], %[[C4]] : (tensor<1x7xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x7xf64>
// CHECK: %[[T3:.+]] = impulse.dynamic_update_slice %[[T2]], %{{.+}}, %[[C0]], %[[C2]]
// CHECK: %[[T4:.+]] = impulse.dynamic_update_slice %[[T3]], %{{.+}}, %[[C0]], %[[C3]]
// CHECK: %[[T5:.+]] = impulse.dynamic_update_slice %[[T4]], %{{.+}}, %[[C0]], %[[C5]]
// CHECK: %[[T6:.+]] = impulse.dynamic_update_slice %[[T5]], %{{.+}}, %[[C0]], %[[C6]]
// CHECK-NEXT: return %[[T6]], %[[ORNG]], %[[W2]] : tensor<1x7xf64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @gibbs_batched_score_tail
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[LOGIT:[^:]+]]: tensor<f64>, %[[SP:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x6xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[C5:.+]] = arith.constant dense<5> : tensor<i64>
// CHECK-DAG: %[[C4:.+]] = arith.constant dense<4> : tensor<i64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[CURSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x6xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[CUR:.+]] = impulse.reshape %[[CURSL]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[P10M:.+]] = arith.mulf %[[LOGIT]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[P10:.+]] = arith.subf %[[P10M]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[P20M:.+]] = arith.mulf %[[LOGIT]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[P20:.+]] = arith.subf %[[P20M]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[P11:.+]] = arith.subf %[[LOGIT]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[P21:.+]] = arith.subf %[[LOGIT]], %[[SP]] : tensor<f64>
// CHECK-NEXT: %[[L1:.+]] = arith.subf %[[P11]], %[[P10]] : tensor<f64>
// CHECK-NEXT: %[[L2:.+]] = arith.subf %[[P21]], %[[P20]] : tensor<f64>
// CHECK: %[[P:.+]] = impulse.logistic %{{.+}} : (tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[ORNG:.+]], %[[U:.+]] = impulse.random %[[RNG]], %[[ZERO]], %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT: %[[TAKE:.+]] = arith.cmpf olt, %[[U]], %[[P]] : tensor<2xf64>
// CHECK-NEXT: %{{.+}} = arith.cmpf ogt, %[[CUR]], %{{.+}} : tensor<2xf64>
// CHECK-NEXT: %[[NEW:.+]] = arith.select %[[TAKE]], %{{.+}}, %{{.+}} : tensor<2xi1>, tensor<2xf64>
// CHECK: %[[DW:.+]] = arith.select %[[TAKE]], %{{.+}}, %{{.+}} : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT: %[[DW1S:.+]] = impulse.slice %[[DW]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[DW1:.+]] = impulse.reshape %[[DW1S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[W1:.+]] = arith.addf %[[W]], %[[DW1]] : tensor<f64>
// CHECK-NEXT: %[[DW2S:.+]] = impulse.slice %[[DW]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[DW2:.+]] = impulse.reshape %[[DW2S]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[W2:.+]] = arith.addf %[[W1]], %[[DW2]] : tensor<f64>
// CHECK-NEXT: %[[ROW:.+]] = impulse.reshape %[[NEW]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[T1:.+]] = impulse.dynamic_update_slice %[[TR]], %[[ROW]], %[[C0]], %[[C1]] : (tensor<1x6xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: %[[TK1S:.+]] = impulse.slice %[[TAKE]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xi1>) -> tensor<1xi1>
// CHECK-NEXT: %[[TK1:.+]] = impulse.reshape %[[TK1S]] : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT: %[[S1:.+]] = arith.select %[[TK1]], %[[P11]], %[[P10]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: %[[S1R:.+]] = impulse.reshape %[[S1]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[S1R]], %[[C0]], %[[C4]] : (tensor<1x6xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: %[[TK2S:.+]] = impulse.slice %[[TAKE]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xi1>) -> tensor<1xi1>
// CHECK-NEXT: %[[TK2:.+]] = impulse.reshape %[[TK2S]] : (tensor<1xi1>) -> tensor<i1>
// CHECK-NEXT: %[[S2:.+]] = arith.select %[[TK2]], %[[P21]], %[[P20]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: %[[S2R:.+]] = impulse.reshape %[[S2]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T3:.+]] = impulse.dynamic_update_slice %[[T2]], %[[S2R]], %[[C0]], %[[C5]] : (tensor<1x6xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: return %[[T3]], %[[ORNG]], %[[W2]] : tensor<1x6xf64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @gibbs_shared_term
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[PB1:.+]] = impulse.logistic %{{.+}} : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[RB1:.+]], %[[UB1:.+]] = impulse.random %[[RNG]], %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[UB1]], %[[PB1]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[PB2:.+]] = impulse.logistic %{{.+}} : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[RB2:.+]], %[[UB2:.+]] = impulse.random %[[RB1]], %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[UB2]], %[[PB2]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: return %{{.+}}, %[[RB2]], %{{.+}} : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @gibbs_dependent_prior
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[PC1:.+]] = impulse.logistic %{{.+}} : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[RC1:.+]], %[[UC1:.+]] = impulse.random %[[RNG]], %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[UC1]], %[[PC1]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[PC2:.+]] = impulse.logistic %{{.+}} : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[RC2:.+]], %[[UC2:.+]] = impulse.random %[[RC1]], %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[UC2]], %[[PC2]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: return %{{.+}}, %[[RC2]], %{{.+}} : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @gibbs_dependent_param_only
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[PD1:.+]] = impulse.logistic %{{.+}} : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[RD1:.+]], %[[UD1:.+]] = impulse.random %[[RNG]], %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[UD1]], %[[PD1]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: %[[PD2:.+]] = impulse.logistic %{{.+}} : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[RD2:.+]], %[[UD2:.+]] = impulse.random %[[RD1]], %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[UD2]], %[[PD2]] : tensor<f64>
// CHECK-NOT: tensor<2xf64>
// CHECK: return %{{.+}}, %[[RD2]], %{{.+}} : tensor<1x4xf64>, tensor<2xui64>, tensor<f64>
