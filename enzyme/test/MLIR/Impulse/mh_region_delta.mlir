// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  // Single-site move on s: dirty terms s and t; u is clean.
  func.func @mh_region_delta_single(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x6xf64>, %weight : tensor<f64>) -> (tensor<1x6xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %tsl = impulse.slice %trace <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>}> : (tensor<1x6xf64>) -> tensor<1x1xf64>
    %tpin = impulse.reshape %tsl : (tensor<1x1xf64>) -> tensor<f64>
    %usl = impulse.slice %trace <{limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>}> : (tensor<1x6xf64>) -> tensor<1x1xf64>
    %upin = impulse.reshape %usl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %acc, %ld, %orng, %w = impulse.infer_region @test(%rng, %mean, %stddev) given %trace state = [%weight] sampler {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>, %sd: tensor<f64>):
      %s:2 = impulse.sample_region(%r, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<1>], fn = @normal, logpdf_fn = @logpdf, name = "s"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t:2 = impulse.sample_region(%s#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<2>], fn = @normal, logpdf_fn = @logpdf, name = "t"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %u:2 = impulse.sample_region(%t#0, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<3>], fn = @normal, logpdf_fn = @logpdf, name = "u"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %u#0, %u#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%sv: tensor<f64>):
      %ds = arith.subf %sv, %mean : tensor<f64>
      %lps = arith.mulf %ds, %ds : tensor<f64>
      %dt = arith.subf %tpin, %sv : tensor<f64>
      %lpt = arith.mulf %dt, %dt : tensor<f64>
      %du = arith.subf %upin, %mean : tensor<f64>
      %lpu = arith.mulf %du, %du : tensor<f64>
      %st = arith.addf %lps, %lpt : tensor<f64>
      %tot = arith.addf %st, %lpu : tensor<f64>
      impulse.yield %tot, %lps, %lpt, %lpu : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<1>]], state_schema = ["weight"]}> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x6xf64>) -> (tensor<1x6xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %acc, %orng, %w : tensor<1x6xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }

  // Block move on s and u: their own terms cancel, so log alpha is t's term.
  func.func @mh_region_delta_block(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x6xf64>, %weight : tensor<f64>) -> (tensor<1x6xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %tsl = impulse.slice %trace <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>}> : (tensor<1x6xf64>) -> tensor<1x1xf64>
    %tpin = impulse.reshape %tsl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %acc, %ld, %orng, %w = impulse.infer_region @test(%rng, %mean, %stddev) given %trace state = [%weight] sampler {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>, %sd: tensor<f64>):
      %s:2 = impulse.sample_region(%r, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<1>], fn = @normal, logpdf_fn = @logpdf, name = "s"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t:2 = impulse.sample_region(%s#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<2>], fn = @normal, logpdf_fn = @logpdf, name = "t"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %u:2 = impulse.sample_region(%t#0, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<3>], fn = @normal, logpdf_fn = @logpdf, name = "u"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %u#0, %u#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%sv: tensor<f64>, %uv: tensor<f64>):
      %ds = arith.subf %sv, %mean : tensor<f64>
      %lps = arith.mulf %ds, %ds : tensor<f64>
      %dt = arith.subf %tpin, %sv : tensor<f64>
      %lpt = arith.mulf %dt, %dt : tensor<f64>
      %du = arith.subf %uv, %mean : tensor<f64>
      %lpu = arith.mulf %du, %du : tensor<f64>
      %st = arith.addf %lps, %lpt : tensor<f64>
      %tot = arith.addf %st, %lpu : tensor<f64>
      impulse.yield %tot, %lps, %lpt, %lpu : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<1>], [#impulse.symbol<3>]], state_schema = ["weight"]}> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x6xf64>) -> (tensor<1x6xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %acc, %orng, %w : tensor<1x6xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }

  // t is dirty through an impulse.if region that uses s (no operand does).
  func.func @mh_region_delta_region_capture(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x6xf64>, %weight : tensor<f64>, %pred : tensor<i1>) -> (tensor<1x6xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %tsl = impulse.slice %trace <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>}> : (tensor<1x6xf64>) -> tensor<1x1xf64>
    %tpin = impulse.reshape %tsl : (tensor<1x1xf64>) -> tensor<f64>
    %usl = impulse.slice %trace <{limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>}> : (tensor<1x6xf64>) -> tensor<1x1xf64>
    %upin = impulse.reshape %usl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %acc, %ld, %orng, %w = impulse.infer_region @test(%rng, %mean, %stddev) given %trace state = [%weight] sampler {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>, %sd: tensor<f64>):
      %s:2 = impulse.sample_region(%r, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<1>], fn = @normal, logpdf_fn = @logpdf, name = "s"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t:2 = impulse.sample_region(%s#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<2>], fn = @normal, logpdf_fn = @logpdf, name = "t"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %u:2 = impulse.sample_region(%t#0, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 <{rng_distribution = #impulse.rng_distribution<NORMAL>}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } <{address = [#impulse.symbol<3>], fn = @normal, logpdf_fn = @logpdf, name = "u"}> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %u#0, %u#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%sv: tensor<f64>):
      %ds = arith.subf %sv, %mean : tensor<f64>
      %lps = arith.mulf %ds, %ds : tensor<f64>
      %lpt = impulse.if (%pred) ({
        %dt = arith.subf %tpin, %sv : tensor<f64>
        %sq = arith.mulf %dt, %dt : tensor<f64>
        impulse.yield %sq : tensor<f64>
      }, {
        %sq = arith.mulf %tpin, %tpin : tensor<f64>
        impulse.yield %sq : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      %du = arith.subf %upin, %mean : tensor<f64>
      %lpu = arith.mulf %du, %du : tensor<f64>
      %st = arith.addf %lps, %lpt : tensor<f64>
      %tot = arith.addf %st, %lpu : tensor<f64>
      impulse.yield %tot, %lps, %lpt, %lpu : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } <{all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<1>]], state_schema = ["weight"]}> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x6xf64>) -> (tensor<1x6xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %acc, %orng, %w : tensor<1x6xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @mh_region_delta_single
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[MEAN:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x6xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[I4:.+]] = arith.constant dense<4> : tensor<i64>
// CHECK-DAG: %[[I3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[I0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK: %[[TSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[TPIN:.+]] = impulse.reshape %[[TSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[RNG1:.+]], %[[S:.+]] = impulse.random %[[RNG]], %[[MEAN]], %[[SD]] <rng_distribution = NORMAL>
// CHECK-NEXT: %[[SR:.+]] = impulse.reshape %[[S]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[P0:.+]] = impulse.dynamic_update_slice %[[TR]], %[[SR]], %[[I0]], %[[I0]]
// CHECK-NEXT: %[[DS:.+]] = arith.subf %[[S]], %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPS:.+]] = arith.mulf %[[DS]], %[[DS]] : tensor<f64>
// CHECK-NEXT: %[[DT:.+]] = arith.subf %[[TPIN]], %[[S]] : tensor<f64>
// CHECK-NEXT: %[[LPT:.+]] = arith.mulf %[[DT]], %[[DT]] : tensor<f64>
// CHECK-NEXT: %[[LPSR:.+]] = impulse.reshape %[[LPS]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[P1:.+]] = impulse.dynamic_update_slice %[[P0]], %[[LPSR]], %[[I0]], %[[I3]]
// CHECK-NEXT: %[[LPTR:.+]] = impulse.reshape %[[LPT]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[P2:.+]] = impulse.dynamic_update_slice %[[P1]], %[[LPTR]], %[[I0]], %[[I4]]
// CHECK-NEXT: %[[OS:.+]] = impulse.slice %[[TR]] <start_indices = [0, 3], limit_indices = [1, 4], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[OSS:.+]] = impulse.reshape %[[OS]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CORR:.+]] = arith.subf %[[LPS]], %[[OSS]] : tensor<f64>
// CHECK-NEXT: %[[OS2:.+]] = impulse.slice %[[TR]] <start_indices = [0, 3], limit_indices = [1, 4], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[OSS2:.+]] = impulse.reshape %[[OS2]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[D1:.+]] = arith.subf %[[LPS]], %[[OSS2]] : tensor<f64>
// CHECK-NEXT: %[[OT:.+]] = impulse.slice %[[TR]] <start_indices = [0, 4], limit_indices = [1, 5], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[OTS:.+]] = impulse.reshape %[[OT]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[D2:.+]] = arith.subf %[[LPT]], %[[OTS]] : tensor<f64>
// CHECK-NEXT: %[[DELTA:.+]] = arith.addf %[[D1]], %[[D2]] : tensor<f64>
// CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[DELTA]], %[[CORR]] : tensor<f64>
// CHECK-NEXT: %[[WNEW:.+]] = arith.addf %[[W]], %[[DELTA]] : tensor<f64>
// CHECK-NEXT: %[[RNG2:.+]], %[[U:.+]] = impulse.random %[[RNG1]], %{{.+}}, %{{.+}} <rng_distribution = UNIFORM>
// CHECK-NEXT: %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
// CHECK-NEXT: %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[ALPHA]] : tensor<f64>
// CHECK-NEXT: %[[VP:.+]] = impulse.slice %[[P2]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[VC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[VU:.+]] = impulse.select %[[ACC]], %[[VP]], %[[VC]] : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T1:.+]] = impulse.dynamic_update_slice %[[TR]], %[[VU]], %[[I0]], %[[I0]] : (tensor<1x6xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: %[[SP:.+]] = impulse.slice %[[P2]] <start_indices = [0, 3], limit_indices = [1, 5], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 3], limit_indices = [1, 5], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SU:.+]] = impulse.select %[[ACC]], %[[SP]], %[[SC]] : (tensor<i1>, tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[SU]], %[[I0]], %[[I3]] : (tensor<1x6xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: %[[WOUT:.+]] = arith.select %[[ACC]], %[[WNEW]], %[[W]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: return %[[T2]], %[[ACC]], %[[RNG2]], %[[WOUT]]

// CHECK-LABEL: func.func @mh_region_delta_block
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[MEAN:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x6xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[I2:.+]] = arith.constant dense<2> : tensor<i64>
// CHECK-DAG: %[[I0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK: %[[RNG1:.+]], %[[S:.+]] = impulse.random %[[RNG]], %[[MEAN]], %[[SD]] <rng_distribution = NORMAL>
// CHECK: %[[RNG2:.+]], %[[UV:.+]] = impulse.random %[[RNG1]], %[[MEAN]], %[[SD]] <rng_distribution = NORMAL>
// CHECK: %[[DS:.+]] = arith.subf %[[S]], %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPS:.+]] = arith.mulf %[[DS]], %[[DS]] : tensor<f64>
// CHECK-NEXT: %[[DT:.+]] = arith.subf %{{.+}}, %[[S]] : tensor<f64>
// CHECK-NEXT: %[[LPT:.+]] = arith.mulf %[[DT]], %[[DT]] : tensor<f64>
// CHECK-NEXT: %[[DU:.+]] = arith.subf %[[UV]], %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPU:.+]] = arith.mulf %[[DU]], %[[DU]] : tensor<f64>
// CHECK: %[[OS2:.+]] = impulse.slice %[[TR]] <start_indices = [0, 3], limit_indices = [1, 4], strides = [1, 1]>
// CHECK-NEXT: %[[OSS2:.+]] = impulse.reshape %[[OS2]]
// CHECK-NEXT: %[[D1:.+]] = arith.subf %[[LPS]], %[[OSS2]] : tensor<f64>
// CHECK-NEXT: %[[OT:.+]] = impulse.slice %[[TR]] <start_indices = [0, 4], limit_indices = [1, 5], strides = [1, 1]>
// CHECK-NEXT: %[[OTS:.+]] = impulse.reshape %[[OT]]
// CHECK-NEXT: %[[D2:.+]] = arith.subf %[[LPT]], %[[OTS]] : tensor<f64>
// CHECK-NEXT: %[[D12:.+]] = arith.addf %[[D1]], %[[D2]] : tensor<f64>
// CHECK-NEXT: %[[OU2:.+]] = impulse.slice %[[TR]] <start_indices = [0, 5], limit_indices = [1, 6], strides = [1, 1]>
// CHECK-NEXT: %[[OUS2:.+]] = impulse.reshape %[[OU2]]
// CHECK-NEXT: %[[D3:.+]] = arith.subf %[[LPU]], %[[OUS2]] : tensor<f64>
// CHECK-NEXT: %[[DELTA:.+]] = arith.addf %[[D12]], %[[D3]] : tensor<f64>
// CHECK-NEXT: %[[WNEW:.+]] = arith.addf %[[W]], %[[DELTA]] : tensor<f64>
// CHECK: %[[ACC:.+]] = arith.cmpf olt, %{{.+}}, %[[D2]] : tensor<f64>
// CHECK-NEXT: %[[VP:.+]] = impulse.slice %[[P:.+]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[VC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[VU:.+]] = impulse.select %[[ACC]], %[[VP]], %[[VC]] : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T1:.+]] = impulse.dynamic_update_slice %[[TR]], %[[VU]], %[[I0]], %[[I0]] : (tensor<1x6xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: %[[WP:.+]] = impulse.slice %[[P]] <start_indices = [0, 2], limit_indices = [1, 6], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[WC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 2], limit_indices = [1, 6], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[WU:.+]] = impulse.select %[[ACC]], %[[WP]], %[[WC]] : (tensor<i1>, tensor<1x4xf64>, tensor<1x4xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[WU]], %[[I0]], %[[I2]] : (tensor<1x6xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<1x6xf64>
// CHECK-NEXT: %[[WOUT:.+]] = arith.select %[[ACC]], %[[WNEW]], %[[W]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: return %[[T2]], %[[ACC]], %{{.+}}, %[[WOUT]]

// CHECK-LABEL: func.func @mh_region_delta_region_capture
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[MEAN:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x6xf64>, %[[W:[^:]+]]: tensor<f64>, %[[PRED:[^:]+]]: tensor<i1>)
// CHECK: %[[RNG1:.+]], %[[S:.+]] = impulse.random %[[RNG]], %[[MEAN]], %[[SD]] <rng_distribution = NORMAL>
// CHECK: %[[DS:.+]] = arith.subf %[[S]], %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPS:.+]] = arith.mulf %[[DS]], %[[DS]] : tensor<f64>
// CHECK-NEXT: %[[LPT:.+]] = impulse.if(%[[PRED]]) ({
// CHECK-NEXT: %[[DT:.+]] = arith.subf %{{.+}}, %[[S]] : tensor<f64>
// CHECK-NEXT: %[[SQ:.+]] = arith.mulf %[[DT]], %[[DT]] : tensor<f64>
// CHECK-NEXT: impulse.yield %[[SQ]] : tensor<f64>
// CHECK-NEXT: }, {
// CHECK: }) : (tensor<i1>) -> tensor<f64>
// CHECK: %[[OT:.+]] = impulse.slice %[[TR]] <start_indices = [0, 4], limit_indices = [1, 5], strides = [1, 1]>
// CHECK-NEXT: %[[OTS:.+]] = impulse.reshape %[[OT]]
// CHECK-NEXT: %[[D2:.+]] = arith.subf %[[LPT]], %[[OTS]] : tensor<f64>
// CHECK-NEXT: %[[DELTA:.+]] = arith.addf %{{.+}}, %[[D2]] : tensor<f64>
// CHECK: impulse.slice %{{.+}} <start_indices = [0, 3], limit_indices = [1, 5], strides = [1, 1]> : (tensor<1x6xf64>) -> tensor<1x2xf64>
// CHECK-NOT: start_indices = array<i64: 0, 5>
// CHECK: return
