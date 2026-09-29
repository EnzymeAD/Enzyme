// RUN: %eopt --expand-impulse %s | FileCheck %s

//     (column 2).

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  func.func @mh_region_bare(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x2xf64>, %weight : tensor<f64>) -> (tensor<1x2xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %tsl = impulse.slice %trace {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
    %tpin = impulse.reshape %tsl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %acc, %ld, %orng, %w = impulse.infer_region @test(%rng, %mean, %stddev) given %trace state = [%weight] sampler {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>, %sd: tensor<f64>):
      %s:2 = impulse.sample_region(%r, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @normal, logpdf_fn = @logpdf, name = "s"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t:2 = impulse.sample_region(%s#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], fn = @normal, logpdf_fn = @logpdf, name = "t"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %t#0, %t#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%sv: tensor<f64>):
      %ds = arith.subf %sv, %mean : tensor<f64>
      %lps = arith.mulf %ds, %ds : tensor<f64>
      %dt = arith.subf %tpin, %sv : tensor<f64>
      %lpt = arith.mulf %dt, %dt : tensor<f64>
      %tot = arith.addf %lps, %lpt : tensor<f64>
      impulse.yield %tot, %lps, %lpt : tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<1>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %acc, %orng, %w : tensor<1x2xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }

  func.func @mh_region_tail(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x4xf64>, %weight : tensor<f64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %tsl = impulse.slice %trace {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
    %tpin = impulse.reshape %tsl : (tensor<1x1xf64>) -> tensor<f64>
    %tr, %acc, %ld, %orng, %w = impulse.infer_region @test(%rng, %mean, %stddev) given %trace state = [%weight] sampler {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>, %sd: tensor<f64>):
      %s:2 = impulse.sample_region(%r, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @normal, logpdf_fn = @logpdf, name = "s"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t:2 = impulse.sample_region(%s#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], fn = @normal, logpdf_fn = @logpdf, name = "t"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %t#0, %t#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%sv: tensor<f64>):
      %ds = arith.subf %sv, %mean : tensor<f64>
      %lps = arith.mulf %ds, %ds : tensor<f64>
      %dt = arith.subf %tpin, %sv : tensor<f64>
      %lpt = arith.mulf %dt, %dt : tensor<f64>
      %tot = arith.addf %lps, %lpt : tensor<f64>
      impulse.yield %tot, %lps, %lpt : tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<1>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %acc, %orng, %w : tensor<1x4xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @mh_region_bare
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[MEAN:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x2xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK: %[[RNG1:.+]], %[[S:.+]] = impulse.random %[[RNG]], %[[MEAN]], %[[SD]] {rng_distribution = #impulse<rng_distribution NORMAL>}
// CHECK: %[[DS:.+]] = arith.subf %[[S]], %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPS:.+]] = arith.mulf %[[DS]], %[[DS]] : tensor<f64>
// CHECK-NEXT: %[[DT:.+]] = arith.subf %{{.+}}, %[[S]] : tensor<f64>
// CHECK-NEXT: %[[LPT:.+]] = arith.mulf %[[DT]], %[[DT]] : tensor<f64>
// CHECK-NEXT: %[[TOT:.+]] = arith.addf %[[LPS]], %[[LPT]] : tensor<f64>
// CHECK-NEXT: %[[OSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[SOLD:.+]] = impulse.reshape %[[OSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[DSO:.+]] = arith.subf %[[SOLD]], %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPSO:.+]] = arith.mulf %[[DSO]], %[[DSO]] : tensor<f64>
// CHECK-NEXT: %[[CORR:.+]] = arith.subf %[[LPS]], %[[LPSO]] : tensor<f64>
// CHECK-NEXT: %[[WDIFF:.+]] = arith.subf %[[TOT]], %[[W]] : tensor<f64>
// CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[WDIFF]], %[[CORR]] : tensor<f64>
// CHECK-NEXT: %{{.+}}, %[[U:.+]] = impulse.random %[[RNG1]], %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>}
// CHECK-NEXT: %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
// CHECK-NEXT: %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[ALPHA]] : tensor<f64>
// CHECK: %{{.+}} = arith.select %[[ACC]], %[[TOT]], %[[W]] : tensor<i1>, tensor<f64>

// CHECK-LABEL: func.func @mh_region_tail
// CHECK-SAME: (%[[RNG:[^:]+]]: tensor<2xui64>, %[[MEAN:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>, %[[TR:[^:]+]]: tensor<1x4xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK: %[[DS:.+]] = arith.subf %{{.+}}, %[[MEAN]] : tensor<f64>
// CHECK-NEXT: %[[LPS:.+]] = arith.mulf %[[DS]], %[[DS]] : tensor<f64>
// CHECK: %[[TOT:.+]] = arith.addf %[[LPS]], %{{.+}} : tensor<f64>
// CHECK: %[[OSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>, strides = array<i64: 1, 1>} : (tensor<1x4xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[SCOLD:.+]] = impulse.reshape %[[OSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CORR:.+]] = arith.subf %[[LPS]], %[[SCOLD]] : tensor<f64>
// CHECK-NEXT: %[[WDIFF:.+]] = arith.subf %[[TOT]], %[[W]] : tensor<f64>
// CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[WDIFF]], %[[CORR]] : tensor<f64>
// CHECK-NEXT: %{{.+}}, %[[U:.+]] = impulse.random %{{.+}}, %{{.+}}, %{{.+}} {rng_distribution = #impulse<rng_distribution UNIFORM>}
// CHECK-NEXT: %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.cmpf olt, %[[LOGU]], %[[ALPHA]] : tensor<f64>
