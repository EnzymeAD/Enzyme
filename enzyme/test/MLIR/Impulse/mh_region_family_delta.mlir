// RUN: %eopt --expand-impulse %s | FileCheck %s

// t1..t4 yield slices of one vector, so their delta is one dot reduction.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  func.func @mh_region_family_delta(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %tvals : tensor<4xf64>, %trace : tensor<1x10xf64>, %weight : tensor<f64>) -> (tensor<1x10xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %tr, %acc, %ld, %orng, %w = impulse.infer_region @test(%rng, %mean, %stddev) given %trace state = [%weight] sampler {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>, %sd: tensor<f64>):
      %s:2 = impulse.sample_region(%r, %m, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<1>], fn = @normal, logpdf_fn = @logpdf, name = "s"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t1:2 = impulse.sample_region(%s#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<2>], fn = @normal, logpdf_fn = @logpdf, name = "t1"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t2:2 = impulse.sample_region(%t1#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<3>], fn = @normal, logpdf_fn = @logpdf, name = "t2"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t3:2 = impulse.sample_region(%t2#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<4>], fn = @normal, logpdf_fn = @logpdf, name = "t3"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      %t4:2 = impulse.sample_region(%t3#0, %s#1, %sd) sampler {
      ^bb0(%a0: tensor<2xui64>, %a1: tensor<f64>, %a2: tensor<f64>):
        %o, %x = impulse.random %a0, %a1, %a2 {rng_distribution = #impulse<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
        impulse.yield %o, %x : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {address = [#impulse.symbol<5>], fn = @normal, logpdf_fn = @logpdf, name = "t4"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %t4#0, %t4#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%sv: tensor<f64>):
      %ds = arith.subf %sv, %mean : tensor<f64>
      %lps = arith.mulf %ds, %ds : tensor<f64>
      %sx = tensor.extract %sv[] : tensor<f64>
      %svec = tensor.splat %sx : tensor<4xf64>
      %dv = arith.subf %tvals, %svec : tensor<4xf64>
      %v = arith.mulf %dv, %dv : tensor<4xf64>
      %v0 = impulse.slice %v {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
      %y0 = impulse.reshape %v0 : (tensor<1xf64>) -> tensor<f64>
      %v1 = impulse.slice %v {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
      %y1 = impulse.reshape %v1 : (tensor<1xf64>) -> tensor<f64>
      %v2 = impulse.slice %v {limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
      %y2 = impulse.reshape %v2 : (tensor<1xf64>) -> tensor<f64>
      %v3 = impulse.slice %v {limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<4xf64>) -> tensor<1xf64>
      %y3 = impulse.reshape %v3 : (tensor<1xf64>) -> tensor<f64>
      %a = arith.addf %lps, %y0 : tensor<f64>
      %b = arith.addf %a, %y1 : tensor<f64>
      %c = arith.addf %b, %y2 : tensor<f64>
      %tot = arith.addf %c, %y3 : tensor<f64>
      impulse.yield %tot, %lps, %y0, %y1, %y2, %y3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
    } {all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>], [#impulse.symbol<5>]], mh_config = #impulse.mh_config, num_samples = 1 : i64, selection = [[#impulse.symbol<1>]], state_schema = ["weight"]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x10xf64>) -> (tensor<1x10xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %acc, %orng, %w : tensor<1x10xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @mh_region_family_delta
// CHECK-DAG: %[[ONES:.+]] = arith.constant dense<1.000000e+00> : tensor<4xf64>
// CHECK: %[[V:.+]] = arith.mulf %{{.+}}, %{{.+}} : tensor<4xf64>
// CHECK: %[[OC:.+]] = impulse.slice %[[TRACE:.+]] {limit_indices = array<i64: 1, 6>, start_indices = array<i64: 0, 5>, strides = array<i64: 1, 1>} : (tensor<1x10xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[OCS:.+]] = impulse.reshape %[[OC]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CORR:.+]] = arith.subf %[[NS:.+]], %[[OCS]] : tensor<f64>
// CHECK-NEXT: %[[OS:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 6>, start_indices = array<i64: 0, 5>, strides = array<i64: 1, 1>} : (tensor<1x10xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[OSS:.+]] = impulse.reshape %[[OS]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[DS:.+]] = arith.subf %[[NS]], %[[OSS]] : tensor<f64>
// CHECK-NEXT: %[[OLD:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 10>, start_indices = array<i64: 0, 6>, strides = array<i64: 1, 1>} : (tensor<1x10xf64>) -> tensor<1x4xf64>
// CHECK-NEXT: %[[OLDV:.+]] = impulse.reshape %[[OLD]] : (tensor<1x4xf64>) -> tensor<4xf64>
// CHECK-NEXT: %[[D:.+]] = arith.subf %[[V]], %[[OLDV]] : tensor<4xf64>
// CHECK-NEXT: %[[FS:.+]] = impulse.dot %[[D]], %[[ONES]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
// CHECK-NEXT: %[[DELTA:.+]] = arith.addf %[[DS]], %[[FS]] : tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.subf %[[DELTA]], %[[CORR]] : tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[DELTA]] : tensor<f64>
