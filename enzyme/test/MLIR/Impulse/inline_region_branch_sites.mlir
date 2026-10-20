// RUN: %eopt --inline-infer-regions %s | FileCheck %s

module {
  func.func private @draw(%rng : tensor<2xui64>, %m : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %rng, %m : tensor<2xui64>, tensor<f64>
  }
  func.func private @lp(%x : tensor<f64>, %m : tensor<f64>) -> tensor<f64> {
    %d = arith.subf %x, %m : tensor<f64>
    %n = arith.mulf %d, %d : tensor<f64>
    return %n : tensor<f64>
  }

  func.func @model(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @draw(%rng, %m) {logpdf = @lp, symbol = #impulse.symbol<1>, name = "a"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z:2 = impulse.sample(%a#0, %logit) {dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<2>, name = "z"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = arith.constant dense<5.000000e-01> : tensor<f64>
    %p = arith.cmpf ogt, %z#1, %half : tensor<f64>
    %if:2 = impulse.if (%p) ({
      %yo:2 = impulse.sample @draw(%z#0, %m) {logpdf = @lp, symbol = #impulse.symbol<3>, name = "yo"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %yo#0, %yo#1 : tensor<2xui64>, tensor<f64>
    }, {
      %yi:2 = impulse.sample @draw(%z#0, %a#1) {logpdf = @lp, symbol = #impulse.symbol<4>, name = "yi"} : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %yi#0, %yi#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @gibbs(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %trace : tensor<1x4xf64>, %obs : tensor<1x2xf64>, %w : tensor<f64>) -> tensor<1x4xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit) given %trace constrained_by %obs state = [%w]
        { gibbs_config = #impulse.gibbs_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          constrained_addresses = [[#impulse.symbol<3>], [#impulse.symbol<4>]],
          state_schema = ["weight"], num_samples = 1 : i64 }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>, tensor<1x2xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x4xf64>
  }

  func.func @mh(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>) -> tensor<1x4xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit) given %trace state = [%w]
        { mh_config = #impulse.mh_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          state_schema = ["weight"], num_samples = 1 : i64 }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x4xf64>
  }
}

// CHECK-LABEL: func.func @gibbs
// CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = impulse.infer_region @model(%{{.+}}) given %{{.+}} constrained_by %{{.+}} state = [%{{.+}}] sampler {
// CHECK: %[[A:.+]]:2 = impulse.sample_region(%{{.+}}) sampler {
// CHECK: } {address = [#impulse.symbol<1>], fn = @draw, logpdf_fn = @lp, name = "a"}
// CHECK: %[[Z:.+]]:2 = impulse.sample_region(%[[A]]#0, %{{.+}}) sampler {
// CHECK: } {address = [#impulse.symbol<2>], dist = #impulse.dist<BERNOULLI>, name = "z", support = #impulse.support<REAL>}
// CHECK-NEXT: %[[P:.+]] = arith.cmpf ogt, %[[Z]]#1, %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[IF:.+]]:2 = impulse.if(%[[P]]) ({
// CHECK-NEXT: %[[YO:.+]]:2 = impulse.sample_region(%[[Z]]#0, %{{.+}}) sampler {
// CHECK: } {address = [#impulse.symbol<3>], fn = @draw, logpdf_fn = @lp, name = "yo"}
// CHECK-NEXT: impulse.yield %[[YO]]#0, %[[YO]]#1 : tensor<2xui64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[YI:.+]]:2 = impulse.sample_region(%[[Z]]#0, %[[A]]#1) sampler {
// CHECK: } {address = [#impulse.symbol<4>], fn = @draw, logpdf_fn = @lp, name = "yi"}
// CHECK-NEXT: impulse.yield %[[YI]]#0, %[[YI]]#1 : tensor<2xui64>, tensor<f64>
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: impulse.yield %[[IF]]#0, %[[IF]]#1 : tensor<2xui64>, tensor<f64>
// CHECK-NEXT: } logpdf {
// CHECK-NEXT: } {{.*}}gibbs_config = #impulse.gibbs_config

// CHECK-LABEL: func.func @mh
// CHECK-NOT: impulse.infer_region
// CHECK: impulse.infer @model(
// CHECK-NOT: impulse.infer_region
// CHECK: return
