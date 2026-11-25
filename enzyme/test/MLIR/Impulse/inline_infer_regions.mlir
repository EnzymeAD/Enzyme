// RUN: %eopt --inline-infer-regions --split-input-file %s | FileCheck %s

// impulse.infer becomes impulse.infer_region with its callee inlined.

module {
  func.func private @model_logpdf(%position : tensor<8xf64>,
                                  %B : tensor<8xf64>) -> tensor<f64> {
    %scaled = arith.mulf %position, %B : tensor<8xf64>
    %sq = impulse.dot %scaled, %scaled <{lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %neg_half = arith.constant dense<-5.000000e-01> : tensor<f64>
    %result = arith.mulf %neg_half, %sq : tensor<f64>
    return %result : tensor<f64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK: impulse.infer_region(%[[RNG:[0-9a-z_]+]], %[[B:[0-9a-z_]+]])
  // CHECK-SAME: logpdf_fn = @model_logpdf
  // CHECK-SAME: sampling {
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[POS:.+]]: tensor<8xf64>):
  // CHECK-NEXT: %[[SCALED:.+]] = arith.mulf %[[POS]], %[[B]] : tensor<8xf64>
  // CHECK-NEXT: %[[SQ:.+]] = impulse.dot %[[SCALED]], %[[SCALED]]
  // CHECK: impulse.yield %{{.+}} : tensor<f64>
  // CHECK-NEXT: } <selection = [], all_addresses = [], nuts_config = <max_tree_depth = 3, max_delta_energy = 1.000000e+03 : f64, adapt_step_size = false, adapt_mass_matrix = false>> : (tensor<2xui64>, tensor<8xf64>, tensor<8xf64>, tensor<f64>, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  // CHECK-NOT: impulse.infer(
  func.func @main(%rng : tensor<2xui64>,
                  %B : tensor<8xf64>)
      -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<0.0> : tensor<1x8xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %imm = arith.constant dense<1.0> : tensor<8xf64>
    %res:4 = "impulse.infer"(%rng, %B, %imm, %step_size, %init_pos) {
      logpdf_fn = @model_logpdf,
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [],
      all_addresses = [],
      num_warmup = 0,
      num_samples = 1,
      operandSegmentSizes = array<i32: 2, 0, 0, 0, 1, 1, 1, 0>
    } : (tensor<2xui64>, tensor<8xf64>, tensor<8xf64>, tensor<f64>, tensor<1x8xf64>)
        -> (tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x8xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// -----

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
    %a:2 = impulse.sample @draw(%rng, %m) <{logpdf_fn = @lp, symbol = #impulse.symbol<1>, name = "a"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %z:2 = impulse.sample(%a#0, %logit) <{dist = #impulse.dist<BERNOULLI>, symbol = #impulse.symbol<2>, name = "z"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %half = arith.constant dense<5.000000e-01> : tensor<f64>
    %p = arith.cmpf ogt, %z#1, %half : tensor<f64>
    %if:2 = impulse.if (%p) ({
      %yo:2 = impulse.sample @draw(%z#0, %m) <{logpdf_fn = @lp, symbol = #impulse.symbol<3>, name = "yo"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %yo#0, %yo#1 : tensor<2xui64>, tensor<f64>
    }, {
      %yi:2 = impulse.sample @draw(%z#0, %a#1) <{logpdf_fn = @lp, symbol = #impulse.symbol<4>, name = "yi"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %yi#0, %yi#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @gibbs(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %trace : tensor<1x4xf64>, %obs : tensor<1x2xf64>, %w : tensor<f64>) -> tensor<1x4xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit) given %trace constrained_by %obs state = [%w]
        <{ gibbs_config = #impulse.gibbs_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          constrained_addresses = [[#impulse.symbol<3>], [#impulse.symbol<4>]],
          state_schema = ["weight"], num_samples = 1 : i64 }>
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>, tensor<1x2xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x4xf64>
  }

  func.func @mh(%rng : tensor<2xui64>, %m : tensor<f64>, %logit : tensor<f64>, %trace : tensor<1x4xf64>, %w : tensor<f64>) -> tensor<1x4xf64> {
    %res:5 = impulse.infer @model(%rng, %m, %logit) given %trace state = [%w]
        <{ mh_config = #impulse.mh_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          state_schema = ["weight"], num_samples = 1 : i64 }>
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0 : tensor<1x4xf64>
  }
}

// CHECK-LABEL: func.func @gibbs
// CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = impulse.infer_region @model(%{{.+}}) given %{{.+}} constrained_by %{{.+}} state = [%{{.+}}] sampling {
// CHECK: %[[A:.+]]:2 = impulse.sample_region(%{{.+}}) sampling {
// CHECK: } <logpdf_fn = @lp, address = [#impulse.symbol<1>]>
// CHECK: %[[Z:.+]]:2 = impulse.sample_region(%[[A]]#0, %{{.+}}) sampling {
// CHECK: } <address = [#impulse.symbol<2>], support = <REAL>, dist = <BERNOULLI>>
// CHECK-NEXT: %[[P:.+]] = arith.cmpf ogt, %[[Z]]#1, %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[IF:.+]]:2 = impulse.if(%[[P]]) ({
// CHECK-NEXT: %[[YO:.+]]:2 = impulse.sample_region(%[[Z]]#0, %{{.+}}) sampling {
// CHECK: } <logpdf_fn = @lp, address = [#impulse.symbol<3>]>
// CHECK-NEXT: impulse.yield %[[YO]]#0, %[[YO]]#1 : tensor<2xui64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[YI:.+]]:2 = impulse.sample_region(%[[Z]]#0, %[[A]]#1) sampling {
// CHECK: } <logpdf_fn = @lp, address = [#impulse.symbol<4>]>
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
