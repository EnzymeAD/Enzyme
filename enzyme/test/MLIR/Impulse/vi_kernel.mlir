// RUN: %eopt --expand-impulse %s | FileCheck %s

// NumPyro SVI rng split order, for parity: state, step, then the guide site.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @svi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<3x1xf64>, tensor<3xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %lr = arith.constant dense<0.1> : tensor<f64>
    %res:11 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace init_step_size = %lr
      { vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = SGD>,
        name = "svi", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 3,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<3x1xf64>, tensor<3xf64>, tensor<3xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<3x1xf64>, tensor<3xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @svi

// CHECK: %[[INIT:.+]]:3 = impulse.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)

// CHECK: %[[LOOP:.+]]:10 = impulse.for(%{{.*}}) to(%{{.*}}) step(%{{.*}}) iter_args(%{{.*}}, %{{.*}}, %[[INIT]]#0,

// CHECK: %[[SPLIT_STATE:.+]]:2 = impulse.randomSplit %{{.*}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[SPLIT_STEP:.+]]:2 = impulse.randomSplit %[[SPLIT_STATE]]#1 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[SPLIT_SITE:.+]]:2 = impulse.randomSplit %[[SPLIT_STEP]]#1 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)

// CHECK-NEXT: %{{.*}}, %[[EPS:.+]] = impulse.random %[[SPLIT_SITE]]#1, %{{.*}}, %{{.*}} {rng_distribution = #impulse.rng_distribution<NORMAL>}

// CHECK: %[[AD:.+]]:4 = enzyme.autodiff_region(%{{.*}}, %{{.*}}, %{{.*}}) {

// CHECK: func.call @test.generate(%{{.*}}, %[[SPLIT_STEP]]#0, %{{.*}}, %{{.*}})

// CHECK: ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]

// CHECK: "enzyme.broadcast"
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.addf

// CHECK-LABEL: func.func @test.generate
