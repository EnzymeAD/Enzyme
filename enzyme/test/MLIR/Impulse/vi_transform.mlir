// RUN: %eopt --expand-impulse --canonicalize %s | FileCheck %s

// REAL + POSITIVE sites: multi-site reassembly (one site takes a bypass).

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @exponential(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @exp_logpdf(%x : tensor<f64>, %rate : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %m:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @normal_logpdf, symbol = #impulse.symbol<1>, name="m", support = #impulse.support<REAL> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %s:2 = impulse.sample @exponential(%m#0, %rate) { logpdf = @exp_logpdf, symbol = #impulse.symbol<2>, name="s", support = #impulse.support<POSITIVE> } : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %m#1, %s#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @svi
  func.func @svi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %rate : tensor<f64>) -> (tensor<3x2xf64>, tensor<3xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0, 1.0]]> : tensor<1x2xf64>
    %lr = arith.constant dense<0.1> : tensor<f64>

    %res:11 = impulse.infer @test(%rng, %mean, %stddev, %rate) given %init_trace init_step_size = %lr
      { vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = SGD>,
        name = "svi", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], num_warmup = 0, num_samples = 3,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<3x2xf64>, tensor<3xf64>, tensor<3xf64>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<3x2xf64>, tensor<3xf64>, tensor<2xui64>
  }
}

// CHECK: %{{.+}}, %[[EPS1:.+]] = impulse.random %{{.+}}, %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
// CHECK-NEXT: %[[EPS_A:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[EPS1]], %{{.+}}, %{{.+}} : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SPLIT2:.+]]:2 = impulse.randomSplit %{{.+}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %{{.+}}, %[[EPS2:.+]] = impulse.random %[[SPLIT2]]#1, %{{.+}}, %{{.+}} {rng_distribution = #impulse.rng_distribution<NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
// CHECK-NEXT: %[[EPS:.+]] = impulse.dynamic_update_slice %[[EPS_A]], %[[EPS2]], %{{.+}}, %{{.+}} : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>

// CHECK: %[[AD:.+]]:3 = enzyme.autodiff_region(%{{.+}}, %{{.+}}, %{{.+}}) {
// CHECK-NEXT: ^bb0(%[[LOC:.+]]: tensor<1x2xf64>, %[[SCALE_UNC:.+]]: tensor<1x2xf64>):

// CHECK-NEXT: %[[SU_EXP:.+]] = math.exp %[[SCALE_UNC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[SCALE:.+]] = math.log1p %[[SU_EXP]] : tensor<1x2xf64>
// CHECK-NEXT: %[[SCALE_EPS:.+]] = arith.mulf %[[SCALE]], %[[EPS]] : tensor<1x2xf64>
// CHECK-NEXT: %[[Z_UNC:.+]] = arith.addf %[[LOC]], %[[SCALE_EPS]] : tensor<1x2xf64>

// CHECK-NEXT: %[[Z_1D:.+]] = impulse.reshape %[[Z_UNC]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[Z_REAL:.+]] = impulse.slice %[[Z_1D]] {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[Z_POS:.+]] = impulse.slice %[[Z_1D]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[C_POS:.+]] = math.exp %[[Z_POS]] : tensor<1xf64>

// CHECK-NEXT: %[[REASM1:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[Z_REAL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[REASM2:.+]] = impulse.dynamic_update_slice %[[REASM1]], %[[C_POS]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[Z_C:.+]] = impulse.reshape %[[REASM2]] : (tensor<2xf64>) -> tensor<1x2xf64>

// CHECK: %[[JAC_1D:.+]] = impulse.reshape %[[Z_UNC]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[JAC_SLICE:.+]] = impulse.slice %[[JAC_1D]] {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[JAC_DOT:.+]] = impulse.dot %[[JAC_SLICE]], %{{.+}} {{.*}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[JAC:.+]] = arith.addf %[[JAC_DOT]], %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[LOG_Q:.+]] = arith.subf %{{.+}}, %[[JAC]] : tensor<f64>

// CHECK-NEXT: %[[TR_SLICE:.+]] = impulse.dynamic_slice %[[Z_C]], %{{.+}}, %{{.+}} {slice_sizes = array<i64: 1, 1>} : (tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK: %[[GEN:.+]]:5 = func.call @test.generate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
// CHECK-NEXT: %[[ELBO:.+]] = arith.subf %[[GEN]]#1, %[[LOG_Q]] : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[ELBO]], %[[GEN]]#2 : tensor<f64>, tensor<2xui64>

// CHECK: ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]

// CHECK-LABEL: func.func @test.generate
// CHECK: call @normal_logpdf
// CHECK: call @exp_logpdf
