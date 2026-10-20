// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @svi_cold(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>)
      -> (tensor<5x1xf64>, tensor<5xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<i64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %lr = arith.constant dense<0.05> : tensor<f64>
    %res:11 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace init_step_size = %lr
      <{ vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
        name = "svi", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 5,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<5x1xf64>, tensor<5xf64>, tensor<5xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3, %res#4, %res#10 : tensor<5x1xf64>, tensor<5xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<i64>
  }

  func.func @svi_resume(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>,
      %loc : tensor<1x1xf64>, %scu : tensor<1x1xf64>, %ml : tensor<1x1xf64>, %vl : tensor<1x1xf64>,
      %ms : tensor<1x1xf64>, %vs : tensor<1x1xf64>, %sc : tensor<i64>)
      -> (tensor<5x1xf64>, tensor<5xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %lr = arith.constant dense<0.05> : tensor<f64>
    %res:11 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      init_step_size = %lr
      state = [%loc, %scu, %ml, %vl, %ms, %vs, %sc]
      <{ vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
        name = "svi", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 5,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<5x1xf64>, tensor<5xf64>, tensor<5xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<5x1xf64>, tensor<5xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @svi_cold
// CHECK: %[[INIT:.+]]:3 = impulse.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[COLDLOOP:.+]]:10 = impulse.for
// CHECK: return %[[COLDLOOP]]#3, %[[COLDLOOP]]#4, %[[COLDLOOP]]#2, %[[COLDLOOP]]#0, %{{.*}} :

// Resume skips the SVI.init split, as NumPyro's initial_optim_state does.
// CHECK-LABEL: func.func @svi_resume
// CHECK-NOT: impulse.randomSplit %arg0
// CHECK: %[[RESUMELOOP:.+]]:10 = impulse.for(%{{.*}}) to(%{{.*}}) step(%{{.*}}) iter_args(%arg3, %arg4, %arg0, %{{.*}}, %{{.*}}, %{{.*}}, %arg5, %arg6, %arg7, %arg8
