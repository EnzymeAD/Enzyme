// RUN: %eopt %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @svi_auto_normal
  func.func @svi_auto_normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<10x1xf64>, tensor<10xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %lr = arith.constant dense<0.05> : tensor<f64>
    // CHECK: impulse.infer @test
    // CHECK-SAME: vi_config = <guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>
    // CHECK-SAME: state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"]
    %res:11 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      init_step_size = %lr
      <{ vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM>,
        name = "svi", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 10,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<10x1xf64>, tensor<10xf64>, tensor<10xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<10x1xf64>, tensor<10xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @svi_auto_mvn_full_config
  func.func @svi_auto_mvn_full_config(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<4x1xf64>, tensor<4xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %lr = arith.constant dense<0.01> : tensor<f64>
    // CHECK: impulse.infer @test
    // CHECK-SAME: vi_config = <guide_kind = AUTO_MULTIVARIATE_NORMAL, optimizer_kind = ADAM, num_particles = 8, init_scale = 2.500000e-01 : f64, adam_b1 = 9.000000e-01 : f64, adam_b2 = {{0.998999[0-9]+|9.990000e-01}} : f64, adam_eps = 1.000000e-08 : f64>
    // CHECK-SAME: state_schema = ["loc", "scale_tril_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale_tril", "opt_v_scale_tril", "opt_step_count"]
    %res:11 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      init_step_size = %lr
      <{ vi_config = #impulse.vi_config<guide_kind = AUTO_MULTIVARIATE_NORMAL, optimizer_kind = ADAM, num_particles = 8, init_scale = 0.25 : f64, adam_b1 = 0.9 : f64, adam_b2 = 0.999 : f64, adam_eps = 1.0e-8 : f64>,
        name = "svi_mvn", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 4,
        state_schema = ["loc", "scale_tril_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale_tril", "opt_v_scale_tril", "opt_step_count"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<4x1xf64>, tensor<4xf64>, tensor<4xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<4x1xf64>, tensor<4xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @svi_sgd
  func.func @svi_sgd(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<3x1xf64>, tensor<3xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %lr = arith.constant dense<0.1> : tensor<f64>
    // CHECK: impulse.infer @test
    // CHECK-SAME: vi_config = <guide_kind = AUTO_NORMAL, optimizer_kind = SGD>
    %res:11 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      init_step_size = %lr
      <{ vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = SGD>,
        name = "svi_sgd", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 3,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<3x1xf64>, tensor<3xf64>, tensor<3xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<3x1xf64>, tensor<3xf64>, tensor<2xui64>
  }
}
