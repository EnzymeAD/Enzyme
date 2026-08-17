// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @expose_both(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>)
      -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:18 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #impulse.nuts_config<max_tree_depth = 5>,
        name = "expose_both", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 4, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>,
            tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>)
    return %res#3, %res#9, %res#10, %res#11, %res#12, %res#13, %res#14, %res#15, %res#16, %res#17
      : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
  }

  func.func @resume_offset(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>,
      %off : tensor<i64>,
      %da0 : tensor<f64>, %da1 : tensor<f64>, %da2 : tensor<f64>, %da3 : tensor<i64>, %da4 : tensor<f64>,
      %wm : tensor<1xf64>, %wm2 : tensor<1xf64>, %wn : tensor<i64>, %widx : tensor<i64>)
      -> (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:18 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      warmup_offset = %off
      adaptation_state_in = %da0, %da1, %da2, %da3, %da4, %wm, %wm2, %wn, %widx
      { nuts_config = #impulse.nuts_config<max_tree_depth = 5>,
        name = "resume_offset", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]],
        num_warmup = 4, total_warmup = 100, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>,
         tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>,
         tensor<1x1xf64>, tensor<f64>, tensor<i64>)
        -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>,
            tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>)
    return %res#3, %res#9, %res#10, %res#11, %res#12, %res#13, %res#14, %res#15, %res#16, %res#17
      : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
  }
}

// CHECK-LABEL: func.func @expose_both
// CHECK: %[[WU:.+]]:16 = impulse.for
// CHECK: %[[SAMP:.+]]:7 = impulse.for(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%[[WU]]#0, %[[WU]]#1, %[[WU]]#2, %[[WU]]#3,
// CHECK: return %[[SAMP]]#3, %[[WU]]#7, %[[WU]]#8, %[[WU]]#9, %[[WU]]#10, %[[WU]]#11, %[[WU]]#12, %[[WU]]#13, %[[WU]]#14, %[[WU]]#15 : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>

// CHECK-LABEL: func.func @resume_offset
// CHECK-DAG: %[[C99:.+]] = arith.constant dense<99> : tensor<i64>
// CHECK: %[[RWU:.+]]:16 = impulse.for(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12 :
// CHECK: arith.addi %arg3, %{{.+}} : tensor<i64>
// CHECK: return %{{.+}}#3, %[[RWU]]#7, %[[RWU]]#8, %[[RWU]]#9, %[[RWU]]#10, %[[RWU]]#11, %[[RWU]]#12, %[[RWU]]#13, %[[RWU]]#14, %[[RWU]]#15 : tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
