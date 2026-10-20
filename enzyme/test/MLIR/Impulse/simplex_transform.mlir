// RUN: %eopt --expand-impulse --canonicalize %s | FileCheck %s

// K = 3 simplex: the trace is 3 wide, the unconstrained position K - 1 = 2.

module {
  func.func private @dirichlet(%rng : tensor<2xui64>, %alpha : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
  func.func private @logpdf(%x : tensor<3xf64>, %alpha : tensor<3xf64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %alpha : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %s:2 = impulse.sample @dirichlet(%rng, %alpha) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<SIMPLEX event_size = 3 : i64> }> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<3xf64>
  }

  // CHECK-LABEL: func.func @nuts
  func.func @nuts(%rng : tensor<2xui64>, %alpha : tensor<3xf64>) -> (tensor<1x3xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.2, 0.3, 0.5]]> : tensor<1x3xf64>
    %inverse_mass_matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %res:4 = impulse.infer @test(%rng, %alpha) given %init_trace
      init_inverse_mass_matrix = %inverse_mass_matrix
      init_step_size = %step_size
      <{ nuts_config = #impulse.nuts_config<max_tree_depth = 2>, name = "nuts", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
      : (tensor<2xui64>, tensor<3xf64>, tensor<1x3xf64>, tensor<2x2xf64>, tensor<f64>) -> (tensor<1x3xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x3xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK: %[[GATHER:.+]] = impulse.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[U1D:.+]] = impulse.reshape %[[GATHER]] : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[USLICE:.+]] = impulse.slice %[[U1D]] <start_indices = [0], limit_indices = [3], strides = [1]> : (tensor<3xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[YMAT:.+]] = impulse.reshape %[[USLICE]] : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[YCROP:.+]] = impulse.slice %[[YMAT]] <start_indices = [0, 0], limit_indices = [1, 2], strides = [1, 1]> : (tensor<1x3xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[CUM:.+]] = impulse.dot %[[YCROP]], %{{.+}} {{.*}} : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[Z1M:.+]] = arith.subf %{{.+}}, %[[CUM]] : tensor<1x2xf64>
// CHECK-NEXT: %[[RATIO:.+]] = arith.divf %[[YCROP]], %[[Z1M]] : tensor<1x2xf64>
// CHECK-NEXT: %[[LOGR:.+]] = math.log %[[RATIO]] : tensor<1x2xf64>
// CHECK-NEXT: %[[X:.+]] = arith.addf %[[LOGR]], %{{.+}} : tensor<1x2xf64>
// CHECK-NEXT: %[[XFLAT:.+]] = impulse.reshape %[[X]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[UPOS:.+]] = impulse.reshape %[[XFLAT]] : (tensor<2xf64>) -> tensor<1x2xf64>

// CHECK: enzyme.autodiff_region(%[[UPOS]], %{{.+}})
// CHECK: %[[XSHC:.+]] = arith.subf %{{.+}}, %{{.+}} : tensor<1x2xf64>
// CHECK-NEXT: %[[ZC:.+]] = impulse.logistic %[[XSHC]] : (tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SPC:.+]] = impulse.log_add_exp %[[XSHC]], %{{.+}} : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[LOG1MZC:.+]] = arith.negf %[[SPC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[LOGPC:.+]] = impulse.dot %[[LOG1MZC]], %{{.+}} {{.*}} : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[PC:.+]] = math.exp %[[LOGPC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[YBODY:.+]] = arith.mulf %[[ZC]], %[[PC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[TOTAL:.+]] = impulse.dot %[[LOG1MZC]], %{{.+}} {{.*}} : (tensor<1x2xf64>, tensor<2x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[REM:.+]] = math.exp %[[TOTAL]] : tensor<1x1xf64>
// CHECK-NEXT: %[[YBUF0:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[YBODY]], %{{.+}}, %{{.+}} : (tensor<1x3xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[YBUF1:.+]] = impulse.dynamic_update_slice %[[YBUF0]], %[[REM]], %{{.+}}, %{{.+}} : (tensor<1x3xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>

// CHECK: impulse.for
// CHECK-SAME: iter_args({{.*}} : tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<1x3xf64>, tensor<1x2xi1>, tensor<1xf64>)
