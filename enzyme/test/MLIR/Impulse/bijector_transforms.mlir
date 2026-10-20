// RUN: %eopt --expand-impulse --canonicalize --split-input-file %s | FileCheck %s

// z = logit(x); logdet = -softplus(z) - softplus(-z).
module {
  func.func private @d(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %p : tensor<f64>) -> tensor<f64>
  func.func @test(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<UNIT_INTERVAL> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @nuts(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.5]]> : tensor<1x1xf64>
    %imm = arith.constant dense<[[1.0]]> : tensor<1x1xf64>
    %ss = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @test(%rng, %p) given %init_trace
      init_inverse_mass_matrix = %imm init_step_size = %ss
      <{ nuts_config = #impulse.nuts_config<max_tree_depth = 1>, name = "nuts", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
      : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @nuts
// CHECK: %[[X:.+]] = impulse.slice %{{.+}} {{.*}} : (tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[OMX:.+]] = arith.subf %{{.+}}, %[[X]] : tensor<1xf64>
// CHECK-NEXT: %[[LX:.+]] = math.log %[[X]] : tensor<1xf64>
// CHECK-NEXT: %[[L1MX:.+]] = math.log %[[OMX]] : tensor<1xf64>
// CHECK-NEXT: %[[Z:.+]] = arith.subf %[[LX]], %[[L1MX]] : tensor<1xf64>
// CHECK: %[[SPN:.+]] = impulse.log_add_exp %{{.+}}, %{{.+}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SPP:.+]] = impulse.log_add_exp %{{.+}}, %{{.+}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[LSP:.+]] = arith.negf %[[SPP]] : tensor<1xf64>
// CHECK-NEXT: %[[LP:.+]] = arith.subf %[[LSP]], %[[SPN]] : tensor<1xf64>
// CHECK-NEXT: %[[SUM:.+]] = impulse.dot %[[LP]], %{{.+}}
// CHECK: enzyme.autodiff_region
// CHECK: impulse.logistic %{{.+}} : (tensor<1xf64>) -> tensor<1xf64>

// -----

// z = logit((x-a)/(b-a)); logdet adds the constant n*log(b-a).
module {
  func.func private @d(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %p : tensor<f64>) -> tensor<f64>
  func.func @test(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<INTERVAL lower = 2.000000e+00 : f64 upper = 5.000000e+00 : f64> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @nuts(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.3]]> : tensor<1x1xf64>
    %imm = arith.constant dense<[[1.0]]> : tensor<1x1xf64>
    %ss = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @test(%rng, %p) given %init_trace
      init_inverse_mass_matrix = %imm init_step_size = %ss
      <{ nuts_config = #impulse.nuts_config<max_tree_depth = 1>, name = "nuts", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
      : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @nuts
// CHECK: %[[X:.+]] = impulse.slice %{{.+}} {{.*}} : (tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SH:.+]] = arith.subf %[[X]], %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[NORM:.+]] = arith.divf %[[SH]], %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[OMN:.+]] = arith.subf %{{.+}}, %[[NORM]] : tensor<1xf64>
// CHECK-NEXT: %[[LN:.+]] = math.log %[[NORM]] : tensor<1xf64>
// CHECK-NEXT: %[[L1MN:.+]] = math.log %[[OMN]] : tensor<1xf64>
// CHECK-NEXT: %[[Z:.+]] = arith.subf %[[LN]], %[[L1MN]] : tensor<1xf64>
// CHECK: %[[SP1:.+]] = impulse.log_add_exp %{{.+}}, %{{.+}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK: %[[SP2:.+]] = impulse.log_add_exp %{{.+}}, %{{.+}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK: %[[LP:.+]] = arith.subf %{{.+}}, %{{.+}} : tensor<1xf64>
// CHECK: %[[SUM:.+]] = impulse.dot %[[LP]], %{{.+}}
// CHECK: arith.addf %[[SUM]], %{{.+}} : tensor<f64>
// CHECK: enzyme.autodiff_region
// CHECK: %[[SIG:.+]] = impulse.logistic %{{.+}} : (tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SC:.+]] = arith.mulf %[[SIG]], %{{.+}} : tensor<1xf64>

// -----

module {
  func.func private @d(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %p : tensor<f64>) -> tensor<f64>
  func.func @test(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<GREATER_THAN lower = 1.000000e+00 : f64> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @nuts(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[2.0]]> : tensor<1x1xf64>
    %imm = arith.constant dense<[[1.0]]> : tensor<1x1xf64>
    %ss = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @test(%rng, %p) given %init_trace
      init_inverse_mass_matrix = %imm init_step_size = %ss
      <{ nuts_config = #impulse.nuts_config<max_tree_depth = 1>, name = "nuts", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
      : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @nuts
// CHECK: %[[X:.+]] = impulse.slice %{{.+}} {{.*}} : (tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SH:.+]] = arith.subf %[[X]], %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[Z:.+]] = math.log %[[SH]] : tensor<1xf64>
// CHECK: impulse.dot %{{.+}}, %{{.+}} {{.*}} -> tensor<f64>
// CHECK: enzyme.autodiff_region
// CHECK: %[[E:.+]] = math.exp %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[C:.+]] = arith.addf %[[E]], %{{.+}} : tensor<1xf64>

// -----

module {
  func.func private @d(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %p : tensor<f64>) -> tensor<f64>
  func.func @test(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<LESS_THAN upper = 4.000000e+00 : f64> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @nuts(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[2.0]]> : tensor<1x1xf64>
    %imm = arith.constant dense<[[1.0]]> : tensor<1x1xf64>
    %ss = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @test(%rng, %p) given %init_trace
      init_inverse_mass_matrix = %imm init_step_size = %ss
      <{ nuts_config = #impulse.nuts_config<max_tree_depth = 1>, name = "nuts", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 1 }>
      : (tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @nuts
// CHECK: %[[X:.+]] = impulse.slice %{{.+}} {{.*}} : (tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SH:.+]] = arith.subf %{{.+}}, %[[X]] : tensor<1xf64>
// CHECK-NEXT: %[[Z:.+]] = math.log %[[SH]] : tensor<1xf64>
// CHECK: impulse.dot %{{.+}}, %{{.+}} {{.*}} -> tensor<f64>
// CHECK: enzyme.autodiff_region
// CHECK: %[[E:.+]] = math.exp %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[C:.+]] = arith.subf %{{.+}}, %[[E]] : tensor<1xf64>
