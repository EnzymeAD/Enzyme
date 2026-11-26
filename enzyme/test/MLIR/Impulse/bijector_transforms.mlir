// RUN: %eopt --expand-impulse --canonicalize --split-input-file %s | FileCheck %s

// HMC, NUTS and VI move constrained sites in unconstrained coordinates.

// z = logit(x); logdet = -softplus(z) - softplus(-z).
module {
  func.func private @d(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %p : tensor<f64>) -> tensor<f64>
  func.func @test(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf_fn = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<UNIT_INTERVAL> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
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
// CHECK: math.log1p %{{.+}} : tensor<1xf64>
// CHECK: math.log1p %{{.+}} : tensor<1xf64>
// CHECK: %[[LSP:.+]] = arith.negf %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[LP:.+]] = arith.subf %[[LSP]], %{{.+}} : tensor<1xf64>
// CHECK-NEXT: %[[SUM:.+]] = impulse.dot %[[LP]], %{{.+}}
// CHECK: enzyme.autodiff_region
// CHECK: impulse.logistic %{{.+}} : (tensor<1xf64>) -> tensor<1xf64>

// -----

// z = logit((x-a)/(b-a)); logdet adds the constant n*log(b-a).
module {
  func.func private @d(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @lp(%x : tensor<f64>, %p : tensor<f64>) -> tensor<f64>
  func.func @test(%rng : tensor<2xui64>, %p : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf_fn = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<INTERVAL, lower = 2.000000e+00 : f64, upper = 5.000000e+00 : f64> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
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
// CHECK: math.log1p %{{.+}} : tensor<1xf64>
// CHECK: math.log1p %{{.+}} : tensor<1xf64>
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
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf_fn = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<GREATER_THAN, lower = 1.000000e+00 : f64> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
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
    %s:2 = impulse.sample @d(%rng, %p) <{ logpdf_fn = @lp, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<LESS_THAN, upper = 4.000000e+00 : f64> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
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

// -----

module {
  func.func private @exponential(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %rate : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @exponential(%rng, %rate) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<POSITIVE> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @exponential(%s#0, %rate) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<2>, name="t", support = #impulse.support<POSITIVE> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @hmc
  func.func @hmc(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[1.0, 1.0]]> : tensor<1x2xf64>

    %inverse_mass_matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %res:4 = impulse.infer @test(%rng, %rate) given %init_trace
      init_inverse_mass_matrix = %inverse_mass_matrix
      init_step_size = %step_size
      <{ hmc_config = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64>, name = "hmc", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], num_warmup = 0, num_samples = 1 }>
      : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<2x2xf64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// CHECK: %[[INIT_TRACE:.+]] = arith.constant dense<1.0{{.*}}> : tensor<1x2xf64>
// CHECK: impulse.dynamic_slice %[[INIT_TRACE]]
// CHECK: impulse.dynamic_update_slice
// CHECK: impulse.dynamic_slice %[[INIT_TRACE]]
// CHECK: %[[EXTRACTED_POS:.+]] = impulse.dynamic_update_slice
// CHECK: %[[FLATTENED:.+]] = impulse.reshape %[[EXTRACTED_POS]] : (tensor<1x2xf64>) -> tensor<2xf64>

// CHECK: %[[SAMPLE1:.+]] = impulse.slice %[[FLATTENED]] <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[LOG1:.+]] = math.log %[[SAMPLE1]] : tensor<1xf64>
// CHECK-NEXT: %[[SAMPLE2:.+]] = impulse.slice %[[FLATTENED]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[LOG2:.+]] = math.log %[[SAMPLE2]] : tensor<1xf64>

// One whole-slice dynamic_update_slice per site, not a per-element scatter.
// CHECK-NEXT: %[[POS1:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[LOG1]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[UNCONSTRAINED_POS:.+]] = impulse.dynamic_update_slice %[[POS1]], %[[LOG2]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[UNCONSTRAINED_POS_2D:.+]] = impulse.reshape %[[UNCONSTRAINED_POS]] : (tensor<2xf64>) -> tensor<1x2xf64>

// The initial generate reads the constrained position.
// CHECK: impulse.dynamic_slice
// CHECK: impulse.dynamic_update_slice
// CHECK: impulse.dynamic_slice
// CHECK: impulse.dynamic_update_slice
// CHECK: %[[INIT_GEN:.+]]:4 = call @test.generate{{.*}}(%{{.+}}, %{{.+}}, %{{.+}})
// CHECK: %[[NEG_WEIGHT:.+]] = arith.negf %[[INIT_GEN]]#1 : tensor<f64>

// CHECK: %[[JAC_POS:.+]] = impulse.reshape %[[UNCONSTRAINED_POS_2D]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[Z1:.+]] = impulse.slice %[[JAC_POS]] <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[SUM1:.+]] = impulse.dot %[[Z1]], %{{.+}} {{.*}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[PARTIAL_SUM:.+]] = arith.addf %[[SUM1]], %{{.+}} : tensor<f64>
// CHECK: %[[Z2:.+]] = impulse.slice %[[JAC_POS]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[SUM2:.+]] = impulse.dot %[[Z2]], %{{.+}} {{.*}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[TOTAL_JACOBIAN:.+]] = arith.addf %[[PARTIAL_SUM]], %[[SUM2]] : tensor<f64>

// CHECK: %[[U0:.+]] = arith.subf %[[NEG_WEIGHT]], %[[TOTAL_JACOBIAN]] : tensor<f64>

// CHECK: %[[AUTODIFF:.+]] = enzyme.autodiff_region(%[[UNCONSTRAINED_POS_2D]], %{{.+}}) {
// CHECK: ^bb0(%[[ARG:.+]]: tensor<1x2xf64>):
// CHECK: %[[ARG_1D:.+]] = impulse.reshape %[[ARG]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[Z_SAMPLE1:.+]] = impulse.slice %[[ARG_1D]] <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[EXP1:.+]] = math.exp %[[Z_SAMPLE1]] : tensor<1xf64>
// CHECK-NEXT: %[[Z_SAMPLE2:.+]] = impulse.slice %[[ARG_1D]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[EXP2:.+]] = math.exp %[[Z_SAMPLE2]] : tensor<1xf64>

// CHECK-NEXT: %[[C_POS1:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[EXP1]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[CONSTRAINED_POS:.+]] = impulse.dynamic_update_slice %[[C_POS1]], %[[EXP2]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: impulse.reshape %[[CONSTRAINED_POS]] : (tensor<2xf64>) -> tensor<1x2xf64>
// CHECK: %[[GEN_RES:.+]]:4 = func.call @test.generate{{.*}}
// CHECK: %[[NEG_GEN_WEIGHT:.+]] = arith.negf %[[GEN_RES]]#1 : tensor<f64>

// CHECK: %[[ARG_1D_2:.+]] = impulse.reshape %[[ARG]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[ARG_Z1:.+]] = impulse.slice %[[ARG_1D_2]] <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[ARG_DOT1:.+]] = impulse.dot %[[ARG_Z1]], %{{.+}}
// CHECK: %[[ARG_PARTIAL:.+]] = arith.addf %[[ARG_DOT1]], %{{.+}} : tensor<f64>
// CHECK: %[[ARG_Z2:.+]] = impulse.slice %[[ARG_1D_2]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[ARG_DOT2:.+]] = impulse.dot %[[ARG_Z2]], %{{.+}}
// CHECK: %[[JAC_SUM:.+]] = arith.addf %[[ARG_PARTIAL]], %[[ARG_DOT2]] : tensor<f64>
// CHECK: %[[ADJUSTED_U:.+]] = arith.subf %[[NEG_GEN_WEIGHT]], %[[JAC_SUM]] : tensor<f64>
// CHECK: enzyme.yield %[[ADJUSTED_U]], %{{.+}} : tensor<f64>, tensor<2xui64>
// CHECK: }

// CHECK: impulse.for

// CHECK: enzyme.autodiff_region
// CHECK: math.exp
// CHECK: math.exp
// CHECK: call @test.generate
// CHECK: enzyme.yield
// CHECK: }

// CHECK: math.exp
// CHECK: arith.minimumf
// CHECK: impulse.random
// CHECK: arith.cmpf olt
// CHECK: %[[FINAL_SELECT:.+]] = impulse.select
// CHECK: %[[FINAL_1D:.+]] = impulse.reshape %[[FINAL_SELECT]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK: %[[FINAL_SAMPLE1:.+]] = impulse.slice %[[FINAL_1D]] <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[FINAL_EXP1:.+]] = math.exp %[[FINAL_SAMPLE1]] : tensor<1xf64>
// CHECK: %[[FINAL_SAMPLE2:.+]] = impulse.slice %[[FINAL_1D]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK: %[[FINAL_EXP2:.+]] = math.exp %[[FINAL_SAMPLE2]] : tensor<1xf64>

// CHECK: impulse.dynamic_update_slice
// CHECK: return

// CHECK-LABEL: func.func @test.generate
// CHECK: impulse.slice %{{.+}} <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]>
// CHECK: impulse.reshape
// CHECK: call @logpdf
// CHECK: impulse.slice %{{.+}} <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]>
// CHECK: impulse.reshape
// CHECK: call @logpdf
// CHECK: return

// -----

// K = 3 simplex: the trace is 3 wide, the unconstrained position K - 1 = 2.

module {
  func.func private @dirichlet(%rng : tensor<2xui64>, %alpha : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
  func.func private @logpdf(%x : tensor<3xf64>, %alpha : tensor<3xf64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %alpha : tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %s:2 = impulse.sample @dirichlet(%rng, %alpha) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s", support = #impulse.support<SIMPLEX, event_size = 3 : i64> }> : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
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
// CHECK-NEXT: %{{.+}} = arith.maximumf %[[XSHC]], %{{.+}} : tensor<1x2xf64>
// CHECK: math.log1p %{{.+}} : tensor<1x2xf64>
// CHECK: %[[LOG1MZC:.+]] = arith.negf %{{.+}} : tensor<1x2xf64>
// CHECK-NEXT: %[[LOGPC:.+]] = impulse.dot %[[LOG1MZC]], %{{.+}} {{.*}} : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[PC:.+]] = math.exp %[[LOGPC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[YBODY:.+]] = arith.mulf %[[ZC]], %[[PC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[TOTAL:.+]] = impulse.dot %[[LOG1MZC]], %{{.+}} {{.*}} : (tensor<1x2xf64>, tensor<2x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[REM:.+]] = math.exp %[[TOTAL]] : tensor<1x1xf64>
// CHECK-NEXT: %[[YBUF0:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[YBODY]], %{{.+}}, %{{.+}} : (tensor<1x3xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[YBUF1:.+]] = impulse.dynamic_update_slice %[[YBUF0]], %[[REM]], %{{.+}}, %{{.+}} : (tensor<1x3xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>

// CHECK: impulse.for
// CHECK-SAME: iter_args({{.*}} : tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<1x3xf64>, tensor<1x2xi1>, tensor<1xf64>)

// -----

// REAL + POSITIVE sites: multi-site reassembly (one site takes a bypass).

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @normal_logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @exponential(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @exp_logpdf(%x : tensor<f64>, %rate : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %m:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @normal_logpdf, symbol = #impulse.symbol<1>, name="m", support = #impulse.support<REAL> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %s:2 = impulse.sample @exponential(%m#0, %rate) <{ logpdf_fn = @exp_logpdf, symbol = #impulse.symbol<2>, name="s", support = #impulse.support<POSITIVE> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %m#1, %s#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @svi
  func.func @svi(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %rate : tensor<f64>) -> (tensor<3x2xf64>, tensor<3xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0, 1.0]]> : tensor<1x2xf64>
    %lr = arith.constant dense<0.1> : tensor<f64>

    %res:11 = impulse.infer @test(%rng, %mean, %stddev, %rate) given %init_trace init_step_size = %lr
      <{ vi_config = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = SGD>,
        name = "svi", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], num_warmup = 0, num_samples = 3,
        state_schema = ["loc", "scale_unconstrained", "opt_m_loc", "opt_v_loc", "opt_m_scale", "opt_v_scale", "opt_step_count"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>) -> (tensor<3x2xf64>, tensor<3xf64>, tensor<3xf64>, tensor<2xui64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<i64>)
    return %res#0, %res#1, %res#3 : tensor<3x2xf64>, tensor<3xf64>, tensor<2xui64>
  }
}

// CHECK: %{{.+}}, %[[EPS1:.+]] = impulse.random %{{.+}}, %{{.+}}, %{{.+}} <rng_distribution = NORMAL> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
// CHECK-NEXT: %[[EPS_A:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[EPS1]], %{{.+}}, %{{.+}} : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SPLIT2:.+]]:2 = impulse.random_split %{{.+}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %{{.+}}, %[[EPS2:.+]] = impulse.random %[[SPLIT2]]#1, %{{.+}}, %{{.+}} <rng_distribution = NORMAL> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
// CHECK-NEXT: %[[EPS:.+]] = impulse.dynamic_update_slice %[[EPS_A]], %[[EPS2]], %{{.+}}, %{{.+}} : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>

// CHECK: %[[AD:.+]]:3 = enzyme.autodiff_region(%{{.+}}, %{{.+}}, %{{.+}}) {
// CHECK-NEXT: ^bb0(%[[LOC:.+]]: tensor<1x2xf64>, %[[SCALE_UNC:.+]]: tensor<1x2xf64>):

// CHECK-NEXT: %[[SU_EXP:.+]] = math.exp %[[SCALE_UNC]] : tensor<1x2xf64>
// CHECK-NEXT: %[[SCALE:.+]] = math.log1p %[[SU_EXP]] : tensor<1x2xf64>
// CHECK-NEXT: %[[SCALE_EPS:.+]] = arith.mulf %[[SCALE]], %[[EPS]] : tensor<1x2xf64>
// CHECK-NEXT: %[[Z_UNC:.+]] = arith.addf %[[LOC]], %[[SCALE_EPS]] : tensor<1x2xf64>

// CHECK-NEXT: %[[Z_1D:.+]] = impulse.reshape %[[Z_UNC]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[Z_REAL:.+]] = impulse.slice %[[Z_1D]] <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[Z_POS:.+]] = impulse.slice %[[Z_1D]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[C_POS:.+]] = math.exp %[[Z_POS]] : tensor<1xf64>

// CHECK-NEXT: %[[REASM1:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[Z_REAL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[REASM2:.+]] = impulse.dynamic_update_slice %[[REASM1]], %[[C_POS]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[Z_C:.+]] = impulse.reshape %[[REASM2]] : (tensor<2xf64>) -> tensor<1x2xf64>

// CHECK: %[[JAC_1D:.+]] = impulse.reshape %[[Z_UNC]] : (tensor<1x2xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[JAC_SLICE:.+]] = impulse.slice %[[JAC_1D]] <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[JAC_DOT:.+]] = impulse.dot %[[JAC_SLICE]], %{{.+}} {{.*}} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[JAC:.+]] = arith.addf %[[JAC_DOT]], %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[LOG_Q:.+]] = arith.subf %{{.+}}, %[[JAC]] : tensor<f64>

// CHECK-NEXT: %[[TR_SLICE:.+]] = impulse.dynamic_slice %[[Z_C]], %{{.+}}, %{{.+}} <slice_sizes = [1, 1]> : (tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK: %[[GEN:.+]]:5 = func.call @test.generate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
// CHECK-NEXT: %[[ELBO:.+]] = arith.subf %[[GEN]]#1, %[[LOG_Q]] : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[ELBO]], %[[GEN]]#2 : tensor<f64>, tensor<2xui64>

// CHECK: ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]

// CHECK-LABEL: func.func @test.generate
// CHECK: call @normal_logpdf
// CHECK: call @exp_logpdf
