// RUN: %eopt --split-input-file %s | %eopt --split-input-file | FileCheck %s

// Impulse ops and attributes print in a form that parses back to the same IR.

module {
  func.func private @normal(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  // CHECK:   func.func @sample(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[RES:.+]]:2 = impulse.sample @normal(%[[SEED]], %[[MEAN]], %[[STDDEV]]) <logpdf_fn = @logpdf, symbol = <3>, name = "r"> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[RES]]#0, %[[RES]]#1 : tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @sample(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %r:2 = impulse.sample @normal(%seed, %mean, %stddev) <{ logpdf_fn = @logpdf, name="r", symbol = #impulse.symbol<3> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %r#0, %r#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK:   func.func @simulate(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[TRACE:.+]], %[[WEIGHT:.+]], %[[OUTPUTS:.+]]:2 = impulse.simulate @sample(%[[SEED]], %[[MEAN]], %[[STDDEV]]) <all_addresses = {{\[}}[#impulse.symbol<3>]], name = "test"> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[TRACE]], %[[WEIGHT]], %[[OUTPUTS]]#0, %[[OUTPUTS]]#1 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @simulate(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:4 = impulse.simulate @sample(%seed, %mean, %stddev) <{ all_addresses = [[#impulse.symbol<3>]], name = "test" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  // CHECK:   func.func @generate(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[CONSTRAINT:.+]] = arith.constant dense<{{.*}}1.5{{.*}}> : tensor<1x1xf64>
  // CHECK-NEXT:    %[[TRACE:.+]], %[[WEIGHT:.+]], %[[OUTPUTS:.+]]:2 = impulse.generate @sample(%[[SEED]], %[[MEAN]], %[[STDDEV]]) constrained_by %[[CONSTRAINT]] <all_addresses = {{\[}}[#impulse.symbol<3>]], constrained_addresses = {{\[}}[#impulse.symbol<3>]], name = "test"> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[TRACE]], %[[WEIGHT]], %[[OUTPUTS]]#0, %[[OUTPUTS]]#1 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @generate(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %constraint = arith.constant dense<1.5> : tensor<1x1xf64>
    %res:4 = impulse.generate @sample(%seed, %mean, %stddev) constrained_by %constraint <{ all_addresses = [[#impulse.symbol<3>]], constrained_addresses = [[#impulse.symbol<3>]], name = "test" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// -----

// Without `fn` the pretty form starts with `(`, which the callee parser skips.

module {
  // CHECK-LABEL: func.func @emitted_site
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[SCALE:.+]]: tensor<f64>)
  // CHECK-NEXT: %[[X:.+]]:2 = impulse.sample(%[[RNG]], %[[SCALE]]) <symbol = <1>, dist = <HALF_NORMAL>, name = "x"> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT: return %[[X]]#0, %[[X]]#1 : tensor<2xui64>, tensor<f64>
  func.func @emitted_site(%rng : tensor<2xui64>, %scale : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %x:2 = impulse.sample(%rng, %scale) <{dist = #impulse.dist<HALF_NORMAL>, symbol = #impulse.symbol<1>, name = "x"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @traced_site
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[SCALE:.+]]: tensor<f64>)
  // CHECK-NEXT: %[[X:.+]]:2 = impulse.sample @emitted_site(%[[RNG]], %[[SCALE]]) <symbol = <1>, name = "x"> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func @traced_site(%rng : tensor<2xui64>, %scale : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %x:2 = impulse.sample @emitted_site(%rng, %scale) <{symbol = #impulse.symbol<1>, name = "x"}> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %x#0, %x#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @logpdf(%x : tensor<2xf64>) -> tensor<f64> {
    %c = arith.constant dense<0.0> : tensor<f64>
    return %c : tensor<f64>
  }

  // CHECK-LABEL: func.func @custom_logpdf_infer
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>)
  // CHECK-NEXT: %[[POS:.+]] = arith.constant dense<{{\[}}[1.000000e+00, -1.000000e+00]]> : tensor<1x2xf64>
  // CHECK-NEXT: %[[STEP:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = impulse.infer(%[[RNG]]) init_step_size = %[[STEP]] logpdf_fn = @logpdf init_position = %[[POS]] <selection = [], all_addresses = [], nuts_config = <max_tree_depth = 3, max_delta_energy = 1.000000e+03 : f64, adapt_step_size = false, adapt_mass_matrix = false>> : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  func.func @custom_logpdf_infer(%rng : tensor<2xui64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<[[1.0, -1.0]]> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer(%rng) init_step_size = %step_size logpdf_fn = @logpdf init_position = %init_pos <{
      nuts_config = #impulse.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0 : f64, adapt_step_size = false, adapt_mass_matrix = false>,
      selection = [], all_addresses = [], num_samples = 1
    }> : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }

  // CHECK-LABEL: func.func @custom_logpdf_infer_region
  // CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[B:.+]]: tensor<2xf64>)
  // CHECK-NEXT: %[[POS:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
  // CHECK-NEXT: %[[STEP:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
  // CHECK-NEXT: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = impulse.infer_region(%[[RNG]], %[[B]]) init_step_size = %[[STEP]] logpdf_fn = @logpdf init_position = %[[POS]] sampling {
  // CHECK-NEXT: } logpdf {
  // CHECK-NEXT: ^bb0(%[[Q:.+]]: tensor<2xf64>):
  // CHECK-NEXT: %[[QB:.+]] = arith.mulf %[[Q]], %[[B]] : tensor<2xf64>
  // CHECK-NEXT: %[[D:.+]] = impulse.dot %[[QB]], %[[QB]] <{{.+}}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
  // CHECK-NEXT: impulse.yield %[[D]] : tensor<f64>
  // CHECK-NEXT: } <selection = [], all_addresses = [], nuts_config = <max_tree_depth = 3>> : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
  func.func @custom_logpdf_infer_region(%rng : tensor<2xui64>, %B : tensor<2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>) {
    %init_pos = arith.constant dense<0.0> : tensor<1x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer_region(%rng, %B) init_step_size = %step_size logpdf_fn = @logpdf init_position = %init_pos sampling {
    } logpdf {
    ^bb0(%q : tensor<2xf64>):
      %qb = arith.mulf %q, %B : tensor<2xf64>
      %d = impulse.dot %qb, %qb <{lhs_contracting_dimensions = array<i64: 0>, rhs_contracting_dimensions = array<i64: 0>}> : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
      impulse.yield %d : tensor<f64>
    } <{
      nuts_config = #impulse.nuts_config<max_tree_depth = 3>,
      selection = [], all_addresses = [], num_samples = 1
    }> : (tensor<2xui64>, tensor<2xf64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>
  }
}

// -----

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
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

// -----

// An elided default followed by a present parameter must parse back.

module {
  // CHECK-LABEL: func.func @configs
  // CHECK-SAME: dist_event = #impulse.dist<STD_NORMAL>
  // CHECK-SAME: dist_full = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>
  // CHECK-SAME: dist_xform = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
  // CHECK-SAME: hmc = #impulse.hmc_config<trajectory_length = 1.000000e+00 : f64, adapt_mass_matrix = false>
  // CHECK-SAME: nuts_default_depth = #impulse.nuts_config<max_delta_energy = 1.000000e+03 : f64, adapt_step_size = false>
  // CHECK-SAME: nuts_depth_only = #impulse.nuts_config<max_tree_depth = 5, adapt_mass_matrix = false>
  // CHECK-SAME: vi_default_particles = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM, init_scale = 1.000000e-01 : f64, adam_b1 = 9.000000e-01 : f64, clip_norm = 1.000000e+01 : f64>
  // CHECK-SAME: vi_minimal = #impulse.vi_config<guide_kind = AUTO_DELTA, optimizer_kind = SGD>
  // CHECK-SAME: vi_particles = #impulse.vi_config<guide_kind = AUTO_MULTIVARIATE_NORMAL, optimizer_kind = RMSPROP, num_particles = 4, rmsprop_gamma = 8.000000e-01 : f64, stable_update = true>
  func.func @configs() attributes {
    vi_default_particles = #impulse.vi_config<guide_kind = AUTO_NORMAL, optimizer_kind = ADAM, init_scale = 0.1 : f64, adam_b1 = 0.9 : f64, clip_norm = 10.0 : f64>,
    vi_minimal = #impulse.vi_config<guide_kind = AUTO_DELTA, optimizer_kind = SGD>,
    vi_particles = #impulse.vi_config<guide_kind = AUTO_MULTIVARIATE_NORMAL, optimizer_kind = RMSPROP, num_particles = 4, rmsprop_gamma = 0.8 : f64, stable_update = true>,
    hmc = #impulse.hmc_config<trajectory_length = 1.0 : f64, adapt_mass_matrix = false>,
    nuts_default_depth = #impulse.nuts_config<max_delta_energy = 1000.0 : f64, adapt_step_size = false>,
    nuts_depth_only = #impulse.nuts_config<max_tree_depth = 5, adapt_mass_matrix = false>,
    dist_event = #impulse.dist<STD_NORMAL>,
    dist_full = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>,
    dist_xform = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
  } {
    return
  }
}
