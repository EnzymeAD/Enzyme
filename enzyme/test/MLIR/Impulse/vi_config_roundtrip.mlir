// RUN: %eopt %s | %eopt | FileCheck %s

// An elided default followed by a present parameter must parse back.

module {
  // CHECK-LABEL: func.func @configs
  // CHECK-SAME: dist_event = #impulse.dist<STD_NORMAL, event_dim = 1>
  // CHECK-SAME: dist_full = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>], event_dim = 1>
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
    dist_event = #impulse.dist<STD_NORMAL, event_dim = 1>,
    dist_full = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>], event_dim = 1>,
    dist_xform = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
  } {
    return
  }
}
