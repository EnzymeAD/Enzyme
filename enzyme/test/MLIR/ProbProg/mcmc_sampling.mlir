// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @sampling_basic(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "sampling_basic", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
  }

  func.func @sampling_thinning(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "sampling_thinning", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 10, thinning = 2 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>
  }

  func.func @sampling_with_warmup(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5>,
        name = "sampling_with_warmup", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 5, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
  }
}

// ============================================================
// sampling_basic: 10 samples, no thinning, no warmup
// ============================================================
// CHECK-LABEL: func.func @sampling_basic
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
// CHECK-DAG: %[[C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant dense<true> : tensor<10xi1>
// CHECK-DAG: %[[SAMP_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<10x1xf64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
//
// --- RNG splits ---
// CHECK: enzyme.randomSplit
// CHECK: enzyme.randomSplit
//
// --- Initial gradient via autodiff ---
// CHECK: enzyme.autodiff_region(%{{.+}}, %{{.+}}) {
// CHECK: ^bb0(%{{.+}}: tensor<1x1xf64>):
// CHECK: func.call @test.generate
// CHECK: arith.negf
// CHECK: enzyme.yield
// CHECK: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}
//
// --- Sampling loop: for i in 0..10 ---
// CHECK: %[[SLOOP:.+]]:6 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C10]] : tensor<i64>)
// CHECK-SAME: iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[SAMP_INIT]], %[[ACC_INIT]] : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>)
// CHECK-SAME: -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK: ^bb0(%[[S_ITER:.+]]: tensor<i64>, %{{.+}}: tensor<1x1xf64>, %{{.+}}: tensor<1x1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<10x1xf64>, %{{.+}}: tensor<10xi1>):
//
// --- Momentum sampling ---
// CHECK: enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
//
// --- Kinetic energy ---
// CHECK: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
//
// --- NUTS tree building ---
// CHECK: enzyme.while_loop
//
// --- Sample storage: update samples buffer ---
// CHECK: enzyme.dynamic_update_slice %{{.+}}, %{{.+}}, %[[S_ITER]], %{{.+}} : (tensor<10x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<10x1xf64>
// CHECK: enzyme.yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK: }
// CHECK: return %[[SLOOP]]#4, %[[SLOOP]]#5, %[[SLOOP]]#3 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
// CHECK: }

// ============================================================
// sampling_thinning: 10 total iterations, thinning=2, output 5 samples
// ============================================================
// CHECK-LABEL: func.func @sampling_thinning
// CHECK-SAME: -> (tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>)
//
// --- Thinning loop ---
// CHECK: enzyme.for_loop
// CHECK-SAME: iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>)
// CHECK-SAME: -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>
//
// --- Thinning logic ---
// CHECK: arith.divsi
// CHECK: arith.cmpi sge
// CHECK: arith.remsi
// CHECK: arith.cmpi eq
// CHECK: arith.andi
//
// --- Sample storage ---
// CHECK: enzyme.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<5x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<5x1xf64>
// CHECK: return {{.*}} : tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>
// CHECK: }

// ============================================================
// sampling_with_warmup: 5 warmup + 10 sampling iterations
// ============================================================
// CHECK-LABEL: func.func @sampling_with_warmup
// CHECK-SAME: -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
//
// --- Warmup loop ---
// CHECK: %[[WARMUP:.+]]:16 = enzyme.for_loop
// CHECK-SAME: iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>)
// CHECK-SAME: -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
//
// --- Dual averaging update ---
// CHECK: arith.subf {{.*}} : tensor<f64>
// CHECK: arith.addf {{.*}} : tensor<f64>
// CHECK: arith.divf {{.*}} : tensor<f64>
// CHECK: math.sqrt
// CHECK: math.exp
//
// --- Welford update ---
// CHECK: enzyme.reshape {{.*}} : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK: arith.subf {{.*}} : tensor<1xf64>
// CHECK: arith.divf {{.*}} : tensor<1xf64>
// CHECK: arith.addf {{.*}} : tensor<1xf64>
//
// --- Window boundary logic ---
// CHECK: enzyme.if
// CHECK: math.sqrt
// CHECK: enzyme.yield
// CHECK: }, {
// CHECK: enzyme.yield
// CHECK: })
//
// --- Post-warmup sampling ---
// CHECK: enzyme.for_loop
// CHECK-SAME: iter_args(%[[WARMUP]]#0, %[[WARMUP]]#1, %[[WARMUP]]#2, %[[WARMUP]]#3
// CHECK-SAME: : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>
//
// --- Momentum with adapted mass matrix ---
// CHECK: enzyme.dot {{.*}} : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
//
// CHECK: return {{.*}} : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
// CHECK: }
