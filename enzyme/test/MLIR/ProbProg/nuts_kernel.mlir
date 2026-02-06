// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @nuts(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "nuts", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @nuts
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
// CHECK-DAG: %[[CKPT_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<3x1xf64>
// CHECK-DAG: %[[C3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
// CHECK-DAG: %[[MAX_DE:.+]] = arith.constant dense<1.000000e+03> : tensor<f64>
//
// --- RNG splits ---
// CHECK: enzyme.randomSplit
// CHECK: enzyme.randomSplit
//
// --- Initial gradient via autodiff ---
// CHECK: enzyme.autodiff_region(%{{.+}}, %[[ONE]]) {
// CHECK: ^bb0(%{{.+}}: tensor<1x1xf64>):
// CHECK: func.call @test.generate
// CHECK: arith.negf
// CHECK: enzyme.yield
// CHECK: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}
//
// --- Sampling loop ---
// CHECK: %[[SLOOP:.+]]:6 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C1]] : tensor<i64>)
// CHECK-SAME: iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[INIT_TRACE]], %{{.+}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>)
// CHECK-SAME: -> tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>
// CHECK: ^bb0(%[[S_ITER:.+]]: tensor<i64>, %[[S_Q:.+]]: tensor<1x1xf64>, %[[S_GRAD:.+]]: tensor<1x1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<1x1xf64>, %{{.+}}: tensor<1xi1>):
//
// --- Momentum sampling ---
// CHECK: enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1x1xf64>)
//
// --- Kinetic energy ---
// CHECK: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
//
// ============================================================
// Main NUTS tree building loop (outer while)
// ============================================================
// CHECK: %[[TREE:.+]]:18 = enzyme.while_loop({{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x1xf64>, tensor<2xui64>) -> {{.*}} condition {
//
// --- Condition: depth < max_tree_depth && !turning && !diverging ---
// CHECK: arith.cmpi slt, %{{.+}}, %[[C3]] : tensor<i64>
// CHECK: arith.xori {{.*}} : tensor<i1>
// CHECK: arith.andi {{.*}} : tensor<i1>
// CHECK: enzyme.yield
// CHECK: } body {
//
// --- Direction sampling ---
// CHECK: enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution UNIFORM>}
// CHECK: arith.cmpf olt, {{.*}} : tensor<f64>
// CHECK: enzyme.randomSplit {{.*}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
//
// --- Subtree size: 2^depth ---
// CHECK: arith.shli {{.*}}, %{{.+}} : tensor<i64>
//
// ============================================================
// Inner subtree building loop ---
// ============================================================
// CHECK: enzyme.while_loop({{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x1xf64>, tensor<2xui64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<i64>) -> {{.*}} condition {
// CHECK: arith.cmpi slt, {{.*}} : tensor<i64>
// CHECK: arith.andi {{.*}} : tensor<i1>
// CHECK: enzyme.yield
// CHECK: } body {
//
// --- Leapfrog step ---
// CHECK: arith.mulf {{.*}} : tensor<1x1xf64>
// CHECK: arith.subf {{.*}} : tensor<1x1xf64>
// CHECK: arith.addf {{.*}} : tensor<1x1xf64>
//
// --- Gradient via autodiff ---
// CHECK: enzyme.autodiff_region
// CHECK: func.call @test.generate
// CHECK: enzyme.yield
// CHECK: }
//
// --- Second half-step momentum update ---
// CHECK: arith.mulf {{.*}} : tensor<1x1xf64>
// CHECK: arith.subf {{.*}} : tensor<1x1xf64>
//
// --- Kinetic energy ---
// CHECK: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
//
// --- Delta energy and divergence check ---
// CHECK: arith.subf {{.*}} : tensor<f64>
// CHECK: arith.cmpf ogt, %{{.+}}, %[[MAX_DE]] : tensor<f64>
//
// --- Tree combination ---
// CHECK: arith.cmpi eq, %{{.+}}, %[[C0]] : tensor<i64>
// CHECK: enzyme.if
// CHECK: enzyme.yield
// CHECK: }, {
// CHECK: enzyme.log_add_exp
// CHECK: enzyme.logistic
// CHECK: enzyme.random
// CHECK: enzyme.select
// CHECK: enzyme.yield
// CHECK: })
//
// --- Checkpoint updates ---
// CHECK: enzyme.popcount
// CHECK: enzyme.dynamic_update_slice {{.*}} : (tensor<3x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<3x1xf64>
//
// --- Iterative turning check loop ---
// CHECK: enzyme.while_loop({{.*}} : tensor<i64>, tensor<i1>) -> tensor<i64>, tensor<i1> condition {
// CHECK: enzyme.yield
// CHECK: } body {
// CHECK: enzyme.dynamic_slice {{.*}} {slice_sizes = array<i64: 1, 1>}
// --- Dynamic termination criterion ---
// CHECK: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
// CHECK: arith.cmpf ole, {{.*}} : tensor<f64>
// CHECK: arith.ori {{.*}} : tensor<i1>
// CHECK: enzyme.yield
// CHECK: }
//
// --- Subtree yield ---
// CHECK: enzyme.yield {{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x1xf64>, tensor<2xui64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<i64>
// CHECK: }
//
// ============================================================
// Tree combination with biased kernel
// ============================================================
// --- Update left/right boundaries ---
// CHECK: enzyme.select {{.*}} : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>)
// CHECK: enzyme.select {{.*}} : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>)
//
// --- Biased transition: exp, min ---
// CHECK: enzyme.log_add_exp
// CHECK: math.exp
// CHECK: arith.minimumf {{.*}}, %[[ONE]]
//
// --- Zero probability when turning/diverging ---
// CHECK: arith.ori {{.*}} : tensor<i1>
// CHECK: arith.select {{.*}}, %[[ZERO_F]]
// CHECK: enzyme.random
// CHECK: arith.cmpf olt
// CHECK: enzyme.select {{.*}} : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>)
//
// --- Turning check on combined tree ---
// CHECK: arith.addf {{.*}} : tensor<1x1xf64>
// CHECK: arith.mulf {{.*}} : tensor<1x1xf64>
// CHECK: arith.subf {{.*}} : tensor<1x1xf64>
// CHECK: enzyme.dot {{.*}} lhs_contracting_dimensions = array<i64: 0, 1>
// CHECK: arith.cmpf ole, {{.*}} : tensor<f64>
// CHECK: arith.ori {{.*}} : tensor<i1>
//
// --- Outer loop yield ---
// CHECK: enzyme.yield {{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x1xf64>, tensor<2xui64>
// CHECK: }
//
// --- Store sample ---
// CHECK: arith.cmpi sge, %[[S_ITER]], %[[C0]]
// CHECK: enzyme.dynamic_update_slice {{.*}} : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK: enzyme.select
//
// --- Sampling loop yield ---
// CHECK: enzyme.yield %[[TREE]]#6, %[[TREE]]#7, %[[TREE]]#8, {{.*}} : tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>
// CHECK: }
// CHECK: return %[[SLOOP]]#4, %[[SLOOP]]#5, %[[SLOOP]]#3 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
// CHECK: }
//
// --- Generated function: test.generate ---
// CHECK-LABEL: func.func @test.generate
// CHECK-SAME: (%{{.+}}: tensor<1x1xf64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK: enzyme.slice %{{.+}} {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>
// CHECK: call @logpdf
// CHECK: return {{.*}} : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
