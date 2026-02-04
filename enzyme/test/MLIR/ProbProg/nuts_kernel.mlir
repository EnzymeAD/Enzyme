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
// CHECK-DAG: %[[NEG1:.+]] = arith.constant dense<-1> : tensor<i64>
// CHECK-DAG: %[[INF:.+]] = arith.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-DAG: %[[NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-DAG: %[[CKPT_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<3x1xf64>
// CHECK-DAG: %[[C3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant dense<true> : tensor<1xi1>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
// CHECK-DAG: %[[EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-DAG: %[[MAX_DE:.+]] = arith.constant dense<1.000000e+03> : tensor<f64>
//
// --- RNG splits ---
// CHECK: %[[SPLIT1:.+]]:2 = enzyme.randomSplit %[[RNG]] : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[SPLIT2:.+]]:3 = enzyme.randomSplit %[[SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
//
// --- Extract initial position ---
// CHECK-NEXT: %[[Q0_SL:.+]] = enzyme.dynamic_slice %[[INIT_TRACE]], %[[C0]], %[[C0]] {slice_sizes = array<i64: 1, 1>} : (tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[Q0:.+]] = enzyme.reshape %[[Q0_SL]] : (tensor<1x1xf64>) -> tensor<1xf64>
//
// --- Initial potential U0 ---
// CHECK-NEXT: %[[Q0_2D:.+]] = enzyme.reshape %[[Q0]] : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[GEN0:.+]]:4 = call @test.generate{{.*}}(%[[Q0_2D]], %[[SPLIT2]]#1, %[[MEAN]], %[[STDDEV]])
// CHECK-NEXT: %[[U0:.+]] = arith.negf %[[GEN0]]#1 : tensor<f64>
//
// --- Initial gradient via autodiff ---
// CHECK-NEXT: %[[AD0:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[ONE]]) {
// CHECK-NEXT: ^bb0(%[[AD0_ARG:.+]]: tensor<1xf64>):
// CHECK-NEXT: %[[AD0_RS:.+]] = enzyme.reshape %[[AD0_ARG]] : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[AD0_GEN:.+]]:4 = func.call @test.generate{{.*}}(%[[AD0_RS]], %[[SPLIT2]]#1, %[[MEAN]], %[[STDDEV]])
// CHECK-NEXT: %[[AD0_NEG:.+]] = arith.negf %[[AD0_GEN]]#1 : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[AD0_NEG]], %[[AD0_GEN]]#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}
//
// --- Sampling loop (num_samples = 1) ---
// CHECK: %[[SLOOP:.+]]:6 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C1]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%[[Q0]], %[[AD0]]#2, %[[U0]], %[[SPLIT2]]#0, %[[INIT_TRACE]], %[[ACC_INIT]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1> {
// CHECK-NEXT: ^bb0(%[[S_ITER:.+]]: tensor<i64>, %[[S_Q:.+]]: tensor<1xf64>, %[[S_GRAD:.+]]: tensor<1xf64>, %[[S_U:.+]]: tensor<f64>, %[[S_RNG:.+]]: tensor<2xui64>, %[[S_SAMP:.+]]: tensor<1x1xf64>, %[[S_ACC:.+]]: tensor<1xi1>):
//
// --- Sample momentum ---
// CHECK-NEXT: %[[SR:.+]]:3 = enzyme.randomSplit %[[S_RNG]]
// CHECK-NEXT: %[[MR:.+]] = enzyme.randomSplit %[[SR]]#1
// CHECK-NEXT: %[[MR2:.+]], %[[P0:.+]] = enzyme.random %[[MR]], %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>}
//
// --- Kinetic energy & Hamiltonian ---
// CHECK-NEXT: %[[KE0_D:.+]] = enzyme.dot %[[P0]], %[[P0]] {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %[[KE0:.+]] = arith.mulf %[[KE0_D]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[H0:.+]] = arith.addf %[[S_U]], %[[KE0]] : tensor<f64>
//
// ============================================================
// Main NUTS tree building loop (outer while)
// ============================================================
// CHECK-NEXT: %[[TREE:.+]]:18 = enzyme.while_loop(%[[S_Q]], %[[P0]], %[[S_GRAD]], %[[S_Q]], %[[P0]], %[[S_GRAD]], %[[S_Q]], %[[S_GRAD]], %[[S_U]], %[[H0]], %[[C0]], %[[ZERO_F]], %[[FALSE]], %[[FALSE]], %[[ZERO_F]], %[[C0]], %[[P0]], %[[SR]]#2 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>) -> {{.*}} condition {
// CHECK-NEXT: ^bb0({{.*}}):
//
// --- Condition: depth < max_tree_depth && !turning && !diverging ---
// CHECK-NEXT: %[[DEPTH_LT:.+]] = arith.cmpi slt, %{{.+}}, %[[C3]] : tensor<i64>
// CHECK-NEXT: %[[NOT_TURN:.+]] = arith.xori %{{.+}}, %[[TRUE]] : tensor<i1>
// CHECK-NEXT: %[[NOT_DIV:.+]] = arith.xori %{{.+}}, %[[TRUE]] : tensor<i1>
// CHECK-NEXT: %[[COND1:.+]] = arith.andi %[[DEPTH_LT]], %[[NOT_TURN]] : tensor<i1>
// CHECK-NEXT: %[[COND:.+]] = arith.andi %[[COND1]], %[[NOT_DIV]] : tensor<i1>
// CHECK-NEXT: enzyme.yield %[[COND]] : tensor<i1>
// CHECK-NEXT: } body {
// CHECK-NEXT: ^bb0({{.*}}):
//
// --- Direction sampling ---
// CHECK-NEXT: %[[DIR_SPLIT:.+]]:3 = enzyme.randomSplit %{{.+}}
// CHECK-NEXT: %[[DIR_RNG:.+]], %[[DIR_U:.+]] = enzyme.random %[[DIR_SPLIT]]#1, %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>}
// CHECK-NEXT: %[[GOING_RIGHT:.+]] = arith.cmpf olt, %[[DIR_U]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[SUB_SPLIT:.+]]:2 = enzyme.randomSplit %[[DIR_SPLIT]]#2
//
// --- Compute subtree size = 2^depth ---
// CHECK-NEXT: %[[SUB_SIZE:.+]] = arith.shli %[[C1]], %{{.+}} : tensor<i64>
//
// ============================================================
// Inner subtree building loop
// ============================================================
// CHECK-NEXT: %[[SUB:.+]]:21 = enzyme.while_loop({{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<i64>) -> {{.*}} condition {
// CHECK-NEXT: ^bb0({{.*}}):
//
// --- Subtree loop condition: leafIdx < subtreeSize && !turning && !diverging ---
// CHECK-NEXT: %{{.+}} = arith.cmpi slt, %{{.+}}, %[[SUB_SIZE]] : tensor<i64>
// CHECK-NEXT: %{{.+}} = arith.xori %{{.+}}, %[[TRUE]] : tensor<i1>
// CHECK-NEXT: %{{.+}} = arith.xori %{{.+}}, %[[TRUE]] : tensor<i1>
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: enzyme.yield %{{.+}} : tensor<i1>
// CHECK-NEXT: } body {
// CHECK-NEXT: ^bb0({{.*}}):
//
// --- Select leaf based on direction ---
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
//
// --- Leapfrog step ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %[[LF_DIR:.+]] = enzyme.select %[[GOING_RIGHT]], %[[EPS]], %[[NEG_EPS]] : (tensor<i1>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.mulf %[[LF_DIR]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Gradient via autodiff ---
// CHECK-NEXT: %{{.+}} = enzyme.autodiff_region
// CHECK: func.call @test.generate
// CHECK: arith.negf
// CHECK: enzyme.yield
// CHECK-NEXT: }
//
// --- Second half-step ---
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.subf
//
// --- New kinetic energy ---
// CHECK-NEXT: %{{.+}} = enzyme.dot %{{.+}}, %{{.+}} {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = arith.mulf %{{.+}}, %[[HALF]] : tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Delta energy for divergence check ---
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = arith.cmpf une
// CHECK-NEXT: %{{.+}} = arith.select
//
// --- Weight and divergence ---
// CHECK-NEXT: %{{.+}} = arith.negf
// CHECK-NEXT: %{{.+}} = arith.cmpf ogt, %{{.+}}, %[[MAX_DE]]
//
// --- Acceptance probability: min(1, exp(-delta)) ---
// CHECK-NEXT: %{{.+}} = arith.negf
// CHECK-NEXT: %{{.+}} = math.exp
// CHECK-NEXT: %{{.+}} = arith.minimumf %{{.+}}, %[[ONE]]
//
// --- Tree combination (enzyme.if for first leaf vs subsequent) ---
// CHECK-NEXT: %{{.+}} = arith.cmpi eq, %{{.+}}, %[[C0]] : tensor<i64>
// CHECK-NEXT: %{{.+}}:18 = enzyme.if
// CHECK: enzyme.yield
// CHECK-NEXT: }, {
// --- Subsequent leaf: log_add_exp, logistic, random, select ---
// CHECK: enzyme.log_add_exp
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = enzyme.logistic
// CHECK-NEXT: %{{.+}} = enzyme.random
// CHECK-NEXT: %{{.+}} = arith.cmpf olt
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK: enzyme.yield
// CHECK-NEXT: })
//
// --- Checkpoint handling ---
// CHECK-NEXT: %{{.+}} = arith.shrui %{{.+}}, %[[C1]] : tensor<i64>
// CHECK-NEXT: %{{.+}} = enzyme.popcount
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.xori %{{.+}}, %[[NEG1]]
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.subi
// CHECK-NEXT: %{{.+}} = enzyme.popcount
// CHECK-NEXT: %{{.+}} = arith.subi
// CHECK-NEXT: %{{.+}} = arith.addi
//
// --- Even leaf: update checkpoints ---
// CHECK-NEXT: %{{.+}} = arith.andi %{{.+}}, %[[C1]]
// CHECK-NEXT: %{{.+}} = arith.cmpi eq, %{{.+}}, %[[C0]]
// CHECK-NEXT: %{{.+}}:2 = enzyme.if
// CHECK: enzyme.reshape
// CHECK-NEXT: enzyme.reshape
// CHECK-NEXT: enzyme.dynamic_update_slice
// CHECK-NEXT: enzyme.dynamic_update_slice
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })
//
// --- Iterative turning check ---
// CHECK-NEXT: %{{.+}}:2 = enzyme.while_loop(%{{.+}}, %[[FALSE]] : tensor<i64>, tensor<i1>) -> tensor<i64>, tensor<i1> condition {
// CHECK-NEXT: ^bb0(%{{.+}}: tensor<i64>, %{{.+}}: tensor<i1>):
// CHECK-NEXT: %{{.+}} = arith.cmpi sge
// CHECK-NEXT: %{{.+}} = arith.xori %{{.+}}, %[[TRUE]]
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: } body {
// CHECK-NEXT: ^bb0(%{{.+}}: tensor<i64>, %{{.+}}: tensor<i1>):
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_slice %{{.+}}, %{{.+}}, %[[C0]] {slice_sizes = array<i64: 1, 1>}
// CHECK-NEXT: %{{.+}} = enzyme.reshape
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_slice %{{.+}}, %{{.+}}, %[[C0]] {slice_sizes = array<i64: 1, 1>}
// CHECK-NEXT: %{{.+}} = enzyme.reshape
// --- Dynamic termination criterion: p_sum_centered = p_sum - (p_ckpt + p) / 2 ---
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.subf
// --- Dot products for turning check ---
// CHECK-NEXT: %{{.+}} = enzyme.dot %{{.+}}, %{{.+}} {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = enzyme.dot %{{.+}}, %{{.+}} {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = arith.cmpf ole, %{{.+}}, %[[ZERO_F]]
// CHECK-NEXT: %{{.+}} = arith.cmpf ole, %{{.+}}, %[[ZERO_F]]
// CHECK-NEXT: %{{.+}} = arith.ori
// CHECK-NEXT: %{{.+}} = arith.subi %{{.+}}, %[[C1]]
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: }
// CHECK-NEXT: %{{.+}} = enzyme.select %{{.+}}, %[[FALSE]], %{{.+}} : (tensor<i1>, tensor<i1>, tensor<i1>)
// CHECK-NEXT: %{{.+}} = arith.addi %{{.+}}, %[[C1]]
//
// --- Subtree yield ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<i64>
// CHECK-NEXT: }
//
// ============================================================
// Tree combination after subtree build (biased kernel)
// ============================================================
// --- Update left/right boundaries based on direction ---
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %[[P_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %[[P_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
// CHECK-NEXT: %{{.+}} = enzyme.select %[[GOING_RIGHT]], %{{.+}}, %{{.+}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>)
//
// --- Biased transition: log_add_exp, exp, min ---
// CHECK-NEXT: %{{.+}} = enzyme.log_add_exp
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = math.exp
// CHECK-NEXT: %{{.+}} = arith.minimumf %{{.+}}, %[[ONE]]
//
// --- Zero probability when turning or diverging ---
// CHECK-NEXT: %{{.+}} = arith.ori
// CHECK-NEXT: %{{.+}} = arith.select %{{.+}}, %[[ZERO_F]]
// CHECK-NEXT: %{{.+}} = enzyme.random
// CHECK-NEXT: %[[BIASED_ACC:.+]] = arith.cmpf olt
// CHECK-NEXT: %{{.+}} = enzyme.select %[[BIASED_ACC]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[BIASED_ACC]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[BIASED_ACC]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[BIASED_ACC]]
//
// --- Depth increment ---
// CHECK-NEXT: %{{.+}} = arith.addi %{{.+}}, %[[C1]]
//
// --- p_sum update ---
// CHECK-NEXT: %[[PSUM_UPD:.+]] = arith.addf
//
// --- Turning check on combined tree: p_sum_centered = p_sum - (p_left + p_right) / 2 ---
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.addf %[[P_LEFT]], %[[P_RIGHT]]
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = enzyme.dot %[[P_LEFT]], %{{.+}} {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = enzyme.dot %[[P_RIGHT]], %{{.+}} {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = arith.cmpf ole, %{{.+}}, %[[ZERO_F]]
// CHECK-NEXT: %{{.+}} = arith.cmpf ole, %{{.+}}, %[[ZERO_F]]
// CHECK-NEXT: %{{.+}} = arith.ori
// CHECK-NEXT: %{{.+}} = arith.ori
// CHECK-NEXT: %{{.+}} = arith.ori
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Outer loop yield ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>
// CHECK-NEXT: }
//
// --- Store sample and accepted flag ---
// CHECK-NEXT: %{{.+}} = arith.cmpi sge, %[[S_ITER]], %[[C0]]
// CHECK-NEXT: %{{.+}} = enzyme.reshape %[[TREE]]#6 : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_update_slice %[[S_SAMP]], %{{.+}}, %[[S_ITER]], %[[C0]]
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = enzyme.reshape %[[TRUE]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_update_slice %[[S_ACC]], %{{.+}}, %[[S_ITER]]
// CHECK-NEXT: %{{.+}} = enzyme.select
//
// --- Sampling loop yield ---
// CHECK-NEXT: enzyme.yield %[[TREE]]#6, %[[TREE]]#7, %[[TREE]]#8, %[[SR]]#0, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SLOOP]]#4, %[[SLOOP]]#5, %[[SLOOP]]#3 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
// CHECK-NEXT: }
//
// --- Generated function: test.generate ---
// CHECK-LABEL: func.func @test.generate
// CHECK-SAME: (%[[G_ARG0:.+]]: tensor<1x1xf64>, %[[G_ARG1:.+]]: tensor<2xui64>, %[[G_ARG2:.+]]: tensor<f64>, %[[G_ARG3:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[G_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[G_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[G_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
// CHECK: %[[G_SLICED:.+]] = enzyme.slice %[[G_ARG0]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[G_VAL:.+]] = enzyme.reshape %[[G_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[G_LP:.+]] = call @logpdf(%[[G_VAL]], %[[G_ARG2]], %[[G_ARG3]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[G_W:.+]] = arith.addf %[[G_LP]], %[[G_ZERO]] : tensor<f64>
// CHECK-NEXT: %[[G_RS:.+]] = enzyme.reshape %[[G_VAL]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[G_TR:.+]] = enzyme.dynamic_update_slice %[[G_TRACE_INIT]], %[[G_RS]], %[[G_C0]], %[[G_C0]] : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK-NEXT: return %[[G_TR]], %[[G_W]], %[[G_ARG1]], %[[G_VAL]] : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
