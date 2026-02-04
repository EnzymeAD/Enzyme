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
// CHECK-DAG: %[[TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant dense<true> : tensor<10xi1>
// CHECK-DAG: %[[SAMP_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<10x1xf64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
//
// --- RNG splits ---
// CHECK: %[[SPLIT1:.+]]:2 = enzyme.randomSplit %[[RNG]]
// CHECK-NEXT: %[[SPLIT2:.+]]:3 = enzyme.randomSplit %[[SPLIT1]]#0
//
// --- Extract initial position ---
// CHECK-NEXT: %[[Q0_SL:.+]] = enzyme.dynamic_slice %[[INIT_TRACE]], %[[C0]], %[[C0]] {slice_sizes = array<i64: 1, 1>}
// CHECK-NEXT: %[[Q0:.+]] = enzyme.reshape %[[Q0_SL]] : (tensor<1x1xf64>) -> tensor<1xf64>
//
// --- Initial potential U0 ---
// CHECK-NEXT: %[[Q0_2D:.+]] = enzyme.reshape %[[Q0]] : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[GEN0:.+]]:4 = call @test.generate{{.*}}(%[[Q0_2D]], %[[SPLIT2]]#1, %[[MEAN]], %[[STDDEV]])
// CHECK-NEXT: %[[U0:.+]] = arith.negf %[[GEN0]]#1 : tensor<f64>
//
// --- Initial gradient via autodiff ---
// CHECK-NEXT: %[[AD0:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[ONE]]) {
// CHECK-NEXT: ^bb0(%{{.+}}: tensor<1xf64>):
// CHECK-NEXT: %{{.+}} = enzyme.reshape
// CHECK-NEXT: %{{.+}} = func.call @test.generate
// CHECK-NEXT: %{{.+}} = arith.negf
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}
//
// --- Sampling loop: for i in 0..10 ---
// CHECK-NEXT: %[[SLOOP:.+]]:6 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C10]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%[[Q0]], %[[AD0]]#2, %[[U0]], %[[SPLIT2]]#0, %[[SAMP_INIT]], %[[ACC_INIT]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK-NEXT: ^bb0(%[[S_ITER:.+]]: tensor<i64>, %[[S_Q:.+]]: tensor<1xf64>, %[[S_GRAD:.+]]: tensor<1xf64>, %[[S_U:.+]]: tensor<f64>, %[[S_RNG:.+]]: tensor<2xui64>, %[[S_SAMP:.+]]: tensor<10x1xf64>, %[[S_ACC:.+]]: tensor<10xi1>):
//
// --- Momentum sampling ---
// CHECK-NEXT: %[[SR:.+]]:3 = enzyme.randomSplit %[[S_RNG]]
// CHECK-NEXT: %[[MR:.+]] = enzyme.randomSplit %[[SR]]#1
// CHECK-NEXT: %[[MR2:.+]], %[[P0:.+]] = enzyme.random %[[MR]], %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>}
//
// --- Kinetic energy & Hamiltonian ---
// CHECK-NEXT: %{{.+}} = enzyme.dot %[[P0]], %[[P0]] {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = arith.mulf %{{.+}}, %[[HALF]]
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building (tested in nuts_kernel.mlir) ---
// CHECK-NEXT: %[[TREE:.+]]:18 = enzyme.while_loop
//
// --- Sample storage: check iter >= 0, update samples/accepted buffers ---
// CHECK: %[[STORE_COND:.+]] = arith.cmpi sge, %[[S_ITER]], %[[C0]] : tensor<i64>
// CHECK-NEXT: %[[Q_2D:.+]] = enzyme.reshape %[[TREE]]#6 : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[UPD_SAMP:.+]] = enzyme.dynamic_update_slice %[[S_SAMP]], %[[Q_2D]], %[[S_ITER]], %[[C0]] : (tensor<10x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<10x1xf64>
// CHECK-NEXT: %[[SEL_SAMP:.+]] = enzyme.select %[[STORE_COND]], %[[UPD_SAMP]], %[[S_SAMP]] : (tensor<i1>, tensor<10x1xf64>, tensor<10x1xf64>) -> tensor<10x1xf64>
// CHECK-NEXT: %[[ACC_1D:.+]] = enzyme.reshape %[[TRUE]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %[[UPD_ACC:.+]] = enzyme.dynamic_update_slice %[[S_ACC]], %[[ACC_1D]], %[[S_ITER]] : (tensor<10xi1>, tensor<1xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK-NEXT: %[[SEL_ACC:.+]] = enzyme.select %[[STORE_COND]], %[[UPD_ACC]], %[[S_ACC]] : (tensor<i1>, tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
//
// --- Sampling loop yield ---
// CHECK-NEXT: enzyme.yield %[[TREE]]#6, %[[TREE]]#7, %[[TREE]]#8, %[[SR]]#0, %[[SEL_SAMP]], %[[SEL_ACC]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SLOOP]]#4, %[[SLOOP]]#5, %[[SLOOP]]#3 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
// CHECK-NEXT: }

// ============================================================
// sampling_thinning: 10 total iterations, thinning=2, output 5 samples
// ============================================================
// CHECK-LABEL: func.func @sampling_thinning
// CHECK-SAME: (%[[T_RNG:.+]]: tensor<2xui64>, %[[T_MEAN:.+]]: tensor<f64>, %[[T_STDDEV:.+]]: tensor<f64>) -> (tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>)
// CHECK-DAG: %[[T_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[T_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[T_ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[T_THIN:.+]] = arith.constant dense<2> : tensor<i64>
// CHECK-DAG: %[[T_C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[T_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[T_ACC_INIT:.+]] = arith.constant dense<true> : tensor<5xi1>
// CHECK-DAG: %[[T_SAMP_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<5x1xf64>
// CHECK-DAG: %[[T_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[T_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[T_INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
//
// --- Init: RNG splits, position extraction, U0, autodiff ---
// CHECK: enzyme.randomSplit
// CHECK: enzyme.dynamic_slice %[[T_INIT_TRACE]]
// CHECK: enzyme.reshape
// CHECK: arith.negf
// CHECK: enzyme.autodiff_region
// CHECK: }
//
// --- Sampling loop: for i in 0..10 ---
// CHECK-NEXT: %[[T_LOOP:.+]]:6 = enzyme.for_loop(%[[T_C0]] : tensor<i64>) to(%[[T_C10]] : tensor<i64>) step(%[[T_C1]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[T_SAMP_INIT]], %[[T_ACC_INIT]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1> {
// CHECK-NEXT: ^bb0(%[[T_ITER:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %[[T_SAMP_BUF:.+]]: tensor<5x1xf64>, %[[T_ACC_BUF:.+]]: tensor<5xi1>):
//
// --- NUTS sampling (tested in nuts_kernel.mlir) ---
// CHECK: %[[T_TREE:.+]]:18 = enzyme.while_loop
//
// --- Thinning: compute storage index and should_store condition ---
// CHECK: %[[STORE_IDX:.+]] = arith.divsi %[[T_ITER]], %[[T_THIN]] : tensor<i64>
// CHECK-NEXT: %[[GE_ZERO:.+]] = arith.cmpi sge, %[[T_ITER]], %[[T_C0]] : tensor<i64>
// CHECK-NEXT: %[[MOD_THIN:.+]] = arith.remsi %[[T_ITER]], %[[T_THIN]] : tensor<i64>
// CHECK-NEXT: %[[MOD_EQ_0:.+]] = arith.cmpi eq, %[[MOD_THIN]], %[[T_C0]] : tensor<i64>
// CHECK-NEXT: %[[SHOULD_STORE:.+]] = arith.andi %[[GE_ZERO]], %[[MOD_EQ_0]] : tensor<i1>
//
// --- Update samples buffer using storage_idx ---
// CHECK-NEXT: %[[T_Q_2D:.+]] = enzyme.reshape %[[T_TREE]]#6 : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T_UPD_SAMP:.+]] = enzyme.dynamic_update_slice %[[T_SAMP_BUF]], %[[T_Q_2D]], %[[STORE_IDX]], %[[T_C0]] : (tensor<5x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<5x1xf64>
// CHECK-NEXT: %[[T_SEL_SAMP:.+]] = enzyme.select %[[SHOULD_STORE]], %[[T_UPD_SAMP]], %[[T_SAMP_BUF]] : (tensor<i1>, tensor<5x1xf64>, tensor<5x1xf64>) -> tensor<5x1xf64>
// CHECK-NEXT: %[[T_ACC_1D:.+]] = enzyme.reshape %[[T_TRUE]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %[[T_UPD_ACC:.+]] = enzyme.dynamic_update_slice %[[T_ACC_BUF]], %[[T_ACC_1D]], %[[STORE_IDX]] : (tensor<5xi1>, tensor<1xi1>, tensor<i64>) -> tensor<5xi1>
// CHECK-NEXT: %[[T_SEL_ACC:.+]] = enzyme.select %[[SHOULD_STORE]], %[[T_UPD_ACC]], %[[T_ACC_BUF]] : (tensor<i1>, tensor<5xi1>, tensor<5xi1>) -> tensor<5xi1>
//
// --- Thinning loop yield ---
// CHECK-NEXT: enzyme.yield %[[T_TREE]]#6, %[[T_TREE]]#7, %[[T_TREE]]#8, %{{.+}}, %[[T_SEL_SAMP]], %[[T_SEL_ACC]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[T_LOOP]]#4, %[[T_LOOP]]#5, %[[T_LOOP]]#3 : tensor<5x1xf64>, tensor<5xi1>, tensor<2xui64>
// CHECK-NEXT: }

// ============================================================
// sampling_with_warmup: 5 warmup + 10 sampling iterations
// ============================================================
// CHECK-LABEL: func.func @sampling_with_warmup
// CHECK-SAME: (%[[W_RNG:.+]]: tensor<2xui64>, %[[W_MEAN:.+]]: tensor<f64>, %[[W_STDDEV:.+]]: tensor<f64>) -> (tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>)
//
// --- Adaptation constants ---
// CHECK-DAG: %[[W_C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[W_ACC_INIT:.+]] = arith.constant dense<true> : tensor<10xi1>
// CHECK-DAG: %[[W_SAMP_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<10x1xf64>
// CHECK-DAG: %[[W_LOG10:.+]] = arith.constant dense<2.3025850929940459> : tensor<f64>
// CHECK-DAG: %[[W_SHRINK:.+]] = arith.constant dense<5.000000e-03> : tensor<f64>
// CHECK-DAG: %[[W_NEG_KAPPA:.+]] = arith.constant dense<-7.500000e-01> : tensor<f64>
// CHECK-DAG: %[[W_GAMMA:.+]] = arith.constant dense<5.000000e-02> : tensor<f64>
// CHECK-DAG: %[[W_TARGET:.+]] = arith.constant dense<8.000000e-01> : tensor<f64>
// CHECK-DAG: %[[W_T0:.+]] = arith.constant dense<1.000000e+01> : tensor<f64>
// CHECK-DAG: %[[W_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[W_FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK-DAG: %[[W_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[W_ZERO_1D:.+]] = arith.constant dense<0.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[W_ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[W_EPS_INIT:.+]] = arith.constant dense<4.44{{.+}}> : tensor<f64>
// CHECK-DAG: %[[W_ONE_1D:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[W_C4:.+]] = arith.constant dense<4> : tensor<i64>
// CHECK-DAG: %[[W_C5:.+]] = arith.constant dense<5> : tensor<i64>
// CHECK-DAG: %[[W_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[W_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[W_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[W_INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
//
// --- Init: RNG splits, position, U0, autodiff ---
// CHECK: enzyme.randomSplit
// CHECK: enzyme.dynamic_slice %[[W_INIT_TRACE]]
// CHECK: enzyme.reshape
// CHECK: arith.negf
// CHECK: enzyme.autodiff_region
// CHECK: }
//
// --- Warmup loop: 16 iter_args ---
// CHECK-NEXT: %[[WARMUP:.+]]:16 = enzyme.for_loop(%[[W_C0]] : tensor<i64>) to(%[[W_C5]] : tensor<i64>) step(%[[W_C1]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[W_ONE_1D]], %[[W_ONE_1D]], %[[W_ZERO_F]], %[[W_ZERO_F]], %[[W_ZERO_F]], %[[W_C0]], %[[W_EPS_INIT]], %[[W_ZERO_1D]], %[[W_ZERO_1D]], %[[W_C0]], %[[W_C0]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK-NEXT: ^bb0(%[[WI:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %[[W_INV_MASS:.+]]: tensor<1xf64>, %[[W_MASS_SQRT:.+]]: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<i64>):
//
// --- Momentum sampling with mass matrix ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}}, %[[W_P_RAW:.+]] = enzyme.random %{{.+}}, %[[W_ZERO_F]], %[[W_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>}
// CHECK-NEXT: %[[W_P_SCALED:.+]] = arith.mulf %[[W_MASS_SQRT]], %[[W_P_RAW]]
// CHECK-NEXT: %[[W_P_MASS:.+]] = arith.mulf %[[W_INV_MASS]], %[[W_P_SCALED]]
// CHECK-NEXT: %{{.+}} = enzyme.dot %[[W_P_SCALED]], %[[W_P_MASS]] {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = arith.mulf %{{.+}}, %[[W_HALF]]
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building (tested in nuts_kernel.mlir) ---
// CHECK-NEXT: %[[W_TREE:.+]]:18 = enzyme.while_loop
//
// --- Dual averaging: compute mean accept prob ---
// CHECK: %{{.+}} = arith.maxsi %[[W_TREE]]#15, %[[W_C1]]
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %[[W_MEAN_AP:.+]] = arith.divf %[[W_TREE]]#14, %{{.+}}
//
// --- Dual averaging: update gradient avg ---
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = arith.subf %[[W_TARGET]], %[[W_MEAN_AP]]
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[W_T0]]
// CHECK-NEXT: %{{.+}} = arith.divf %[[W_ONE]], %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.subf %[[W_ONE]], %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Dual averaging: log step size ---
// CHECK-NEXT: %{{.+}} = math.sqrt
// CHECK-NEXT: %{{.+}} = arith.divf %{{.+}}, %[[W_GAMMA]]
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %[[W_LOG_SS:.+]] = arith.subf
//
// --- Dual averaging: averaged log step size ---
// CHECK-NEXT: %{{.+}} = math.powf %{{.+}}, %[[W_NEG_KAPPA]]
// CHECK-NEXT: %{{.+}} = arith.subf %[[W_ONE]], %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Step size from dual averaging: exp ---
// CHECK-NEXT: %[[W_SS_CUR:.+]] = math.exp %[[W_LOG_SS]]
// CHECK-NEXT: %[[W_SS_AVG:.+]] = math.exp
//
// --- Window boundary check ---
// CHECK-NEXT: %[[W_IS_END:.+]] = arith.cmpi eq, %[[WI]], %[[W_C4]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[W_IS_END]], %[[W_SS_AVG]], %[[W_SS_CUR]]
//
// --- Clamp step size ---
// CHECK-NEXT: %{{.+}} = arith.maximumf
// CHECK-NEXT: %{{.+}} = arith.minimumf
//
// --- Welford in-window check ---
// CHECK-NEXT: %{{.+}} = arith.cmpi sgt
// CHECK-NEXT: %{{.+}} = arith.cmpi slt
// CHECK-NEXT: %[[W_IN_WINDOW:.+]] = arith.andi
//
// --- Welford update: delta, new_mean, delta2, new_m2 ---
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.subf %[[W_TREE]]#6, %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.divf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = arith.subf %[[W_TREE]]#6, %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Conditional Welford update ---
// CHECK-NEXT: %{{.+}} = enzyme.select %[[W_IN_WINDOW]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[W_IN_WINDOW]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[W_IN_WINDOW]]
//
// --- Window schedule logic ---
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = arith.andi
//
// --- Window boundary: finalize Welford & reinit (enzyme.if) ---
// CHECK-NEXT: %{{.+}}:10 = enzyme.if
// CHECK: math.sqrt
// CHECK: arith.divf
// CHECK: math.log
// CHECK: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })
//
// --- Warmup loop yield ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK-NEXT: }
//
// --- Post-warmup sampling loop: uses adapted params from warmup ---
// CHECK-NEXT: %[[W_SLOOP:.+]]:6 = enzyme.for_loop(%[[W_C0]] : tensor<i64>) to(%[[W_C10]] : tensor<i64>) step(%[[W_C1]] : tensor<i64>) iter_args(%[[WARMUP]]#0, %[[WARMUP]]#1, %[[WARMUP]]#2, %[[WARMUP]]#3, %[[W_SAMP_INIT]], %[[W_ACC_INIT]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK-NEXT: ^bb0(%[[WS_ITER:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %[[WS_SAMP:.+]]: tensor<10x1xf64>, %[[WS_ACC:.+]]: tensor<10xi1>):
//
// --- Momentum sampling with adapted mass matrix ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}}, %{{.+}} = enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>}
// CHECK-NEXT: %{{.+}} = arith.mulf %[[WARMUP]]#6
// CHECK-NEXT: %{{.+}} = arith.mulf %[[WARMUP]]#5
// CHECK-NEXT: %{{.+}} = enzyme.dot
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building ---
// CHECK-NEXT: %[[WS_TREE:.+]]:18 = enzyme.while_loop
//
// --- Sample storage ---
// CHECK: %[[WS_STORE:.+]] = arith.cmpi sge, %[[WS_ITER]], %[[W_C0]]
// CHECK-NEXT: %{{.+}} = enzyme.reshape %[[WS_TREE]]#6 : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_update_slice %[[WS_SAMP]], %{{.+}}, %[[WS_ITER]], %[[W_C0]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[WS_STORE]]
// CHECK-NEXT: %{{.+}} = enzyme.reshape %[[W_TRUE]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_update_slice %[[WS_ACC]], %{{.+}}, %[[WS_ITER]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[WS_STORE]]
//
// --- Sampling loop yield ---
// CHECK-NEXT: enzyme.yield %[[WS_TREE]]#6, %[[WS_TREE]]#7, %[[WS_TREE]]#8, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[W_SLOOP]]#4, %[[W_SLOOP]]#5, %[[W_SLOOP]]#3 : tensor<10x1xf64>, tensor<10xi1>, tensor<2xui64>
// CHECK-NEXT: }
