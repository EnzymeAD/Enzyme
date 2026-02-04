// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // adapt_step_size = true, adapt_mass_matrix = true
  func.func @warmup_both(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5>,
        name = "warmup_both", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // adapt_step_size = true, adapt_mass_matrix = false
  func.func @warmup_step_only(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = true, adapt_mass_matrix = false>,
        name = "warmup_step_only", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // adapt_step_size = false, adapt_mass_matrix = true
  func.func @warmup_mass_only(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = true>,
        name = "warmup_mass_only", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // adapt_step_size = false, adapt_mass_matrix = false
  func.func @warmup_none(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "warmup_none", selection = [[#enzyme.symbol<1>]], all_addresses = [[#enzyme.symbol<1>]], num_warmup = 10, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
  }
}

// ============================================================
// warmup_both: adapt_step_size=true, adapt_mass_matrix=true
// 16 iter_args: q, grad, U, rng, stepSize, invMass, massSqrt,
//   daGradAvg, daLogSSAvg, daGradientAvg, daStepCount, daProxCenter,
//   welfordMean, welfordM2, welfordN, windowIdx
// ============================================================
// CHECK-LABEL: func.func @warmup_both
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
//
// --- Constants ---
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant dense<true> : tensor<1xi1>
// CHECK-DAG: %[[LOG10:.+]] = arith.constant dense<2.3025850929940459> : tensor<f64>
// CHECK-DAG: %[[SHRINK:.+]] = arith.constant dense<5.000000e-03> : tensor<f64>
// CHECK-DAG: %[[FMAX:.+]] = arith.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK-DAG: %[[FMIN:.+]] = arith.constant dense<4.940660e-324> : tensor<f64>
// CHECK-DAG: %[[NEG_KAPPA:.+]] = arith.constant dense<-7.500000e-01> : tensor<f64>
// CHECK-DAG: %[[GAMMA:.+]] = arith.constant dense<5.000000e-02> : tensor<f64>
// CHECK-DAG: %[[TARGET:.+]] = arith.constant dense<8.000000e-01> : tensor<f64>
// CHECK-DAG: %[[T0:.+]] = arith.constant dense<1.000000e+01> : tensor<f64>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-DAG: %[[FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO_1D:.+]] = arith.constant dense<0.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[EPS_INIT:.+]] = arith.constant dense<4.44{{.+}}> : tensor<f64>
// CHECK-DAG: %[[ONE_1D:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[C9:.+]] = arith.constant dense<9> : tensor<i64>
// CHECK-DAG: %[[C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
//
// --- Init: RNG splits ---
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
// --- Warmup loop: 16 iter_args ---
// CHECK-NEXT: %[[WARMUP:.+]]:16 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C10]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ONE_1D]], %[[ONE_1D]], %[[ZERO_F]], %[[ZERO_F]], %[[ZERO_F]], %[[C0]], %[[EPS_INIT]], %[[ZERO_1D]], %[[ZERO_1D]], %[[C0]], %[[C0]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK-NEXT: ^bb0(%[[WI:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %[[INV_MASS:.+]]: tensor<1xf64>, %[[MASS_SQRT:.+]]: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<i64>):
//
// --- Momentum sampling with mass matrix ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}}, %[[P_RAW:.+]] = enzyme.random %{{.+}}, %[[ZERO_F]], %[[ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>}
// CHECK-NEXT: %[[P_SCALED:.+]] = arith.mulf %[[MASS_SQRT]], %[[P_RAW]]
// CHECK-NEXT: %[[P_MASS:.+]] = arith.mulf %[[INV_MASS]], %[[P_SCALED]]
// CHECK-NEXT: %{{.+}} = enzyme.dot %[[P_SCALED]], %[[P_MASS]] {{{.*}}lhs_contracting_dimensions = array<i64: 0>{{.*}}}
// CHECK-NEXT: %{{.+}} = arith.mulf %{{.+}}, %[[HALF]]
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building (tested in nuts_kernel.mlir) ---
// CHECK-NEXT: %[[TREE:.+]]:18 = enzyme.while_loop
//
// --- Dual averaging: compute mean accept prob ---
// CHECK: %{{.+}} = arith.maxsi %[[TREE]]#15, %[[C1]]
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %[[MEAN_AP:.+]] = arith.divf %[[TREE]]#14, %{{.+}}
//
// --- Dual averaging: update step count & gradient avg ---
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = arith.subf %[[TARGET]], %[[MEAN_AP]]
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[T0]]
// CHECK-NEXT: %{{.+}} = arith.divf %[[ONE]], %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.subf %[[ONE]], %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Dual averaging: log step size ---
// CHECK-NEXT: %{{.+}} = math.sqrt
// CHECK-NEXT: %{{.+}} = arith.divf %{{.+}}, %[[GAMMA]]
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %[[LOG_SS:.+]] = arith.subf
//
// --- Dual averaging: averaged log step size ---
// CHECK-NEXT: %{{.+}} = math.powf %{{.+}}, %[[NEG_KAPPA]]
// CHECK-NEXT: %{{.+}} = arith.subf %[[ONE]], %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Step size from dual averaging ---
// CHECK-NEXT: %[[SS_CUR:.+]] = math.exp %[[LOG_SS]]
// CHECK-NEXT: %[[SS_AVG:.+]] = math.exp
//
// --- Window boundary check: select averaged at boundary ---
// CHECK-NEXT: %[[IS_END:.+]] = arith.cmpi eq, %[[WI]], %[[C9]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[IS_END]], %[[SS_AVG]], %[[SS_CUR]]
//
// --- Clamp step size ---
// CHECK-NEXT: %{{.+}} = arith.maximumf %{{.+}}, %[[FMIN]]
// CHECK-NEXT: %{{.+}} = arith.minimumf %{{.+}}, %[[FMAX]]
//
// --- Welford in-window check ---
// CHECK-NEXT: %{{.+}} = arith.cmpi sgt
// CHECK-NEXT: %{{.+}} = arith.cmpi slt
// CHECK-NEXT: %[[IN_WINDOW:.+]] = arith.andi
//
// --- Welford update: delta, new_mean, delta2, new_m2 ---
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.subf %[[TREE]]#6, %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.divf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = arith.subf %[[TREE]]#6, %{{.+}}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Conditional Welford update ---
// CHECK-NEXT: %{{.+}} = enzyme.select %[[IN_WINDOW]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[IN_WINDOW]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[IN_WINDOW]]
//
// --- Window schedule logic ---
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = arith.andi
//
// --- Window boundary: finalize Welford & reinit DA (enzyme.if) ---
// CHECK-NEXT: %{{.+}}:10 = enzyme.if
// CHECK: arith.subi
// CHECK-NEXT: arith.sitofp
// CHECK-NEXT: "enzyme.broadcast"
// CHECK-NEXT: arith.divf
// CHECK-NEXT: arith.sitofp
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.divf
// CHECK-NEXT: "enzyme.broadcast"
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.divf %[[SHRINK]]
// CHECK-NEXT: "enzyme.broadcast"
// CHECK-NEXT: arith.addf
// CHECK-NEXT: math.sqrt
// CHECK-NEXT: arith.divf %[[ONE_1D]]
// CHECK-NEXT: math.log
// CHECK-NEXT: arith.addf %{{.+}}, %[[LOG10]]
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })
//
// --- Warmup loop yield ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK-NEXT: }
//
// --- Post-warmup sampling loop: 6 iter_args with adapted params ---
// CHECK-NEXT: %[[SLOOP:.+]]:6 = enzyme.for_loop(%[[C0]] : tensor<i64>) to(%[[C1]] : tensor<i64>) step(%[[C1]] : tensor<i64>) iter_args(%[[WARMUP]]#0, %[[WARMUP]]#1, %[[WARMUP]]#2, %[[WARMUP]]#3, %{{.+}}, %[[ACC_INIT]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1> {
// CHECK-NEXT: ^bb0(%[[SI:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %[[S_SAMP:.+]]: tensor<1x1xf64>, %[[S_ACC:.+]]: tensor<1xi1>):
//
// --- Momentum with adapted mass matrix from warmup ---
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
// CHECK-NEXT: %[[S_TREE:.+]]:18 = enzyme.while_loop
//
// --- Sample storage ---
// CHECK: %[[S_STORE:.+]] = arith.cmpi sge, %[[SI]], %[[C0]]
// CHECK-NEXT: %{{.+}} = enzyme.reshape %[[S_TREE]]#6 : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_update_slice %[[S_SAMP]], %{{.+}}, %[[SI]], %[[C0]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[S_STORE]]
// CHECK-NEXT: %{{.+}} = enzyme.reshape %[[TRUE]] : (tensor<i1>) -> tensor<1xi1>
// CHECK-NEXT: %{{.+}} = enzyme.dynamic_update_slice %[[S_ACC]], %{{.+}}, %[[SI]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[S_STORE]]
//
// --- Sampling yield ---
// CHECK-NEXT: enzyme.yield %[[S_TREE]]#6, %[[S_TREE]]#7, %[[S_TREE]]#8, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SLOOP]]#4, %[[SLOOP]]#5, %[[SLOOP]]#3 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
// CHECK-NEXT: }

// ============================================================
// warmup_step_only: adapt_step_size=true, adapt_mass_matrix=false
// 13 iter_args: no welford_mean, welford_m2, welford_n
// ============================================================
// CHECK-LABEL: func.func @warmup_step_only
// CHECK-SAME: (%{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
//
// CHECK-DAG: %[[S_LOG10:.+]] = arith.constant dense<2.3025850929940459> : tensor<f64>
// CHECK-DAG: %[[S_FMAX:.+]] = arith.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK-DAG: %[[S_FMIN:.+]] = arith.constant dense<4.940660e-324> : tensor<f64>
// CHECK-DAG: %[[S_NEG_KAPPA:.+]] = arith.constant dense<-7.500000e-01> : tensor<f64>
// CHECK-DAG: %[[S_GAMMA:.+]] = arith.constant dense<5.000000e-02> : tensor<f64>
// CHECK-DAG: %[[S_TARGET:.+]] = arith.constant dense<8.000000e-01> : tensor<f64>
// CHECK-DAG: %[[S_T0:.+]] = arith.constant dense<1.000000e+01> : tensor<f64>
// CHECK-DAG: %[[S_EPS_INIT:.+]] = arith.constant dense<4.44{{.+}}> : tensor<f64>
// CHECK-DAG: %[[S_C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[S_C9:.+]] = arith.constant dense<9> : tensor<i64>
// CHECK-DAG: %[[S_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[S_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[S_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[S_ONE_1D:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[S_ZERO_F:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
//
// --- Init ---
// CHECK: enzyme.randomSplit
// CHECK: enzyme.dynamic_slice
// CHECK: enzyme.reshape
// CHECK: arith.negf
// CHECK: enzyme.autodiff_region
// CHECK: }
//
// --- Warmup loop: 13 iter_args (no Welford mean/m2/n) ---
// CHECK-NEXT: %[[S_WARMUP:.+]]:13 = enzyme.for_loop(%[[S_C0]] : tensor<i64>) to(%[[S_C10]] : tensor<i64>) step(%[[S_C1]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[S_ONE_1D]], %[[S_ONE_1D]], %[[S_ZERO_F]], %[[S_ZERO_F]], %[[S_ZERO_F]], %[[S_C0]], %[[S_EPS_INIT]], %[[S_C0]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64> {
// CHECK-NEXT: ^bb0(%[[S_WI:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>):
//
// --- Momentum sampling ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}}, %{{.+}} = enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = enzyme.dot
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building ---
// CHECK-NEXT: %[[S_WTREE:.+]]:18 = enzyme.while_loop
//
// --- Dual averaging update (same as warmup_both) ---
// CHECK: %{{.+}} = arith.maxsi %[[S_WTREE]]#15, %[[S_C1]]
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = arith.divf %[[S_WTREE]]#14
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = arith.subf %[[S_TARGET]]
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[S_T0]]
// CHECK-NEXT: %{{.+}} = arith.divf %[[S_ONE]]
// CHECK-NEXT: %{{.+}} = arith.subf %[[S_ONE]]
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = math.sqrt
// CHECK-NEXT: %{{.+}} = arith.divf %{{.+}}, %[[S_GAMMA]]
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.subf
// CHECK-NEXT: %{{.+}} = math.powf %{{.+}}, %[[S_NEG_KAPPA]]
// CHECK-NEXT: %{{.+}} = arith.subf %[[S_ONE]]
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = math.exp
// CHECK-NEXT: %{{.+}} = math.exp
//
// --- Window boundary: select averaged at end ---
// CHECK-NEXT: %{{.+}} = arith.cmpi eq, %[[S_WI]], %[[S_C9]]
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = arith.maximumf %{{.+}}, %[[S_FMIN]]
// CHECK-NEXT: %{{.+}} = arith.minimumf %{{.+}}, %[[S_FMAX]]
//
// --- NO Welford update (adapt_mass_matrix=false) ---
// --- Window schedule logic ---
// CHECK-NEXT: %{{.+}} = arith.cmpi sgt
// CHECK-NEXT: %{{.+}} = arith.cmpi slt
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = arith.andi
//
// --- Window boundary enzyme.if: 7 results (no Welford) ---
// CHECK-NEXT: %{{.+}}:7 = enzyme.if
// CHECK: math.log
// CHECK: arith.addf %{{.+}}, %[[S_LOG10]]
// CHECK: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })
//
// --- Warmup yield: 13 values ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>
// CHECK-NEXT: }
//
// --- Post-warmup sampling loop ---
// CHECK-NEXT: %[[S_SLOOP:.+]]:6 = enzyme.for_loop(%[[S_C0]] : tensor<i64>) to(%[[S_C1]] : tensor<i64>) step(%[[S_C1]] : tensor<i64>) iter_args(%[[S_WARMUP]]#0, %[[S_WARMUP]]#1, %[[S_WARMUP]]#2, %[[S_WARMUP]]#3, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1> {
// CHECK: enzyme.while_loop
// CHECK: return %[[S_SLOOP]]#4, %[[S_SLOOP]]#5, %[[S_SLOOP]]#3 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
// CHECK-NEXT: }

// ============================================================
// warmup_mass_only: adapt_step_size=false, adapt_mass_matrix=true
// 16 iter_args: same layout as warmup_both but step size unchanged
// ============================================================
// CHECK-LABEL: func.func @warmup_mass_only
// CHECK-SAME: (%{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
//
// CHECK-DAG: %[[M_SHRINK:.+]] = arith.constant dense<5.000000e-03> : tensor<f64>
// CHECK-DAG: %[[M_FMAX:.+]] = arith.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK-DAG: %[[M_FMIN:.+]] = arith.constant dense<4.940660e-324> : tensor<f64>
// CHECK-DAG: %[[M_C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[M_C9:.+]] = arith.constant dense<9> : tensor<i64>
// CHECK-DAG: %[[M_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[M_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[M_ONE_1D:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[M_ZERO_1D:.+]] = arith.constant dense<0.000000e+00> : tensor<1xf64>
// CHECK-DAG: %[[M_EPS_INIT:.+]] = arith.constant dense<4.44{{.+}}> : tensor<f64>
//
// --- Init ---
// CHECK: enzyme.randomSplit
// CHECK: arith.negf
// CHECK: enzyme.autodiff_region
// CHECK: }
//
// --- Warmup loop: 16 iter_args (includes Welford state) ---
// CHECK-NEXT: %[[M_WARMUP:.+]]:16 = enzyme.for_loop(%[[M_C0]] : tensor<i64>) to(%[[M_C10]] : tensor<i64>) step(%[[M_C1]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[M_ONE_1D]], %[[M_ONE_1D]], %{{.+}}, %{{.+}}, %{{.+}}, %[[M_C0]], %[[M_EPS_INIT]], %[[M_ZERO_1D]], %[[M_ZERO_1D]], %[[M_C0]], %[[M_C0]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK-NEXT: ^bb0(%[[M_WI:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %[[M_INV:.+]]: tensor<1xf64>, %[[M_SQRT:.+]]: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<i64>):
//
// --- Momentum sampling with mass matrix ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}}, %{{.+}} = enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>}
// CHECK-NEXT: %{{.+}} = arith.mulf %[[M_SQRT]]
// CHECK-NEXT: %{{.+}} = arith.mulf %[[M_INV]]
// CHECK-NEXT: %{{.+}} = enzyme.dot
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building ---
// CHECK-NEXT: %[[M_TREE:.+]]:18 = enzyme.while_loop
//
// --- Step size: trivially select same value (no DA) ---
// CHECK: %[[M_IS_END:.+]] = arith.cmpi eq, %[[M_WI]], %[[M_C9]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[M_IS_END]]
// CHECK-NEXT: %{{.+}} = arith.maximumf %{{.+}}, %[[M_FMIN]]
// CHECK-NEXT: %{{.+}} = arith.minimumf %{{.+}}, %[[M_FMAX]]
//
// --- Welford in-window check ---
// CHECK-NEXT: %{{.+}} = arith.cmpi sgt
// CHECK-NEXT: %{{.+}} = arith.cmpi slt
// CHECK-NEXT: %[[M_IN_WIN:.+]] = arith.andi
//
// --- Welford update ---
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = arith.sitofp
// CHECK-NEXT: %{{.+}} = "enzyme.broadcast"
// CHECK-NEXT: %{{.+}} = arith.subf %[[M_TREE]]#6
// CHECK-NEXT: %{{.+}} = arith.divf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK-NEXT: %{{.+}} = arith.subf %[[M_TREE]]#6
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- Conditional Welford update ---
// CHECK-NEXT: %{{.+}} = enzyme.select %[[M_IN_WIN]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[M_IN_WIN]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[M_IN_WIN]]
//
// --- Window schedule ---
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = arith.andi
//
// --- Window boundary enzyme.if: 10 results (includes Welford finalization) ---
// CHECK-NEXT: %{{.+}}:10 = enzyme.if
// CHECK: math.sqrt
// CHECK: arith.divf %[[M_ONE_1D]]
// CHECK: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })
//
// --- Warmup yield: 16 values ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK-NEXT: }
//
// --- Post-warmup sampling with adapted mass matrix ---
// CHECK-NEXT: %[[M_SLOOP:.+]]:6 = enzyme.for_loop(%[[M_C0]] : tensor<i64>) to(%[[M_C1]] : tensor<i64>) step(%[[M_C1]] : tensor<i64>) iter_args(%[[M_WARMUP]]#0, %[[M_WARMUP]]#1, %[[M_WARMUP]]#2, %[[M_WARMUP]]#3, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1> {
// CHECK: arith.mulf %[[M_WARMUP]]#6
// CHECK-NEXT: %{{.+}} = arith.mulf %[[M_WARMUP]]#5
// CHECK: enzyme.while_loop
// CHECK: return %[[M_SLOOP]]#4, %[[M_SLOOP]]#5, %[[M_SLOOP]]#3 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
// CHECK-NEXT: }

// ============================================================
// warmup_none: adapt_step_size=false, adapt_mass_matrix=false
// 13 iter_args, trivial enzyme.if (both branches identical)
// ============================================================
// CHECK-LABEL: func.func @warmup_none
// CHECK-SAME: (%{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>) -> (tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>)
//
// CHECK-DAG: %[[N_FMAX:.+]] = arith.constant dense<1.7976931348623157E+308> : tensor<f64>
// CHECK-DAG: %[[N_FMIN:.+]] = arith.constant dense<4.940660e-324> : tensor<f64>
// CHECK-DAG: %[[N_C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[N_C9:.+]] = arith.constant dense<9> : tensor<i64>
// CHECK-DAG: %[[N_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[N_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[N_EPS_INIT:.+]] = arith.constant dense<4.44{{.+}}> : tensor<f64>
// CHECK-DAG: %[[N_ONE_1D:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
//
// --- Init ---
// CHECK: enzyme.randomSplit
// CHECK: arith.negf
// CHECK: enzyme.autodiff_region
// CHECK: }
//
// --- Warmup loop: 13 iter_args ---
// CHECK-NEXT: %[[N_WARMUP:.+]]:13 = enzyme.for_loop(%[[N_C0]] : tensor<i64>) to(%[[N_C10]] : tensor<i64>) step(%[[N_C1]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[N_ONE_1D]], %[[N_ONE_1D]], %{{.+}}, %{{.+}}, %{{.+}}, %[[N_C0]], %[[N_EPS_INIT]], %[[N_C0]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64> {
// CHECK-NEXT: ^bb0(%[[N_WI:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<i64>):
//
// --- Momentum sampling ---
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}} = enzyme.randomSplit
// CHECK-NEXT: %{{.+}}, %{{.+}} = enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution NORMAL>}
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = enzyme.dot
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
//
// --- NUTS tree building ---
// CHECK-NEXT: %{{.+}}:18 = enzyme.while_loop
//
// --- Step size: trivial select (no dual averaging) ---
// CHECK: %[[N_IS_END:.+]] = arith.cmpi eq, %[[N_WI]], %[[N_C9]]
// CHECK-NEXT: %{{.+}} = enzyme.select %[[N_IS_END]]
// CHECK-NEXT: %{{.+}} = arith.maximumf %{{.+}}, %[[N_FMIN]]
// CHECK-NEXT: %{{.+}} = arith.minimumf %{{.+}}, %[[N_FMAX]]
//
// --- Window schedule logic (no Welford) ---
// CHECK-NEXT: %{{.+}} = arith.cmpi sgt
// CHECK-NEXT: %{{.+}} = arith.cmpi slt
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.cmpi eq
// CHECK-NEXT: %{{.+}} = arith.andi
// CHECK-NEXT: %{{.+}} = arith.addi
// CHECK-NEXT: %{{.+}} = enzyme.select
// CHECK-NEXT: %{{.+}} = arith.andi
//
// --- Trivial enzyme.if: both branches yield identical values ---
// CHECK-NEXT: %{{.+}}:7 = enzyme.if
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })
//
// --- Warmup yield: 13 values ---
// CHECK-NEXT: enzyme.yield {{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i64>
// CHECK-NEXT: }
//
// --- Post-warmup sampling ---
// CHECK-NEXT: %[[N_SLOOP:.+]]:6 = enzyme.for_loop(%[[N_C0]] : tensor<i64>) to(%[[N_C1]] : tensor<i64>) step(%[[N_C1]] : tensor<i64>) iter_args(%[[N_WARMUP]]#0, %[[N_WARMUP]]#1, %[[N_WARMUP]]#2, %[[N_WARMUP]]#3, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<1x1xf64>, tensor<1xi1> {
// CHECK: enzyme.while_loop
// CHECK: return %[[N_SLOOP]]#4, %[[N_SLOOP]]#5, %[[N_SLOOP]]#3 : tensor<1x1xf64>, tensor<1xi1>, tensor<2xui64>
// CHECK-NEXT: }
