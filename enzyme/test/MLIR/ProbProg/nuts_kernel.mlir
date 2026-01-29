// RUN: %eopt --probprog %s | FileCheck %s


module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @nuts(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 3, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "nuts", selection = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 1 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK: %[[CST_NEG1:.+]] = arith.constant dense<-1> : tensor<i64>
// CHECK: %[[CST_INF:.+]] = arith.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK: %[[CST_NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK: %[[CST_1_I64:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[CST_CKPT_ZEROS:.+]] = arith.constant dense<0.000000e+00> : tensor<3x1xf64>
// CHECK: %[[CST_3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK: %[[CST_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK: %[[CST_FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK: %[[CST_0_I64:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK: %[[CST_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK: %[[CST_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[CST_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK: %[[CST_MAX_DELTA:.+]] = arith.constant dense<1.000000e+03> : tensor<f64>
// CHECK: %[[CST_EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>

// CHECK: %[[INIT_TRACE:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK: %[[RNG_SPLIT1:.+]]:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[RNG_SPLIT2:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[Q0:.+]] = enzyme.getFlattenedSamplesFromTrace %[[INIT_TRACE]] {selection = {{.*}}} : (!enzyme.Trace) -> tensor<1xf64>
// CHECK: %[[WEIGHT:.+]] = enzyme.getWeightFromTrace %[[INIT_TRACE]] : (!enzyme.Trace) -> tensor<f64>
// CHECK: %[[U0:.+]] = arith.negf %[[WEIGHT]] : tensor<f64>

// CHECK: %[[INIT_GRAD:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[CST_ONE]]) {
// CHECK: ^bb0(%[[AD_ARG:.+]]: tensor<1xf64>):
// CHECK:   %[[UPDATE_CALL:.+]]:3 = func.call @test.update{{.*}}(%[[INIT_TRACE]], %[[AD_ARG]], %[[RNG_SPLIT2]]#1, %arg1, %arg2)
// CHECK:   %[[NEG_W:.+]] = arith.negf %[[UPDATE_CALL]]#1 : tensor<f64>
// CHECK:   enzyme.yield %[[NEG_W]], %[[UPDATE_CALL]]#2 : tensor<f64>, tensor<2xui64>
// CHECK: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}

// CHECK: %[[RNG_SPLIT3:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT2]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK: %[[RNG_SPLIT4:.+]] = enzyme.randomSplit %[[RNG_SPLIT3]]#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK: %[[RNG_OUT:.+]], %[[P0:.+]] = enzyme.random %[[RNG_SPLIT4]], %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

// CHECK: %[[P0_DOT:.+]] = enzyme.dot %[[P0]], %[[P0]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK: %[[K0:.+]] = arith.mulf %[[P0_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK: %[[H0:.+]] = arith.addf %[[U0]], %[[K0]] : tensor<f64>

// Main tree building loop
// CHECK: %[[TREE:.+]]:18 = enzyme.while_loop(%[[Q0]], %[[P0]], %[[INIT_GRAD]]#2, %[[Q0]], %[[P0]], %[[INIT_GRAD]]#2, %[[Q0]], %[[INIT_GRAD]]#2, %[[U0]], %[[H0]], %[[CST_0_I64]], %[[CST_ZERO]], %[[CST_FALSE]], %[[CST_FALSE]], %[[CST_ZERO]], %[[CST_0_I64]], %[[P0]], %[[RNG_SPLIT3]]#2 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>) -> tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64> condition {
// CHECK: ^bb0({{.*}}):

// Main loop condition: depth < max_tree_depth && !turning && !diverging
// CHECK:   %[[DEPTH_LT:.+]] = arith.cmpi slt, %{{.*}}, %[[CST_3]] : tensor<i64>
// CHECK:   %[[NOT_TURN:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK:   %[[NOT_DIV:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK:   %[[COND1:.+]] = arith.andi %[[DEPTH_LT]], %[[NOT_TURN]] : tensor<i1>
// CHECK:   %[[COND2:.+]] = arith.andi %[[COND1]], %[[NOT_DIV]] : tensor<i1>
// CHECK:   enzyme.yield %[[COND2]] : tensor<i1>
// CHECK: } body {
// CHECK: ^bb0({{.*}}):

// Direction sampling
// CHECK:   %[[DIR_SPLIT:.+]]:3 = enzyme.randomSplit %{{.*}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK:   %[[DIR_RNG:.+]], %[[DIR_RAND:.+]] = enzyme.random %[[DIR_SPLIT]]#1, %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK:   %[[GOING_RIGHT:.+]] = arith.cmpf olt, %[[DIR_RAND]], %[[CST_HALF]] : tensor<f64>
// CHECK:   %[[SUBTREE_SPLIT:.+]]:2 = enzyme.randomSplit %[[DIR_SPLIT]]#2 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)

// CHECK:   %[[NUM_PROPS:.+]] = arith.shli %[[CST_1_I64]], %{{.*}} : tensor<i64>

// Subtree building loop
// CHECK:   %[[SUBTREE:.+]]:21 = enzyme.while_loop({{.*}} : {{.*}}) -> {{.*}} condition {
// CHECK:   ^bb0({{.*}}):

// Subtree condition: num_proposals < 2^depth && !turning && !diverging
// CHECK:     %[[SUB_LT:.+]] = arith.cmpi slt, %{{.*}}, %[[NUM_PROPS]] : tensor<i64>
// CHECK:     %[[SUB_NOT_TURN:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK:     %[[SUB_NOT_DIV:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK:     %[[SUB_COND1:.+]] = arith.andi %[[SUB_LT]], %[[SUB_NOT_TURN]] : tensor<i1>
// CHECK:     %[[SUB_COND2:.+]] = arith.andi %[[SUB_COND1]], %[[SUB_NOT_DIV]] : tensor<i1>
// CHECK:     enzyme.yield %[[SUB_COND2]] : tensor<i1>
// CHECK:   } body {
// CHECK:   ^bb0({{.*}}):

// Select leaf based on direction
// CHECK:     %[[SEL_Q:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:     %[[SEL_P:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:     %[[SEL_GRAD:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:     %[[LEAF_SPLIT:.+]]:2 = enzyme.randomSplit %{{.*}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)

// Leapfrog step
// CHECK:     %[[STEP_SEL:.+]] = enzyme.select %[[GOING_RIGHT]], %[[CST_EPS]], %[[CST_NEG_EPS]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK:     %[[STEP_BCAST:.+]] = "enzyme.broadcast"(%[[STEP_SEL]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:     %[[HALF_STEP:.+]] = arith.mulf %[[STEP_SEL]], %[[CST_HALF]] : tensor<f64>
// CHECK:     %[[HALF_STEP_BCAST:.+]] = "enzyme.broadcast"(%[[HALF_STEP]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:     %[[GRAD_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[SEL_GRAD]] : tensor<1xf64>
// CHECK:     %[[P_HALF:.+]] = arith.subf %[[SEL_P]], %[[GRAD_SCALED]] : tensor<1xf64>
// CHECK:     %[[P_HALF_SCALED:.+]] = arith.mulf %[[STEP_BCAST]], %[[P_HALF]] : tensor<1xf64>
// CHECK:     %[[Q_NEW:.+]] = arith.addf %[[SEL_Q]], %[[P_HALF_SCALED]] : tensor<1xf64>

// Gradient (inside loop)
// CHECK:     %[[GRAD_RES:.+]]:3 = enzyme.autodiff_region(%[[Q_NEW]], %[[CST_ONE]]) {
// CHECK:     ^bb0({{.*}}):
// CHECK:       func.call @test.update{{.*}}
// CHECK:       arith.negf
// CHECK:       enzyme.yield
// CHECK:     }

// Second half-step: p_new = p_half - (eps/2) * grad_new
// CHECK:     %[[GRAD_NEW_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[GRAD_RES]]#2 : tensor<1xf64>
// CHECK:     %[[P_NEW:.+]] = arith.subf %[[P_HALF]], %[[GRAD_NEW_SCALED]] : tensor<1xf64>

// Energy calculations and divergence check
// Kinetic energy: K = 0.5 * p^T * p
// CHECK:     %[[P_DOT:.+]] = enzyme.dot %[[P_NEW]], %[[P_NEW]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:     %[[K_NEW:.+]] = arith.mulf %[[P_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK:     %[[H_NEW:.+]] = arith.addf %[[GRAD_RES]]#0, %[[K_NEW]] : tensor<f64>

// Delta energy for divergence: H - H0
// CHECK:     %[[DELTA_E:.+]] = arith.subf %[[H_NEW]], %[[H0]] : tensor<f64>

// NaN handling: if delta != delta, set to inf
// CHECK:     %[[IS_NAN:.+]] = arith.cmpf une, %[[DELTA_E]], %[[DELTA_E]] : tensor<f64>
// CHECK:     %[[DELTA_SAFE:.+]] = arith.select %[[IS_NAN]], %[[CST_INF]], %[[DELTA_E]] : tensor<i1>, tensor<f64>

// Weight = -delta_energy
// CHECK:     %[[NEG_DELTA:.+]] = arith.negf %[[DELTA_SAFE]] : tensor<f64>

// Divergence check: delta > max_delta_energy
// CHECK:     %[[DIVERGING:.+]] = arith.cmpf ogt, %[[DELTA_SAFE]], %[[CST_MAX_DELTA]] : tensor<f64>

// Acceptance probability: min(1, exp(-delta))
// CHECK:     arith.negf %[[DELTA_SAFE]] : tensor<f64>
// CHECK:     math.exp
// CHECK:     %[[ACCEPT_PROB:.+]] = arith.minimumf {{.*}}, %[[CST_ONE]] : tensor<f64>

// Tree combination: first leaf vs subsequent leaves
// CHECK:     %[[IS_FIRST:.+]] = arith.cmpi eq, %{{.*}}, %[[CST_0_I64]] : tensor<i64>
// CHECK:     %[[IF_RES:.+]]:18 = enzyme.if(%[[IS_FIRST]]) ({

// First leaf case: initialize tree state with current leaf values
// CHECK:       enzyme.yield %[[Q_NEW]], %[[P_NEW]], %[[GRAD_RES]]#2, %[[Q_NEW]], %[[P_NEW]], %[[GRAD_RES]]#2, %[[Q_NEW]], %[[GRAD_RES]]#2, %[[GRAD_RES]]#0, %[[H_NEW]], %[[CST_0_I64]], %[[NEG_DELTA]], %[[CST_FALSE]], %[[DIVERGING]], %[[ACCEPT_PROB]], %[[CST_1_I64]], %[[P_NEW]], %[[GRAD_RES]]#1 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>
// CHECK:     }, {

// Subsequent leaves: combine trees using uniform transition kernel
// CHECK:       %[[COMB_Q_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[Q_NEW]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[COMB_P_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[P_NEW]] : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[COMB_GRAD_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[GRAD_RES]]#2 : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[COMB_Q_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %[[Q_NEW]], %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[COMB_P_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %[[P_NEW]], %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[COMB_GRAD_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %[[GRAD_RES]]#2, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>

// Log weight combination: log(exp(w1) + exp(w2))
// CHECK:       %[[COMB_WEIGHT:.+]] = enzyme.log_add_exp %{{.*}}, %[[NEG_DELTA]] : (tensor<f64>, tensor<f64>) -> tensor<f64>

// Uniform transition probability: sigmoid(new_weight - current_weight)
// CHECK:       %[[WEIGHT_DIFF2:.+]] = arith.subf %[[NEG_DELTA]], %{{.*}} : tensor<f64>
// CHECK:       %[[TRANS_PROB2:.+]] = enzyme.logistic %[[WEIGHT_DIFF2]] : (tensor<f64>) -> tensor<f64>

// Random selection for proposal
// CHECK:       %[[UNIF_RNG:.+]], %[[UNIF_RAND:.+]] = enzyme.random %{{.*}}, %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK:       %[[ACCEPT_PROP:.+]] = arith.cmpf olt, %[[UNIF_RAND]], %[[TRANS_PROB2]] : tensor<f64>

// Select proposal based on acceptance
// CHECK:       %[[SEL_PROP_Q:.+]] = enzyme.select %[[ACCEPT_PROP]], %[[Q_NEW]], %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[SEL_PROP_GRAD:.+]] = enzyme.select %[[ACCEPT_PROP]], %[[GRAD_RES]]#2, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:       %[[SEL_PROP_U:.+]] = enzyme.select %[[ACCEPT_PROP]], %[[GRAD_RES]]#0, %{{.*}} : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK:       %[[SEL_PROP_H:.+]] = enzyme.select %[[ACCEPT_PROP]], %[[H_NEW]], %{{.*}} : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>

// Update tree metadata
// CHECK:       %[[INC_DEPTH:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK:       %[[OR_DIV:.+]] = arith.ori %{{.*}}, %[[DIVERGING]] : tensor<i1>
// CHECK:       %[[ADD_ACCEPT:.+]] = arith.addf %{{.*}}, %[[ACCEPT_PROB]] : tensor<f64>
// CHECK:       %[[INC_PROPS:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK:       %[[ADD_PSUM:.+]] = arith.addf %{{.*}}, %[[P_NEW]] : tensor<1xf64>

// CHECK:       enzyme.yield %[[COMB_Q_LEFT]], %[[COMB_P_LEFT]], %[[COMB_GRAD_LEFT]], %[[COMB_Q_RIGHT]], %[[COMB_P_RIGHT]], %[[COMB_GRAD_RIGHT]], %[[SEL_PROP_Q]], %[[SEL_PROP_GRAD]], %[[SEL_PROP_U]], %[[SEL_PROP_H]], %[[INC_DEPTH]], %[[COMB_WEIGHT]], %{{.*}}, %[[OR_DIV]], %[[ADD_ACCEPT]], %[[INC_PROPS]], %[[ADD_PSUM]], %[[UNIF_RNG]] : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>
// CHECK:     }) : (tensor<i1>) -> (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>)

// Checkpoint index calculations
// CHECK:     %[[SHRUI:.+]] = arith.shrui %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK:     %[[POPCOUNT1:.+]] = enzyme.popcount %[[SHRUI]] : (tensor<i64>) -> tensor<i64>
// CHECK:     %[[ADDI1:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK:     %[[XORI:.+]] = arith.xori %{{.*}}, %[[CST_NEG1]] : tensor<i64>
// CHECK:     %[[ANDI1:.+]] = arith.andi %[[XORI]], %[[ADDI1]] : tensor<i64>
// CHECK:     %[[SUBI:.+]] = arith.subi %[[ANDI1]], %[[CST_1_I64]] : tensor<i64>
// CHECK:     %[[POPCOUNT2:.+]] = enzyme.popcount %[[SUBI]] : (tensor<i64>) -> tensor<i64>
// CHECK:     %[[DIFF:.+]] = arith.subi %[[POPCOUNT1]], %[[POPCOUNT2]] : tensor<i64>
// CHECK:     %[[IDX_MIN:.+]] = arith.addi %[[DIFF]], %[[CST_1_I64]] : tensor<i64>

// Checkpoint update
// CHECK:     %[[LEAF_MOD:.+]] = arith.andi %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK:     %[[IS_EVEN:.+]] = arith.cmpi eq, %[[LEAF_MOD]], %[[CST_0_I64]] : tensor<i64>
// CHECK:     %[[CKPT_IF:.+]]:2 = enzyme.if(%[[IS_EVEN]]) ({
// CHECK:       %{{.*}} = enzyme.dynamic_update %{{.*}}, %[[POPCOUNT1]], %[[P_NEW]] : (tensor<3x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<3x1xf64>
// CHECK:       %{{.*}} = enzyme.dynamic_update %{{.*}}, %[[POPCOUNT1]], %[[IF_RES]]#16 : (tensor<3x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<3x1xf64>
// CHECK:       enzyme.yield
// CHECK:     }, {
// CHECK:       enzyme.yield
// CHECK:     })

// Iterative turning check
// CHECK:     %[[TURN_LOOP:.+]]:2 = enzyme.while_loop(%[[POPCOUNT1]], %[[CST_FALSE]] : tensor<i64>, tensor<i1>) -> tensor<i64>, tensor<i1> condition {
// CHECK:     ^bb0(%[[TURN_IDX:.+]]: tensor<i64>, %[[TURN_FLAG:.+]]: tensor<i1>):
// CHECK:       %[[GE:.+]] = arith.cmpi sge, %[[TURN_IDX]], %[[IDX_MIN]] : tensor<i64>
// CHECK:       %[[NOT_TURN2:.+]] = arith.xori %[[TURN_FLAG]], %[[CST_TRUE]] : tensor<i1>
// CHECK:       %[[TURN_COND:.+]] = arith.andi %[[GE]], %[[NOT_TURN2]] : tensor<i1>
// CHECK:       enzyme.yield %[[TURN_COND]] : tensor<i1>
// CHECK:     } body {
// CHECK:     ^bb0(%[[TURN_IDX2:.+]]: tensor<i64>, %[[TURN_FLAG2:.+]]: tensor<i1>):

// Extract checkpoint momentum
// CHECK:       %[[P_CKPT:.+]] = enzyme.dynamic_extract %[[CKPT_IF]]#0, %[[TURN_IDX2]] : (tensor<3x1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK:       %[[PSUM_CKPT:.+]] = enzyme.dynamic_extract %[[CKPT_IF]]#1, %[[TURN_IDX2]] : (tensor<3x1xf64>, tensor<i64>) -> tensor<1xf64>

// Compute subtree momentum sum: p_sum - pSumCkpt + pCkpt
// CHECK:       %[[SUB1:.+]] = arith.subf %[[IF_RES]]#16, %[[PSUM_CKPT]] : tensor<1xf64>
// CHECK:       %[[SUBTREE_PSUM:.+]] = arith.addf %[[SUB1]], %[[P_CKPT]] : tensor<1xf64>

// Dynamic termination criterion: p_sum_centered = p_sum - (p_left + p_right) / 2
// CHECK:       %[[HALF_BCAST:.+]] = "enzyme.broadcast"(%[[CST_HALF]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:       %[[P_SUM2:.+]] = arith.addf %[[P_CKPT]], %[[P_NEW]] : tensor<1xf64>
// CHECK:       %[[P_AVG:.+]] = arith.mulf %[[HALF_BCAST]], %[[P_SUM2]] : tensor<1xf64>
// CHECK:       %[[P_CENTERED:.+]] = arith.subf %[[SUBTREE_PSUM]], %[[P_AVG]] : tensor<1xf64>

// Check turning condition via dot products
// CHECK:       %[[DOT_LEFT:.+]] = enzyme.dot %[[P_CKPT]], %[[P_CENTERED]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:       %[[DOT_RIGHT:.+]] = enzyme.dot %[[P_NEW]], %[[P_CENTERED]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:       %[[LE_LEFT:.+]] = arith.cmpf ole, %[[DOT_LEFT]], %[[CST_ZERO]] : tensor<f64>
// CHECK:       %[[LE_RIGHT:.+]] = arith.cmpf ole, %[[DOT_RIGHT]], %[[CST_ZERO]] : tensor<f64>
// CHECK:       %[[TURN_DETECTED:.+]] = arith.ori %[[LE_LEFT]], %[[LE_RIGHT]] : tensor<i1>

// CHECK:       %[[NEXT_IDX:.+]] = arith.subi %[[TURN_IDX2]], %[[CST_1_I64]] : tensor<i64>
// CHECK:       enzyme.yield %[[NEXT_IDX]], %[[TURN_DETECTED]] : tensor<i64>, tensor<i1>
// CHECK:     }

// Select turning (skip on first leaf)
// CHECK:     %[[FINAL_TURN:.+]] = enzyme.select %[[IS_FIRST]], %[[CST_FALSE]], %[[TURN_LOOP]]#1 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>

// Increment leaf index
// CHECK:     %[[NEXT_LEAF:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>

// Subtree body yield
// CHECK:     enzyme.yield {{.*}}
// CHECK:   }

// Tree combination after subtree build
// CHECK:   %[[SEL_Q_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[SUBTREE]]#0 : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[SEL_P_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[SUBTREE]]#1 : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[SEL_GRAD_LEFT:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[SUBTREE]]#2 : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[SEL_Q_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %[[SUBTREE]]#3, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[SEL_P_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %[[SUBTREE]]#4, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[SEL_GRAD_RIGHT:.+]] = enzyme.select %[[GOING_RIGHT]], %[[SUBTREE]]#5, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>

// Log weight combination
// CHECK:   %[[COMBINED_WEIGHT:.+]] = enzyme.log_add_exp %{{.*}}, %[[SUBTREE]]#11 : (tensor<f64>, tensor<f64>) -> tensor<f64>

// Biased transition probability: min(1, exp(new_weight - current_weight))
// CHECK:   %[[WEIGHT_DIFF:.+]] = arith.subf %[[SUBTREE]]#11, %{{.*}} : tensor<f64>
// CHECK:   %[[EXP_DIFF:.+]] = math.exp %[[WEIGHT_DIFF]] : tensor<f64>
// CHECK:   %[[TRANS_PROB:.+]] = arith.minimumf %[[EXP_DIFF]], %[[CST_ONE]] : tensor<f64>

// Zero probability when turning or diverging
// CHECK:   %[[TURN_OR_DIV:.+]] = arith.ori %[[SUBTREE]]#12, %[[SUBTREE]]#13 : tensor<i1>
// CHECK:   %[[ZEROED_PROB:.+]] = arith.select %[[TURN_OR_DIV]], %[[CST_ZERO]], %[[TRANS_PROB]] : tensor<i1>, tensor<f64>

// Random selection
// CHECK:   %[[TRANS_RNG:.+]], %[[TRANS_RAND:.+]] = enzyme.random %{{.*}}, %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK:   %[[ACCEPT_TRANS:.+]] = arith.cmpf olt, %[[TRANS_RAND]], %[[ZEROED_PROB]] : tensor<f64>

// Select proposal
// CHECK:   %[[FINAL_Q:.+]] = enzyme.select %[[ACCEPT_TRANS]], %[[SUBTREE]]#6, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[FINAL_GRAD:.+]] = enzyme.select %[[ACCEPT_TRANS]], %[[SUBTREE]]#7, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK:   %[[FINAL_U:.+]] = enzyme.select %[[ACCEPT_TRANS]], %[[SUBTREE]]#8, %{{.*}} : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK:   %[[FINAL_H:.+]] = enzyme.select %[[ACCEPT_TRANS]], %[[SUBTREE]]#9, %{{.*}} : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>

// Increment depth
// CHECK:   %[[NEW_DEPTH:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>

// Update p_sum (first addf - accumulate)
// CHECK:   %[[PSUM_ACCUM:.+]] = arith.addf %{{.*}}, %[[SUBTREE]]#16 : tensor<1xf64>

// Final turning check on combined tree
// CHECK:   %[[HALF_BCAST2:.+]] = "enzyme.broadcast"(%[[CST_HALF]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK:   %[[P_LEFT_RIGHT:.+]] = arith.addf %[[SEL_P_LEFT]], %[[SEL_P_RIGHT]] : tensor<1xf64>
// CHECK:   %[[P_AVG2:.+]] = arith.mulf %[[HALF_BCAST2]], %[[P_LEFT_RIGHT]] : tensor<1xf64>
// CHECK:   %[[P_CENTERED2:.+]] = arith.subf %[[PSUM_ACCUM]], %[[P_AVG2]] : tensor<1xf64>
// CHECK:   %[[DOT_LEFT2:.+]] = enzyme.dot %[[SEL_P_LEFT]], %[[P_CENTERED2]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:   %[[DOT_RIGHT2:.+]] = enzyme.dot %[[SEL_P_RIGHT]], %[[P_CENTERED2]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK:   %[[LE_LEFT2:.+]] = arith.cmpf ole, %[[DOT_LEFT2]], %[[CST_ZERO]] : tensor<f64>
// CHECK:   %[[LE_RIGHT2:.+]] = arith.cmpf ole, %[[DOT_RIGHT2]], %[[CST_ZERO]] : tensor<f64>
// CHECK:   %[[COMBINED_TURN:.+]] = arith.ori %[[LE_LEFT2]], %[[LE_RIGHT2]] : tensor<i1>

// Merge turning and diverging flags
// CHECK:   %[[FINAL_TURN:.+]] = arith.ori %[[SUBTREE]]#12, %[[COMBINED_TURN]] : tensor<i1>
// CHECK:   %[[FINAL_DIV:.+]] = arith.ori %{{.*}}, %[[SUBTREE]]#13 : tensor<i1>

// Accumulate accept probs and num proposals
// CHECK:   %[[ACCUM_ACCEPT:.+]] = arith.addf %{{.*}}, %[[SUBTREE]]#14 : tensor<f64>
// CHECK:   %[[ACCUM_PROPS:.+]] = arith.addi %{{.*}}, %[[SUBTREE]]#15 : tensor<i64>

// Final p_sum update
// CHECK:   %[[FINAL_PSUM:.+]] = arith.addf %{{.*}}, %[[SUBTREE]]#16 : tensor<1xf64>

// CHECK:   enzyme.yield %[[SEL_Q_LEFT]], %[[SEL_P_LEFT]], %[[SEL_GRAD_LEFT]], %[[SEL_Q_RIGHT]], %[[SEL_P_RIGHT]], %[[SEL_GRAD_RIGHT]], %[[FINAL_Q]], %[[FINAL_GRAD]], %[[FINAL_U]], %[[FINAL_H]], %[[NEW_DEPTH]], %[[COMBINED_WEIGHT]], %[[FINAL_TURN]], %[[FINAL_DIV]], %[[ACCUM_ACCEPT]], %[[ACCUM_PROPS]], %[[FINAL_PSUM]], %{{.*}} : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>
// CHECK: }

