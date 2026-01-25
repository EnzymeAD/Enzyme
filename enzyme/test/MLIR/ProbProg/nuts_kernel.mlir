// RUN: %eopt --probprog %s | FileCheck %s

// Test NUTS (No-U-Turn Sampler) kernel internals with full coverage.
// Focus on: tree building structure, subtree iteration, turning detection.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // ============================================================================
  // NUTS with max_tree_depth = 3 for manageable output
  // ============================================================================
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

// ============================================================================
// NUTS Kernel Structure - Full Coverage CHECK patterns
// ============================================================================
// CHECK-LABEL: func.func @nuts

// ============================================================================
// 1. Constants
// ============================================================================
// CHECK-NEXT: %[[CST_NEG1:.+]] = arith.constant dense<-1> : tensor<i64>
// CHECK-NEXT: %[[CST_INF:.+]] = arith.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-NEXT: %[[CST_NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-NEXT: %[[CST_1_I64:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[CST_CKPT_ZEROS:.+]] = arith.constant dense<0.000000e+00> : tensor<3x1xf64>
// CHECK-NEXT: %[[CST_3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-NEXT: %[[CST_TRUE:.+]] = arith.constant dense<true> : tensor<i1>
// CHECK-NEXT: %[[CST_FALSE:.+]] = arith.constant dense<false> : tensor<i1>
// CHECK-NEXT: %[[CST_0_I64:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[CST_HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT: %[[CST_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[CST_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[CST_MAX_DELTA:.+]] = arith.constant dense<1.000000e+03> : tensor<f64>
// CHECK-NEXT: %[[CST_EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>

// ============================================================================
// 2. Initial state extraction from trace
// ============================================================================
// CHECK-NEXT: %[[INIT_TRACE:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT: %[[RNG_SPLIT1:.+]]:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[RNG_SPLIT2:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT1]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[Q0:.+]] = enzyme.getFlattenedSamplesFromTrace %[[INIT_TRACE]] {selection = {{.*}}} : (!enzyme.Trace) -> tensor<1xf64>
// CHECK-NEXT: %[[WEIGHT:.+]] = enzyme.getWeightFromTrace %[[INIT_TRACE]] : (!enzyme.Trace) -> tensor<f64>
// CHECK-NEXT: %[[U0:.+]] = arith.negf %[[WEIGHT]] : tensor<f64>

// ============================================================================
// 3. Initial gradient computation via autodiff_region
// ============================================================================
// CHECK-NEXT: %[[INIT_GRAD:.+]]:3 = enzyme.autodiff_region(%[[Q0]], %[[CST_ONE]]) {
// CHECK-NEXT: ^bb0(%[[AD_ARG:.+]]: tensor<1xf64>):
// CHECK-NEXT: %[[UPDATE_CALL:.+]]:3 = func.call @test.update{{.*}}(%[[INIT_TRACE]], %[[AD_ARG]], %[[RNG_SPLIT2]]#1, %arg1, %arg2)
// CHECK-NEXT: %[[NEG_W:.+]] = arith.negf %[[UPDATE_CALL]]#1 : tensor<f64>
// CHECK-NEXT: enzyme.yield %[[NEG_W]], %[[UPDATE_CALL]]#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]}

// ============================================================================
// 4. Momentum sampling from N(0, I)
// ============================================================================
// CHECK-NEXT: %[[RNG_SPLIT3:.+]]:3 = enzyme.randomSplit %[[RNG_SPLIT2]]#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[RNG_SPLIT4:.+]] = enzyme.randomSplit %[[RNG_SPLIT3]]#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT: %[[RNG_OUT:.+]], %[[P0:.+]] = enzyme.random %[[RNG_SPLIT4]], %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

// ============================================================================
// 5. Initial kinetic energy and Hamiltonian: H0 = U0 + 0.5 * p^T * p
// ============================================================================
// CHECK-NEXT: %[[P0_DOT:.+]] = enzyme.dot %[[P0]], %[[P0]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[K0:.+]] = arith.mulf %[[P0_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK-NEXT: %[[H0:.+]] = arith.addf %[[U0]], %[[K0]] : tensor<f64>

// ============================================================================
// 6. Main tree building loop (outer while_loop)
// TreeState: 18 fields - left/right leaves (q,p,grad x2), proposal (q,grad,U,H),
//            depth, weight, turning, diverging, sum_accept_probs, num_proposals, p_sum, rng
// ============================================================================
// CHECK-NEXT: %[[TREE:.+]]:18 = enzyme.while_loop(%[[Q0]], %[[P0]], %[[INIT_GRAD]]#2, %[[Q0]], %[[P0]], %[[INIT_GRAD]]#2, %[[Q0]], %[[INIT_GRAD]]#2, %[[U0]], %[[H0]], %[[CST_0_I64]], %[[CST_ZERO]], %[[CST_FALSE]], %[[CST_FALSE]], %[[CST_ZERO]], %[[CST_0_I64]], %[[P0]], %[[RNG_SPLIT3]]#2 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64>) -> tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<2xui64> condition {
// CHECK-NEXT: ^bb0({{.*}}):

// Main loop condition: depth < max_tree_depth && !turning && !diverging
// CHECK-NEXT: %[[DEPTH_LT:.+]] = arith.cmpi slt, %{{.*}}, %[[CST_3]] : tensor<i64>
// CHECK-NEXT: %[[NOT_TURN:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK-NEXT: %[[NOT_DIV:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK-NEXT: %[[COND1:.+]] = arith.andi %[[DEPTH_LT]], %[[NOT_TURN]] : tensor<i1>
// CHECK-NEXT: %[[COND2:.+]] = arith.andi %[[COND1]], %[[NOT_DIV]] : tensor<i1>
// CHECK-NEXT: enzyme.yield %[[COND2]] : tensor<i1>
// CHECK-NEXT: } body {
// CHECK-NEXT: ^bb0({{.*}}):

// ============================================================================
// 6a. Direction sampling (uniform random -> compare to 0.5)
// ============================================================================
// CHECK-NEXT: %[[DIR_SPLIT:.+]]:3 = enzyme.randomSplit %{{.*}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT: %[[DIR_RNG:.+]], %[[DIR_RAND:.+]] = enzyme.random %[[DIR_SPLIT]]#1, %[[CST_ZERO]], %[[CST_ONE]] {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[GOING_RIGHT:.+]] = arith.cmpf olt, %[[DIR_RAND]], %[[CST_HALF]] : tensor<f64>
// CHECK-NEXT: %[[SUBTREE_SPLIT:.+]]:2 = enzyme.randomSplit %[[DIR_SPLIT]]#2 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)

// Number of proposals for this depth: 2^depth
// CHECK-NEXT: %[[NUM_PROPS:.+]] = arith.shli %[[CST_1_I64]], %{{.*}} : tensor<i64>

// ============================================================================
// 7. Subtree building loop (inner while_loop)
// 21 iter_args: TreeState(18) + pCkpts, pSumCkpts, leafIdx
// ============================================================================
// CHECK-NEXT: %[[SUBTREE:.+]]:21 = enzyme.while_loop({{.*}} : {{.*}}) -> {{.*}} condition {
// CHECK-NEXT: ^bb0({{.*}}):

// Subtree condition: num_proposals < 2^depth && !turning && !diverging
// CHECK-NEXT: %[[SUB_LT:.+]] = arith.cmpi slt, %{{.*}}, %[[NUM_PROPS]] : tensor<i64>
// CHECK-NEXT: %[[SUB_NOT_TURN:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK-NEXT: %[[SUB_NOT_DIV:.+]] = arith.xori %{{.*}}, %[[CST_TRUE]] : tensor<i1>
// CHECK-NEXT: %[[SUB_COND1:.+]] = arith.andi %[[SUB_LT]], %[[SUB_NOT_TURN]] : tensor<i1>
// CHECK-NEXT: %[[SUB_COND2:.+]] = arith.andi %[[SUB_COND1]], %[[SUB_NOT_DIV]] : tensor<i1>
// CHECK-NEXT: enzyme.yield %[[SUB_COND2]] : tensor<i1>
// CHECK-NEXT: } body {
// CHECK-NEXT: ^bb0({{.*}}):

// ============================================================================
// 7a. Select leaf based on direction
// ============================================================================
// CHECK-NEXT: %[[SEL_Q:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SEL_P:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[SEL_GRAD:.+]] = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %{{.*}} : (tensor<i1>, tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[LEAF_SPLIT:.+]]:2 = enzyme.randomSplit %{{.*}} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)

// ============================================================================
// 7b. Leapfrog step: select direction, half-step p, full-step q, gradient, half-step p
// ============================================================================
// CHECK-NEXT: %[[STEP_SEL:.+]] = enzyme.select %[[GOING_RIGHT]], %[[CST_EPS]], %[[CST_NEG_EPS]] : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[STEP_BCAST:.+]] = "enzyme.broadcast"(%[[STEP_SEL]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[HALF_STEP:.+]] = arith.mulf %[[STEP_SEL]], %[[CST_HALF]] : tensor<f64>
// CHECK-NEXT: %[[HALF_STEP_BCAST:.+]] = "enzyme.broadcast"(%[[HALF_STEP]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[GRAD_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[SEL_GRAD]] : tensor<1xf64>
// CHECK-NEXT: %[[P_HALF:.+]] = arith.subf %[[SEL_P]], %[[GRAD_SCALED]] : tensor<1xf64>
// CHECK-NEXT: %[[P_HALF_SCALED:.+]] = arith.mulf %[[STEP_BCAST]], %[[P_HALF]] : tensor<1xf64>
// CHECK-NEXT: %[[Q_NEW:.+]] = arith.addf %[[SEL_Q]], %[[P_HALF_SCALED]] : tensor<1xf64>

// Gradient via autodiff_region
// CHECK-NEXT: %[[GRAD_RES:.+]]:3 = enzyme.autodiff_region(%[[Q_NEW]], %[[CST_ONE]]) {
// CHECK-NEXT: ^bb0({{.*}}):
// CHECK-NEXT: %{{.*}} = func.call @test.update{{.*}}
// CHECK-NEXT: %{{.*}} = arith.negf
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: }

// Second half-step: p_new = p_half - (eps/2) * grad_new
// CHECK-NEXT: %[[GRAD_NEW_SCALED:.+]] = arith.mulf %[[HALF_STEP_BCAST]], %[[GRAD_RES]]#2 : tensor<1xf64>
// CHECK-NEXT: %[[P_NEW:.+]] = arith.subf %[[P_HALF]], %[[GRAD_NEW_SCALED]] : tensor<1xf64>

// ============================================================================
// 7c. Energy calculations and divergence check
// ============================================================================
// Kinetic energy: K = 0.5 * p^T * p
// CHECK-NEXT: %[[P_DOT:.+]] = enzyme.dot %[[P_NEW]], %[[P_NEW]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[K_NEW:.+]] = arith.mulf %[[P_DOT]], %[[CST_HALF]] : tensor<f64>
// CHECK-NEXT: %[[H_NEW:.+]] = arith.addf %[[GRAD_RES]]#0, %[[K_NEW]] : tensor<f64>

// Delta energy for divergence: H - H0
// CHECK-NEXT: %[[DELTA_E:.+]] = arith.subf %[[H_NEW]], %[[H0]] : tensor<f64>

// Handle NaN: if delta != delta, set to inf
// CHECK-NEXT: %[[IS_NAN:.+]] = arith.cmpf une, %[[DELTA_E]], %[[DELTA_E]] : tensor<f64>
// CHECK-NEXT: %[[DELTA_SAFE:.+]] = arith.select %[[IS_NAN]], %[[CST_INF]], %[[DELTA_E]] : tensor<i1>, tensor<f64>

// Weight = -delta_energy
// CHECK-NEXT: %[[NEG_DELTA:.+]] = arith.negf %[[DELTA_SAFE]] : tensor<f64>

// Divergence check: delta > max_delta_energy
// CHECK-NEXT: %[[DIVERGING:.+]] = arith.cmpf ogt, %[[DELTA_SAFE]], %[[CST_MAX_DELTA]] : tensor<f64>

// Acceptance probability: min(1, exp(-delta))
// CHECK-NEXT: %[[NEG_DELTA2:.+]] = arith.negf %[[DELTA_SAFE]] : tensor<f64>
// CHECK-NEXT: %[[EXP_NEG:.+]] = math.exp %[[NEG_DELTA2]] : tensor<f64>
// CHECK-NEXT: %[[ACCEPT_PROB:.+]] = arith.minimumf %[[EXP_NEG]], %[[CST_ONE]] : tensor<f64>

// ============================================================================
// 7d. Tree combination with enzyme.if for first leaf vs subsequent
// ============================================================================
// CHECK-NEXT: %[[IS_FIRST:.+]] = arith.cmpi eq, %{{.*}}, %[[CST_0_I64]] : tensor<i64>
// CHECK-NEXT: %[[IF_RES:.+]]:18 = enzyme.if(%[[IS_FIRST]]) ({

// First leaf case: initialize tree state
// CHECK-NEXT: enzyme.yield {{.*}}
// CHECK-NEXT: }, {

// Subsequent leaves: combine trees using uniform kernel
// CHECK: enzyme.log_add_exp
// CHECK: enzyme.logistic
// CHECK: enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution UNIFORM>}
// CHECK: arith.cmpf olt
// CHECK: enzyme.select
// CHECK: enzyme.yield
// CHECK-NEXT: })

// ============================================================================
// 7e. Checkpoint index calculations (leafIdxToCheckpointIdxs)
// ============================================================================
// CHECK-NEXT: %[[SHRUI:.+]] = arith.shrui %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK-NEXT: %[[POPCOUNT1:.+]] = enzyme.popcount %[[SHRUI]] : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT: %[[ADDI1:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK-NEXT: %[[XORI:.+]] = arith.xori %{{.*}}, %[[CST_NEG1]] : tensor<i64>
// CHECK-NEXT: %[[ANDI1:.+]] = arith.andi %[[XORI]], %[[ADDI1]] : tensor<i64>
// CHECK-NEXT: %[[SUBI:.+]] = arith.subi %[[ANDI1]], %[[CST_1_I64]] : tensor<i64>
// CHECK-NEXT: %[[POPCOUNT2:.+]] = enzyme.popcount %[[SUBI]] : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT: %[[DIFF:.+]] = arith.subi %[[POPCOUNT1]], %[[POPCOUNT2]] : tensor<i64>
// CHECK-NEXT: %[[IDX_MIN:.+]] = arith.addi %[[DIFF]], %[[CST_1_I64]] : tensor<i64>

// ============================================================================
// 7f. Checkpoint update (when leafIdx is even)
// ============================================================================
// CHECK-NEXT: %[[LEAF_MOD:.+]] = arith.andi %{{.*}}, %[[CST_1_I64]] : tensor<i64>
// CHECK-NEXT: %[[IS_EVEN:.+]] = arith.cmpi eq, %[[LEAF_MOD]], %[[CST_0_I64]] : tensor<i64>
// CHECK-NEXT: %[[CKPT_IF:.+]]:2 = enzyme.if(%[[IS_EVEN]]) ({
// CHECK-NEXT: %{{.*}} = enzyme.dynamic_update %{{.*}}, %[[POPCOUNT1]], %[[P_NEW]] : (tensor<3x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<3x1xf64>
// CHECK-NEXT: %{{.*}} = enzyme.dynamic_update %{{.*}}, %[[POPCOUNT1]], %[[IF_RES]]#16 : (tensor<3x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<3x1xf64>
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: }, {
// CHECK-NEXT: enzyme.yield
// CHECK-NEXT: })

// ============================================================================
// 7g. Iterative turning check (innermost while_loop)
// ============================================================================
// CHECK-NEXT: %[[TURN_LOOP:.+]]:2 = enzyme.while_loop(%[[POPCOUNT1]], %[[CST_FALSE]] : tensor<i64>, tensor<i1>) -> tensor<i64>, tensor<i1> condition {
// CHECK-NEXT: ^bb0(%[[TURN_IDX:.+]]: tensor<i64>, %[[TURN_FLAG:.+]]: tensor<i1>):
// CHECK-NEXT: %[[GE:.+]] = arith.cmpi sge, %[[TURN_IDX]], %[[IDX_MIN]] : tensor<i64>
// CHECK-NEXT: %[[NOT_TURN2:.+]] = arith.xori %[[TURN_FLAG]], %[[CST_TRUE]] : tensor<i1>
// CHECK-NEXT: %[[TURN_COND:.+]] = arith.andi %[[GE]], %[[NOT_TURN2]] : tensor<i1>
// CHECK-NEXT: enzyme.yield %[[TURN_COND]] : tensor<i1>
// CHECK-NEXT: } body {
// CHECK-NEXT: ^bb0(%[[TURN_IDX2:.+]]: tensor<i64>, %[[TURN_FLAG2:.+]]: tensor<i1>):

// Extract checkpoint momentum
// CHECK-NEXT: %[[P_CKPT:.+]] = enzyme.dynamic_extract %[[CKPT_IF]]#0, %[[TURN_IDX2]] : (tensor<3x1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT: %[[PSUM_CKPT:.+]] = enzyme.dynamic_extract %[[CKPT_IF]]#1, %[[TURN_IDX2]] : (tensor<3x1xf64>, tensor<i64>) -> tensor<1xf64>

// Compute subtree momentum sum: p_sum - pSumCkpt + pCkpt
// CHECK-NEXT: %[[SUB1:.+]] = arith.subf %[[IF_RES]]#16, %[[PSUM_CKPT]] : tensor<1xf64>
// CHECK-NEXT: %[[SUBTREE_PSUM:.+]] = arith.addf %[[SUB1]], %[[P_CKPT]] : tensor<1xf64>

// Dynamic termination criterion: p_sum_centered = p_sum - (p_left + p_right) / 2
// CHECK-NEXT: %[[HALF_BCAST:.+]] = "enzyme.broadcast"(%[[CST_HALF]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[P_SUM2:.+]] = arith.addf %[[P_CKPT]], %[[P_NEW]] : tensor<1xf64>
// CHECK-NEXT: %[[P_AVG:.+]] = arith.mulf %[[HALF_BCAST]], %[[P_SUM2]] : tensor<1xf64>
// CHECK-NEXT: %[[P_CENTERED:.+]] = arith.subf %[[SUBTREE_PSUM]], %[[P_AVG]] : tensor<1xf64>

// Check turning condition via dot products
// CHECK-NEXT: %[[DOT_LEFT:.+]] = enzyme.dot %[[P_CKPT]], %[[P_CENTERED]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[DOT_RIGHT:.+]] = enzyme.dot %[[P_NEW]], %[[P_CENTERED]] {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<1xf64>, tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[LE_LEFT:.+]] = arith.cmpf ole, %[[DOT_LEFT]], %[[CST_ZERO]] : tensor<f64>
// CHECK-NEXT: %[[LE_RIGHT:.+]] = arith.cmpf ole, %[[DOT_RIGHT]], %[[CST_ZERO]] : tensor<f64>
// CHECK-NEXT: %[[TURN_DETECTED:.+]] = arith.ori %[[LE_LEFT]], %[[LE_RIGHT]] : tensor<i1>

// CHECK-NEXT: %[[NEXT_IDX:.+]] = arith.subi %[[TURN_IDX2]], %[[CST_1_I64]] : tensor<i64>
// CHECK-NEXT: enzyme.yield %[[NEXT_IDX]], %[[TURN_DETECTED]] : tensor<i64>, tensor<i1>
// CHECK-NEXT: }

// Select turning (skip on first leaf)
// CHECK-NEXT: %[[FINAL_TURN:.+]] = enzyme.select %[[IS_FIRST]], %[[CST_FALSE]], %[[TURN_LOOP]]#1 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>

// Increment leaf index
// CHECK-NEXT: %[[NEXT_LEAF:.+]] = arith.addi %{{.*}}, %[[CST_1_I64]] : tensor<i64>

// Subtree body yield
// CHECK-NEXT: enzyme.yield {{.*}}
// CHECK-NEXT: }

// ============================================================================
// 8. Tree combination after subtree build (biased transition kernel)
// ============================================================================
// Update left/right leaves based on direction
// CHECK-NEXT: %{{.*}} = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[SUBTREE]]#0
// CHECK-NEXT: %{{.*}} = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[SUBTREE]]#1
// CHECK-NEXT: %{{.*}} = enzyme.select %[[GOING_RIGHT]], %{{.*}}, %[[SUBTREE]]#2
// CHECK-NEXT: %{{.*}} = enzyme.select %[[GOING_RIGHT]], %[[SUBTREE]]#3, %{{.*}}
// CHECK-NEXT: %{{.*}} = enzyme.select %[[GOING_RIGHT]], %[[SUBTREE]]#4, %{{.*}}
// CHECK-NEXT: %{{.*}} = enzyme.select %[[GOING_RIGHT]], %[[SUBTREE]]#5, %{{.*}}

// Log weight combination
// CHECK-NEXT: %{{.*}} = enzyme.log_add_exp %{{.*}}, %[[SUBTREE]]#11

// Biased transition probability: min(1, exp(new_weight - current_weight))
// CHECK-NEXT: %{{.*}} = arith.subf %[[SUBTREE]]#11, %{{.*}}
// CHECK-NEXT: %{{.*}} = math.exp
// CHECK-NEXT: %{{.*}} = arith.minimumf {{.*}}, %[[CST_ONE]]

// Zero probability when turning or diverging
// CHECK-NEXT: %{{.*}} = arith.ori %[[SUBTREE]]#12, %[[SUBTREE]]#13
// CHECK-NEXT: %{{.*}} = arith.select {{.*}}, %[[CST_ZERO]], %{{.*}}

// Random selection
// CHECK-NEXT: %{{.*}}, %{{.*}} = enzyme.random {{.*}} {rng_distribution = #enzyme<rng_distribution UNIFORM>}
// CHECK-NEXT: %{{.*}} = arith.cmpf olt

// Select proposal
// CHECK-NEXT: %{{.*}} = enzyme.select
// CHECK-NEXT: %{{.*}} = enzyme.select
// CHECK-NEXT: %{{.*}} = enzyme.select
// CHECK-NEXT: %{{.*}} = enzyme.select

// Increment depth
// CHECK: arith.addi {{.*}}, %[[CST_1_I64]]

// Update p_sum
// CHECK: arith.addf

// Final turning check on combined tree
// CHECK: "enzyme.broadcast"(%[[CST_HALF]])
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: enzyme.dot
// CHECK: enzyme.dot
// CHECK: arith.cmpf ole
// CHECK: arith.cmpf ole
// CHECK: arith.ori

// Main body yield
// CHECK: enzyme.yield {{.*}}
// CHECK-NEXT: }

// ============================================================================
// 9. Final trace creation
// ============================================================================
// CHECK-NEXT: %[[FINAL_TRACE:.+]]:3 = call @test.update(%[[INIT_TRACE]], %[[TREE]]#6, %[[RNG_SPLIT3]]#0, %arg1, %arg2)
// CHECK-NEXT: return %[[FINAL_TRACE]]#0, %[[CST_TRUE]], %[[FINAL_TRACE]]#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
