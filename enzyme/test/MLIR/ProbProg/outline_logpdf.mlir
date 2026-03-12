// RUN: %eopt --outline-mcmc-regions %s | FileCheck %s

// ============================================================================
// Test 1: Basic outline of unified logpdf region (no free values)
// ============================================================================

// CHECK-LABEL: func.func @test_basic_outline
// Initial position extracted from trace via slice/update ops
// CHECK: enzyme.dynamic_slice
// CHECK: enzyme.dynamic_update_slice
// CHECK: enzyme.dynamic_slice
// CHECK: %[[INIT_POS:.+]] = enzyme.dynamic_update_slice
// MCMCOp in pure logpdf_fn mode (no @fn reference before the parens)
// CHECK: enzyme.mcmc(
// CHECK-SAME: logpdf_fn = @[[LOGPDF:[a-zA-Z0-9_]+]]
// CHECK-SAME: initial_position = %[[INIT_POS]]
//
// Outlined logpdf wrapper function
// CHECK: func.func private @[[LOGPDF]](%[[POS:[^:]+]]: tensor<1x2xf64>) -> tensor<f64>
// Slice position into two scalar components
// CHECK: %[[S0:.+]] = enzyme.dynamic_slice %[[POS]]
// CHECK-NEXT: %[[X0:.+]] = enzyme.reshape %[[S0]]
// CHECK: %[[S1:.+]] = enzyme.dynamic_slice %[[POS]]
// CHECK-NEXT: %[[X1:.+]] = enzyme.reshape %[[S1]]
// Logpdf body: -x0 + -x1
// CHECK: %[[N0:.+]] = arith.negf %[[X0]]
// CHECK-NEXT: %[[N1:.+]] = arith.negf %[[X1]]
// CHECK-NEXT: %[[SUM:.+]] = arith.addf %[[N0]], %[[N1]]
// CHECK-NEXT: return %[[SUM]]

// ============================================================================
// Test 2: Logpdf with free values (hoisted op from enclosing scope)
// ============================================================================

// CHECK-LABEL: func.func @test_outline_with_free_values
// Hoisted log computed before mcmc_region
// CHECK: %[[LOG:.+]] = math.log
// CHECK: enzyme.mcmc(
// CHECK-SAME: logpdf_fn = @[[LP2:[a-zA-Z0-9_]+]]
// The logpdf function receives the hoisted log as an extra parameter
// CHECK: func.func private @[[LP2]](%[[P2:[^:]+]]: tensor<1x2xf64>, %[[FLOG:[^:]+]]: tensor<f64>) -> tensor<f64>
// Body uses the hoisted log value
// CHECK: arith.subf %{{.*}}, %[[FLOG]]
// CHECK: arith.subf %{{.*}}, %[[FLOG]]
// CHECK: arith.addf
// CHECK-NEXT: return

module {
  // Test 1: Basic (no free values)
  func.func @test_basic_outline(
      %rng : tensor<2xui64>,
      %mu : tensor<f64>,
      %trace : tensor<1x2xf64>) {

    %step_size = arith.constant dense<0.1> : tensor<f64>

    %result:8 = enzyme.mcmc_region(%rng, %mu) given %trace
        step_size = %step_size {
    ^bb0(%r: tensor<2xui64>, %m: tensor<f64>):
      %x0:2 = enzyme.sample_region(%r, %m) sampler {
      ^bb0(%sr: tensor<2xui64>, %sm: tensor<f64>):
        enzyme.yield %sr, %sm : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {symbol = #enzyme.symbol<0>}
        : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

      %x1:2 = enzyme.sample_region(%x0#0, %m) sampler {
      ^bb0(%sr2: tensor<2xui64>, %sm2: tensor<f64>):
        enzyme.yield %sr2, %sm2 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {symbol = #enzyme.symbol<1>}
        : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

      enzyme.yield %x1#0, %x1#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%lp_x0: tensor<f64>, %lp_x1: tensor<f64>):
      %neg0 = arith.negf %lp_x0 : tensor<f64>
      %neg1 = arith.negf %lp_x1 : tensor<f64>
      %total = arith.addf %neg0, %neg1 : tensor<f64>
      enzyme.yield %total : tensor<f64>
    } attributes {
      selection = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      nuts_config = #enzyme.nuts_config<max_tree_depth = 10, max_delta_energy = 1000.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64,
      num_position_args = 2 : i64, position_size = 2 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>,
            tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>,
            tensor<f64>, tensor<1x2xf64>)
    return
  }

  // Test 2: Free values (hoisted math.log from enclosing scope)
  func.func @test_outline_with_free_values(
      %rng : tensor<2xui64>,
      %sigma : tensor<f64>,
      %trace : tensor<1x2xf64>) {

    %step_size = arith.constant dense<0.1> : tensor<f64>
    // This math.log is hoisted by SICM before the mcmc_region.
    // The logpdf region captures it as a free value.
    %log_sigma = math.log %sigma : tensor<f64>

    %result:8 = enzyme.mcmc_region(%rng, %sigma) given %trace
        step_size = %step_size {
    ^bb0(%r: tensor<2xui64>, %s: tensor<f64>):
      %x0:2 = enzyme.sample_region(%r, %s) sampler {
      ^bb0(%sr: tensor<2xui64>, %ss: tensor<f64>):
        enzyme.yield %sr, %ss : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {symbol = #enzyme.symbol<0>}
        : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

      %x1:2 = enzyme.sample_region(%x0#0, %s) sampler {
      ^bb0(%sr2: tensor<2xui64>, %ss2: tensor<f64>):
        enzyme.yield %sr2, %ss2 : tensor<2xui64>, tensor<f64>
      } logpdf {
      } {symbol = #enzyme.symbol<1>}
        : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

      enzyme.yield %x1#0, %x1#1 : tensor<2xui64>, tensor<f64>
    } logpdf {
    ^bb0(%lp_x0: tensor<f64>, %lp_x1: tensor<f64>):
      // Both sites subtract the hoisted log_sigma (CSE'd free value)
      %sub0 = arith.subf %lp_x0, %log_sigma : tensor<f64>
      %sub1 = arith.subf %lp_x1, %log_sigma : tensor<f64>
      %total = arith.addf %sub0, %sub1 : tensor<f64>
      enzyme.yield %total : tensor<f64>
    } attributes {
      selection = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      all_addresses = [[#enzyme.symbol<0>], [#enzyme.symbol<1>]],
      nuts_config = #enzyme.nuts_config<max_tree_depth = 10, max_delta_energy = 1000.0>,
      num_samples = 1 : i64, thinning = 1 : i64, num_warmup = 0 : i64,
      num_position_args = 2 : i64, position_size = 2 : i64
    } : (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
        -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>,
            tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>,
            tensor<f64>, tensor<1x2xf64>)
    return
  }
}
