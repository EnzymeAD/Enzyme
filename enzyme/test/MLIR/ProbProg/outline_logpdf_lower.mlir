// RUN: %eopt --outline-mcmc-regions --probprog %s | FileCheck %s

// ============================================================================
// Test: Full pipeline for outlined unified logpdf
//   mcmc_region (with unified logpdf) → outline → probprog lowering
// ============================================================================
//
// Verifies that the outlined logpdf wrapper function is correctly called
// during NUTS lowering: initial U0 computation, gradient via autodiff_region,
// and leapfrog integration inside the tree-building loop.

// CHECK-LABEL: func.func @test_outline_lower_nuts
// Position extracted from trace → outline builds initial_position
// CHECK: enzyme.dynamic_slice
// CHECK: enzyme.dynamic_update_slice
// CHECK: enzyme.dynamic_slice
// CHECK: %[[INIT_POS:.+]] = enzyme.dynamic_update_slice
// Initial U0: call logpdf, negf
// CHECK: call @[[LOGPDF:[a-zA-Z0-9_]+]](%[[INIT_POS]])
// CHECK-NEXT: %[[U0:.+]] = arith.negf
// Gradient via autodiff_region
// CHECK: enzyme.autodiff_region
// CHECK: func.call @[[LOGPDF]]
// CHECK-NEXT: %[[NEG:.+]] = arith.negf
// CHECK-NEXT: enzyme.yield
// NUTS sample loop
// CHECK: enzyme.for_loop
// Gradient inside tree building
// CHECK: enzyme.autodiff_region
// CHECK: func.call @[[LOGPDF]]
// CHECK-NEXT: %{{.+}} = arith.negf
// CHECK-NEXT: enzyme.yield

// The outlined logpdf wrapper function
// CHECK: func.func private @[[LOGPDF]](%[[POS:[^:]+]]: tensor<1x2xf64>) -> tensor<f64>
// Slice position into per-site scalars
// CHECK: %[[S0:.+]] = enzyme.dynamic_slice %[[POS]]
// CHECK-NEXT: %[[X0:.+]] = enzyme.reshape %[[S0]]
// CHECK: %[[S1:.+]] = enzyme.dynamic_slice %[[POS]]
// CHECK-NEXT: %[[X1:.+]] = enzyme.reshape %[[S1]]
// Logpdf body: -x0 + -x1
// CHECK: %[[N0:.+]] = arith.negf %[[X0]]
// CHECK-NEXT: %[[N1:.+]] = arith.negf %[[X1]]
// CHECK-NEXT: %[[SUM:.+]] = arith.addf %[[N0]], %[[N1]]
// CHECK-NEXT: return %[[SUM]]

module {
  func.func @test_outline_lower_nuts(
      %rng : tensor<2xui64>,
      %mu : tensor<f64>,
      %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {

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
    return %result#0, %result#1, %result#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }

  // ==========================================================================
  // Test 2: Logpdf with free values (hoisted math.log from enclosing scope)
  //   The logpdf wrapper receives the hoisted value as an extra parameter.
  //   Lowering must pass it through as an MCMCOp input.
  // ==========================================================================

  // CHECK-LABEL: func.func @test_outline_lower_free_values
  // Hoisted log before mcmc
  // CHECK: %[[LOG:.+]] = math.log
  // Initial U0: call logpdf with position AND free value
  // CHECK: call @[[LP2:[a-zA-Z0-9_]+]](%{{.+}}, %[[LOG]])
  // CHECK-NEXT: arith.negf
  // Gradient via autodiff_region
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @[[LP2]](%{{.+}}, %[[LOG]])
  // CHECK-NEXT: arith.negf
  // CHECK-NEXT: enzyme.yield
  // NUTS sample loop
  // CHECK: enzyme.for_loop
  // Gradient inside tree building
  // CHECK: enzyme.autodiff_region
  // CHECK: func.call @[[LP2]](%{{.+}}, %[[LOG]])
  // CHECK-NEXT: arith.negf
  // CHECK-NEXT: enzyme.yield
  //
  // The logpdf wrapper with free value parameter
  // CHECK: func.func private @[[LP2]](%[[P2:[^:]+]]: tensor<1x2xf64>, %[[FLOG:[^:]+]]: tensor<f64>) -> tensor<f64>
  // CHECK: arith.subf %{{.*}}, %[[FLOG]]
  // CHECK: arith.subf %{{.*}}, %[[FLOG]]
  // CHECK: arith.addf
  // CHECK-NEXT: return

  func.func @test_outline_lower_free_values(
      %rng : tensor<2xui64>,
      %sigma : tensor<f64>,
      %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>) {

    %step_size = arith.constant dense<0.1> : tensor<f64>
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
    return %result#0, %result#1, %result#2 : tensor<1x2xf64>, tensor<1xi1>, tensor<2xui64>
  }
}
