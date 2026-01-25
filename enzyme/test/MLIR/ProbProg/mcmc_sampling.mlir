// RUN: %eopt --probprog %s | FileCheck %s

// Test the outermost sampling loop structure for MCMC.
// We focus on the sampling loop structure (iter_args, buffer collection), not kernel internals.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  // ============================================================================
  // Test 1: Basic sampling without warmup
  // Sampling loop collects samples into a buffer tensor
  // ============================================================================
  func.func @sampling_basic(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<10xi1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "sampling_basic", selection = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<10xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<10xi1>, tensor<2xui64>
  }

  // ============================================================================
  // Test 2: Sampling with thinning
  // With thinning > 1, only every nth sample is collected
  // ============================================================================
  func.func @sampling_thinning(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<5xi1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5, max_delta_energy = 1000.0, adapt_step_size = false, adapt_mass_matrix = false>,
        name = "sampling_thinning", selection = [[#enzyme.symbol<1>]], num_warmup = 0, num_samples = 10, thinning = 2 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<5xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<5xi1>, tensor<2xui64>
  }

  // ============================================================================
  // Test 3: Sampling after warmup
  // Warmup loop runs first, then sampling loop collects samples
  // ============================================================================
  func.func @sampling_with_warmup(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<10xi1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      step_size = %step_size
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 5>,
        name = "sampling_with_warmup", selection = [[#enzyme.symbol<1>]], num_warmup = 5, num_samples = 10 }
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<10xi1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<10xi1>, tensor<2xui64>
  }
}

// ============================================================================
// CHECK patterns for Test 1: Basic sampling
// Sampling loop has 6 iter_args: [q, grad, U, rng, samplesBuffer, acceptedBuffer]
// ============================================================================
// CHECK-LABEL: func.func @sampling_basic
// Sampling loop iterating num_samples times
// CHECK: %[[SAMPLES_LOOP:.+]]:6 = enzyme.for_loop(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// Loop body: position (q), gradient, potential energy (U), rng, samples buffer, accepted buffer
// CHECK-NEXT: ^bb0(%[[ITER_IDX:.+]]: tensor<i64>, %[[Q:.+]]: tensor<1xf64>, %[[GRAD:.+]]: tensor<1xf64>, %[[U:.+]]: tensor<f64>, %[[RNG:.+]]: tensor<2xui64>, %[[SAMPLES_BUF:.+]]: tensor<10x1xf64>, %[[ACCEPTED_BUF:.+]]: tensor<10xi1>):
// Sample storage uses enzyme.dynamic_update to add samples to buffer
// CHECK: %[[UPD_SAMPLES:.+]] = enzyme.dynamic_update %[[SAMPLES_BUF]], %[[ITER_IDX]], %{{.+}} : (tensor<10x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<10x1xf64>
// Conditional storage based on iteration index matching collection criteria
// CHECK: %[[SEL_SAMPLES:.+]] = enzyme.select %{{.+}}, %[[UPD_SAMPLES]], %[[SAMPLES_BUF]] : (tensor<i1>, tensor<10x1xf64>, tensor<10x1xf64>) -> tensor<10x1xf64>
// CHECK: %[[UPD_ACCEPTED:.+]] = enzyme.dynamic_update %[[ACCEPTED_BUF]], %[[ITER_IDX]], %{{.+}} : (tensor<10xi1>, tensor<i64>, tensor<i1>) -> tensor<10xi1>
// CHECK: %[[SEL_ACCEPTED:.+]] = enzyme.select %{{.+}}, %[[UPD_ACCEPTED]], %[[ACCEPTED_BUF]] : (tensor<i1>, tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
// CHECK: enzyme.yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[SEL_SAMPLES]], %[[SEL_ACCEPTED]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// After loop: use enzyme.recover_sample to unflatten samples back to original shape
// CHECK: enzyme.recover_sample %[[SAMPLES_LOOP]]#4[0] : tensor<10x1xf64> -> tensor<10xf64>

// ============================================================================
// CHECK patterns for Test 2: Sampling with thinning
// With thinning=2 and num_samples=10, only 5 samples collected
// ============================================================================
// CHECK-LABEL: func.func @sampling_thinning
// Buffer shape is [collected_samples, position_size] = [5, 1]
// CHECK: %[[THIN_LOOP:.+]]:6 = enzyme.for_loop(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1> {
// CHECK: enzyme.yield
// Collected samples shape: [5] (10 samples / 2 thinning)
// CHECK: enzyme.recover_sample %[[THIN_LOOP]]#4[0] : tensor<5x1xf64> -> tensor<5xf64>

// ============================================================================
// CHECK patterns for Test 3: Sampling with warmup
// Two for_loops: first for warmup, second for sampling
// ============================================================================
// CHECK-LABEL: func.func @sampling_with_warmup
// First loop: warmup (16 iter_args with both adaptations enabled)
// CHECK: %[[WARMUP_LOOP:.+]]:16 = enzyme.for_loop(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(
// CHECK: enzyme.yield
// Second loop: sampling (6 iter_args)
// CHECK: %[[SAMPLE_LOOP:.+]]:6 = enzyme.for_loop(%{{.+}} : tensor<i64>) to(%{{.+}} : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%[[WARMUP_LOOP]]#0, %[[WARMUP_LOOP]]#1, %[[WARMUP_LOOP]]#2, %[[WARMUP_LOOP]]#3, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK: enzyme.yield
