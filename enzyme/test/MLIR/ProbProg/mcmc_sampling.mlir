// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

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

// CHECK-LABEL: func.func @sampling_basic
// CHECK: %[[INIT_TRACE:.+]] = enzyme.initTrace
// CHECK: %[[INIT_POS:.+]] = enzyme.getFlattenedSamplesFromTrace %[[INIT_TRACE]]
// CHECK: %[[AUTODIFF:.+]]:3 = enzyme.autodiff_region
// CHECK: %[[SAMPLES_LOOP:.+]]:6 = enzyme.for_loop(%[[C0:.+]] : tensor<i64>) to(%[[NUM_SAMPLES:.+]] : tensor<i64>) step(%[[C1:.+]] : tensor<i64>) iter_args(%[[INIT_Q:.+]], %[[INIT_GRAD:.+]], %[[INIT_U:.+]], %[[INIT_RNG:.+]], %[[SAMPLES_INIT:.+]], %[[ACCEPTED_INIT:.+]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK-NEXT: ^bb0(%[[ITER_IDX:.+]]: tensor<i64>, %[[Q:.+]]: tensor<1xf64>, %[[GRAD:.+]]: tensor<1xf64>, %[[U:.+]]: tensor<f64>, %[[RNG:.+]]: tensor<2xui64>, %[[SAMPLES_BUF:.+]]: tensor<10x1xf64>, %[[ACCEPTED_BUF:.+]]: tensor<10xi1>):
// CHECK: %[[NUTS_RESULT:.+]]:18 = enzyme.while_loop
// CHECK: %[[STORAGE_COND:.+]] = arith.cmpi sge, %[[ITER_IDX]], %[[C0:.+]] : tensor<i64>
// CHECK: %[[UPDATED_SAMPLES:.+]] = enzyme.dynamic_update %[[SAMPLES_BUF]], %[[ITER_IDX]], %[[PROPOSAL:.+]] : (tensor<10x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<10x1xf64>
// CHECK: %[[SELECTED_SAMPLES:.+]] = enzyme.select %[[STORAGE_COND]], %[[UPDATED_SAMPLES]], %[[SAMPLES_BUF]] : (tensor<i1>, tensor<10x1xf64>, tensor<10x1xf64>) -> tensor<10x1xf64>
// CHECK: %[[UPDATED_ACCEPTED:.+]] = enzyme.dynamic_update %[[ACCEPTED_BUF]], %[[ITER_IDX]], %[[ACCEPT_VAL:.+]] : (tensor<10xi1>, tensor<i64>, tensor<i1>) -> tensor<10xi1>
// CHECK: %[[SELECTED_ACCEPTED:.+]] = enzyme.select %[[STORAGE_COND]], %[[UPDATED_ACCEPTED]], %[[ACCEPTED_BUF]] : (tensor<i1>, tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
// CHECK: enzyme.yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[SELECTED_SAMPLES]], %[[SELECTED_ACCEPTED]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>
// CHECK: }
// CHECK: %[[NEW_TRACE:.+]] = enzyme.initTrace
// CHECK: %[[RECOVERED:.+]] = enzyme.recover_sample %[[SAMPLES_LOOP]]#4[0] : tensor<10x1xf64> -> tensor<10xf64>
// CHECK: %[[TRACE_WITH_SAMPLES:.+]] = enzyme.addSampleToTrace %[[RECOVERED]] into %[[NEW_TRACE]] {symbol = #enzyme.symbol<1>}
// CHECK: %[[ORIG_WEIGHT:.+]] = enzyme.getWeightFromTrace %[[INIT_TRACE]]
// CHECK: %[[FINAL_TRACE:.+]] = enzyme.addWeightToTrace %[[ORIG_WEIGHT]] into %[[TRACE_WITH_SAMPLES]]

// CHECK-LABEL: func.func @sampling_thinning
// CHECK: %[[THIN_INIT_TRACE:.+]] = enzyme.initTrace
// CHECK: %[[THIN_INIT_POS:.+]] = enzyme.getFlattenedSamplesFromTrace
// CHECK: %[[THIN_LOOP:.+]]:6 = enzyme.for_loop(%[[TC0:.+]] : tensor<i64>) to(%[[TNUM_SAMPLES:.+]] : tensor<i64>) step(%[[TC1:.+]] : tensor<i64>) iter_args(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1> {
// CHECK: ^bb0(%[[THIN_ITER:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %[[THIN_SAMPLES_BUF:.+]]: tensor<5x1xf64>, %[[THIN_ACCEPTED_BUF:.+]]: tensor<5xi1>):
// CHECK: %[[STORAGE_IDX:.+]] = arith.divsi %[[THIN_ITER]], %[[THINNING:.+]] : tensor<i64>
// CHECK: %[[GE_ZERO:.+]] = arith.cmpi sge, %[[THIN_ITER]], %[[TC0:.+]] : tensor<i64>
// CHECK: %[[MOD_THIN:.+]] = arith.remsi %[[THIN_ITER]], %[[THINNING]] : tensor<i64>
// CHECK: %[[MOD_EQ_ZERO:.+]] = arith.cmpi eq, %[[MOD_THIN]], %[[TC0:.+]] : tensor<i64>
// CHECK: %[[SHOULD_STORE:.+]] = arith.andi %[[GE_ZERO]], %[[MOD_EQ_ZERO]] : tensor<i1>
// CHECK: %[[THIN_UPD_SAMPLES:.+]] = enzyme.dynamic_update %[[THIN_SAMPLES_BUF]], %[[STORAGE_IDX]], %{{.+}} : (tensor<5x1xf64>, tensor<i64>, tensor<1xf64>) -> tensor<5x1xf64>
// CHECK: %[[THIN_SEL_SAMPLES:.+]] = enzyme.select %[[SHOULD_STORE]], %[[THIN_UPD_SAMPLES]], %[[THIN_SAMPLES_BUF]] : (tensor<i1>, tensor<5x1xf64>, tensor<5x1xf64>) -> tensor<5x1xf64>
// CHECK: %[[THIN_UPD_ACCEPTED:.+]] = enzyme.dynamic_update %[[THIN_ACCEPTED_BUF]], %[[STORAGE_IDX]], %{{.+}} : (tensor<5xi1>, tensor<i64>, tensor<i1>) -> tensor<5xi1>
// CHECK: %[[THIN_SEL_ACCEPTED:.+]] = enzyme.select %[[SHOULD_STORE]], %[[THIN_UPD_ACCEPTED]], %[[THIN_ACCEPTED_BUF]] : (tensor<i1>, tensor<5xi1>, tensor<5xi1>) -> tensor<5xi1>
// CHECK: enzyme.yield %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[THIN_SEL_SAMPLES]], %[[THIN_SEL_ACCEPTED]] : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<5x1xf64>, tensor<5xi1>
// CHECK: }
// CHECK: enzyme.recover_sample %[[THIN_LOOP]]#4[0] : tensor<5x1xf64> -> tensor<5xf64>

// CHECK-LABEL: func.func @sampling_with_warmup
// CHECK: %[[W_INIT_TRACE:.+]] = enzyme.initTrace
// CHECK: %[[WARMUP_LOOP:.+]]:16 = enzyme.for_loop(%[[WC0:.+]] : tensor<i64>) to(%[[NUM_WARMUP:.+]] : tensor<i64>) step(%[[WC1:.+]] : tensor<i64>) iter_args(
// CHECK-SAME: tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>
// CHECK-SAME: ) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64> {
// CHECK: enzyme.yield
// CHECK: }
// CHECK: %[[SAMPLE_LOOP:.+]]:6 = enzyme.for_loop(%[[SC0:.+]] : tensor<i64>) to(%[[S_NUM_SAMPLES:.+]] : tensor<i64>) step(%[[SC1:.+]] : tensor<i64>) iter_args(%[[WARMUP_LOOP]]#0, %[[WARMUP_LOOP]]#1, %[[WARMUP_LOOP]]#2, %[[WARMUP_LOOP]]#3, %{{.+}}, %{{.+}} : tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1>) -> tensor<1xf64>, tensor<1xf64>, tensor<f64>, tensor<2xui64>, tensor<10x1xf64>, tensor<10xi1> {
// CHECK: ^bb0(%[[S_ITER:.+]]: tensor<i64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<1xf64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<2xui64>, %[[S_SAMPLES_BUF:.+]]: tensor<10x1xf64>, %[[S_ACCEPTED_BUF:.+]]: tensor<10xi1>):
// CHECK: enzyme.while_loop
// CHECK: %[[S_STORAGE_COND:.+]] = arith.cmpi sge, %[[S_ITER]], %[[SC0:.+]] : tensor<i64>
// CHECK: enzyme.dynamic_update %[[S_SAMPLES_BUF]], %[[S_ITER]]
// CHECK: enzyme.select
// CHECK: enzyme.dynamic_update %[[S_ACCEPTED_BUF]], %[[S_ITER]]
// CHECK: enzyme.select
// CHECK: enzyme.yield
// CHECK: }
// CHECK: enzyme.recover_sample %[[SAMPLE_LOOP]]#4[0] : tensor<10x1xf64> -> tensor<10xf64>
