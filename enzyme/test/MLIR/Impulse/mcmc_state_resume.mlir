// RUN: %eopt --expand-impulse %s | FileCheck %s

// Resumed run: no InitHMC gradient recompute, so no autodiff before warmup.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @resume(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>,
      %pos : tensor<1x1xf64>, %grad : tensor<1x1xf64>, %pe : tensor<f64>, %ss : tensor<f64>, %imm : tensor<1x1xf64>,
      %da0 : tensor<f64>, %da1 : tensor<f64>, %da2 : tensor<f64>, %da3 : tensor<i64>, %da4 : tensor<f64>,
      %wm : tensor<1xf64>, %wm2 : tensor<1xf64>, %wn : tensor<i64>, %widx : tensor<i64>, %off : tensor<i64>)
      -> (tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>,
          tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>, tensor<i64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:19 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      init_step_size = %step_size
      state = [%pos, %grad, %pe, %ss, %imm, %da0, %da1, %da2, %da3, %da4, %wm, %wm2, %wn, %widx, %off]
      <{ nuts_config = #impulse.nuts_config<max_tree_depth = 5>,
        name = "resume", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]],
        num_warmup = 4, total_warmup = 100, num_samples = 1,
        state_schema = ["position", "gradient", "potential_energy", "step_size", "inverse_mass_matrix", "da_log_step_size", "da_log_step_size_avg", "da_gradient_avg", "da_step_count", "da_prox_center", "welford_mean", "welford_m2", "welford_n", "window_idx", "warmup_offset"] }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>,
         tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>,
         tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>, tensor<i64>,
         tensor<1x1xf64>, tensor<f64>)
        -> (tensor<1x1xf64>, tensor<1x2xi1>, tensor<1xf64>, tensor<2xui64>,
            tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>,
            tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>, tensor<i64>)
    return %res#3, %res#4, %res#5, %res#6, %res#7, %res#8, %res#9, %res#10, %res#11, %res#12, %res#13, %res#14, %res#15, %res#16, %res#17, %res#18
      : tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>,
        tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>, tensor<i64>
  }
}

// CHECK-LABEL: func.func @resume(
// CHECK-SAME: %[[RNG:.+]]: tensor<2xui64>, %{{.+}}: tensor<f64>, %{{.+}}: tensor<f64>, %[[POS:.+]]: tensor<1x1xf64>, %[[GRAD:.+]]: tensor<1x1xf64>, %[[PE:.+]]: tensor<f64>, %[[SS:.+]]: tensor<f64>, %[[IMM:.+]]: tensor<1x1xf64>, %[[DA0:.+]]: tensor<f64>, %[[DA1:.+]]: tensor<f64>, %[[DA2:.+]]: tensor<f64>, %[[DA3:.+]]: tensor<i64>, %[[DA4:.+]]: tensor<f64>, %[[WM:.+]]: tensor<1xf64>, %[[WM2:.+]]: tensor<1xf64>, %[[WN:.+]]: tensor<i64>, %[[WIDX:.+]]: tensor<i64>, %[[OFF:.+]]: tensor<i64>) ->
// CHECK-NOT: enzyme.autodiff_region
// CHECK: %[[SQRT:.+]] = impulse.triangular_solve
// CHECK: %[[WU:.+]]:16 = impulse.for(%{{.+}} : tensor<i64>) to(%[[C4:.+]] : tensor<i64>) step(%{{.+}} : tensor<i64>) iter_args(%[[POS]], %[[GRAD]], %[[PE]], %[[RNG]], %[[SS]], %[[IMM]], %[[SQRT]], %[[DA0]], %[[DA1]], %[[DA2]], %[[DA3]], %[[DA4]], %[[WM]], %[[WM2]], %[[WN]], %[[WIDX]] :
// CHECK: arith.addi %[[OFF]], %{{.+}} : tensor<i64>
// CHECK: %[[SAMP:.+]]:7 = impulse.for({{.*}}) iter_args(%[[WU]]#0, %[[WU]]#1, %[[WU]]#2, %[[WU]]#3,
// CHECK: %[[OFFOUT:.+]] = arith.addi %[[OFF]], %[[C4]] : tensor<i64>
// CHECK: return %[[SAMP]]#3, %[[SAMP]]#0, %[[SAMP]]#1, %[[SAMP]]#2, %[[WU]]#4, %[[WU]]#5, %[[WU]]#7, %[[WU]]#8, %[[WU]]#9, %[[WU]]#10, %[[WU]]#11, %[[WU]]#12, %[[WU]]#13, %[[WU]]#14, %[[WU]]#15, %[[OFFOUT]] : tensor<2xui64>, tensor<1x1xf64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<1xf64>, tensor<1xf64>, tensor<i64>, tensor<i64>, tensor<i64>
