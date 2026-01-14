// RUN: %eopt --probprog --canonicalize %s --mlir-print-ir-after=canonicalize | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %unused = enzyme.initTrace : !enzyme.Trace
    %init_trace = enzyme.initTrace : !enzyme.Trace

    %inverse_mass_matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      inverse_mass_matrix = %inverse_mass_matrix
      step_size = %step_size
      { hmc_config = #enzyme.hmc_config<trajectory_length = 1.000000e+00 : f64>, name = "hmc", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<2x2xf64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK:  func.func @hmc(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
// CHECK-NEXT:    %cst = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_0 = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_1 = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_2 = arith.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %cst_3 = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_4 = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_5 = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_6 = arith.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_7 = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_8 = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:    %2:3 = enzyme.randomSplit %1#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:    %3 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : (!enzyme.Trace) -> tensor<2xf64>
// CHECK-NEXT:    %4 = enzyme.getWeightFromTrace %0 : (!enzyme.Trace) -> tensor<f64>
// CHECK-NEXT:    %5 = arith.negf %4 : tensor<f64>
// CHECK-NEXT:    %6 = enzyme.autodiff_region(%3, %cst_7) {
// CHECK-NEXT:    ^bb0(%arg3: tensor<2xf64>):
// CHECK-NEXT:      %27:3 = func.call @test.update_1(%0, %arg3, %2#1, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %28 = arith.negf %27#1 : tensor<f64>
// CHECK-NEXT:      enzyme.yield %28, %27#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_constnoneed>]} : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:    %7:3 = enzyme.randomSplit %2#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:    %8 = enzyme.randomSplit %7#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT:    %output_rng_state, %result = enzyme.random %8, %cst_4, %cst_7 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:    %9 = enzyme.cholesky %cst_8 : (tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %10 = enzyme.triangular_solve %9, %result {transpose_a = #enzyme<transpose TRANSPOSE>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %11 = enzyme.dot %cst_8, %10 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %12 = enzyme.dot %10, %11 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %13 = arith.mulf %12, %cst_3 : tensor<f64>
// CHECK-NEXT:    %14 = arith.addf %5, %13 : tensor<f64>
// CHECK-NEXT:    %15:5 = enzyme.for_loop(%cst_1 : tensor<i64>) to(%cst_6 : tensor<i64>) step(%cst_0 : tensor<i64>) iter_args(%3, %10, %6, %5, %7#2 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<2xui64> {
// CHECK-NEXT:    ^bb0(%arg3: tensor<i64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<f64>, %arg8: tensor<2xui64>):
// CHECK-NEXT:      %27 = enzyme.select %cst_2, %cst_5, %cst : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %28 = "enzyme.broadcast"(%27) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:      %29 = arith.mulf %27, %cst_3 : tensor<f64>
// CHECK-NEXT:      %30 = "enzyme.broadcast"(%29) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:      %31 = arith.mulf %30, %arg6 : tensor<2xf64>
// CHECK-NEXT:      %32 = arith.subf %arg5, %31 : tensor<2xf64>
// CHECK-NEXT:      %33 = enzyme.dot %cst_8, %32 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %34 = arith.mulf %28, %33 : tensor<2xf64>
// CHECK-NEXT:      %35 = arith.addf %arg4, %34 : tensor<2xf64>
// CHECK-NEXT:      %36:3 = enzyme.autodiff_region(%35, %cst_7) {
// CHECK-NEXT:      ^bb0(%arg9: tensor<2xf64>):
// CHECK-NEXT:        %39:3 = func.call @test.update_0(%0, %arg9, %arg8, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:        %40 = arith.negf %39#1 : tensor<f64>
// CHECK-NEXT:        enzyme.yield %40, %39#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %37 = arith.mulf %30, %36#2 : tensor<2xf64>
// CHECK-NEXT:      %38 = arith.subf %32, %37 : tensor<2xf64>
// CHECK-NEXT:      enzyme.yield %35, %38, %36#2, %36#0, %36#1 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %16 = enzyme.dot %cst_8, %15#1 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %17 = enzyme.dot %15#1, %16 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %18 = arith.mulf %17, %cst_3 : tensor<f64>
// CHECK-NEXT:    %19 = arith.addf %15#3, %18 : tensor<f64>
// CHECK-NEXT:    %20 = arith.subf %14, %19 : tensor<f64>
// CHECK-NEXT:    %21 = math.exp %20 : tensor<f64>
// CHECK-NEXT:    %22 = arith.minimumf %21, %cst_7 : tensor<f64>
// CHECK-NEXT:    %output_rng_state_9, %result_10 = enzyme.random %15#4, %cst_4, %cst_7 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %23 = arith.cmpf olt, %result_10, %22 : tensor<f64>
// CHECK-NEXT:    %24 = enzyme.select %23, %15#0, %3 : (tensor<i1>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %25:3 = call @test.update(%0, %24, %7#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:    %26 = enzyme.select %23, %25#0, %0 : (tensor<i1>, !enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// CHECK-NEXT:    return %26, %23, %25#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CHECK-NEXT:  }