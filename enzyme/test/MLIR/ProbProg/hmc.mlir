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
      { hmc_config = #enzyme.hmc_config<num_steps = 10>, name = "hmc", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>, !enzyme.Trace, tensor<2x2xf64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK:  func.func @hmc(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
// CHECK-NEXT:    %cst = arith.constant dense<5.000000e-02> : tensor<f64>
// CHECK-NEXT:    %cst_0 = arith.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_1 = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_2 = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_3 = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_4 = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_5 = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_6 = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_7 = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:    %2:3 = enzyme.randomSplit %1#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:    %3 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : (!enzyme.Trace) -> tensor<2xf64>
// CHECK-NEXT:    %4 = enzyme.getWeightFromTrace %0 : (!enzyme.Trace) -> tensor<f64>
// CHECK-NEXT:    %5 = arith.negf %4 : tensor<f64>
// CHECK-NEXT:    %6 = enzyme.autodiff_region(%3, %cst_5) {
// CHECK-NEXT:    ^bb0(%arg3: tensor<2xf64>):
// CHECK-NEXT:      %28:3 = func.call @test.update_1(%0, %arg3, %2#1, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %29 = arith.negf %28#1 : tensor<f64>
// CHECK-NEXT:      enzyme.yield %29, %28#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_constnoneed>]} : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:    %7:3 = enzyme.randomSplit %2#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:    %8 = enzyme.randomSplit %7#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT:    %output_rng_state, %result = enzyme.random %8, %cst_4, %cst_5 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:    %9 = enzyme.cholesky_solve %cst_7, %cst_7 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %10 = enzyme.dot %9, %result {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %11 = enzyme.dot %cst_7, %10 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %12 = enzyme.dot %10, %11 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %13 = arith.mulf %12, %cst_3 : tensor<f64>
// CHECK-NEXT:    %14 = arith.addf %5, %13 : tensor<f64>
// CHECK-NEXT:    %15:5 = enzyme.for_loop(%cst_2 : tensor<i64>) to(%cst_0 : tensor<i64>) step(%cst_1 : tensor<i64>) iter_args(%3, %10, %6, %5, %7#2 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<2xui64> {
// CHECK-NEXT:    ^bb0(%arg3: tensor<i64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<f64>, %arg8: tensor<2xui64>):
// CHECK-NEXT:      %28 = "enzyme.broadcast"(%cst_6) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:      %29 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:      %30 = arith.mulf %29, %arg6 : tensor<2xf64>
// CHECK-NEXT:      %31 = arith.subf %arg5, %30 : tensor<2xf64>
// CHECK-NEXT:      %32 = enzyme.dot %cst_7, %31 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %33 = arith.mulf %28, %32 : tensor<2xf64>
// CHECK-NEXT:      %34 = arith.addf %arg4, %33 : tensor<2xf64>
// CHECK-NEXT:      %35:3 = enzyme.autodiff_region(%34, %cst_5) {
// CHECK-NEXT:      ^bb0(%arg9: tensor<2xf64>):
// CHECK-NEXT:        %38:3 = func.call @test.update_0(%0, %arg9, %arg8, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:        %39 = arith.negf %38#1 : tensor<f64>
// CHECK-NEXT:        enzyme.yield %39, %38#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %36 = arith.mulf %29, %35#2 : tensor<2xf64>
// CHECK-NEXT:      %37 = arith.subf %31, %36 : tensor<2xf64>
// CHECK-NEXT:      enzyme.yield %34, %37, %35#2, %35#0, %35#1 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %16 = enzyme.dot %cst_7, %15#1 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %17 = enzyme.dot %15#1, %16 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %18 = arith.mulf %17, %cst_3 : tensor<f64>
// CHECK-NEXT:    %19 = arith.addf %15#3, %18 : tensor<f64>
// CHECK-NEXT:    %20 = arith.subf %14, %19 : tensor<f64>
// CHECK-NEXT:    %21 = math.exp %20 : tensor<f64>
// CHECK-NEXT:    %22 = arith.minimumf %21, %cst_5 : tensor<f64>
// CHECK-NEXT:    %output_rng_state_8, %result_9 = enzyme.random %15#4, %cst_4, %cst_5 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %23 = arith.cmpf olt, %result_9, %22 : tensor<f64>
// CHECK-NEXT:    %24 = "enzyme.broadcast"(%23) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:    %25 = arith.select %24, %15#0, %3 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:    %26:3 = call @test.update(%0, %25, %7#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:    %27 = enzyme.selectTrace %23, %26#0, %0 : (tensor<i1>, !enzyme.Trace, !enzyme.Trace) -> !enzyme.Trace
// CHECK-NEXT:    return %27, %23, %26#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CHECK-NEXT:  }
