// RUN: %eopt --probprog --canonicalize %s | FileCheck %s

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
    %num_steps = arith.constant dense<10> : tensor<i64>

    %res:3 = enzyme.mcmc algorithm = HMC @test(%rng, %mean, %stddev) given %init_trace
      inverse_mass_matrix = %inverse_mass_matrix : tensor<2x2xf64>
      step_size = %step_size : tensor<f64>
      num_steps = %num_steps : tensor<i64>
      { name = "hmc", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK:  func.func @hmc(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
// CHECK-NEXT:    %cst = arith.constant dense<5.000000e-02> : tensor<f64>
// CHECK-NEXT:    %cst_0 = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_1 = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_2 = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_3 = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_4 = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_5 = arith.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_6 = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_7 = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : tensor<2xf64>
// CHECK-NEXT:    %2 = enzyme.getWeightFromTrace %0 : tensor<f64>
// CHECK-NEXT:    %3 = arith.negf %2 : tensor<f64>
// CHECK-NEXT:    %4 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT:    %output_rng_state, %result = enzyme.random %4, %cst_4, %cst_3 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:    %5 = enzyme.cholesky_solve %cst_7, %cst_7 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %6 = enzyme.dot %5, %result {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %7 = enzyme.dot %cst_7, %6 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %8 = enzyme.dot %6, %7 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %9 = arith.mulf %8, %cst_2 : tensor<f64>
// CHECK-NEXT:    %10 = arith.addf %3, %9 : tensor<f64>
// CHECK-NEXT:    %11:2 = enzyme.autodiff_region(%1, %cst_3) {
// CHECK-NEXT:    ^bb0(%arg3: tensor<2xf64>):
// CHECK-NEXT:      %24:3 = func.call @test.update_1(%0, %arg3, %output_rng_state, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %25 = arith.negf %24#1 : tensor<f64>
// CHECK-NEXT:      enzyme.yield %25, %24#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:    %12:4 = enzyme.for_loop(%cst_1 : tensor<i64>) to(%cst_5 : tensor<i64>) step(%cst_0 : tensor<i64>) iter_args(%1, %6, %11#1, %11#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64> {
// CHECK-NEXT:    ^bb0(%arg3: tensor<i64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xui64>):
// CHECK-NEXT:      %24 = "enzyme.broadcast"(%cst_6) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:      %25 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:      %26 = arith.mulf %25, %arg6 : tensor<2xf64>
// CHECK-NEXT:      %27 = arith.subf %arg5, %26 : tensor<2xf64>
// CHECK-NEXT:      %28 = enzyme.dot %cst_7, %27 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %29 = arith.mulf %24, %28 : tensor<2xf64>
// CHECK-NEXT:      %30 = arith.addf %arg4, %29 : tensor<2xf64>
// CHECK-NEXT:      %31:2 = enzyme.autodiff_region(%30, %cst_3) {
// CHECK-NEXT:      ^bb0(%arg8: tensor<2xf64>):
// CHECK-NEXT:        %34:3 = func.call @test.update_0(%0, %arg8, %arg7, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:        %35 = arith.negf %34#1 : tensor<f64>
// CHECK-NEXT:        enzyme.yield %35, %34#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %32 = arith.mulf %25, %31#1 : tensor<2xf64>
// CHECK-NEXT:      %33 = arith.subf %27, %32 : tensor<2xf64>
// CHECK-NEXT:      enzyme.yield %30, %33, %31#1, %31#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %13:3 = call @test.update(%0, %12#0, %12#3, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:    %14 = arith.negf %13#1 : tensor<f64>
// CHECK-NEXT:    %15 = enzyme.dot %cst_7, %12#1 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %16 = enzyme.dot %12#1, %15 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %17 = arith.mulf %16, %cst_2 : tensor<f64>
// CHECK-NEXT:    %18 = arith.addf %14, %17 : tensor<f64>
// CHECK-NEXT:    %19 = arith.subf %10, %18 : tensor<f64>
// CHECK-NEXT:    %20 = math.exp %19 : tensor<f64>
// CHECK-NEXT:    %21 = arith.minimumf %20, %cst_3 : tensor<f64>
// CHECK-NEXT:    %output_rng_state_8, %result_9 = enzyme.random %13#2, %cst_4, %cst_3 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %22 = arith.cmpf olt, %result_9, %21 : tensor<f64>
// CHECK-NEXT:    %23 = enzyme.selectTrace %22, %13#0, %0 : tensor<i1>
// CHECK-NEXT:    return %23, %22, %output_rng_state_8 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CHECK-NEXT:  }

// CHECK:  func.func @test.update(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1 = enzyme.unflatten_slice %arg1[0] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:    %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:    %4 = enzyme.addSampleToTrace(%1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %5 = enzyme.unflatten_slice %arg1[1] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:    %6 = call @logpdf(%5, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %7 = arith.addf %3, %6 : tensor<f64>
// CHECK-NEXT:    %8 = enzyme.addSampleToTrace(%5 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:    %10 = enzyme.addRetvalToTrace(%5 : tensor<f64>) into %9
// CHECK-NEXT:    return %10, %7, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:  }