// RUN: %eopt --probprog %s --mlir-print-ir-after=probprog | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @nuts(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %init_trace = enzyme.initTrace : !enzyme.Trace

    %inverse_mass_matrix = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>

    %res:3 = enzyme.mcmc algorithm = NUTS @test(%rng, %mean, %stddev) given %init_trace
      inverse_mass_matrix = %inverse_mass_matrix : tensor<2x2xf64>
      step_size = %step_size : tensor<f64>
      { name = "nuts", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
    return %res#0, %res#1, %res#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }
}

// CHECK:   func.func @nuts(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
// CHECK-NEXT:      %cst = arith.constant dense<-1> : tensor<i64>
// CHECK-NEXT:      %cst_0 = arith.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-NEXT:      %cst_1 = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-NEXT:      %cst_2 = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT:      %cst_3 = arith.constant dense<0.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:      %cst_4 = arith.constant dense<10> : tensor<i64>
// CHECK-NEXT:      %cst_5 = arith.constant dense<true> : tensor<i1>
// CHECK-NEXT:      %cst_6 = arith.constant dense<1.000000e+03> : tensor<f64>
// CHECK-NEXT:      %cst_7 = arith.constant dense<false> : tensor<i1>
// CHECK-NEXT:      %cst_8 = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %cst_9 = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:      %cst_10 = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:      %cst_11 = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:      %cst_12 = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-NEXT:      %cst_13 = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CHECK-NEXT:      %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:      %1:2 = enzyme.randomSplit %arg0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:      %2:3 = enzyme.randomSplit %1#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:      %3 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : tensor<2xf64>
// CHECK-NEXT:      %4 = enzyme.getWeightFromTrace %0 : tensor<f64>
// CHECK-NEXT:      %5 = arith.negf %4 : tensor<f64>
// CHECK-NEXT:      %6 = enzyme.randomSplit %2#2 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT:      %output_rng_state, %result = enzyme.random %6, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %7:2 = enzyme.autodiff_region(%3, %cst_10) {
// CHECK-NEXT:      ^bb0(%arg3: tensor<2xf64>):
// CHECK-NEXT:        %18:3 = func.call @test.update_1(%0, %arg3, %2#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:        %19 = arith.negf %18#1 : tensor<f64>
// CHECK-NEXT:        enzyme.yield %19, %18#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %8:3 = enzyme.randomSplit %7#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:      %9 = enzyme.randomSplit %8#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT:      %output_rng_state_14, %result_15 = enzyme.random %9, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %10 = enzyme.cholesky_solve %cst_13, %cst_13 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:      %11 = enzyme.dot %10, %result_15 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %12 = enzyme.dot %cst_13, %11 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %13 = enzyme.dot %11, %12 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:      %14 = arith.mulf %13, %cst_9 : tensor<f64>
// CHECK-NEXT:      %15 = arith.addf %5, %14 : tensor<f64>
// CHECK-NEXT:      %16:18 = enzyme.while_loop(%3, %11, %7#1, %3, %11, %7#1, %3, %7#1, %5, %15, %cst_8, %cst_11, %cst_7, %cst_7, %cst_11, %cst_8, %11, %8#2 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64> condition {
// CHECK-NEXT:      ^bb0(%arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<i64>, %arg14: tensor<f64>, %arg15: tensor<i1>, %arg16: tensor<i1>, %arg17: tensor<f64>, %arg18: tensor<i64>, %arg19: tensor<2xf64>, %arg20: tensor<2xui64>):
// CHECK-NEXT:        %18 = arith.cmpi slt, %arg13, %cst_4 : tensor<i64>
// CHECK-NEXT:        %19 = arith.xori %arg15, %cst_5 : tensor<i1>
// CHECK-NEXT:        %20 = arith.xori %arg16, %cst_5 : tensor<i1>
// CHECK-NEXT:        %21 = arith.andi %18, %19 : tensor<i1>
// CHECK-NEXT:        %22 = arith.andi %21, %20 : tensor<i1>
// CHECK-NEXT:        enzyme.yield %22 : tensor<i1>
// CHECK-NEXT:      } body {
// CHECK-NEXT:      ^bb0(%arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<i64>, %arg14: tensor<f64>, %arg15: tensor<i1>, %arg16: tensor<i1>, %arg17: tensor<f64>, %arg18: tensor<i64>, %arg19: tensor<2xf64>, %arg20: tensor<2xui64>):
// CHECK-NEXT:        %18:3 = enzyme.randomSplit %arg20 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:        %output_rng_state_16, %result_17 = enzyme.random %18#1, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:        %19 = arith.cmpf olt, %result_17, %cst_9 : tensor<f64>
// CHECK-NEXT:        %20:2 = enzyme.randomSplit %18#2 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:        %21 = arith.shli %cst_2, %arg13 : tensor<i64>
// CHECK-NEXT:        %22:21 = enzyme.while_loop(%arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst_8, %arg19, %20#0, %cst_3, %cst_3, %cst_8 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<i64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<i64> condition {
// CHECK-NEXT:        ^bb0(%arg21: tensor<2xf64>, %arg22: tensor<2xf64>, %arg23: tensor<2xf64>, %arg24: tensor<2xf64>, %arg25: tensor<2xf64>, %arg26: tensor<2xf64>, %arg27: tensor<2xf64>, %arg28: tensor<2xf64>, %arg29: tensor<f64>, %arg30: tensor<f64>, %arg31: tensor<i64>, %arg32: tensor<f64>, %arg33: tensor<i1>, %arg34: tensor<i1>, %arg35: tensor<f64>, %arg36: tensor<i64>, %arg37: tensor<2xf64>, %arg38: tensor<2xui64>, %arg39: tensor<10x2xf64>, %arg40: tensor<10x2xf64>, %arg41: tensor<i64>):
// CHECK-NEXT:          %60 = arith.cmpi slt, %arg36, %21 : tensor<i64>
// CHECK-NEXT:          %61 = arith.xori %arg33, %cst_5 : tensor<i1>
// CHECK-NEXT:          %62 = arith.xori %arg34, %cst_5 : tensor<i1>
// CHECK-NEXT:          %63 = arith.andi %60, %61 : tensor<i1>
// CHECK-NEXT:          %64 = arith.andi %63, %62 : tensor<i1>
// CHECK-NEXT:          enzyme.yield %64 : tensor<i1>
// CHECK-NEXT:        } body {
// CHECK-NEXT:        ^bb0(%arg21: tensor<2xf64>, %arg22: tensor<2xf64>, %arg23: tensor<2xf64>, %arg24: tensor<2xf64>, %arg25: tensor<2xf64>, %arg26: tensor<2xf64>, %arg27: tensor<2xf64>, %arg28: tensor<2xf64>, %arg29: tensor<f64>, %arg30: tensor<f64>, %arg31: tensor<i64>, %arg32: tensor<f64>, %arg33: tensor<i1>, %arg34: tensor<i1>, %arg35: tensor<f64>, %arg36: tensor<i64>, %arg37: tensor<2xf64>, %arg38: tensor<2xui64>, %arg39: tensor<10x2xf64>, %arg40: tensor<10x2xf64>, %arg41: tensor<i64>):
// CHECK-NEXT:          %60 = "enzyme.broadcast"(%19) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %61 = arith.select %60, %arg24, %arg21 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %62 = arith.select %60, %arg25, %arg22 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %63 = arith.select %60, %arg26, %arg23 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %64:2 = enzyme.randomSplit %arg38 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:          %65 = arith.select %19, %cst_12, %cst_1 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %66 = "enzyme.broadcast"(%65) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:          %67 = arith.mulf %65, %cst_9 : tensor<f64>
// CHECK-NEXT:          %68 = "enzyme.broadcast"(%67) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:          %69 = arith.mulf %68, %63 : tensor<2xf64>
// CHECK-NEXT:          %70 = arith.subf %62, %69 : tensor<2xf64>
// CHECK-NEXT:          %71 = enzyme.dot %cst_13, %70 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:          %72 = arith.mulf %66, %71 : tensor<2xf64>
// CHECK-NEXT:          %73 = arith.addf %61, %72 : tensor<2xf64>
// CHECK-NEXT:          %74:3 = enzyme.autodiff_region(%73, %cst_10) {
// CHECK-NEXT:          ^bb0(%arg42: tensor<2xf64>):
// CHECK-NEXT:            %150:3 = func.call @test.update_0(%0, %arg42, %64#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:            %151 = arith.negf %150#1 : tensor<f64>
// CHECK-NEXT:            enzyme.yield %151, %150#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:          } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:          %75 = arith.mulf %68, %74#2 : tensor<2xf64>
// CHECK-NEXT:          %76 = arith.subf %70, %75 : tensor<2xf64>
// CHECK-NEXT:          %77 = enzyme.dot %cst_13, %76 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:          %78 = enzyme.dot %76, %77 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:          %79 = arith.mulf %78, %cst_9 : tensor<f64>
// CHECK-NEXT:          %80 = arith.addf %74#0, %79 : tensor<f64>
// CHECK-NEXT:          %81 = arith.subf %80, %15 : tensor<f64>
// CHECK-NEXT:          %82 = arith.cmpf une, %81, %81 : tensor<f64>
// CHECK-NEXT:          %83 = arith.select %82, %cst_0, %81 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %84 = arith.negf %83 : tensor<f64>
// CHECK-NEXT:          %85 = arith.cmpf ogt, %83, %cst_6 : tensor<f64>
// CHECK-NEXT:          %86 = arith.negf %83 : tensor<f64>
// CHECK-NEXT:          %87 = math.exp %86 : tensor<f64>
// CHECK-NEXT:          %88 = arith.minimumf %87, %cst_10 : tensor<f64>
// CHECK-NEXT:          %89 = "enzyme.broadcast"(%19) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %90 = arith.select %89, %arg21, %73 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %91 = arith.select %89, %arg22, %76 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %92 = arith.select %89, %arg23, %74#2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %93 = arith.select %89, %73, %arg24 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %94 = arith.select %89, %76, %arg25 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %95 = arith.select %89, %74#2, %arg26 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %96 = enzyme.log_add_exp %arg32, %84 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:          %97 = arith.subf %84, %arg32 : tensor<f64>
// CHECK-NEXT:          %98 = arith.negf %97 : tensor<f64>
// CHECK-NEXT:          %99 = math.exp %98 : tensor<f64>
// CHECK-NEXT:          %100 = arith.addf %99, %cst_10 : tensor<f64>
// CHECK-NEXT:          %101 = arith.divf %cst_10, %100 : tensor<f64>
// CHECK-NEXT:          %output_rng_state_20, %result_21 = enzyme.random %64#1, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:          %102 = arith.cmpf olt, %result_21, %101 : tensor<f64>
// CHECK-NEXT:          %103 = "enzyme.broadcast"(%102) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %104 = arith.select %103, %73, %arg27 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %105 = arith.select %103, %74#2, %arg28 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %106 = arith.select %102, %74#0, %arg29 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %107 = arith.select %102, %80, %arg30 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %108 = arith.addi %arg31, %cst_2 : tensor<i64>
// CHECK-NEXT:          %109 = arith.ori %arg34, %85 : tensor<i1>
// CHECK-NEXT:          %110 = arith.addf %arg35, %88 : tensor<f64>
// CHECK-NEXT:          %111 = arith.addi %arg36, %cst_2 : tensor<i64>
// CHECK-NEXT:          %112 = arith.addf %arg37, %76 : tensor<2xf64>
// CHECK-NEXT:          %113 = arith.cmpi eq, %arg36, %cst_8 : tensor<i64>
// CHECK-NEXT:          %114 = "enzyme.broadcast"(%113) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %115 = arith.select %114, %73, %90 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %116 = arith.select %114, %76, %91 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %117 = arith.select %114, %74#2, %92 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %118 = arith.select %114, %73, %93 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %119 = arith.select %114, %76, %94 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %120 = arith.select %114, %74#2, %95 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %121 = arith.select %114, %73, %104 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %122 = arith.select %114, %74#2, %105 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %123 = arith.select %114, %76, %112 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %124 = arith.select %113, %74#0, %106 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %125 = arith.select %113, %80, %107 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %126 = arith.select %113, %cst_8, %108 : tensor<i1>, tensor<i64>
// CHECK-NEXT:          %127 = arith.select %113, %84, %96 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %128 = arith.select %113, %85, %109 : tensor<i1>, tensor<i1>
// CHECK-NEXT:          %129 = arith.select %113, %88, %110 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %130 = arith.select %113, %cst_2, %111 : tensor<i1>, tensor<i64>
// CHECK-NEXT:          %131 = arith.shrui %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          %132 = enzyme.popcount %131 : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:          %133 = arith.addi %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          %134 = arith.xori %arg41, %cst : tensor<i64>
// CHECK-NEXT:          %135 = arith.andi %134, %133 : tensor<i64>
// CHECK-NEXT:          %136 = arith.subi %135, %cst_2 : tensor<i64>
// CHECK-NEXT:          %137 = enzyme.popcount %136 : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:          %138 = arith.subi %132, %137 : tensor<i64>
// CHECK-NEXT:          %139 = arith.addi %138, %cst_2 : tensor<i64>
// CHECK-NEXT:          %140 = arith.andi %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          %141 = arith.cmpi eq, %140, %cst_8 : tensor<i64>
// CHECK-NEXT:          %142 = enzyme.dynamic_update %arg39[%132] = %76 : (tensor<10x2xf64>, tensor<i64>, tensor<2xf64>) -> tensor<10x2xf64>
// CHECK-NEXT:          %143 = enzyme.dynamic_update %arg40[%132] = %123 : (tensor<10x2xf64>, tensor<i64>, tensor<2xf64>) -> tensor<10x2xf64>
// CHECK-NEXT:          %144 = "enzyme.broadcast"(%141) <{shape = array<i64: 10, 2>}> : (tensor<i1>) -> tensor<10x2xi1>
// CHECK-NEXT:          %145 = arith.select %144, %142, %arg39 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:          %146 = arith.select %144, %143, %arg40 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:          %147:2 = enzyme.while_loop(%132, %cst_7 : tensor<i64>, tensor<i1>) -> tensor<i64>, tensor<i1> condition {
// CHECK-NEXT:          ^bb0(%arg42: tensor<i64>, %arg43: tensor<i1>):
// CHECK-NEXT:            %150 = arith.cmpi sge, %arg42, %139 : tensor<i64>
// CHECK-NEXT:            %151 = arith.xori %arg43, %cst_5 : tensor<i1>
// CHECK-NEXT:            %152 = arith.andi %150, %151 : tensor<i1>
// CHECK-NEXT:            enzyme.yield %152 : tensor<i1>
// CHECK-NEXT:          } body {
// CHECK-NEXT:          ^bb0(%arg42: tensor<i64>, %arg43: tensor<i1>):
// CHECK-NEXT:            %150 = enzyme.dynamic_extract %145[%arg42] : (tensor<10x2xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:            %151 = enzyme.dynamic_extract %146[%arg42] : (tensor<10x2xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:            %152 = arith.subf %123, %151 : tensor<2xf64>
// CHECK-NEXT:            %153 = arith.addf %152, %150 : tensor<2xf64>
// CHECK-NEXT:            %154 = enzyme.dot %cst_13, %150 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:            %155 = enzyme.dot %cst_13, %76 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:            %156 = "enzyme.broadcast"(%cst_9) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:            %157 = arith.addf %150, %76 : tensor<2xf64>
// CHECK-NEXT:            %158 = arith.mulf %156, %157 : tensor<2xf64>
// CHECK-NEXT:            %159 = arith.subf %153, %158 : tensor<2xf64>
// CHECK-NEXT:            %160 = enzyme.dot %154, %159 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:            %161 = enzyme.dot %155, %159 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:            %162 = arith.cmpf ole, %160, %cst_11 : tensor<f64>
// CHECK-NEXT:            %163 = arith.cmpf ole, %161, %cst_11 : tensor<f64>
// CHECK-NEXT:            %164 = arith.ori %162, %163 : tensor<i1>
// CHECK-NEXT:            %165 = arith.subi %arg42, %cst_2 : tensor<i64>
// CHECK-NEXT:            enzyme.yield %165, %164 : tensor<i64>, tensor<i1>
// CHECK-NEXT:          }
// CHECK-NEXT:          %148 = arith.select %113, %cst_7, %147#1 : tensor<i1>, tensor<i1>
// CHECK-NEXT:          %149 = arith.addi %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          enzyme.yield %115, %116, %117, %118, %119, %120, %121, %122, %124, %125, %126, %127, %148, %128, %129, %130, %123, %64#0, %145, %146, %149 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %23 = "enzyme.broadcast"(%19) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:        %24 = arith.select %23, %arg3, %22#0 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %25 = arith.select %23, %arg4, %22#1 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %26 = arith.select %23, %arg5, %22#2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %27 = arith.select %23, %22#3, %arg6 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %28 = arith.select %23, %22#4, %arg7 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %29 = arith.select %23, %22#5, %arg8 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %30 = enzyme.log_add_exp %arg14, %22#11 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:        %31 = arith.subf %22#11, %arg14 : tensor<f64>
// CHECK-NEXT:        %32 = math.exp %31 : tensor<f64>
// CHECK-NEXT:        %33 = arith.minimumf %32, %cst_10 : tensor<f64>
// CHECK-NEXT:        %34 = arith.ori %22#12, %22#13 : tensor<i1>
// CHECK-NEXT:        %35 = arith.select %34, %cst_11, %33 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %output_rng_state_18, %result_19 = enzyme.random %20#1, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:        %36 = arith.cmpf olt, %result_19, %35 : tensor<f64>
// CHECK-NEXT:        %37 = "enzyme.broadcast"(%36) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:        %38 = arith.select %37, %22#6, %arg9 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %39 = arith.select %37, %22#7, %arg10 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %40 = arith.select %36, %22#8, %arg11 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %41 = arith.select %36, %22#9, %arg12 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %42 = arith.addi %arg13, %cst_2 : tensor<i64>
// CHECK-NEXT:        %43 = arith.addf %arg19, %22#16 : tensor<2xf64>
// CHECK-NEXT:        %44 = enzyme.dot %cst_13, %25 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %45 = enzyme.dot %cst_13, %28 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %46 = "enzyme.broadcast"(%cst_9) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:        %47 = arith.addf %25, %28 : tensor<2xf64>
// CHECK-NEXT:        %48 = arith.mulf %46, %47 : tensor<2xf64>
// CHECK-NEXT:        %49 = arith.subf %43, %48 : tensor<2xf64>
// CHECK-NEXT:        %50 = enzyme.dot %44, %49 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %51 = enzyme.dot %45, %49 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %52 = arith.cmpf ole, %50, %cst_11 : tensor<f64>
// CHECK-NEXT:        %53 = arith.cmpf ole, %51, %cst_11 : tensor<f64>
// CHECK-NEXT:        %54 = arith.ori %52, %53 : tensor<i1>
// CHECK-NEXT:        %55 = arith.ori %22#12, %54 : tensor<i1>
// CHECK-NEXT:        %56 = arith.ori %arg16, %22#13 : tensor<i1>
// CHECK-NEXT:        %57 = arith.addf %arg17, %22#14 : tensor<f64>
// CHECK-NEXT:        %58 = arith.addi %arg18, %22#15 : tensor<i64>
// CHECK-NEXT:        %59 = arith.addf %arg19, %22#16 : tensor<2xf64>
// CHECK-NEXT:        enzyme.yield %24, %25, %26, %27, %28, %29, %38, %39, %40, %41, %42, %30, %55, %56, %57, %58, %59, %18#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %17:3 = call @test.update(%0, %16#6, %16#17, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      return %17#0, %cst_5, %17#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @test.update(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:      %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:      %1 = enzyme.unflatten_slice %arg1[0] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:      %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:      %4 = enzyme.addSampleToTrace(%1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:      %5 = enzyme.unflatten_slice %arg1[1] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:      %6 = call @logpdf(%5, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %7 = arith.addf %3, %6 : tensor<f64>
// CHECK-NEXT:      %8 = enzyme.addSampleToTrace(%5 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:      %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:      %10 = enzyme.addRetvalToTrace(%5 : tensor<f64>) into %9
// CHECK-NEXT:      return %10, %7, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @test.update_0(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:      %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:      %1 = enzyme.unflatten_slice %arg1[0] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:      %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:      %4 = enzyme.addSampleToTrace(%1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:      %5 = enzyme.unflatten_slice %arg1[1] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:      %6 = call @logpdf(%5, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %7 = arith.addf %3, %6 : tensor<f64>
// CHECK-NEXT:      %8 = enzyme.addSampleToTrace(%5 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:      %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:      %10 = enzyme.addRetvalToTrace(%5 : tensor<f64>) into %9
// CHECK-NEXT:      return %10, %7, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @test.update_1(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CHECK-NEXT:      %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:      %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:      %1 = enzyme.unflatten_slice %arg1[0] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:      %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:      %4 = enzyme.addSampleToTrace(%1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:      %5 = enzyme.unflatten_slice %arg1[1] : tensor<2xf64> -> tensor<f64>
// CHECK-NEXT:      %6 = call @logpdf(%5, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %7 = arith.addf %3, %6 : tensor<f64>
// CHECK-NEXT:      %8 = enzyme.addSampleToTrace(%5 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:      %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:      %10 = enzyme.addRetvalToTrace(%5 : tensor<f64>) into %9
// CHECK-NEXT:      return %10, %7, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }