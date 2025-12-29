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

    %res:3 = enzyme.mcmc @test(%rng, %mean, %stddev) given %init_trace
      inverse_mass_matrix = %inverse_mass_matrix : tensor<2x2xf64>
      step_size = %step_size : tensor<f64>
      { nuts_config = #enzyme.nuts_config<max_tree_depth = 10>, name = "nuts", selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]] } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>)
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
// CHECK-NEXT:      %6:2 = enzyme.autodiff_region(%3, %cst_10) {
// CHECK-NEXT:      ^bb0(%arg3: tensor<2xf64>):
// CHECK-NEXT:        %17:3 = func.call @test.update_1(%0, %arg3, %2#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:        %18 = arith.negf %17#1 : tensor<f64>
// CHECK-NEXT:        enzyme.yield %18, %17#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %7:3 = enzyme.randomSplit %6#0 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:      %8 = enzyme.randomSplit %7#1 : (tensor<2xui64>) -> tensor<2xui64>
// CHECK-NEXT:      %output_rng_state, %result = enzyme.random %8, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:      %9 = enzyme.cholesky_solve %cst_13, %cst_13 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:      %10 = enzyme.dot %9, %result {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %11 = enzyme.dot %cst_13, %10 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:      %12 = enzyme.dot %10, %11 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:      %13 = arith.mulf %12, %cst_9 : tensor<f64>
// CHECK-NEXT:      %14 = arith.addf %5, %13 : tensor<f64>
// CHECK-NEXT:      %15:18 = enzyme.while_loop(%3, %10, %6#1, %3, %10, %6#1, %3, %6#1, %5, %14, %cst_8, %cst_11, %cst_7, %cst_7, %cst_11, %cst_8, %10, %7#2 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64> condition {
// CHECK-NEXT:      ^bb0(%arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<i64>, %arg14: tensor<f64>, %arg15: tensor<i1>, %arg16: tensor<i1>, %arg17: tensor<f64>, %arg18: tensor<i64>, %arg19: tensor<2xf64>, %arg20: tensor<2xui64>):
// CHECK-NEXT:        %17 = arith.cmpi slt, %arg13, %cst_4 : tensor<i64>
// CHECK-NEXT:        %18 = arith.xori %arg15, %cst_5 : tensor<i1>
// CHECK-NEXT:        %19 = arith.xori %arg16, %cst_5 : tensor<i1>
// CHECK-NEXT:        %20 = arith.andi %17, %18 : tensor<i1>
// CHECK-NEXT:        %21 = arith.andi %20, %19 : tensor<i1>
// CHECK-NEXT:        enzyme.yield %21 : tensor<i1>
// CHECK-NEXT:      } body {
// CHECK-NEXT:      ^bb0(%arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<i64>, %arg14: tensor<f64>, %arg15: tensor<i1>, %arg16: tensor<i1>, %arg17: tensor<f64>, %arg18: tensor<i64>, %arg19: tensor<2xf64>, %arg20: tensor<2xui64>):
// CHECK-NEXT:        %17:3 = enzyme.randomSplit %arg20 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:        %output_rng_state_14, %result_15 = enzyme.random %17#1, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:        %18 = arith.cmpf olt, %result_15, %cst_9 : tensor<f64>
// CHECK-NEXT:        %19:2 = enzyme.randomSplit %17#2 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:        %20 = arith.shli %cst_2, %arg13 : tensor<i64>
// CHECK-NEXT:        %21:21 = enzyme.while_loop(%arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst_8, %arg19, %19#0, %cst_3, %cst_3, %cst_8 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<i64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<i64> condition {
// CHECK-NEXT:        ^bb0(%arg21: tensor<2xf64>, %arg22: tensor<2xf64>, %arg23: tensor<2xf64>, %arg24: tensor<2xf64>, %arg25: tensor<2xf64>, %arg26: tensor<2xf64>, %arg27: tensor<2xf64>, %arg28: tensor<2xf64>, %arg29: tensor<f64>, %arg30: tensor<f64>, %arg31: tensor<i64>, %arg32: tensor<f64>, %arg33: tensor<i1>, %arg34: tensor<i1>, %arg35: tensor<f64>, %arg36: tensor<i64>, %arg37: tensor<2xf64>, %arg38: tensor<2xui64>, %arg39: tensor<10x2xf64>, %arg40: tensor<10x2xf64>, %arg41: tensor<i64>):
// CHECK-NEXT:          %59 = arith.cmpi slt, %arg36, %20 : tensor<i64>
// CHECK-NEXT:          %60 = arith.xori %arg33, %cst_5 : tensor<i1>
// CHECK-NEXT:          %61 = arith.xori %arg34, %cst_5 : tensor<i1>
// CHECK-NEXT:          %62 = arith.andi %59, %60 : tensor<i1>
// CHECK-NEXT:          %63 = arith.andi %62, %61 : tensor<i1>
// CHECK-NEXT:          enzyme.yield %63 : tensor<i1>
// CHECK-NEXT:        } body {
// CHECK-NEXT:        ^bb0(%arg21: tensor<2xf64>, %arg22: tensor<2xf64>, %arg23: tensor<2xf64>, %arg24: tensor<2xf64>, %arg25: tensor<2xf64>, %arg26: tensor<2xf64>, %arg27: tensor<2xf64>, %arg28: tensor<2xf64>, %arg29: tensor<f64>, %arg30: tensor<f64>, %arg31: tensor<i64>, %arg32: tensor<f64>, %arg33: tensor<i1>, %arg34: tensor<i1>, %arg35: tensor<f64>, %arg36: tensor<i64>, %arg37: tensor<2xf64>, %arg38: tensor<2xui64>, %arg39: tensor<10x2xf64>, %arg40: tensor<10x2xf64>, %arg41: tensor<i64>):
// CHECK-NEXT:          %59 = "enzyme.broadcast"(%18) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %60 = arith.select %59, %arg24, %arg21 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %61 = arith.select %59, %arg25, %arg22 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %62 = arith.select %59, %arg26, %arg23 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %63:2 = enzyme.randomSplit %arg38 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:          %64 = arith.select %18, %cst_12, %cst_1 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %65 = "enzyme.broadcast"(%64) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:          %66 = arith.mulf %64, %cst_9 : tensor<f64>
// CHECK-NEXT:          %67 = "enzyme.broadcast"(%66) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:          %68 = arith.mulf %67, %62 : tensor<2xf64>
// CHECK-NEXT:          %69 = arith.subf %61, %68 : tensor<2xf64>
// CHECK-NEXT:          %70 = enzyme.dot %cst_13, %69 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:          %71 = arith.mulf %65, %70 : tensor<2xf64>
// CHECK-NEXT:          %72 = arith.addf %60, %71 : tensor<2xf64>
// CHECK-NEXT:          %73:3 = enzyme.autodiff_region(%72, %cst_10) {
// CHECK-NEXT:          ^bb0(%arg42: tensor<2xf64>):
// CHECK-NEXT:            %149:3 = func.call @test.update_0(%0, %arg42, %63#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:            %150 = arith.negf %149#1 : tensor<f64>
// CHECK-NEXT:            enzyme.yield %150, %149#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:          } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:          %74 = arith.mulf %67, %73#2 : tensor<2xf64>
// CHECK-NEXT:          %75 = arith.subf %69, %74 : tensor<2xf64>
// CHECK-NEXT:          %76 = enzyme.dot %cst_13, %75 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:          %77 = enzyme.dot %75, %76 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:          %78 = arith.mulf %77, %cst_9 : tensor<f64>
// CHECK-NEXT:          %79 = arith.addf %73#0, %78 : tensor<f64>
// CHECK-NEXT:          %80 = arith.subf %79, %14 : tensor<f64>
// CHECK-NEXT:          %81 = arith.cmpf une, %80, %80 : tensor<f64>
// CHECK-NEXT:          %82 = arith.select %81, %cst_0, %80 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %83 = arith.negf %82 : tensor<f64>
// CHECK-NEXT:          %84 = arith.cmpf ogt, %82, %cst_6 : tensor<f64>
// CHECK-NEXT:          %85 = arith.negf %82 : tensor<f64>
// CHECK-NEXT:          %86 = math.exp %85 : tensor<f64>
// CHECK-NEXT:          %87 = arith.minimumf %86, %cst_10 : tensor<f64>
// CHECK-NEXT:          %88 = "enzyme.broadcast"(%18) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %89 = arith.select %88, %arg21, %72 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %90 = arith.select %88, %arg22, %75 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %91 = arith.select %88, %arg23, %73#2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %92 = arith.select %88, %72, %arg24 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %93 = arith.select %88, %75, %arg25 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %94 = arith.select %88, %73#2, %arg26 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %95 = enzyme.log_add_exp %arg32, %83 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:          %96 = arith.subf %83, %arg32 : tensor<f64>
// CHECK-NEXT:          %97 = arith.negf %96 : tensor<f64>
// CHECK-NEXT:          %98 = math.exp %97 : tensor<f64>
// CHECK-NEXT:          %99 = arith.addf %98, %cst_10 : tensor<f64>
// CHECK-NEXT:          %100 = arith.divf %cst_10, %99 : tensor<f64>
// CHECK-NEXT:          %output_rng_state_18, %result_19 = enzyme.random %63#1, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:          %101 = arith.cmpf olt, %result_19, %100 : tensor<f64>
// CHECK-NEXT:          %102 = "enzyme.broadcast"(%101) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %103 = arith.select %102, %72, %arg27 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %104 = arith.select %102, %73#2, %arg28 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %105 = arith.select %101, %73#0, %arg29 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %106 = arith.select %101, %79, %arg30 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %107 = arith.addi %arg31, %cst_2 : tensor<i64>
// CHECK-NEXT:          %108 = arith.ori %arg34, %84 : tensor<i1>
// CHECK-NEXT:          %109 = arith.addf %arg35, %87 : tensor<f64>
// CHECK-NEXT:          %110 = arith.addi %arg36, %cst_2 : tensor<i64>
// CHECK-NEXT:          %111 = arith.addf %arg37, %75 : tensor<2xf64>
// CHECK-NEXT:          %112 = arith.cmpi eq, %arg36, %cst_8 : tensor<i64>
// CHECK-NEXT:          %113 = "enzyme.broadcast"(%112) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:          %114 = arith.select %113, %72, %89 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %115 = arith.select %113, %75, %90 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %116 = arith.select %113, %73#2, %91 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %117 = arith.select %113, %72, %92 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %118 = arith.select %113, %75, %93 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %119 = arith.select %113, %73#2, %94 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %120 = arith.select %113, %72, %103 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %121 = arith.select %113, %73#2, %104 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %122 = arith.select %113, %75, %111 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:          %123 = arith.select %112, %73#0, %105 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %124 = arith.select %112, %79, %106 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %125 = arith.select %112, %cst_8, %107 : tensor<i1>, tensor<i64>
// CHECK-NEXT:          %126 = arith.select %112, %83, %95 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %127 = arith.select %112, %84, %108 : tensor<i1>, tensor<i1>
// CHECK-NEXT:          %128 = arith.select %112, %87, %109 : tensor<i1>, tensor<f64>
// CHECK-NEXT:          %129 = arith.select %112, %cst_2, %110 : tensor<i1>, tensor<i64>
// CHECK-NEXT:          %130 = arith.shrui %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          %131 = enzyme.popcount %130 : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:          %132 = arith.addi %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          %133 = arith.xori %arg41, %cst : tensor<i64>
// CHECK-NEXT:          %134 = arith.andi %133, %132 : tensor<i64>
// CHECK-NEXT:          %135 = arith.subi %134, %cst_2 : tensor<i64>
// CHECK-NEXT:          %136 = enzyme.popcount %135 : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:          %137 = arith.subi %131, %136 : tensor<i64>
// CHECK-NEXT:          %138 = arith.addi %137, %cst_2 : tensor<i64>
// CHECK-NEXT:          %139 = arith.andi %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          %140 = arith.cmpi eq, %139, %cst_8 : tensor<i64>
// CHECK-NEXT:          %141 = enzyme.dynamic_update %arg39[%131] = %75 : (tensor<10x2xf64>, tensor<i64>, tensor<2xf64>) -> tensor<10x2xf64>
// CHECK-NEXT:          %142 = enzyme.dynamic_update %arg40[%131] = %122 : (tensor<10x2xf64>, tensor<i64>, tensor<2xf64>) -> tensor<10x2xf64>
// CHECK-NEXT:          %143 = "enzyme.broadcast"(%140) <{shape = array<i64: 10, 2>}> : (tensor<i1>) -> tensor<10x2xi1>
// CHECK-NEXT:          %144 = arith.select %143, %141, %arg39 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:          %145 = arith.select %143, %142, %arg40 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:          %146:2 = enzyme.while_loop(%131, %cst_7 : tensor<i64>, tensor<i1>) -> tensor<i64>, tensor<i1> condition {
// CHECK-NEXT:          ^bb0(%arg42: tensor<i64>, %arg43: tensor<i1>):
// CHECK-NEXT:            %149 = arith.cmpi sge, %arg42, %138 : tensor<i64>
// CHECK-NEXT:            %150 = arith.xori %arg43, %cst_5 : tensor<i1>
// CHECK-NEXT:            %151 = arith.andi %149, %150 : tensor<i1>
// CHECK-NEXT:            enzyme.yield %151 : tensor<i1>
// CHECK-NEXT:          } body {
// CHECK-NEXT:          ^bb0(%arg42: tensor<i64>, %arg43: tensor<i1>):
// CHECK-NEXT:            %149 = enzyme.dynamic_extract %144[%arg42] : (tensor<10x2xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:            %150 = enzyme.dynamic_extract %145[%arg42] : (tensor<10x2xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:            %151 = arith.subf %122, %150 : tensor<2xf64>
// CHECK-NEXT:            %152 = arith.addf %151, %149 : tensor<2xf64>
// CHECK-NEXT:            %153 = enzyme.dot %cst_13, %149 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:            %154 = enzyme.dot %cst_13, %75 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:            %155 = "enzyme.broadcast"(%cst_9) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:            %156 = arith.addf %149, %75 : tensor<2xf64>
// CHECK-NEXT:            %157 = arith.mulf %155, %156 : tensor<2xf64>
// CHECK-NEXT:            %158 = arith.subf %152, %157 : tensor<2xf64>
// CHECK-NEXT:            %159 = enzyme.dot %153, %158 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:            %160 = enzyme.dot %154, %158 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:            %161 = arith.cmpf ole, %159, %cst_11 : tensor<f64>
// CHECK-NEXT:            %162 = arith.cmpf ole, %160, %cst_11 : tensor<f64>
// CHECK-NEXT:            %163 = arith.ori %161, %162 : tensor<i1>
// CHECK-NEXT:            %164 = arith.subi %arg42, %cst_2 : tensor<i64>
// CHECK-NEXT:            enzyme.yield %164, %163 : tensor<i64>, tensor<i1>
// CHECK-NEXT:          }
// CHECK-NEXT:          %147 = arith.select %112, %cst_7, %146#1 : tensor<i1>, tensor<i1>
// CHECK-NEXT:          %148 = arith.addi %arg41, %cst_2 : tensor<i64>
// CHECK-NEXT:          enzyme.yield %114, %115, %116, %117, %118, %119, %120, %121, %123, %124, %125, %126, %147, %127, %128, %129, %122, %63#0, %144, %145, %148 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<i64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %22 = "enzyme.broadcast"(%18) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:        %23 = arith.select %22, %arg3, %21#0 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %24 = arith.select %22, %arg4, %21#1 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %25 = arith.select %22, %arg5, %21#2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %26 = arith.select %22, %21#3, %arg6 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %27 = arith.select %22, %21#4, %arg7 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %28 = arith.select %22, %21#5, %arg8 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %29 = enzyme.log_add_exp %arg14, %21#11 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:        %30 = arith.subf %21#11, %arg14 : tensor<f64>
// CHECK-NEXT:        %31 = math.exp %30 : tensor<f64>
// CHECK-NEXT:        %32 = arith.minimumf %31, %cst_10 : tensor<f64>
// CHECK-NEXT:        %33 = arith.ori %21#12, %21#13 : tensor<i1>
// CHECK-NEXT:        %34 = arith.select %33, %cst_11, %32 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %output_rng_state_16, %result_17 = enzyme.random %19#1, %cst_11, %cst_10 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:        %35 = arith.cmpf olt, %result_17, %34 : tensor<f64>
// CHECK-NEXT:        %36 = "enzyme.broadcast"(%35) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:        %37 = arith.select %36, %21#6, %arg9 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %38 = arith.select %36, %21#7, %arg10 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %39 = arith.select %35, %21#8, %arg11 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %40 = arith.select %35, %21#9, %arg12 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %41 = arith.addi %arg13, %cst_2 : tensor<i64>
// CHECK-NEXT:        %42 = arith.addf %arg19, %21#16 : tensor<2xf64>
// CHECK-NEXT:        %43 = enzyme.dot %cst_13, %24 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %44 = enzyme.dot %cst_13, %27 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %45 = "enzyme.broadcast"(%cst_9) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:        %46 = arith.addf %24, %27 : tensor<2xf64>
// CHECK-NEXT:        %47 = arith.mulf %45, %46 : tensor<2xf64>
// CHECK-NEXT:        %48 = arith.subf %42, %47 : tensor<2xf64>
// CHECK-NEXT:        %49 = enzyme.dot %43, %48 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %50 = enzyme.dot %44, %48 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %51 = arith.cmpf ole, %49, %cst_11 : tensor<f64>
// CHECK-NEXT:        %52 = arith.cmpf ole, %50, %cst_11 : tensor<f64>
// CHECK-NEXT:        %53 = arith.ori %51, %52 : tensor<i1>
// CHECK-NEXT:        %54 = arith.ori %21#12, %53 : tensor<i1>
// CHECK-NEXT:        %55 = arith.ori %arg16, %21#13 : tensor<i1>
// CHECK-NEXT:        %56 = arith.addf %arg17, %21#14 : tensor<f64>
// CHECK-NEXT:        %57 = arith.addi %arg18, %21#15 : tensor<i64>
// CHECK-NEXT:        %58 = arith.addf %arg19, %21#16 : tensor<2xf64>
// CHECK-NEXT:        enzyme.yield %23, %24, %25, %26, %27, %28, %37, %38, %39, %40, %41, %29, %54, %55, %56, %57, %58, %17#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %16:3 = call @test.update(%0, %15#6, %15#17, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      return %16#0, %cst_5, %16#2 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CHECK-NEXT:    }
