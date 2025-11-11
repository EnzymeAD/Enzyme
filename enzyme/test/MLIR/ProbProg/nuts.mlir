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

// CHECK:  func.func @nuts(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
// CHECK-NEXT:    %cst = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_0 = arith.constant dense<1.000000e+03> : tensor<f64>
// CHECK-NEXT:    %cst_1 = arith.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %cst_2 = arith.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %cst_3 = arith.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %cst_4 = arith.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %cst_5 = arith.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_6 = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_7 = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_8 = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %cst_9 = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK-NEXT:    %cst_10 = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : tensor<2xf64>
// CHECK-NEXT:    %2 = enzyme.getWeightFromTrace %0 : tensor<f64>
// CHECK-NEXT:    %3 = arith.negf %2 : tensor<f64>
// CHECK-NEXT:    %4 = enzyme.cholesky_solve %cst_10, %cst_10 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CHECK-NEXT:    %output_rng_state, %result = enzyme.random %arg0, %cst_8, %cst_7 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:    %5 = enzyme.dot %4, %result {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %6 = enzyme.dot %cst_10, %5 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    %7 = enzyme.dot %5, %6 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:    %8 = arith.mulf %7, %cst_6 : tensor<f64>
// CHECK-NEXT:    %9 = arith.addf %3, %8 : tensor<f64>
// CHECK-NEXT:    %10:2 = enzyme.autodiff_region(%1, %cst_7) {
// CHECK-NEXT:    ^bb0(%arg3: tensor<2xf64>):
// CHECK-NEXT:      %18:3 = func.call @test.update_1(%0, %arg3, %output_rng_state, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:      %19 = arith.negf %18#1 : tensor<f64>
// CHECK-NEXT:      enzyme.yield %19, %18#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:    %11:18 = enzyme.while_loop(%1, %5, %10#1, %1, %5, %10#1, %1, %10#1, %3, %9, %cst_5, %cst_8, %cst_3, %cst_3, %cst_7, %cst_4, %5, %10#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64> condition {
// CHECK-NEXT:    ^bb0(%arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<i64>, %arg14: tensor<f64>, %arg15: tensor<i1>, %arg16: tensor<i1>, %arg17: tensor<f64>, %arg18: tensor<i64>, %arg19: tensor<2xf64>, %arg20: tensor<2xui64>):
// CHECK-NEXT:      %18 = arith.cmpi slt, %arg13, %cst_1 : tensor<i64>
// CHECK-NEXT:      %19 = arith.xori %arg15, %cst_2 : tensor<i1>
// CHECK-NEXT:      %20 = arith.xori %arg16, %cst_2 : tensor<i1>
// CHECK-NEXT:      %21 = arith.andi %18, %19 : tensor<i1>
// CHECK-NEXT:      %22 = arith.andi %21, %20 : tensor<i1>
// CHECK-NEXT:      enzyme.yield %22 : tensor<i1>
// CHECK-NEXT:    } body {
// CHECK-NEXT:    ^bb0(%arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>, %arg11: tensor<f64>, %arg12: tensor<f64>, %arg13: tensor<i64>, %arg14: tensor<f64>, %arg15: tensor<i1>, %arg16: tensor<i1>, %arg17: tensor<f64>, %arg18: tensor<i64>, %arg19: tensor<2xf64>, %arg20: tensor<2xui64>):
// CHECK-NEXT:      %18:2 = enzyme.randomSplit %arg20 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:      %output_rng_state_13, %result_14 = enzyme.random %18#0, %cst_8, %cst_7 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:      %19 = arith.cmpf ogt, %result_14, %cst_6 : tensor<f64>
// CHECK-NEXT:      %20 = arith.addi %arg13, %cst_4 : tensor<i64>
// CHECK-NEXT:      %21:18 = enzyme.while_loop(%arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %18#1 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64> condition {
// CHECK-NEXT:      ^bb0(%arg21: tensor<2xf64>, %arg22: tensor<2xf64>, %arg23: tensor<2xf64>, %arg24: tensor<2xf64>, %arg25: tensor<2xf64>, %arg26: tensor<2xf64>, %arg27: tensor<2xf64>, %arg28: tensor<2xf64>, %arg29: tensor<f64>, %arg30: tensor<f64>, %arg31: tensor<i64>, %arg32: tensor<f64>, %arg33: tensor<i1>, %arg34: tensor<i1>, %arg35: tensor<f64>, %arg36: tensor<i64>, %arg37: tensor<2xf64>, %arg38: tensor<2xui64>):
// CHECK-NEXT:        %50 = arith.cmpi slt, %arg31, %20 : tensor<i64>
// CHECK-NEXT:        %51 = arith.xori %arg33, %cst_2 : tensor<i1>
// CHECK-NEXT:        %52 = arith.xori %arg34, %cst_2 : tensor<i1>
// CHECK-NEXT:        %53 = arith.andi %50, %51 : tensor<i1>
// CHECK-NEXT:        %54 = arith.andi %53, %52 : tensor<i1>
// CHECK-NEXT:        enzyme.yield %54 : tensor<i1>
// CHECK-NEXT:      } body {
// CHECK-NEXT:      ^bb0(%arg21: tensor<2xf64>, %arg22: tensor<2xf64>, %arg23: tensor<2xf64>, %arg24: tensor<2xf64>, %arg25: tensor<2xf64>, %arg26: tensor<2xf64>, %arg27: tensor<2xf64>, %arg28: tensor<2xf64>, %arg29: tensor<f64>, %arg30: tensor<f64>, %arg31: tensor<i64>, %arg32: tensor<f64>, %arg33: tensor<i1>, %arg34: tensor<i1>, %arg35: tensor<f64>, %arg36: tensor<i64>, %arg37: tensor<2xf64>, %arg38: tensor<2xui64>):
// CHECK-NEXT:        %50 = "enzyme.broadcast"(%19) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:        %51 = arith.select %50, %arg24, %arg21 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %52 = arith.select %50, %arg25, %arg22 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %53 = arith.select %50, %arg26, %arg23 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %54:3 = enzyme.randomSplit %arg38 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:        %55 = arith.select %19, %cst_9, %cst : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %56 = "enzyme.broadcast"(%55) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:        %57 = arith.mulf %55, %cst_6 : tensor<f64>
// CHECK-NEXT:        %58 = "enzyme.broadcast"(%57) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
// CHECK-NEXT:        %59 = arith.mulf %58, %53 : tensor<2xf64>
// CHECK-NEXT:        %60 = arith.subf %52, %59 : tensor<2xf64>
// CHECK-NEXT:        %61 = enzyme.dot %cst_10, %60 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %62 = arith.mulf %56, %61 : tensor<2xf64>
// CHECK-NEXT:        %63 = arith.addf %51, %62 : tensor<2xf64>
// CHECK-NEXT:        %64:3 = enzyme.autodiff_region(%63, %cst_7) {
// CHECK-NEXT:        ^bb0(%arg39: tensor<2xf64>):
// CHECK-NEXT:          %106:3 = func.call @test.update_0(%0, %arg39, %54#0, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:          %107 = arith.negf %106#1 : tensor<f64>
// CHECK-NEXT:          enzyme.yield %107, %106#2 : tensor<f64>, tensor<2xui64>
// CHECK-NEXT:        } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<2xf64>)
// CHECK-NEXT:        %65 = arith.mulf %58, %64#2 : tensor<2xf64>
// CHECK-NEXT:        %66 = arith.subf %60, %65 : tensor<2xf64>
// CHECK-NEXT:        %67 = enzyme.dot %cst_10, %66 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %68 = enzyme.dot %66, %67 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %69 = arith.mulf %68, %cst_6 : tensor<f64>
// CHECK-NEXT:        %70 = arith.addf %64#0, %69 : tensor<f64>
// CHECK-NEXT:        %71 = arith.subf %70, %9 : tensor<f64>
// CHECK-NEXT:        %72 = arith.cmpf ogt, %71, %cst_0 : tensor<f64>
// CHECK-NEXT:        %73 = arith.negf %71 : tensor<f64>
// CHECK-NEXT:        %74 = math.exp %73 : tensor<f64>
// CHECK-NEXT:        %75 = arith.minimumf %74, %cst_7 : tensor<f64>
// CHECK-NEXT:        %76 = arith.select %50, %arg21, %63 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %77 = arith.select %50, %arg22, %66 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %78 = arith.select %50, %arg23, %64#2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %79 = arith.select %50, %63, %arg24 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %80 = arith.select %50, %66, %arg25 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %81 = arith.select %50, %64#2, %arg26 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %82 = enzyme.log_add_exp %arg32, %73 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:        %83 = arith.subf %73, %arg32 : tensor<f64>
// CHECK-NEXT:        %84 = arith.negf %83 : tensor<f64>
// CHECK-NEXT:        %85 = math.exp %84 : tensor<f64>
// CHECK-NEXT:        %86 = arith.addf %85, %cst_7 : tensor<f64>
// CHECK-NEXT:        %87 = arith.divf %cst_7, %86 : tensor<f64>
// CHECK-NEXT:        %output_rng_state_17, %result_18 = enzyme.random %54#1, %cst_8, %cst_7 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:        %88 = arith.cmpf olt, %result_18, %87 : tensor<f64>
// CHECK-NEXT:        %89 = "enzyme.broadcast"(%88) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:        %90 = arith.select %89, %63, %arg27 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %91 = arith.select %89, %64#2, %arg28 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:        %92 = arith.select %88, %64#0, %arg29 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %93 = arith.select %88, %70, %arg30 : tensor<i1>, tensor<f64>
// CHECK-NEXT:        %94 = arith.addi %arg31, %cst_4 : tensor<i64>
// CHECK-NEXT:        %95 = arith.ori %arg34, %72 : tensor<i1>
// CHECK-NEXT:        %96 = arith.addf %arg35, %75 : tensor<f64>
// CHECK-NEXT:        %97 = arith.addi %arg36, %cst_4 : tensor<i64>
// CHECK-NEXT:        %98 = arith.addf %arg37, %66 : tensor<2xf64>
// CHECK-NEXT:        %99 = enzyme.dot %cst_10, %77 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %100 = enzyme.dot %cst_10, %80 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:        %101 = enzyme.dot %98, %99 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %102 = enzyme.dot %98, %100 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CHECK-NEXT:        %103 = arith.cmpf olt, %101, %cst_8 : tensor<f64>
// CHECK-NEXT:        %104 = arith.cmpf olt, %102, %cst_8 : tensor<f64>
// CHECK-NEXT:        %105 = arith.ori %103, %104 : tensor<i1>
// CHECK-NEXT:        enzyme.yield %76, %77, %78, %79, %80, %81, %90, %91, %92, %93, %94, %82, %105, %95, %96, %97, %98, %arg38 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %22:2 = enzyme.randomSplit %21#17 : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CHECK-NEXT:      %23 = "enzyme.broadcast"(%19) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:      %24 = arith.select %23, %arg3, %21#0 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %25 = arith.select %23, %arg4, %21#1 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %26 = arith.select %23, %arg5, %21#2 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %27 = arith.select %23, %21#3, %arg6 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %28 = arith.select %23, %21#4, %arg7 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %29 = arith.select %23, %21#5, %arg8 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %30 = enzyme.log_add_exp %arg14, %21#11 : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %31 = arith.subf %21#11, %arg14 : tensor<f64>
// CHECK-NEXT:      %32 = arith.negf %31 : tensor<f64>
// CHECK-NEXT:      %33 = math.exp %32 : tensor<f64>
// CHECK-NEXT:      %34 = arith.addf %33, %cst_7 : tensor<f64>
// CHECK-NEXT:      %35 = arith.divf %cst_7, %34 : tensor<f64>
// CHECK-NEXT:      %36 = arith.ori %21#12, %21#13 : tensor<i1>
// CHECK-NEXT:      %37 = arith.select %36, %cst_8, %35 : tensor<i1>, tensor<f64>
// CHECK-NEXT:      %output_rng_state_15, %result_16 = enzyme.random %22#0, %cst_8, %cst_7 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:      %38 = arith.cmpf olt, %result_16, %37 : tensor<f64>
// CHECK-NEXT:      %39 = "enzyme.broadcast"(%38) <{shape = array<i64: 2>}> : (tensor<i1>) -> tensor<2xi1>
// CHECK-NEXT:      %40 = arith.select %39, %21#6, %arg9 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %41 = arith.select %39, %21#7, %arg10 : tensor<2xi1>, tensor<2xf64>
// CHECK-NEXT:      %42 = arith.select %38, %21#8, %arg11 : tensor<i1>, tensor<f64>
// CHECK-NEXT:      %43 = arith.select %38, %21#9, %arg12 : tensor<i1>, tensor<f64>
// CHECK-NEXT:      %44 = arith.addi %arg13, %cst_4 : tensor<i64>
// CHECK-NEXT:      %45 = arith.ori %arg15, %21#12 : tensor<i1>
// CHECK-NEXT:      %46 = arith.ori %arg16, %21#13 : tensor<i1>
// CHECK-NEXT:      %47 = arith.addf %arg17, %21#14 : tensor<f64>
// CHECK-NEXT:      %48 = arith.addi %arg18, %21#15 : tensor<i64>
// CHECK-NEXT:      %49 = arith.addf %arg19, %21#16 : tensor<2xf64>
// CHECK-NEXT:      enzyme.yield %24, %25, %26, %27, %28, %29, %40, %41, %42, %43, %44, %30, %45, %46, %47, %48, %49, %22#1 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xui64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %12:3 = call @test.update(%0, %11#6, %11#17, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CHECK-NEXT:    %13 = arith.subf %9, %11#9 : tensor<f64>
// CHECK-NEXT:    %14 = math.exp %13 : tensor<f64>
// CHECK-NEXT:    %15 = arith.minimumf %14, %cst_7 : tensor<f64>
// CHECK-NEXT:    %output_rng_state_11, %result_12 = enzyme.random %12#2, %cst_8, %cst_7 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %16 = arith.cmpf olt, %result_12, %15 : tensor<f64>
// CHECK-NEXT:    %17 = enzyme.selectTrace %16, %12#0, %0 : tensor<i1>
// CHECK-NEXT:    return %17, %16, %output_rng_state_11 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CHECK-NEXT:  }
