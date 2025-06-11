// RUN: %eopt %s --split-input-file --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops)" | FileCheck %s

func.func @nested_if(%arg0: f64, %arg1: i1, %arg2: i1) -> f64 {
  %cst = arith.constant 4.000000e+00 : f64
  %0 = scf.if %arg1 -> (f64) {
    %1 = arith.mulf %arg0, %arg0 : f64
    scf.yield %1 : f64
  } else {
    %1 = scf.if %arg2 -> (f64) {
      %3 = math.cos %arg0 : f64
      scf.yield %3 : f64
    } else {
      scf.yield %cst : f64
    }
    %2 = arith.addf %1, %arg0 : f64
    scf.yield %2 : f64
  }
  return %0 : f64
}

func.func @dnested_if(%arg0: f64, %arg1: i1, %arg2: i1, %dres: f64) -> f64 {
  %res = enzyme.autodiff @nested_if(%arg0, %arg1, %arg2, %dres)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (f64, i1, i1, f64) -> f64
  return %res : f64
}
// CHECK: func.func private @diffenested_if(%arg0: f64, %arg1: i1, %arg2: i1, %arg3: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %1 = "enzyme.init"() : () -> !enzyme.Cache<i1>
// CHECK-NEXT:    %2 = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%2, %cst) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %3 = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %4 = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    scf.if %arg1 {
// CHECK-NEXT:      "enzyme.push"(%4, %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:      "enzyme.push"(%3, %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:    } else {
// CHECK-NEXT:      "enzyme.push"(%1, %arg2) : (!enzyme.Cache<i1>, i1) -> ()
// CHECK-NEXT:      scf.if %arg2 {
// CHECK-NEXT:        "enzyme.push"(%0, %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %5 = arith.addf %arg3, %cst : f64
// CHECK-NEXT:    scf.if %arg1 {
// CHECK-NEXT:      %7 = arith.addf %5, %cst : f64
// CHECK-NEXT:      %8 = "enzyme.pop"(%4) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %9 = "enzyme.pop"(%3) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %10 = arith.mulf %7, %9 : f64
// CHECK-NEXT:      %11 = "enzyme.get"(%2) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:      %12 = arith.addf %11, %10 : f64
// CHECK-NEXT:      "enzyme.set"(%2, %12) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %13 = arith.mulf %7, %8 : f64
// CHECK-NEXT:      %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:      "enzyme.set"(%2, %14) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %7 = arith.addf %5, %cst : f64
// CHECK-NEXT:      %8 = arith.addf %7, %cst : f64
// CHECK-NEXT:      %9 = "enzyme.get"(%2) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:      %10 = arith.addf %9, %7 : f64
// CHECK-NEXT:      "enzyme.set"(%2, %10) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %11 = "enzyme.pop"(%1) : (!enzyme.Cache<i1>) -> i1
// CHECK-NEXT:      scf.if %11 {
// CHECK-NEXT:        %12 = arith.addf %8, %cst : f64
// CHECK-NEXT:        %13 = "enzyme.pop"(%0) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:        %14 = math.sin %13 : f64
// CHECK-NEXT:        %15 = arith.negf %14 : f64
// CHECK-NEXT:        %16 = arith.mulf %12, %15 : f64
// CHECK-NEXT:        %17 = "enzyme.get"(%2) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:        %18 = arith.addf %17, %16 : f64
// CHECK-NEXT:        "enzyme.set"(%2, %18) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %6 = "enzyme.get"(%2) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    return %6 : f64
// CHECK-NEXT:  }

// -----


func.func @some_res_inactive(%arg0: f64, %arg1: i1) -> (f64, i64) {
  %0:2 = scf.if %arg1 -> (f64, i64) {
    %1 = arith.mulf %arg0, %arg0 : f64
    %2 = arith.fptosi %1 : f64 to i64
    scf.yield %1, %2 : f64, i64
  } else {
    %1 = math.cos %arg0 : f64
    %2 = arith.addf %1, %arg0 : f64
    %3 = arith.fptosi %2 : f64 to i64
    scf.yield %2, %3 : f64, i64
  }
  return %0#0, %0#1 : f64, i64
}

func.func @dsome_res_inactive(%x: f64, %cond: i1, %dres: f64) -> f64 {
  %true = arith.constant true
  %res = enzyme.autodiff @some_res_inactive(%x, %cond, %dres)
    {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_constnoneed>]
    } : (f64, i1, f64) -> (f64)
  return %res : f64
}
// CHECK: func.func private @diffesome_res_inactive(%arg0: f64, %arg1: i1, %arg2: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %1 = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%1, %cst) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %2 = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %3 = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    scf.if %arg1 {
// CHECK-NEXT:      "enzyme.push"(%3, %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:      "enzyme.push"(%2, %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:    } else {
// CHECK-NEXT:      "enzyme.push"(%0, %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = arith.addf %arg2, %cst : f64
// CHECK-NEXT:    scf.if %arg1 {
// CHECK-NEXT:      %6 = arith.addf %4, %cst : f64
// CHECK-NEXT:      %7 = "enzyme.pop"(%3) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %8 = "enzyme.pop"(%2) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %9 = arith.mulf %6, %8 : f64
// CHECK-NEXT:      %10 = "enzyme.get"(%1) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:      %11 = arith.addf %10, %9 : f64
// CHECK-NEXT:      "enzyme.set"(%1, %11) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %12 = arith.mulf %6, %7 : f64
// CHECK-NEXT:      %13 = arith.addf %11, %12 : f64
// CHECK-NEXT:      "enzyme.set"(%1, %13) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %6 = arith.addf %4, %cst : f64
// CHECK-NEXT:      %7 = arith.addf %6, %cst : f64
// CHECK-NEXT:      %8 = "enzyme.get"(%1) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:      %9 = arith.addf %8, %6 : f64
// CHECK-NEXT:      "enzyme.set"(%1, %9) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %10 = "enzyme.pop"(%0) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %11 = math.sin %10 : f64
// CHECK-NEXT:      %12 = arith.negf %11 : f64
// CHECK-NEXT:      %13 = arith.mulf %7, %12 : f64
// CHECK-NEXT:      %14 = arith.addf %9, %13 : f64
// CHECK-NEXT:      "enzyme.set"(%1, %14) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    %5 = "enzyme.get"(%1) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    return %5 : f64
// CHECK-NEXT:  }
