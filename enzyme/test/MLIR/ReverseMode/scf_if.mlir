// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops)" | FileCheck %s

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
