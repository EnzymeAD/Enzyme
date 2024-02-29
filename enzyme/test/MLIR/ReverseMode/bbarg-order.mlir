// RUN: %eopt --enzyme --remove-unnecessary-enzyme-ops -canonicalize %s | FileCheck %s

module {
  func.func @bbargs(%x: f64) -> f64 {
    %cst1 = arith.constant 1.000000e+00 : f64
    %2 = arith.addf %x, %cst1 : f64
    cf.br ^bb1(%2 : f64)
  ^bb1(%1 : f64):
    %cst = arith.constant 0.000000e+00 : f64
    %flag = arith.cmpf ult, %1, %cst : f64
    cf.cond_br %flag, ^bb1(%1 : f64), ^bb2(%1 : f64)
  ^bb2(%ret : f64):
    return %ret : f64
  }

  func.func @diff(%x: f64, %dres: f64) -> f64 {
    %r = enzyme.autodiff @bbargs(%x, %dres) { activity=[#enzyme<activity enzyme_out>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @diffebbargs(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %1 = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%1, %cst_0) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %2 = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %3 = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%3, %cst_0) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %4 = arith.addf %arg0, %cst : f64
// CHECK-NEXT:    "enzyme.push"(%2, %c0_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb1(%4 : f64)
// CHECK-NEXT:  ^bb1(%5: f64):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    %6 = arith.cmpf ult, %5, %cst_0 : f64
// CHECK-NEXT:    "enzyme.push"(%2, %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    "enzyme.push"(%0, %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.cond_br %6, ^bb1(%5 : f64), ^bb2
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %7 = arith.addf %arg1, %cst_0 : f64
// CHECK-NEXT:    %8 = "enzyme.pop"(%0) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %9 = arith.cmpi eq, %8, %c-1_i32 : i32
// CHECK-NEXT:    %10 = arith.select %9, %7, %cst_0 : f64
// CHECK-NEXT:    %11 = "enzyme.get"(%1) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %12 = arith.addf %11, %10 : f64
// CHECK-NEXT:    "enzyme.set"(%1, %12) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    cf.br ^bb3
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:    %13 = "enzyme.pop"(%2) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %14 = "enzyme.get"(%1) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    "enzyme.set"(%1, %cst_0) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %15 = arith.cmpi eq, %13, %c-1_i32 : i32
// CHECK-NEXT:    %16 = arith.select %15, %14, %cst_0 : f64
// CHECK-NEXT:    %17 = "enzyme.get"(%1) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %18 = arith.addf %17, %16 : f64
// CHECK-NEXT:    "enzyme.set"(%1, %18) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %19 = arith.cmpi eq, %13, %c-1_i32 : i32
// CHECK-NEXT:    %20 = arith.select %19, %14, %cst_0 : f64
// CHECK-NEXT:    %21 = "enzyme.get"(%3) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %22 = arith.addf %21, %20 : f64
// CHECK-NEXT:    "enzyme.set"(%3, %22) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    cf.switch %13 : i32, [
// CHECK-NEXT:      default: ^bb3,
// CHECK-NEXT:      0: ^bb4
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %23 = "enzyme.get"(%3) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    "enzyme.set"(%3, %cst_0) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %24 = arith.addf %23, %cst_0 : f64
// CHECK-NEXT:    return %24 : f64
// CHECK-NEXT:  }
