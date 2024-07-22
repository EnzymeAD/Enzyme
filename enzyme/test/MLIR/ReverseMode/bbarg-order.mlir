// RUN: %eopt --enzyme -canonicalize --remove-unnecessary-enzyme-ops -canonicalize %s | FileCheck %s

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
    %r = enzyme.autodiff @bbargs(%x, %dres) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @diffebbargs(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %1 = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %2 = arith.addf %arg0, %cst : f64
// CHECK-NEXT:    "enzyme.push"(%1, %c0_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb1(%2 : f64)
// CHECK-NEXT:  ^bb1(%3: f64):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    %4 = arith.cmpf ult, %3, %cst_0 : f64
// CHECK-NEXT:    "enzyme.push"(%1, %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    "enzyme.push"(%0, %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.cond_br %4, ^bb1(%3 : f64), ^bb2
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %5 = arith.addf %arg1, %cst_0 : f64
// CHECK-NEXT:    %6 = "enzyme.pop"(%0) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %7 = arith.cmpi eq, %6, %c-1_i32 : i32
// CHECK-NEXT:    %8 = arith.select %7, %5, %cst_0 : f64
// CHECK-NEXT:    %9 = arith.addf %8, %cst_0 : f64
// CHECK-NEXT:    cf.br ^bb3
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:    %10 = "enzyme.pop"(%1) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %11 = arith.cmpi eq, %10, %c-1_i32 : i32
// CHECK-NEXT:    %12 = arith.select %11, %9, %cst_0 : f64
// CHECK-NEXT:    %13 = arith.addf %12, %cst_0 : f64
// CHECK-NEXT:    cf.switch %10 : i32, [
// CHECK-NEXT:      default: ^bb3,
// CHECK-NEXT:      0: ^bb4
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %14 = arith.addf %13, %cst_0 : f64
// CHECK-NEXT:    return %14 : f64
// CHECK-NEXT:  }
