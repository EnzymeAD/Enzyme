// RUN: %eopt -remove-unnecessary-enzyme-ops %s | FileCheck %s

// This pop cannot be removed even though we know the first popped value with be -1
// the other pops will be conditional

module {
  func.func private @diffebbargs(%arg0: f64) {
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %3 = "enzyme.init"() : () -> !enzyme.Cache<i32>
    "enzyme.push"(%3, %c0_i32) : (!enzyme.Cache<i32>, i32) -> ()
    cf.br ^bb1(%arg0 : f64)
  ^bb1(%7: f64):  // 2 preds: ^bb0, ^bb1
    %8 = arith.cmpf ult, %7, %cst : f64
    "enzyme.push"(%3, %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
    cf.cond_br %8, ^bb1(%7 : f64), ^bb4
  ^bb4:  // 2 preds: ^bb3, ^bb4
    %18 = "enzyme.pop"(%3) : (!enzyme.Cache<i32>) -> i32
    cf.switch %18 : i32, [
      default: ^bb4,
      0: ^bb5
    ]
  ^bb5:  // pred: ^bb4
    return
  }
}

// CHECK:  func.func private @diffebbargs(%arg0: f64) {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    "enzyme.push"(%0, %c0_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb1(%arg0 : f64)
// CHECK-NEXT:  ^bb1(%1: f64):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    %2 = arith.cmpf ult, %1, %cst : f64
// CHECK-NEXT:    "enzyme.push"(%0, %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.cond_br %2, ^bb1(%1 : f64), ^bb2
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %3 = "enzyme.pop"(%0) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    cf.switch %3 : i32, [
// CHECK-NEXT:      default: ^bb2,
// CHECK-NEXT:      0: ^bb3
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    return
// CHECK-NEXT:  }