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
    %r = enzyme.autodiff @bbargs(%x, %dres) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @diffebbargs(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %1 = arith.addf %arg0, %cst : f64
// CHECK-NEXT:    cf.br ^bb1(%1, %c0_i32 : f64, i32)
// CHECK-NEXT:  ^bb1(%2: f64, %3: i32):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    "enzyme.push"(%0, %3) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    %4 = arith.cmpf ult, %2, %cst_0 : f64
// CHECK-NEXT:    cf.cond_br %4, ^bb1(%2, %c-1_i32 : f64, i32), ^bb2
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    %5 = "enzyme.pop"(%0) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %[[CMP:.+]] = arith.cmpi eq, %5, %c0_i32 : i32
// CHECK-NEXT:    %[[GRAD:.+]] = arith.select %[[CMP]], %arg1, %cst_0 : f64
// CHECK-NEXT:    cf.switch %5 : i32, [
// CHECK-NEXT:      default: ^bb2,
// CHECK-NEXT:      0: ^bb3
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    return %[[GRAD]] : f64
// CHECK-NEXT:  }
