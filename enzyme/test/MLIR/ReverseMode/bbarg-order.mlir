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

// CHECK:  func.func private @diffebbargs(%[[X:.*]]: f64, %[[DRES:.*]]: f64) -> f64 {
// CHECK-NEXT:    %[[CM1:.*]] = arith.constant -1 : i32
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[BRCACHE:.*]] = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %[[PREDCACHE:.*]] = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK-NEXT:    %[[ADD:.*]] = arith.addf %[[X]], %[[ONE]] : f64
// CHECK-NEXT:    "enzyme.push"(%[[PREDCACHE]], %[[C0]]) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb1(%[[ADD]] : f64)
// CHECK-NEXT:  ^bb1(%[[ARG:.*]]: f64):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    %[[FLAG:.*]] = arith.cmpf ult, %[[ARG]], %[[ZERO]] : f64
// CHECK-NEXT:    "enzyme.push"(%[[PREDCACHE]], %[[CM1]]) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    "enzyme.push"(%[[BRCACHE]], %[[CM1]]) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.cond_br %[[FLAG]], ^bb1(%[[ARG]] : f64), ^bb2
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %[[BRIDX:.*]] = "enzyme.pop"(%[[BRCACHE]]) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %[[ISRET:.*]] = arith.cmpi eq, %[[BRIDX]], %[[CM1]] : i32
// CHECK-NEXT:    %[[DRET:.*]] = arith.select %[[ISRET]], %[[DRES]], %[[ZERO]] : f64
// CHECK-NEXT:    cf.br ^bb3
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:    %[[PREDIDX:.*]] = "enzyme.pop"(%[[PREDCACHE]]) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %[[ISENTRY:.*]] = arith.cmpi eq, %[[PREDIDX]], %[[C0]] : i32
// CHECK-NEXT:    %[[DX:.*]] = arith.select %[[ISENTRY]], %[[DRET]], %[[ZERO]] : f64
// CHECK-NEXT:    cf.switch %[[PREDIDX]] : i32, [
// CHECK-NEXT:      default: ^bb3,
// CHECK-NEXT:      0: ^bb4
// CHECK-NEXT:    ]
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    return %[[DX]] : f64
// CHECK-NEXT:  }
