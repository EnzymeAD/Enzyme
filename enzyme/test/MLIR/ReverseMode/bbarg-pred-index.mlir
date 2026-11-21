// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize %s | FileCheck %s

// The adjoint of a block argument flows back along the edge control came in
// on, so each edge must be selected by the branch index its predecessor
// recorded, not by the argument's position. Here both edges set argument 0:
// d/dx is 2 through ^a and 3 through ^b.
module {
  func.func @f(%x: f64, %c: i1) -> f64 {
    cf.cond_br %c, ^a, ^b
  ^a:
    %two = arith.constant 2.0 : f64
    %xa = arith.mulf %x, %two : f64
    cf.br ^join(%xa : f64)
  ^b:
    %three = arith.constant 3.0 : f64
    %xb = arith.mulf %x, %three : f64
    cf.br ^join(%xb : f64)
  ^join(%v: f64):
    return %v : f64
  }
  func.func @df(%x: f64, %c: i1, %dr: f64) -> f64 {
    %r = enzyme.autodiff @f(%x, %c, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, i1, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:      %[[X:.+]]: f64, %[[C:.+]]: i1, %[[DR:.+]]: f64) -> f64 {
// CHECK-DAG:     %[[THREE:.+]] = arith.constant 3.000000e+00 : f64
// CHECK-DAG:     %[[TWO:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:     %[[IDX_A:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[IDX_B:.+]] = arith.constant -1 : i32
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:         %[[FLAGS:.+]] = "enzyme.init"() : () -> !enzyme.Cache<i32>
// CHECK:         cf.cond_br %[[C]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK:         "enzyme.push"(%[[FLAGS]], %[[IDX_A]])
// CHECK:       ^bb2:
// CHECK:         "enzyme.push"(%[[FLAGS]], %[[IDX_B]])
// CHECK:       ^bb3:
// CHECK:         %[[FLAG:.+]] = "enzyme.pop"(%[[FLAGS]])
// CHECK:         %[[IS_B:.+]] = arith.cmpi eq, %[[FLAG]], %[[IDX_B]] : i32
// CHECK:         %[[D_B:.+]] = arith.select %[[IS_B]], %[[DR]], %[[ZERO]] : f64
// CHECK:         %[[IS_A:.+]] = arith.cmpi eq, %[[FLAG]], %[[IDX_A]] : i32
// CHECK:         %[[D_A:.+]] = arith.select %[[IS_A]], %[[DR]], %[[ZERO]] : f64
// CHECK:         cf.switch %[[FLAG]] : i32, [
// CHECK-NEXT:      default: ^bb4,
// CHECK-NEXT:      0: ^bb5
// CHECK-NEXT:    ]
// CHECK:       ^bb4:
// CHECK:         arith.mulf %[[D_B]], %{{.+}}
// CHECK:       ^bb5:
// CHECK:         arith.mulf %[[D_A]], %{{.+}}
