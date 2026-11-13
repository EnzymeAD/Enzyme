// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// The reverse of a call passes the adjoint of its result to the callee's
// derivative and then clears it. In a loop the same adjoint is reused by every
// iteration of the reverse sweep, so one left uncleared would be passed again,
// added to, by each later iteration's call.
module {
  func.func @h(%v: f64) -> f64 {
    %r = arith.mulf %v, %v : f64
    return %r : f64
  }
  func.func @g(%x: f64) -> f64 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %zero = arith.constant 0.0 : f64
    cf.br ^h(%zero, %c0 : f64, i32)
  ^h(%s: f64, %i: i32):
    %cond = arith.cmpi slt, %i, %c3 : i32
    cf.cond_br %cond, ^body, ^exit
  ^body:
    %y = func.call @h(%x) : (f64) -> f64
    %s2 = arith.addf %s, %y : f64
    %i2 = arith.addi %i, %c1 : i32
    cf.br ^h(%s2, %i2 : f64, i32)
  ^exit:
    return %s : f64
  }
  func.func @dg(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @g(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @diffeg(
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:         %[[V:.+]] = "enzyme.pop"(%{{.+}}) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:    %[[DY:.+]] = "enzyme.get"(%[[GY:.+]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    "enzyme.set"(%[[GY]], %[[ZERO]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %{{.+}} = call @diffeh(%[[V]], %[[DY]]) : (f64, f64) -> f64
