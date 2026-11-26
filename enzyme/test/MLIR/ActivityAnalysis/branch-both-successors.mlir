// RUN: %eopt --print-activity-analysis="dataflow=false annotate=true funcs=f" %s 2>&1 | FileCheck %s

// A value a branch passes along two edges flows into both block arguments.

module {
  func.func private @f(%x: f64, %n: i32) -> f64 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant 1.0 : f64
    cf.br ^outer(%one, %c0 : f64, i32)
  ^outer(%s: f64, %j: i32):
    %go = arith.cmpi slt, %j, %n : i32
    cf.cond_br %go, ^body(%s : f64), ^done(%s : f64)
  ^body(%t: f64):
    %s2 = arith.mulf %t, %x : f64
    %j2 = arith.addi %j, %c1 : i32
    cf.br ^outer(%s2, %j2 : f64, i32)
  ^done(%r: f64):
    return %r : f64
  }
}

// CHECK: func.func private @f({{.*}}enzyme.arg_icv0 = false
// CHECK: arith.mulf {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
