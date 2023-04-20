// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @infinite(%x : f64, %y : f64) -> f64 {
    cf.br ^bb1(%x: f64)

  ^bb1(%r: f64):
    %c1 = arith.constant 1.0 : f64
    %sum = arith.addf %r, %c1 : f64
    %c = arith.cmpf ult, %x, %y : f64
    cf.cond_br %c, ^bb1(%r : f64), ^bb2

   ^bb2:
    return %sum : f64
  }
  func.func @dsq(%x : f64, %dx : f64, %y : f64, %dy : f64) -> f64 {
    %r = enzyme.fwddiff @infinite(%x, %dx, %y, %dy) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>] } : (f64, f64, f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffeinfinite
// CHECK-SAME: (%[[ARG0:.+]]: f64, %[[ARG1:.+]]: f64, %[[ARG2:.+]]: f64, %[[ARG3:.+]]: f64) -> f64 {
// CHECK:    cf.br ^[[BB1:.+]](%[[ARG0]], %[[ARG1]] : f64, f64)
// CHECK:  ^[[BB1]](%[[V0:.+]]: f64, %[[V1:.+]]: f64):
// CHECK:    cf.cond_br %{{.*}}, ^[[BB1]](%[[V0]], %[[V1]] : f64, f64), ^[[BB2:.+]]
// CHECK:  ^[[BB2]]:
// CHECK:    return %[[V1]]
