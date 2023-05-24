// RUN: %eopt --enzyme %s | FileCheck %s

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

// CHECK:    func.func @diff(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[i0:.+]] = call @diffebbargs(%[[arg0]], %[[arg1]]) : (f64, f64) -> f64
// CHECK-NEXT:    return %[[i0:.+]]
// CHECK:    func.func private @diffebbargs(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {

// There should be exactly one block with two f64 args, and their values should be accumulated
// in the shadow.
// CHECK:    ^[[BBMULTI:.+]](%[[fst:.+]]: f64, %[[snd:.+]]: f64):
// CHECK-NEXT:    "enzyme.set"(%[[shadow:.+]], %[[fst]])
// CHECK-NEXT:    %[[before:.+]] = "enzyme.get"(%[[shadow]])
// CHECK-NEXT:    %[[after:.+]] = arith.addf %[[snd]], %[[before]]
// CHECK-NEXT:    "enzyme.set"(%[[shadow]], %[[after]])
