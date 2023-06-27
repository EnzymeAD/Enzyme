// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @ppow(%x: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 10 : index
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%r_it = %cst) -> f64 {
      %r_next = arith.mulf %r_it, %x : f64
      scf.yield %r_next : f64
    }
    return %r : f64
  }

  func.func @dppow(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @ppow(%x, %dr) { activity=[#enzyme<activity enzyme_out>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:    func.func private @diffeppow(%[[x:.+]]: f64, %[[dr:.+]]: f64) -> f64

// Make sure the right values are being cached in the primal
// CHECK:        %[[one:.+]] = arith.constant 1.0
// CHECK:        scf.for %[[iv:.+]] = %c0 to %c10 step %c1 iter_args(%[[r_it:.+]] = %[[one]])
// CHECK-NEXT:       "enzyme.push"(%[[xcache:.+]], %[[x]])
// CHECK-NEXT:       "enzyme.push"(%[[rcache:.+]], %[[r_it]])

// Ensure the right value is yielded in the adjoint
// CHECK:        "enzyme.set"(%[[rshadow:.+]], %[[dr]])
// CHECK:        %[[dr:.+]] = "enzyme.get"(%[[rshadow]])
// CHECK:        scf.for %[[iv:.+]] = %[[lb:.+]] to %[[ub:.+]] step %[[step:.+]] iter_args(%[[dr_it:.+]] = %[[dr]])
// CHECK-NEXT:       "enzyme.set"(%[[rshadow:.+]], %[[dr_it]])
// CHECK-NEXT:       %[[dr_it:.+]] = "enzyme.get"(%[[rshadow]])
// CHECK-NEXT:       %[[x:.+]] = "enzyme.pop"(%[[xcache]])
// CHECK-NEXT:       %[[dr_next:.+]] = arith.mulf %[[dr_it]], %[[x]]
// CHECK-NEXT:       "enzyme.set"(%[[rshadow:.+]], %[[dr_next]])
// CHECK-NEXT:       %[[r_cached:.+]] = "enzyme.pop"(%[[rcache]])
// CHECK-NEXT:       %[[dx_next:.+]] = arith.mulf %[[dr_it]], %[[r_cached]]
// CHECK-NEXT:       %[[dx0:.+]] = "enzyme.get"(%[[xshadow:.+]]) :
// CHECK-NEXT:       %[[dx1:.+]] = arith.addf %[[dx0]], %[[dx_next]]
// CHECK-NEXT:       "enzyme.set"(%[[xshadow]], %[[dx1]])
// CHECK-NEXT:       %[[dr_next:.+]] = "enzyme.get"(%[[rshadow]])
// CHECK-NEXT:       scf.yield %[[dr_next]]
// CHECK:        %[[final:.+]] = "enzyme.get"(%[[xshadow]])
// CHECK-NEXT:   return %[[final]]
