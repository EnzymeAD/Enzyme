// RUN: %eopt --enzyme -canonicalize --remove-unnecessary-enzyme-ops -enzyme-simplify-math -canonicalize %s | FileCheck %s

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
    %r = enzyme.autodiff @ppow(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @diffeppow(%[[x:.+]]: f64, %[[dr:.+]]: f64) -> f64 {
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.0
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[xshadow:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%[[xshadow]], %[[zero]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %[[itshadow:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%[[itshadow]], %[[zero]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %[[xcache:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %[[rcache:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %[[rshadow:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    "enzyme.set"(%[[rshadow]], %[[zero]]) : (!enzyme.Gradient<f64>, f64) -> ()

// CHECK-NEXT:    %{{.+}} = scf.for %[[iv:.+]] = %c0 to %c10 step %c1 iter_args(%[[r_it:.+]] = %[[one]]) -> (f64) {
// CHECK-NEXT:      "enzyme.push"(%[[rcache]], %[[r_it]]) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:      "enzyme.push"(%[[xcache]], %[[x]]) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:      %[[fwd:.+]] = arith.mulf %[[r_it]], %[[x]] : f64
// CHECK-NEXT:      scf.yield %[[fwd]] : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    "enzyme.set"(%[[rshadow]], %[[dr]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    scf.for %[[div:.+]] = %c0 to %c10 step %c1 {
// CHECK-NEXT:      %[[dr_it:.+]] = "enzyme.get"(%[[rshadow]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:      "enzyme.set"(%[[rshadow]], %[[zero]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %[[r_cached:.+]] = "enzyme.pop"(%[[rcache]]) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %[[x_cached:.+]] = "enzyme.pop"(%[[xcache]]) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:      %[[dr_next:.+]] = arith.mulf %[[dr_it]], %[[x_cached]]
// CHECK-NEXT:      %[[previts:.+]] = "enzyme.get"(%[[itshadow]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:      %[[postits:.+]] = arith.addf %[[previts]], %[[dr_next]] : f64
// CHECK-NEXT:      "enzyme.set"(%[[itshadow]], %[[postits]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %[[dx_next:.+]] = arith.mulf %[[dr_it]], %[[r_cached]] : f64
// CHECK-NEXT:      %[[dx0:.+]] = "enzyme.get"(%[[xshadow]]) :
// CHECK-NEXT:      %[[dx1:.+]] = arith.addf %[[dx0]], %[[dx_next]]
// CHECK-NEXT:      "enzyme.set"(%[[xshadow]], %[[dx1]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %[[divp1:.+]] = arith.addi %[[div]], %c1 : index
// CHECK-NEXT:      %[[last:.+]] = arith.cmpi sge, %[[divp1]], %c10 : index
// CHECK-NEXT:      "enzyme.set"(%[[itshadow]], %[[zero]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:      %[[sel:.+]] = arith.select %[[last]], %[[zero]], %12 : f64
// CHECK-NEXT:      "enzyme.set"(%[[itshadow]], %[[sel]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[final:.+]] = "enzyme.get"(%[[xshadow]])
// CHECK-NEXT:    return %[[final]]