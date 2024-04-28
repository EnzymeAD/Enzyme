// RUN: %eopt --enzyme %s | FileCheck %s
// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s --check-prefix=REM
// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s --check-prefix=FIN

module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}


// CHECK:  func.func @dsquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %0 = call @diffesquare(%arg0, %arg1) : (f64, f64) -> f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// CHECK:  func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[dx:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    %[[c0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    "enzyme.set"(%[[dx]], %[[c0]]) : (!enzyme.Gradient<f64>, f64) -> ()

// CHECK-NEXT:    %[[lhscache:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:    %[[rhscache:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f64>

// CHECK-NEXT:    %[[dy:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:    %[[c1:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    "enzyme.set"(%[[dy]], %[[c1]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    "enzyme.push"(%[[rhscache]], %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:    "enzyme.push"(%[[lhscache]], %arg0) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:    %[[mul:.+]] = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:    cf.br ^bb1

// CHECK:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %[[prevdret0:.+]] = "enzyme.get"(%[[dy]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %[[postdret0:.+]] = arith.addf %[[prevdret0]], %arg1 : f64
// CHECK-NEXT:    "enzyme.set"(%[[dy]], %[[postdret0]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %[[prevdret:.+]] = "enzyme.get"(%[[dy]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %[[c2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    "enzyme.set"(%[[dy]], %[[c2]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %[[postlhs:.+]] = "enzyme.pop"(%[[rhscache]]) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:    %[[postrhs:.+]] = "enzyme.pop"(%[[lhscache]]) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:    %[[dlhs:.+]] = arith.mulf %[[prevdret]], %[[postrhs]] : f64
// CHECK-NEXT:    %[[prevdx1:.+]] = "enzyme.get"(%[[dx]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %[[postdx1:.+]] = arith.addf %[[prevdx1]], %[[dlhs]] : f64
// CHECK-NEXT:    "enzyme.set"(%[[dx]], %[[postdx1]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %[[drhs:.+]] = arith.mulf %[[prevdret]], %[[postlhs]] : f64
// CHECK-NEXT:    %[[prevdx2:.+]] = "enzyme.get"(%[[dx]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    %[[postdx2:.+]] = arith.addf %[[prevdx2]], %[[drhs]] : f64
// CHECK-NEXT:    "enzyme.set"(%[[dx]], %[[postdx2]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:    %[[res:.+]] = "enzyme.get"(%[[dx]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }


// REM:  func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64 {
// REM-NEXT:    %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// REM-NEXT:    %[[a1:.+]] = arith.addf %arg1, %[[cst]] : f64
// REM-NEXT:    %[[a2:.+]] = arith.mulf %[[a1]], %arg0 : f64
// REM-NEXT:    %[[a3:.+]] = arith.addf %[[a2]], %[[cst]] : f64
// REM-NEXT:    %[[a4:.+]] = arith.mulf %[[a1]], %arg0 : f64
// REM-NEXT:    %[[a5:.+]] = arith.addf %[[a3]], %[[a4]] : f64
// REM-NEXT:    return %[[a5]] : f64
// REM-NEXT:  }

// FIN:  func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64 {
// FIN-NEXT:    %0 = arith.mulf %arg1, %arg0 : f64
// FIN-NEXT:    %1 = arith.addf %0, %0 : f64
// FIN-NEXT:    return %1 : f64
// FIN-NEXT:  }