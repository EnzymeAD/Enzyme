// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64, %c : i32) -> f64 {
    %c2 = arith.constant 2.000000e+00 : f64
    %c10 = arith.constant 10.000000e+00 : f64
    %r:2 = scf.execute_region -> (f64, f64) {
      cf.switch %c : i32, [
      default: ^bb5,
      104: ^bb3,
      113: ^bb4(%c10 : f64)
    ]
  ^bb4(%z : f64):  // pred: ^bb2
    %x2 = arith.mulf %x, %x : f64
    scf.yield %x2, %z : f64, f64
  ^bb3:
    %x3 = arith.addf %x, %x : f64
    scf.yield %x3, %c2 : f64, f64
  ^bb5:
    cf.br ^bb4(%x : f64)
    }
    %res = arith.mulf %r#0, %r#1 : f64
    return %res : f64
  }
  func.func @dsq(%x : f64, %dx : f64, %c : i32) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, i32) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffesquare(%[[x:.+]]: f64, %[[dx:.+]]: f64, %[[c:.+]]: i32) -> f64 {
// CHECK-DAG:     %[[cst2:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:     %[[cst10:.+]] = arith.constant 1.000000e+01 : f64
// CHECK-DAG:     %[[cst0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %[[r0:.+]]:4 = scf.execute_region -> (f64, f64, f64, f64) {
// CHECK-NEXT:     %[[cst02:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:       cf.switch %[[c]] : i32, [
// CHECK-NEXT:         default: ^bb3,
// CHECK-NEXT:         104: ^bb2,
// CHECK-NEXT:         113: ^bb1(%[[cst10]], %[[cst02]] : f64, f64)
// CHECK-NEXT:       ]
// CHECK-NEXT:     ^bb1(%[[a3:.+]]: f64, %[[da3:.+]]: f64):  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:       %[[a4:.+]] = arith.mulf %[[dx]], %[[x]] : f64
// CHECK-NEXT:       %[[a5:.+]] = arith.mulf %[[dx]], %[[x]] : f64
// CHECK-NEXT:       %[[a6:.+]] = arith.addf %[[a4]], %[[a5]] : f64
// CHECK-NEXT:       %[[a7:.+]] = arith.mulf %[[x]], %[[x]] : f64
// CHECK-NEXT:       scf.yield %[[a7]], %[[a6]], %[[a3]], %[[da3]] : f64, f64, f64, f64
// CHECK-NEXT:     ^bb2:  // pred: ^bb0
// CHECK-NEXT:       %[[b8:.+]] = arith.addf %[[dx]], %[[dx]] : f64
// CHECK-NEXT:       %[[b9:.+]] = arith.addf %[[x]], %[[x]] : f64
// CHECK-NEXT:       scf.yield %[[b9]], %[[b8]], %[[cst2]], %[[cst0]] : f64, f64, f64, f64
// CHECK-NEXT:     ^bb3:  // pred: ^bb0
// CHECK-NEXT:       cf.br ^bb1(%[[x]], %[[dx]] : f64, f64)
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[r1:.+]] = arith.mulf %[[r0]]#1, %[[r0]]#2 : f64
// CHECK-NEXT:     %[[r2:.+]] = arith.mulf %[[r0]]#3, %[[r0]]#0 : f64
// CHECK-NEXT:     %[[r3:.+]] = arith.addf %[[r1]], %[[r2]] : f64
// CHECK-NEXT:     %[[r4:.+]] = arith.mulf %[[r0]]#0, %[[r0]]#2 : f64
// CHECK-NEXT:     return %[[r3]] : f64
// CHECK-NEXT:   }
