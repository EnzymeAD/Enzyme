// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64, %c : i1) -> f64 {
    %c2 = arith.constant 2.000000e+00 : f64
    %c10 = arith.constant 10.000000e+00 : f64
    %r:2 = scf.if %c -> (f64, f64) {
       %mul = arith.mulf %x, %x : f64
       scf.yield %mul, %c2 : f64, f64
    } else {
       %add = arith.addf %x, %x : f64
       scf.yield %add, %c10 : f64, f64
    }
    %res = arith.mulf %r#0, %r#1 : f64
    return %res : f64
  }
  func.func @dsq(%x : f64, %dx : f64, %c : i1) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, i1) -> (f64)
    return %r : f64
  }
}


// CHECK:  func.func private @fwddiffesquare(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64, %[[arg2:.+]]: i1) -> f64 {
// CHECK-DAG:    %[[cst2:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:    %[[cst10:.+]] = arith.constant 1.000000e+01 : f64
// CHECK-NEXT:    %[[r0:.+]]:3 = scf.if %[[arg2]] -> (f64, f64, f64) {
// CHECK-NEXT:      %[[t3:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:      %[[t4:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:      %[[t5:.+]] = arith.addf %[[t3]], %[[t4]] : f64
// CHECK-NEXT:      %[[t6:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:      scf.yield %[[t6]], %[[t5]], %[[cst2]] : f64, f64, f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[e3:.+]] = arith.addf %arg1, %arg1 : f64
// CHECK-NEXT:      %[[e4:.+]] = arith.addf %arg0, %arg0 : f64
// CHECK-NEXT:      scf.yield %[[e4]], %[[e3]], %[[cst10]] : f64, f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[r1:.+]] = arith.mulf %[[r0]]#1, %[[r0]]#2 : f64
// CHECK-NEXT:    %[[r2:.+]] = arith.mulf %[[r0]]#0, %[[r0]]#2 : f64
// CHECK-NEXT:    return %[[r1]] : f64
// CHECK-NEXT:  }
