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
  func.func @dsq(%x : f64, %dx : tensor<2xf64>, %c : i1) -> tensor<2xf64> {
    %r = enzyme.fwddiff @square(%x, %dx, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_dupnoneed>], width=2 } : (f64, tensor<2xf64>, i1) -> (tensor<2xf64>)
    return %r : tensor<2xf64>
  }
}

// CHECK:  func.func private @fwddiffe2square(%[[arg0:.+]]: f64, %[[arg1:.+]]: tensor<2xf64>, %[[arg2:.+]]: i1) -> tensor<2xf64> {
// CHECK-DAG:    %[[cst2:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:    %[[cst10:.+]] = arith.constant 1.000000e+01 : f64
// CHECK-NEXT:    %[[r0:.+]]:3 = scf.if %[[arg2]] -> (f64, tensor<2xf64>, f64) {
// CHECK-NEXT:      %[[t4:.+]] = "enzyme.broadcast"(%[[arg0]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:      %[[t5:.+]] = arith.mulf %[[arg1]], %[[t4]] : tensor<2xf64>
// CHECK-NEXT:      %[[t6:.+]] = "enzyme.broadcast"(%[[arg0]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:      %[[t7:.+]] = arith.mulf %[[arg1]], %[[t6]] : tensor<2xf64>
// CHECK-NEXT:      %[[t8:.+]] = arith.addf %[[t5]], %[[t7]] : tensor<2xf64>
// CHECK-NEXT:      %[[t9:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:      scf.yield %[[t9]], %[[t8]], %[[cst2]] : f64, tensor<2xf64>, f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[e4:.+]] = arith.addf %[[arg1]], %[[arg1]] : tensor<2xf64>
// CHECK-NEXT:      %[[e5:.+]] = arith.addf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:      scf.yield %[[e5]], %[[e4]], %[[cst10]] : f64, tensor<2xf64>, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[r1:.+]] = "enzyme.broadcast"(%[[r0]]#2) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %[[r2:.+]] = arith.mulf %[[r0]]#1, %[[r1]] : tensor<2xf64>
// CHECK-NEXT:    %[[r3:.+]] = arith.mulf %[[r0]]#0, %[[r0]]#2 : f64
// CHECK-NEXT:    return %[[r2]] : tensor<2xf64>
// CHECK-NEXT:  }
