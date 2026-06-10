// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @while(%x : f64) -> f64 {
    %cst = arith.constant 10.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    %r:2 = scf.while (%arg1 = %c0, %arg2 = %cst) : (index, f64) -> (index, f64) {
      %1 = arith.cmpi slt, %arg1, %c10 : index
      scf.condition(%1) %arg1, %arg2 : index, f64
    } do {
    ^bb0(%arg1: index, %arg2: f64):
      %1 = arith.addi %arg1, %c1 : index
      %2 = arith.addf %arg2, %x : f64
      scf.yield %1, %2 : index, f64
    }
    return %r#1 : f64
  }
  func.func @dwhile(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @while(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r : f64
  }
  // CHECK: @fwddiffewhile
  // CHECK: (%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
  // CHECK:   %[[cst:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK:   %[[cst_0:.+]] = arith.constant 1.000000e+01 : f64
  // CHECK:   %[[c0:.+]] = arith.constant 0 : index
  // CHECK:   %[[c1:.+]] = arith.constant 1 : index
  // CHECK:   %[[c10:.+]] = arith.constant 10 : index
  // CHECK:   %[[r0:.+]]:3 = scf.while (%[[arg2:.+]] = %[[c0]], %[[arg3:.+]] = %[[cst_0]], %[[arg4:.+]] = %[[cst]]) : (index, f64, f64) -> (index, f64, f64) {
  // CHECK:     %[[v1:.+]] = arith.cmpi slt, %[[arg2]], %[[c10]] : index
  // CHECK:     scf.condition(%[[v1]]) %[[arg2]], %[[arg3]], %[[arg4]] : index, f64, f64
  // CHECK:   } do {
  // CHECK:   ^bb0(%[[arg2:.+]]: index, %[[arg3:.+]]: f64, %[[arg4:.+]]: f64):
  // CHECK:     %[[v1:.+]] = arith.addi %[[arg2]], %[[c1]] : index
  // CHECK:     %[[v2:.+]] = arith.addf %[[arg4]], %[[arg1]] : f64
  // CHECK:     %[[v3:.+]] = arith.addf %[[arg3]], %[[arg0]] : f64
  // CHECK:     scf.yield %[[v1]], %[[v3]], %[[v2]] : index, f64, f64
  // CHECK:   }
  // CHECK:   return %[[r0]]#2 : f64
  // CHECK: }
}
