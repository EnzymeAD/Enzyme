// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @loop(%x : f64) -> f64 {
    %cst = arith.constant 10.000000e+00 : f64
    %r = affine.for %arg1 = 0 to 10 step 1 iter_args(%arg2 = %cst) -> (f64) {
      %n = arith.addf %arg2, %x : f64
      affine.yield %n : f64
    }
    return %r : f64
  }
  func.func @dloop(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @loop(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r : f64
  }
  // CHECK: @fwddiffeloop
  // CHECK: (%[[arg0:.+]]: f64, %[[arg1:.+]]: f64)
  // CHECK: %[[cst:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: %[[cst_0:.+]] = arith.constant 1.000000e+01 : f64
  // CHECK: %[[r0:.+]]:2 = affine.for %{{.*}} = 0 to 10 iter_args(%[[arg3:.+]] = %[[cst_0]], %[[arg4:.+]] = %[[cst]]) -> (f64, f64) {
  // CHECK:   %[[v1:.+]] = arith.addf %[[arg4]], %[[arg1]] : f64
  // CHECK:   %[[v2:.+]] = arith.addf %[[arg3]], %[[arg0]] : f64
  // CHECK:   affine.yield %[[v2]], %[[v1]] : f64, f64
  // CHECK: }
  // CHECK: return %[[r0]]#1 : f64

  func.func @if_then_else(%x : f64, %c : i1) -> f64 {
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
  func.func @dif_then_else(%x : f64, %dx : f64, %c : i1) -> f64 {
    %r = enzyme.fwddiff @if_then_else(%x, %dx, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, i1) -> (f64)
    return %r : f64
  }
  // CHECK: @fwddiffeif_then_else
  // CHECK: (%[[arg0:.+]]: f64, %[[arg1:.+]]: f64, %[[arg2:.+]]: i1)
  // CHECK: %[[cst:.+]] = arith.constant 2.000000e+00 : f64
  // CHECK: %[[cst_0:.+]] = arith.constant 1.000000e+01 : f64
  // CHECK: %[[r0:.+]]:3 = scf.if %[[arg2]] -> (f64, f64, f64) {
  // CHECK:   %[[v3:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
  // CHECK:   %[[v4:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
  // CHECK:   %[[v5:.+]] = arith.addf %[[v3]], %[[v4]] : f64
  // CHECK:   %[[v6:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
  // CHECK:   scf.yield %[[v6]], %[[v5]], %[[cst]] : f64, f64, f64
  // CHECK: } else {
  // CHECK:   %[[v3:.+]] = arith.addf %[[arg1]], %[[arg1]] : f64
  // CHECK:   %[[v4:.+]] = arith.addf %[[arg0]], %[[arg0]] : f64
  // CHECK:   scf.yield %[[v4]], %[[v3]], %[[cst_0]] : f64, f64, f64
  // CHECK: }
  // CHECK: %[[v1:.+]] = arith.mulf %[[r0]]#1, %[[r0]]#2 : f64
  // CHECK: %[[v2:.+]] = arith.mulf %[[r0]]#0, %[[r0]]#2 : f64
  // CHECK: return %[[v1]] : f64

  func.func @if_then(%x : f64, %c : i1) -> f64 {
    %c2 = arith.constant 2.000000e+00 : f64
    %c10 = arith.constant 10.000000e+00 : f64
    %mem = memref.alloc() : memref<1xf64>
    affine.store %c2, %mem[0] : memref<1xf64>
    scf.if %c {
      %mul = arith.mulf %x, %x : f64
      affine.store %mul, %mem[0] : memref<1xf64>
    }
    %r = affine.load %mem[0] : memref<1xf64>
    %res = arith.mulf %c2, %r : f64
    return %res : f64
  }
  func.func @dif_then(%x : f64, %dx : f64, %c : i1) -> f64 {
    %r = enzyme.fwddiff @if_then(%x, %dx, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, i1) -> (f64)
    return %r : f64
  }
  // CHECK: @fwddiffeif_then
  // CHECK: (%[[arg0:.+]]: f64, %[[arg1:.+]]: f64, %[[arg2:.+]]: i1) -> f64 {
  // CHECK-DAG: %[[cst2:.+]] = arith.constant 2.000000e+00 : f64
  // CHECK-DAG: %[[cst1:.+]] = arith.constant 1.000000e+01 : f64
  // CHECK: %[[alloc:.+]] = memref.alloc() : memref<1xf64>
  // CHECK: %[[alloc_2:.+]] = memref.alloc() : memref<1xf64>
  // CHECK-DAG: %[[cst0:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: affine.store %[[cst0]], %[[alloc]][0] : memref<1xf64>
  // CHECK: affine.store %[[cst2]], %[[alloc_2]][0] : memref<1xf64>
  // CHECK: scf.if %[[arg2]] {
  // CHECK:   %[[v4:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
  // CHECK:   %[[v5:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
  // CHECK:   %[[v6:.+]] = arith.addf %[[v4]], %[[v5]] : f64
  // CHECK:   %[[v7:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
  // CHECK:   affine.store %[[v6]], %[[alloc]][0] : memref<1xf64>
  // CHECK:   affine.store %[[v7]], %[[alloc_2]][0] : memref<1xf64>
  // CHECK: }
  // CHECK: %[[v0:.+]] = affine.load %[[alloc]][0] : memref<1xf64>
  // CHECK: %[[v1:.+]] = affine.load %[[alloc_2]][0] : memref<1xf64>
  // CHECK: %[[v2:.+]] = arith.mulf %[[v0]], %[[cst2]] : f64
  // CHECK: %[[v3:.+]] = arith.mulf %[[cst2]], %[[v1]] : f64
  // CHECK: return %[[v2]] : f64
}
