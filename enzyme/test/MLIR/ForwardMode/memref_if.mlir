// RUN: %eopt --enzyme %s | FileCheck %s

// A memref that is unconditionally initialized with a constant and then
// conditionally overwritten with an active value. In forward mode the shadow
// memref must be zero-initialized before the conditional store, otherwise the
// load on the not-taken path reads uninitialized shadow memory and the returned
// tangent is garbage instead of 0.
//
// This is the memref analogue of the affine.store `@if_then` test.
//
// XFAIL: *
// FIXME: forward-mode differentiation of memref.store does not currently emit
// the shadow zero-initialization (`memref.store %cst0, %[[alloc]]`) before the
// conditional store, so the not-taken path reads an uninitialized shadow. When
// this is fixed the test will XPASS; remove the XFAIL line above.

module {
  func.func @if_then(%x : f64, %c : i1) -> f64 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2.000000e+00 : f64
    %mem = memref.alloc() : memref<1xf64>
    memref.store %c2, %mem[%c0] : memref<1xf64>
    scf.if %c {
      %mul = arith.mulf %x, %x : f64
      memref.store %mul, %mem[%c0] : memref<1xf64>
    }
    %r = memref.load %mem[%c0] : memref<1xf64>
    %res = arith.mulf %c2, %r : f64
    return %res : f64
  }
  func.func @dif_then(%x : f64, %dx : f64, %c : i1) -> f64 {
    %r = enzyme.fwddiff @if_then(%x, %dx, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, i1) -> (f64)
    return %r : f64
  }
}

// CHECK: @fwddiffeif_then
// CHECK: (%[[arg0:.+]]: f64, %[[arg1:.+]]: f64, %[[arg2:.+]]: i1) -> f64 {
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[cst2:.+]] = arith.constant 2.000000e+00 : f64
// CHECK: %[[alloc:.+]] = memref.alloc() : memref<1xf64>
// CHECK: %[[alloc_0:.+]] = memref.alloc() : memref<1xf64>
// CHECK-DAG: %[[cst0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK: memref.store %[[cst0]], %[[alloc]][%[[c0]]] : memref<1xf64>
// CHECK: memref.store %[[cst2]], %[[alloc_0]][%[[c0]]] : memref<1xf64>
// CHECK: scf.if %[[arg2]] {
// CHECK:   %[[v4:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK:   %[[v5:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK:   %[[v6:.+]] = arith.addf %[[v4]], %[[v5]] : f64
// CHECK:   %[[v7:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK:   memref.store %[[v6]], %[[alloc]][%[[c0]]] : memref<1xf64>
// CHECK:   memref.store %[[v7]], %[[alloc_0]][%[[c0]]] : memref<1xf64>
// CHECK: }
// CHECK: %[[v0:.+]] = memref.load %[[alloc]][%[[c0]]] : memref<1xf64>
// CHECK: %[[v1:.+]] = memref.load %[[alloc_0]][%[[c0]]] : memref<1xf64>
// CHECK: %[[v2:.+]] = arith.mulf %[[v0]], %[[cst2]] : f64
// CHECK: return %[[v2]] : f64
