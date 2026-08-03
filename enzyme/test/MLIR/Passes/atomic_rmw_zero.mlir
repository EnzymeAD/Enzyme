// RUN: %eopt --split-input-file --canonicalize %s | FileCheck %s

// Accumulating a zero leaves memory alone, so the read-modify-write is only a
// read -- and if nobody wanted the read either, it is nothing at all.

func.func @unused(%m: memref<4xf64>, %i: index) {
  %z = arith.constant 0.000000e+00 : f64
  %0 = enzyme.atomic_rmw addf %z, %m[%i] monotonic fastmath<fast> : (f64, memref<4xf64>) -> f64
  return
}

// CHECK-LABEL: func.func @unused
// CHECK-NEXT:    return

// -----

func.func @used(%m: memref<4xf64>, %i: index) -> f64 {
  %z = arith.constant -0.000000e+00 : f64
  %0 = enzyme.atomic_rmw addf %z, %m[%i] monotonic fastmath<nsz> : (f64, memref<4xf64>) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @used
// CHECK-NEXT:    %[[v:.+]] = memref.load
// CHECK-NEXT:    return %[[v]]

// -----

func.func @affine_used(%m: memref<4xf64>, %i: index) -> f64 {
  %z = arith.constant 0.000000e+00 : f64
  %0 = enzyme.affine_atomic_rmw addf %z, %m, (affine_map<(d0) -> (d0)>)[%i] fastmath<fast> : (f64, memref<4xf64>) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @affine_used
// CHECK-NEXT:    %[[v:.+]] = affine.load
// CHECK-NEXT:    return %[[v]]

// -----

// An integer accumulation of zero needs no flags to be nothing.

func.func @integer(%m: memref<4xi64>, %i: index) {
  %z = arith.constant 0 : i64
  %0 = enzyme.atomic_rmw addi %z, %m[%i] monotonic : (i64, memref<4xi64>) -> i64
  return
}

// CHECK-LABEL: func.func @integer
// CHECK-NEXT:    return

// -----

// Signed zeros matter without nsz: adding +0.0 to a stored -0.0 changes it.

func.func @no_nsz(%m: memref<4xf64>, %i: index) {
  %z = arith.constant 0.000000e+00 : f64
  %0 = enzyme.atomic_rmw addf %z, %m[%i] monotonic : (f64, memref<4xf64>) -> f64
  return
}

// CHECK-LABEL: func.func @no_nsz
// CHECK:         enzyme.atomic_rmw

// -----

// A plain load cannot stand in for a read an observer could order against.

func.func @ordered(%m: memref<4xf64>, %i: index) -> f64 {
  %z = arith.constant 0.000000e+00 : f64
  %0 = enzyme.atomic_rmw addf %z, %m[%i] seq_cst fastmath<fast> : (f64, memref<4xf64>) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @ordered
// CHECK:         enzyme.atomic_rmw

// -----

func.func @nonzero(%m: memref<4xf64>, %i: index, %v: f64) {
  %0 = enzyme.atomic_rmw addf %v, %m[%i] monotonic fastmath<fast> : (f64, memref<4xf64>) -> f64
  return
}

// CHECK-LABEL: func.func @nonzero
// CHECK:         enzyme.atomic_rmw
