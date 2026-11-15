// RUN: %eopt --canonicalize --split-input-file %s | FileCheck %s

// A memref.cast that only makes the memref's sizes dynamic folds into the
// atomics, as it does into loads and stores.

func.func @fold_cast(%m: memref<72xf64, 3>, %v: f64, %i: index) -> f64 {
  %c = memref.cast %m : memref<72xf64, 3> to memref<?xf64, 3>
  %r = enzyme.affine_atomic_rmw addf %v, %c, (affine_map<(d0) -> (d0 + 1)>) [%i] monotonic : (f64, memref<?xf64, 3>) -> f64
  return %r : f64
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:   func.func @fold_cast(
// CHECK-SAME:                         %[[ARG0:[a-z0-9]+]]: memref<72xf64, 3>,
// CHECK-SAME:                         %[[ARG1:[a-z0-9]+]]: f64,
// CHECK-SAME:                         %[[ARG2:[a-z0-9]+]]: index) -> f64 {
// CHECK:           %[[RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[ARG1]], %[[ARG0]], (#[[$ATTR_0]]) {{\[}}%[[ARG2]]] monotonic : (f64, memref<72xf64, 3>) -> f64
// CHECK:           return %[[RMW_0]] : f64
// CHECK:         }

// -----

func.func @fold_cast_plain(%m: memref<72xf64, 3>, %v: f64, %i: index) -> f64 {
  %c = memref.cast %m : memref<72xf64, 3> to memref<?xf64, 3>
  %r = enzyme.atomic_rmw addf %v, %c[%i] monotonic : (f64, memref<?xf64, 3>) -> f64
  return %r : f64
}

// CHECK-LABEL:   func.func @fold_cast_plain(
// CHECK-SAME:                               %[[ARG0:[a-z0-9]+]]: memref<72xf64, 3>,
// CHECK-SAME:                               %[[ARG1:[a-z0-9]+]]: f64,
// CHECK-SAME:                               %[[ARG2:[a-z0-9]+]]: index) -> f64 {
// CHECK:           %[[RMW_0:.*]] = enzyme.atomic_rmw addf %[[ARG1]], %[[ARG0]]{{\[}}%[[ARG2]]] monotonic : (f64, memref<72xf64, 3>) -> f64
// CHECK:           return %[[RMW_0]] : f64
// CHECK:         }
