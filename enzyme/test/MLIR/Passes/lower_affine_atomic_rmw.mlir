// RUN: %eopt %s --lower-affine-atomic-rmw | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @affine2(%arg0: f32, %arg2: memref<?xf32>) {
    affine.parallel (%arg5) = (0) to (4) {
      %5 = enzyme.affine_atomic_rmw addf %arg0, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
      affine.yield
    }
    return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @affine2(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?xf32>) {
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (4) {
// CHECK:             %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]])
// CHECK:             %[[ATOMIC_RMW_0:.*]] = memref.atomic_rmw addf %[[ARG0]], %[[ARG1]]{{\[}}%[[APPLY_0]]] : (f32, memref<?xf32>) -> f32
// CHECK:           }
// CHECK:           return
// CHECK:         }

