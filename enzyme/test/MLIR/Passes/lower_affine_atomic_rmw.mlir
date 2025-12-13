// RUN: %eopt %s --lower-affine-atomic-rmw | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @affine2(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<4xf32>
    affine.parallel (%arg5) = (0) to (4) {
      %1 = affine.load %arg1[%arg5] : memref<?xf32>
      affine.store %1, %alloc[%arg5] : memref<4xf32>
      %2 = arith.mulf %1, %arg0 : f32
      affine.store %2, %arg3[%arg5] : memref<?xf32>
    }
    %0 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
      %1 = affine.load %alloc[%arg5] : memref<4xf32>
      %2 = affine.load %arg4[%arg5] : memref<?xf32>
      affine.store %cst, %arg4[%arg5] : memref<?xf32>
      %3 = arith.mulf %2, %arg0 : f32
      %4 = arith.mulf %2, %1 : f32
      %5 = enzyme.affine_atomic_rmw addf %3, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
      affine.yield %4 : f32
    }
    memref.dealloc %alloc : memref<4xf32>
    return
  }
}

// CHECK-LABEL:   func.func @affine2(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:[^ :]+]]: memref<?xf32>,
// CHECK-SAME:      %[[ARG2:[^ :]+]]: memref<?xf32>,
// CHECK-SAME:      %[[ARG3:[^ :]+]]: memref<?xf32>,
// CHECK-SAME:      %[[ARG4:[^ :]+]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (4) {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<4xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:             affine.store %[[MULF_0]], %[[ARG3]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:           }
// CHECK:           %[[PARALLEL_0:.*]] = affine.parallel (%[[VAL_1:.*]]) = (0) to (4) reduce ("addf") -> (f32) {
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<4xf32>
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG4]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             memref.store %[[CONSTANT_2]], %[[ARG4]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[ARG0]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:             %[[ATOMIC_RMW_0:.*]] = memref.atomic_rmw addf %[[MULF_1]], %[[ARG2]]{{\[}}%[[VAL_1]]] : (f32, memref<?xf32>) -> f32
// CHECK:             affine.yield %[[MULF_2]] : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<4xf32>
// CHECK:           return
// CHECK:         }

