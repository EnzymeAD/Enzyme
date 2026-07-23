// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,flatten-enzyme-caches,cse,canonicalize)" | FileCheck %s

func.func private @flatten(%b0: index, %b1: index, %b2: index, %x: memref<?xf32>) {
  affine.parallel (%iv, %jv, %kv) = (0, 0, 0) to (%b0, %b1, %b2) {
    %load = affine.load %x[%iv * symbol(%b1) * symbol(%b2) + %jv * symbol(%b2) + %kv] : memref<?xf32>
    %mulf = arith.mulf %load, %load : f32
    affine.store %mulf, %x[%iv * symbol(%b1) * symbol(%b2) + %jv * symbol(%b2) + %kv] : memref<?xf32>
  }
  return
}

func.func @dflatten(%b0: index, %b1: index, %b2: index, %x: memref<?xf32>, %dx: memref<?xf32>) {
  enzyme.autodiff @flatten(%b0, %b1, %b2, %x, %dx) {
    activity = [
      #enzyme<activity enzyme_const>,
      #enzyme<activity enzyme_const>,
      #enzyme<activity enzyme_const>,
      #enzyme<activity enzyme_dup>
    ],
    ret_activity = []
  } : (index, index, index, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2)[s0, s1] -> (d1 * s1 + d2 + (d0 * s0) * s1)
// CHECK-LABEL:   func.func private @diffeflatten(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: memref<?xf32>,
// CHECK-SAME:      %[[ARG4:.*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG0]], %[[ARG1]] : index
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[MULI_0]], %[[ARG2]] : index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[MULI_1]]) : memref<?xf32>
// CHECK:           affine.parallel (%[[VAL_0:.*]], %[[VAL_1:.*]], %[[VAL_2:.*]]) = (0, 0, 0) to (symbol(%[[ARG0]]), symbol(%[[ARG1]]), symbol(%[[ARG2]])) {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG3]]{{\[}}%[[VAL_1]] * symbol(%[[ARG2]]) + %[[VAL_2]] + (%[[VAL_0]] * symbol(%[[ARG1]])) * symbol(%[[ARG2]])] : memref<?xf32>
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[VAL_0]], %[[ARG1]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[MULI_2]], %[[VAL_1]] : index
// CHECK:             %[[MULI_3:.*]] = arith.muli %[[ADDI_0]], %[[ARG2]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[MULI_3]], %[[VAL_2]] : index
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[ADDI_1]]] : memref<?xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[LOAD_0]] : f32
// CHECK:             affine.store %[[MULF_0]], %[[ARG3]]{{\[}}%[[VAL_1]] * symbol(%[[ARG2]]) + %[[VAL_2]] + (%[[VAL_0]] * symbol(%[[ARG1]])) * symbol(%[[ARG2]])] : memref<?xf32>
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) = (0, 0, 0) to (symbol(%[[ARG0]]), symbol(%[[ARG1]]), symbol(%[[ARG2]])) {
// CHECK:             %[[MULI_4:.*]] = arith.muli %[[VAL_3]], %[[ARG1]] : index
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[MULI_4]], %[[VAL_4]] : index
// CHECK:             %[[MULI_5:.*]] = arith.muli %[[ADDI_2]], %[[ARG2]] : index
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[MULI_5]], %[[VAL_5]] : index
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[ADDI_3]]] : memref<?xf32>
// CHECK:             %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]){{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG4]]{{\[}}%[[APPLY_0]]] : memref<?xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[LOAD_2]], %[[CONSTANT_0]] : f32
// CHECK:             memref.store %[[CONSTANT_0]], %[[ARG4]]{{\[}}%[[APPLY_0]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[ADDF_0]], %[[LOAD_1]] : f32
// CHECK:             %[[ADDF_1:.*]] = arith.addf %[[MULF_1]], %[[CONSTANT_0]] : f32
// CHECK:             %[[ADDF_2:.*]] = arith.addf %[[ADDF_1]], %[[MULF_1]] : f32
// CHECK:             %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[ADDF_2]], %[[ARG4]]{{\[}}%[[APPLY_0]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?xf32>
// CHECK:           return
// CHECK:         }
