// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" --split-input-file | FileCheck %s

func.func private @some_res_inactive(%x: f32, %ub: index) -> (f32) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index

  %sum_0 = arith.constant 1.0 : f32
  %sum, %unused = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0, %inactive = %sum_0) -> (f32, f32) {
    %sum_next = arith.mulf %sum_iter, %x : f32
    %inactive_next = arith.addf %sum_next, %inactive : f32
    scf.yield %sum_next, %inactive_next : f32, f32
  }
  return %sum : f32
}

func.func @dsome_res_inactive(%x: f32, %ub: index, %dr: f32) -> (f32) {
  %dx = enzyme.autodiff @some_res_inactive(%x, %ub, %dr) {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f32, index, f32) -> f32
  return %dx : f32
}

// CHECK-LABEL:   func.func private @diffesome_res_inactive(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[ARG1]]) : memref<?xf32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[ARG1]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_0]], %[[VAL_2:.*]] = %[[CONSTANT_0]]) -> (f32, f32) {
// CHECK:             memref.store %[[VAL_1]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[VAL_1]], %[[ARG0]] : f32
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[MULF_0]], %[[VAL_2]] : f32
// CHECK:             scf.yield %[[MULF_0]], %[[ADDF_0]] : f32, f32
// CHECK:           }
// CHECK:           %[[FOR_1:.*]]:2 = scf.for %[[VAL_3:.*]] = %[[CONSTANT_2]] to %[[ARG1]] step %[[CONSTANT_1]] iter_args(%[[VAL_4:.*]] = %[[ARG2]], %[[VAL_5:.*]] = %[[CONSTANT_3]]) -> (f32, f32) {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[ARG1]], %[[CONSTANT_1]] : index
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[VAL_3]] : index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[SUBI_1]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[VAL_4]], %[[ARG0]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[VAL_4]], %[[LOAD_0]] : f32
// CHECK:             %[[ADDF_1:.*]] = arith.addf %[[VAL_5]], %[[MULF_2]] : f32
// CHECK:             scf.yield %[[MULF_1]], %[[ADDF_1]] : f32, f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?xf32>
// CHECK:           return %[[VAL_6:.*]]#1 : f32
// CHECK:         }

// -----

func.func private @affine_some_res_inactive(%x: f32, %ub: index) -> (f32) {
  %sum_0 = arith.constant 1.0 : f32
  %sum, %unused = affine.for %iv = 0 to %ub
      iter_args(%sum_iter = %sum_0, %inactive = %sum_0) -> (f32, f32) {
    %sum_next = arith.mulf %sum_iter, %x : f32
    %inactive_next = arith.addf %sum_next, %inactive : f32
    affine.yield %sum_next, %inactive_next : f32, f32
  }
  return %sum : f32
}

func.func @daffine_res_inactive(%x: f32, %ub: index, %dr: f32) -> (f32) {
  %dx = enzyme.autodiff @affine_some_res_inactive(%x, %ub, %dr) {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f32, index, f32) -> f32
  return %dx : f32
}

// CHECK-LABEL:   func.func private @diffeaffine_some_res_inactive(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[ARG1]]) : memref<?xf32>
// CHECK:           %[[FOR_0:.*]]:2 = affine.for %[[VAL_0:.*]] = 0 to %[[ARG1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]], %[[VAL_2:.*]] = %[[CONSTANT_1]]) -> (f32, f32) {
// CHECK:             memref.store %[[VAL_1]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[VAL_1]], %[[ARG0]] : f32
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[MULF_0]], %[[VAL_2]] : f32
// CHECK:             affine.yield %[[MULF_0]], %[[ADDF_0]] : f32, f32
// CHECK:           }
// CHECK:           %[[FOR_1:.*]]:2 = affine.for %[[VAL_3:.*]] = 0 to %[[ARG1]] iter_args(%[[VAL_4:.*]] = %[[ARG2]], %[[VAL_5:.*]] = %[[CONSTANT_2]]) -> (f32, f32) {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[ARG1]], %[[CONSTANT_0]] : index
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[VAL_3]] : index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[SUBI_1]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[VAL_4]], %[[ARG0]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[VAL_4]], %[[LOAD_0]] : f32
// CHECK:             %[[ADDF_1:.*]] = arith.addf %[[VAL_5]], %[[MULF_2]] : f32
// CHECK:             affine.yield %[[MULF_1]], %[[ADDF_1]] : f32, f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?xf32>
// CHECK:           return %[[VAL_6:.*]]#1 : f32
// CHECK:         }
