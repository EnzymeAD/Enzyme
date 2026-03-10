// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s
func.func @foo(%x: memref<?xf32> {llvm.noalias}, %y: memref<?xf32> {llvm.noalias}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%arg9) = (%c0) to (%c4) step (%c1) {
    %0 = memref.load %x[%arg9] : memref<?xf32>
    %1 = arith.mulf %0, %0 : f32
    memref.store %1, %y[%arg9] : memref<?xf32>
    scf.reduce
  }
  return
}

func.func @dfoo(%x: memref<?xf32> {llvm.noalias}, %dx: memref<?xf32> {llvm.noalias}, %y: memref<?xf32> {llvm.noalias}, %dy: memref<?xf32> {llvm.noalias}) {
  enzyme.autodiff @foo(%x, %dx, %y, %dy) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK-LABEL:   func.func private @diffefoo(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32> {llvm.noalias}, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32> {llvm.noalias}, %[[ARG3:.*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           scf.parallel (%[[VAL_0:.*]]) = (%[[CONSTANT_2]]) to (%[[CONSTANT_0]]) step (%[[CONSTANT_1]]) {
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[LOAD_0]] : f32
// CHECK:             memref.store %[[MULF_0]], %[[ARG2]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             scf.reduce
// CHECK:           }
// CHECK:           scf.parallel (%[[VAL_1:.*]]) = (%[[CONSTANT_2]]) to (%[[CONSTANT_0]]) step (%[[CONSTANT_1]]) {
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ARG0]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             memref.store %[[CONSTANT_3]], %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[MULF_1]], %[[MULF_2]] : f32
// CHECK:             %[[ATOMIC_RMW_0:.*]] = memref.atomic_rmw addf %[[ADDF_0]], %[[ARG1]]{{\[}}%[[VAL_1]]] : (f32, memref<?xf32>) -> f32
// CHECK:             scf.reduce
// CHECK:           }
// CHECK:           return
// CHECK:         }
