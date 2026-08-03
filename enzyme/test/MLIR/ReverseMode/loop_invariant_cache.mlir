// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme{dataflow markReadonly},canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" --split-input-file | FileCheck %s

func.func private @loop_invariant_cache(%ub0: index, %ub1: index, %x: memref<?x?xf32> {llvm.noalias}, %y: memref<?x?xf32> {llvm.noalias}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  %alloca = memref.alloca() : memref<8xf32>

  scf.for %iv = %c0 to %c8 step %c1 {
    %iv_i32 = arith.index_cast %iv : index to i32
    %ivf = arith.sitofp %iv_i32 : i32 to f32
    memref.store %ivf, %alloca[%iv] : memref<8xf32>
  }

  scf.for %iv = %c0 to %ub0 step %c1 {
    scf.for %jv = %c0 to %ub1 step %c1 {
      %xval = memref.load %x[%iv, %jv] : memref<?x?xf32>
      %idx = arith.remsi %jv, %c8 : index
      %allocaval = memref.load %alloca[%idx] : memref<8xf32>
      %mul = arith.mulf %allocaval, %xval : f32
      memref.store %mul, %y[%iv, %jv] : memref<?x?xf32>
    }
  }
  memref.store %zero, %alloca[%c0] : memref<8xf32>
  return
}

func.func @dloop_invariant_cache(%ub0: index, %ub1: index, %x: memref<?x?xf32>, %dx: memref<?x?xf32>, %y: memref<?x?xf32>, %dy: memref<?x?xf32>) {
  enzyme.autodiff @loop_invariant_cache(%ub0, %ub1, %x, %dx, %y, %dy) {
    activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (index, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// CHECK-LABEL:   func.func private @diffeloop_invariant_cache(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: memref<?x?xf32> {llvm.noalias}, %[[ARG3:.*]]: memref<?x?xf32>, %[[ARG4:.*]]: memref<?x?xf32> {llvm.noalias}, %[[ARG5:.*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() : memref<8xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : index to i32
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[INDEX_CAST_0]] : i32 to f32
// CHECK:             memref.store %[[SITOFP_0]], %[[ALLOCA_0]]{{\[}}%[[VAL_0]]] : memref<8xf32>
// CHECK:           }
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[ARG0]]) : memref<?x8xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[ARG0]] step %[[CONSTANT_1]] {
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_1]], 0] [1, 8] [1, 1] : memref<?x8xf32> to memref<8xf32, strided<[1], offset: ?>>
// CHECK:             scf.for %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] {
// CHECK:               %[[LOAD_0:.*]] = memref.load %[[ALLOCA_0]]{{\[}}%[[VAL_2]]] : memref<8xf32>
// CHECK:               memref.store %[[LOAD_0]], %[[SUBVIEW_0]]{{\[}}%[[VAL_2]]] : memref<8xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_2]] to %[[ARG1]] step %[[CONSTANT_1]] {
// CHECK:               %[[LOAD_1:.*]] = memref.load %[[ARG2]]{{\[}}%[[VAL_1]], %[[VAL_3]]] {enzyme.readonly} : memref<?x?xf32>
// CHECK:               %[[REMSI_0:.*]] = arith.remsi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK:               %[[LOAD_2:.*]] = memref.load %[[ALLOCA_0]]{{\[}}%[[REMSI_0]]] : memref<8xf32>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               memref.store %[[MULF_0]], %[[ARG4]]{{\[}}%[[VAL_1]], %[[VAL_3]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           memref.store %[[CONSTANT_3]], %[[ALLOCA_0]]{{\[}}%[[CONSTANT_2]]] : memref<8xf32>
// CHECK:           scf.for %[[VAL_4:.*]] = %[[CONSTANT_2]] to %[[ARG0]] step %[[CONSTANT_1]] {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[ARG0]], %[[CONSTANT_1]] : index
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[VAL_4]] : index
// CHECK:             %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[SUBI_1]], 0] [1, 8] [1, 1] : memref<?x8xf32> to memref<8xf32, strided<[1], offset: ?>>
// CHECK:             scf.for %[[VAL_5:.*]] = %[[CONSTANT_2]] to %[[ARG1]] step %[[CONSTANT_1]] {
// CHECK:               %[[SUBI_2:.*]] = arith.subi %[[ARG1]], %[[CONSTANT_1]] : index
// CHECK:               %[[SUBI_3:.*]] = arith.subi %[[SUBI_2]], %[[VAL_5]] : index
// CHECK:               %[[REMSI_1:.*]] = arith.remsi %[[SUBI_3]], %[[CONSTANT_0]] : index
// CHECK:               %[[LOAD_3:.*]] = memref.load %[[SUBVIEW_1]]{{\[}}%[[REMSI_1]]] : memref<8xf32, strided<[1], offset: ?>>
// CHECK:               %[[LOAD_4:.*]] = memref.load %[[ARG5]]{{\[}}%[[SUBI_1]], %[[SUBI_3]]] : memref<?x?xf32>
// CHECK:               memref.store %[[CONSTANT_3]], %[[ARG5]]{{\[}}%[[SUBI_1]], %[[SUBI_3]]] : memref<?x?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_4]], %[[LOAD_3]] fastmath<fast> : f32
// CHECK:               %[[LOAD_5:.*]] = memref.load %[[ARG3]]{{\[}}%[[SUBI_1]], %[[SUBI_3]]] : memref<?x?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_5]], %[[MULF_1]] fastmath<fast> : f32
// CHECK:               memref.store %[[ADDF_0]], %[[ARG3]]{{\[}}%[[SUBI_1]], %[[SUBI_3]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?x8xf32>
// CHECK:           return
// CHECK:         }
