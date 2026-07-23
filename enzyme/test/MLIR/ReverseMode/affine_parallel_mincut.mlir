// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" --split-input-file | FileCheck %s

// From the perspective of minimizing values, the optimal mincut stores 2 values
func.func private @reproducer(%cond: i1, %srcGrid: memref<?xf32>, %dstGrid: memref<?xf32>) {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %three = arith.constant 3.0 : f32
  %c1 = arith.constant 1 : index
  affine.parallel (%iv) = (0) to (100) {
    %idx0 = arith.addi %iv, %c1 : index
    %idx1 = arith.addi %idx0, %c1 : index
    %tmpN = memref.load %srcGrid[%iv] : memref<?xf32>
    %tmpK = memref.load %srcGrid[%idx0] : memref<?xf32>
    %tmpL = memref.load %srcGrid[%idx1] : memref<?xf32>

    %ifres = scf.if %cond -> (f32) {
      scf.yield %zero : f32
    } else {
      %add0 = arith.addf %tmpN, %tmpN : f32
      %add1 = arith.addf %add0, %tmpK : f32
      %rho = arith.addf %add1, %tmpL : f32
      %ux = arith.addf %add1, %add0 : f32

      // diff-use: %ux and %rho required in reverse
      %ux2 = arith.divf %ux, %rho : f32

      %u2 = arith.mulf %ux2, %ux2 : f32
      %tmp_base = arith.mulf %two, %rho : f32
      %tmp1 = arith.mulf %two, %tmp_base : f32

      %tmpN2 = arith.mulf %one, %tmpN : f32
      // diff-use: %tmp1 and %ux2 required, but can be computed from %ux and %rho
      %tmpN2_1 = arith.mulf %tmp1, %ux2 : f32
      %tmpN2_2 = arith.addf %tmpN2, %tmpN2_1 : f32
      scf.yield %tmpN2_2 : f32
    }
    memref.store %ifres, %dstGrid[%iv] : memref<?xf32>
  }
  return
}

func.func @dreproducer(%cond: i1, %src: memref<?xf32>, %dsrc: memref<?xf32>, %dst: memref<?xf32>, %ddst: memref<?xf32>) {
  enzyme.autodiff @reproducer(%cond, %src, %dsrc, %dst, %ddst) {
    activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (i1, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  return
}

// CHECK-LABEL:   func.func private @differeproducer(
// CHECK-SAME:      %[[ARG0:.*]]: i1,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?xf32>,  %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<?xf32>, %[[ARG4:.*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<100xf32>
// CHECK:           %[[ALLOC_1:.*]] = memref.alloc() : memref<100xf32>
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (100) {
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG1]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ARG1]]{{\[}}%[[ADDI_0]]] : memref<?xf32>
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG1]]{{\[}}%[[ADDI_1]]] : memref<?xf32>
// CHECK:             %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f32) {
// CHECK:               scf.yield %[[CONSTANT_3]] : f32
// CHECK:             } else {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[LOAD_0]] : f32
// CHECK:               %[[ADDF_1:.*]] = arith.addf %[[ADDF_0]], %[[LOAD_1]] : f32
// CHECK:               %[[ADDF_2:.*]] = arith.addf %[[ADDF_1]], %[[LOAD_2]] : f32
// CHECK:               memref.store %[[ADDF_2]], %[[ALLOC_1]]{{\[}}%[[VAL_0]]] : memref<100xf32>
// CHECK:               %[[ADDF_3:.*]] = arith.addf %[[ADDF_1]], %[[ADDF_0]] : f32
// CHECK:               memref.store %[[ADDF_3]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<100xf32>
// CHECK:               %[[DIVF_0:.*]] = arith.divf %[[ADDF_3]], %[[ADDF_2]] : f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[ADDF_2]], %[[CONSTANT_2]] : f32
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[MULF_0]], %[[CONSTANT_2]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[MULF_1]], %[[DIVF_0]] : f32
// CHECK:               %[[ADDF_4:.*]] = arith.addf %[[LOAD_0]], %[[MULF_2]] : f32
// CHECK:               scf.yield %[[ADDF_4]] : f32
// CHECK:             }
// CHECK:             memref.store %[[IF_0]], %[[ARG3]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_1:.*]]) = (0) to (100) {
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_1]] : index
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK:             %[[LOAD_3:.*]] = memref.load %[[ARG4]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             memref.store %[[CONSTANT_3]], %[[ARG4]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[IF_1:.*]]:3 = scf.if %[[ARG0]] -> (f32, f32, f32) {
// CHECK:               scf.yield %[[CONSTANT_3]], %[[CONSTANT_3]], %[[CONSTANT_3]] : f32, f32, f32
// CHECK:             } else {
// CHECK:               %[[LOAD_4:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<100xf32>
// CHECK:               %[[LOAD_5:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[VAL_1]]] : memref<100xf32>
// CHECK:               %[[DIVF_1:.*]] = arith.divf %[[LOAD_4]], %[[LOAD_5]] : f32
// CHECK:               %[[MULF_3:.*]] = arith.mulf %[[LOAD_5]], %[[CONSTANT_2]] : f32
// CHECK:               %[[MULF_4:.*]] = arith.mulf %[[MULF_3]], %[[CONSTANT_2]] : f32
// CHECK:               %[[MULF_5:.*]] = arith.mulf %[[LOAD_3]], %[[DIVF_1]] : f32
// CHECK:               %[[MULF_6:.*]] = arith.mulf %[[LOAD_3]], %[[MULF_4]] : f32
// CHECK:               %[[MULF_7:.*]] = arith.mulf %[[MULF_5]], %[[CONSTANT_2]] : f32
// CHECK:               %[[MULF_8:.*]] = arith.mulf %[[MULF_7]], %[[CONSTANT_2]] : f32
// CHECK:               %[[DIVF_2:.*]] = arith.divf %[[MULF_6]], %[[LOAD_5]] : f32
// CHECK:               %[[DIVF_3:.*]] = arith.divf %[[MULF_6]], %[[LOAD_5]] : f32
// CHECK:               %[[DIVF_4:.*]] = arith.divf %[[LOAD_4]], %[[LOAD_5]] : f32
// CHECK:               %[[MULF_9:.*]] = arith.mulf %[[DIVF_3]], %[[DIVF_4]] : f32
// CHECK:               %[[NEGF_0:.*]] = arith.negf %[[MULF_9]] : f32
// CHECK:               %[[ADDF_5:.*]] = arith.addf %[[MULF_8]], %[[NEGF_0]] : f32
// CHECK:               %[[ADDF_6:.*]] = arith.addf %[[DIVF_2]], %[[ADDF_5]] : f32
// CHECK:               %[[ADDF_7:.*]] = arith.addf %[[DIVF_2]], %[[ADDF_6]] : f32
// CHECK:               %[[ADDF_8:.*]] = arith.addf %[[LOAD_3]], %[[ADDF_7]] : f32
// CHECK:               %[[ADDF_9:.*]] = arith.addf %[[ADDF_8]], %[[ADDF_7]] : f32
// CHECK:               scf.yield %[[ADDF_9]], %[[ADDF_5]], %[[ADDF_6]] : f32, f32, f32
// CHECK:             }
// CHECK:             %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[VAL_2:.*]]#1, %[[ARG2]]{{\[}}%[[ADDI_3]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:             %[[ATOMIC_RMW_1:.*]] = enzyme.atomic_rmw addf %[[VAL_2]]#2, %[[ARG2]]{{\[}}%[[ADDI_2]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:             %[[ATOMIC_RMW_2:.*]] = enzyme.atomic_rmw addf %[[VAL_2]]#0, %[[ARG2]]{{\[}}%[[VAL_1]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_1]] : memref<100xf32>
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<100xf32>
// CHECK:           return
// CHECK:         }

// -----

// mincut should avoid caching intermediate values that can be re-computed from sources
func.func @redundant(%x: memref<?xf32>, %y: memref<?xf32>, %len: index) {
  affine.parallel (%iv) = (0) to (%len) {
    // ld0, ld1 are sources
    %ld0 = affine.load %x[%iv] : memref<?xf32>
    %ld1 = affine.load %x[%iv + 1] : memref<?xf32>

    // addf is an intermediate that can be re-computed, so we should not store it
    %addf = arith.addf %ld0, %ld1 : f32

    // these sins introduce sinks
    %sin0 = math.sin %ld0 : f32
    %sin1 = math.sin %ld1 : f32
    %sin2 = math.sin %addf : f32
    %acc0 = arith.addf %sin0, %sin1 : f32    
    %acc1 = arith.addf %acc0, %sin2 : f32
    affine.store %acc1, %y[%iv] : memref<?xf32>
  }
  return
}

func.func @dredundant(%x: memref<?xf32>, %dx: memref<?xf32>, %y: memref<?xf32>, %dy: memref<?xf32>) {
  %c4 = arith.constant 4 : index
  enzyme.autodiff @redundant(%x, %dx, %y, %dy, %c4) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
    ret_activity = []
  } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
  return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 + 1)
// CHECK-LABEL:   func.func private @differedundant(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<?xf32>,
// CHECK-SAME:      %[[ARG4:.*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[ARG4]]) : memref<?xf32>
// CHECK:           %[[ALLOC_1:.*]] = memref.alloc(%[[ARG4]]) : memref<?xf32>
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (symbol(%[[ARG4]])) {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG0]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[LOAD_1:.*]] = affine.load %[[ARG0]]{{\[}}%[[VAL_0]] + 1] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_1]], %[[ALLOC_1]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[LOAD_1]] : f32
// CHECK:             %[[SIN_0:.*]] = math.sin %[[LOAD_0]] : f32
// CHECK:             %[[SIN_1:.*]] = math.sin %[[LOAD_1]] : f32
// CHECK:             %[[SIN_2:.*]] = math.sin %[[ADDF_0]] : f32
// CHECK:             %[[ADDF_1:.*]] = arith.addf %[[SIN_0]], %[[SIN_1]] : f32
// CHECK:             %[[ADDF_2:.*]] = arith.addf %[[ADDF_1]], %[[SIN_2]] : f32
// CHECK:             affine.store %[[ADDF_2]], %[[ARG2]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:           }
// CHECK:           affine.parallel (%[[VAL_1:.*]]) = (0) to (symbol(%[[ARG4]])) {
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[LOAD_3:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[ADDF_3:.*]] = arith.addf %[[LOAD_2]], %[[LOAD_3]] : f32
// CHECK:             %[[LOAD_4:.*]] = memref.load %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             memref.store %[[CONSTANT_0]], %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             %[[COS_0:.*]] = math.cos %[[ADDF_3]] : f32
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_4]], %[[COS_0]] : f32
// CHECK:             %[[COS_1:.*]] = math.cos %[[LOAD_3]] : f32
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[LOAD_4]], %[[COS_1]] : f32
// CHECK:             %[[COS_2:.*]] = math.cos %[[LOAD_2]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[LOAD_4]], %[[COS_2]] : f32
// CHECK:             %[[ADDF_4:.*]] = arith.addf %[[MULF_2]], %[[MULF_0]] : f32
// CHECK:             %[[ADDF_5:.*]] = arith.addf %[[MULF_1]], %[[MULF_0]] : f32
// CHECK:             %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]])
// CHECK:             %[[ATOMIC_RMW_0:.*]] = enzyme.atomic_rmw addf %[[ADDF_5]], %[[ARG1]]{{\[}}%[[APPLY_0]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:             %[[ATOMIC_RMW_1:.*]] = enzyme.atomic_rmw addf %[[ADDF_4]], %[[ARG1]]{{\[}}%[[VAL_1]]] monotonic : (f32, memref<?xf32>) -> f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_1]] : memref<?xf32>
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?xf32>
// CHECK:           return
// CHECK:         }
