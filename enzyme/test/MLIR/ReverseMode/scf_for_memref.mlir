// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --flatten-enzyme-caches --canonicalize --split-input-file | FileCheck %s

func.func @reduce(%x: f32, %ub: index) -> (f32) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %ub_inner = arith.constant 4 : index

  // Initial sum set to 0.
  %sum_0 = arith.constant 1.0 : f32
  // iter_args binds initial values to the loop's region arguments.
  %sum2 = scf.for %iv2 = %lb to %ub step %step
      iter_args(%sum_iter2 = %sum_0) -> (f32) {
    %sum = scf.for %iv = %lb to %ub_inner step %step
        iter_args(%sum_iter = %sum_iter2) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %x : f32
      // Yield current iteration sum to next iteration %sum_iter or to %sum
      // if final iteration.
      scf.yield %sum_next : f32
    }
    scf.yield %sum : f32
  }
  return %sum2 : f32
}

func.func @dreduce(%x: f32, %ub: index, %dres: f32) -> (f32) {
  %res = enzyme.autodiff @reduce(%x, %ub, %dres) {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f32, index, f32) -> f32
  return %res : f32
}

// CHECK:  func.func private @differeduce(%arg0: f32, %arg1: index, %arg2: f32) -> f32 {
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%arg1) : memref<?x4xf32>
// CHECK-NEXT:    %[[v0:.+]] = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:      %subview = memref.subview %alloc[%arg3, 0] [1, 4] [1, 1] : memref<?x4xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v2:.+]] = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (f32) {
// CHECK-NEXT:        memref.store %arg6, %subview[%arg5] : memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v3:.+]] = arith.mulf %arg6, %arg0 : f32
// CHECK-NEXT:        scf.yield %[[v3]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v2]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v1:.+]]:2 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %arg2, %arg5 = %cst_0) -> (f32, f32) {

// CHECK-NEXT:      %[[nm1:.+]] = arith.subi %arg1, %c1 : index 
// CHECK-NEXT:      %[[idx1:.+]] = arith.subi %[[nm1]], %arg3 : index 


// CHECK-NEXT:      %subview = memref.subview %alloc[%[[idx1]], 0] [1, 4] [1, 1] : memref<?x4xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v2:.+]]:2 = scf.for %[[arg7:.+]] = %c0 to %c4 step %c1 iter_args(%[[arg8:.+]] = %arg4, %[[arg9:.+]] = %arg5) -> (f32, f32) {

// CHECK-NEXT:        %[[idx2:.+]] = arith.subi %c3, %[[arg7]] : index 

// CHECK-NEXT:        %[[v4:.+]] = memref.load %subview[%[[idx2]]] : memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v5:.+]] = arith.mulf %[[arg8]], %arg0 : f32
// CHECK-NEXT:        %[[v6:.+]] = arith.mulf %[[arg8]], %[[v4]] : f32
// CHECK-NEXT:        %[[v7:.+]] = arith.addf %[[arg9]], %[[v6]] : f32
// CHECK-NEXT:        scf.yield %[[v5]], %[[v7]] : f32, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v2]]#0, %[[v2]]#1 : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?x4xf32>
// CHECK-NEXT:    return %[[v1]]#1 : f32
// CHECK-NEXT:  }

// -----

func.func private @reverse_index(%lb: index, %ub: index, %x: memref<?xf32>) -> f32 {
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %poison = ub.poison : f32
  // Verify correct loop indices. e.g. if the loop iterates [3, 4, 5, 6],
  // the reversed loop indices are [6, 5, 4, 3] while the *canonical*
  // reversed indices are [3, 2, 1, 0]. Memref accesses should use the
  // reversed indices while caches should use the canonical reversed indices.
  %res:2 = scf.for %iv = %lb to %ub step %c1 iter_args(%acc = %zero, %acc2 = %poison) -> (f32, f32) {
    %ld = memref.load %x[%iv] : memref<?xf32>
    %mulf = arith.mulf %ld, %ld : f32
    memref.store %mulf, %x[%iv] : memref<?xf32>
    %addf = arith.addf %mulf, %acc : f32
    scf.yield %addf, %addf : f32, f32
  }
  return %res#1 : f32
}

func.func @dreverse_index(%lb: index, %ub: index, %x: memref<?xf32>, %dx: memref<?xf32>, %dr: f32) {
    enzyme.autodiff @reverse_index(%lb, %ub, %x, %dx, %dr) {
      activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (index, index, memref<?xf32>, memref<?xf32>, f32) -> ()
    return
}

// CHECK-LABEL:   func.func private @differeverse_index(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<?xf32>,
// CHECK-SAME:      %[[ARG4:.*]]: f32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ARG1]], %[[ARG0]] : index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[SUBI_0]]) : memref<?xf32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_1]]) -> (f32) {
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[VAL_0]], %[[ARG0]] : index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG2]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[SUBI_1]]] : memref<?xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[LOAD_0]] : f32
// CHECK:             memref.store %[[MULF_0]], %[[ARG2]]{{\[}}%[[VAL_0]]] : memref<?xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[MULF_0]], %[[VAL_1]] : f32
// CHECK:             scf.yield %[[ADDF_0]] : f32
// CHECK:           }
// CHECK:           %[[SUBI_2:.*]] = arith.subi %[[ARG1]], %[[ARG0]] : index
// CHECK:           %[[FOR_1:.*]]:2 = scf.for %[[VAL_2:.*]] = %[[ARG0]] to %[[ARG1]] step %[[CONSTANT_0]] iter_args(%[[VAL_3:.*]] = %[[CONSTANT_1]], %[[VAL_4:.*]] = %[[ARG4]]) -> (f32, f32) {
// CHECK:             %[[SUBI_3:.*]] = arith.subi %[[VAL_2]], %[[ARG0]] : index
// CHECK:             %[[SUBI_4:.*]] = arith.subi %[[SUBI_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[RCANON_IV:.*]] = arith.subi %[[SUBI_4]], %[[SUBI_3]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : index
// CHECK:             %[[SUBI_6:.*]] = arith.subi %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[R_IV:.*]] = arith.subi %[[SUBI_6]], %[[VAL_2]] : index
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[RCANON_IV]]] : memref<?xf32>
                      // The gradient signal should be added together
// CHECK:             %[[ADDF_1:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG3]]{{\[}}%[[R_IV]]] : memref<?xf32>
// CHECK:             %[[ADDF_2:.*]] = arith.addf %[[ADDF_1]], %[[LOAD_2]] : f32
// CHECK:             memref.store %[[CONSTANT_1]], %[[ARG3]]{{\[}}%[[R_IV]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[ADDF_2]], %[[LOAD_1]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[ADDF_2]], %[[LOAD_1]] : f32
// CHECK:             %[[ADDF_3:.*]] = arith.addf %[[MULF_1]], %[[MULF_2]] : f32
// CHECK:             %[[LOAD_3:.*]] = memref.load %[[ARG3]]{{\[}}%[[R_IV]]] : memref<?xf32>
// CHECK:             %[[ADDF_4:.*]] = arith.addf %[[LOAD_3]], %[[ADDF_3]] : f32
// CHECK:             memref.store %[[ADDF_4]], %[[ARG3]]{{\[}}%[[R_IV]]] : memref<?xf32>
// CHECK:             scf.yield %[[ADDF_1]], %[[CONSTANT_1]] : f32, f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?xf32>
// CHECK:           return
// CHECK:         }

// -----

func.func @reduce_i32(%x: f32, %ub: i32) -> (f32) {
  %lb = arith.constant 0 : i32
  %step = arith.constant 1 : i32
  %ub_inner = arith.constant 4 : index

  %sum_0 = arith.constant 1.0 : f32
  %sum = scf.for %iv2 = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) : i32 {
    %sum_next = arith.mulf %sum_iter, %x : f32
    scf.yield %sum_next : f32
  }
  return %sum : f32
}

func.func @dreduce(%x: f32, %ub: i32, %dres: f32) -> (f32) {
  %res = enzyme.autodiff @reduce_i32(%x, %ub, %dres) {
    activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f32, i32, f32) -> f32
  return %res : f32
}

// CHECK-LABEL:   func.func private @differeduce_i32(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: i32,
// CHECK-SAME:      %[[ARG2:.*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[INDEX_CAST_0]]) : memref<?xf32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[ARG1]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_0]]) -> (f32)  : i32 {
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             memref.store %[[VAL_1]], %[[ALLOC_0]]{{\[}}%[[INDEX_CAST_1]]] : memref<?xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[VAL_1]], %[[ARG0]] : f32
// CHECK:             scf.yield %[[MULF_0]] : f32
// CHECK:           }
// CHECK:           %[[FOR_1:.*]]:2 = scf.for %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[ARG1]] step %[[CONSTANT_1]] iter_args(%[[VAL_3:.*]] = %[[ARG2]], %[[VAL_4:.*]] = %[[CONSTANT_3]]) -> (f32, f32)  : i32 {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[ARG1]], %[[CONSTANT_1]] : i32
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[VAL_2]] : i32
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[SUBI_1]] : i32 to index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[INDEX_CAST_2]]] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[VAL_3]], %[[ARG0]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[VAL_3]], %[[LOAD_0]] : f32
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[MULF_2]] : f32
// CHECK:             scf.yield %[[MULF_1]], %[[ADDF_0]] : f32, f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<?xf32>
// CHECK:           return %[[VAL_5:.*]]#1 : f32
// CHECK:         }
