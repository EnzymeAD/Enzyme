// RUN: %eopt %s --enzyme-wrap="infn=affine outfn= argTys=enzyme_active,enzyme_dup,enzyme_dup retTys= mode=ReverseModeCombined" --enzyme-wrap="infn=scf outfn= argTys=enzyme_active,enzyme_dup,enzyme_dup retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s

module {
  func.func @affine(%a: f32,
                    %x: memref<?xf32>,
                    %y: memref<?xf32>) {
    affine.parallel (%arg9) = (0) to (1) {
      %0 = affine.load %x[0] : memref<?xf32>
      %1 = arith.mulf %0, %a : f32
      affine.store %1, %y[0] : memref<?xf32>
    }
    return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<() -> (0)>
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:.*]]: f32,
// CHECK-SAME:                      %[[ARG1:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:.*]]: memref<?xf32>) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<1xf32>
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (1) {
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[ARG1]][0] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<1xf32>
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:             affine.store %[[MULF_0]], %[[ARG3]][0] : memref<?xf32>
// CHECK:           }
// CHECK:           %[[PARALLEL_0:.*]] = affine.parallel (%[[VAL_1:.*]]) = (0) to (1) reduce ("addf") -> (f32) {
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<1xf32>
// CHECK:             %[[LOAD_2:.*]] = affine.load %[[ARG4]][0] : memref<?xf32>
// CHECK:             affine.store %[[CONSTANT_0]], %[[ARG4]][0] : memref<?xf32>
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[ARG0]] : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:             %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[MULF_1]], %[[ARG2]], (#[[$ATTR_0]]) [] : (f32, memref<?xf32>) -> f32
// CHECK:             affine.yield %[[MULF_2]] : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<1xf32>
// CHECK:           return %[[PARALLEL_0]] : f32
// CHECK:         }

