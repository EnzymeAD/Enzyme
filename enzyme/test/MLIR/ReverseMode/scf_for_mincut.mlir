// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme{dataflow markReadonly},canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" --split-input-file | FileCheck %s

module {
  func.func @main(%arg0: f32) -> (f32) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index

    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %arg0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %sum_iter : f32
      %cos_next = math.cos %sum_next : f32
      scf.yield %cos_next : f32
    } {enzyme.cache_use_tensor}

    return %sum : f32
  }

  func.func @dmain(%arg0: f32, %dsum: f32) -> f32 {
    %darg0 = enzyme.autodiff @main(%arg0, %dsum) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f32, f32) -> f32
    return %darg0 : f32
  }
}

// CHECK:  func.func private @diffemain(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[v0:.+]] = tensor.empty() : tensor<10xf32>
// CHECK-NEXT:    %[[for:.+]]:2 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %arg0, %arg4 = %[[v0]]) -> (f32, tensor<10xf32>) {
// CHECK-NEXT:      %[[cache:.+]] = tensor.insert %arg3 into %arg4[%arg2] : tensor<10xf32>
// CHECK-NEXT:      %[[v3:.+]] = arith.mulf %arg3, %arg3 : f32
// CHECK-NEXT:      %[[v4:.+]] = math.cos %[[v3]] : f32
// CHECK-NEXT:      scf.yield %[[v4]], %[[cache]] : f32, tensor<10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[revFor:.+]] = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %arg1) -> (f32) {

// CHECK-NEXT:      %[[ridx:.+]] = arith.subi %c9, %arg2 : index 

// CHECK-NEXT:      %[[cache:.+]] = tensor.extract %[[for]]#1[%[[ridx]]] : tensor<10xf32>
// CHECK-NEXT:      %[[v3:.+]] = arith.mulf %[[cache]], %[[cache]] : f32
// CHECK-NEXT:      %[[v4:.+]] = math.sin %[[v3]] fastmath<fast> : f32
// CHECK-NEXT:      %[[v5:.+]] = arith.negf %[[v4]] fastmath<fast> : f32
// CHECK-NEXT:      %[[v6:.+]] = arith.mulf %arg3, %[[v5]] fastmath<fast> : f32
// CHECK-NEXT:      %[[v7:.+]] = arith.mulf %[[v6]], %extracted fastmath<fast> : f32
// CHECK-NEXT:      %[[v8:.+]] = arith.mulf %[[v6]], %extracted fastmath<fast> : f32
// CHECK-NEXT:      %[[v9:.+]] = arith.addf %[[v7]], %[[v8]] fastmath<fast> : f32
// CHECK-NEXT:      scf.yield %[[v9]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[revFor:.+]] : f32
// CHECK-NEXT:  }

// -----

func.func private @recompute_if(%cond: memref<?xi1>, %x: f32) -> f32 {
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index

  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%mul_iter = %x) -> (f32) {
    %cond_it = memref.load %cond[%iv] : memref<?xi1>
    %sin = math.sin %mul_iter : f32
    %ifres = scf.if %cond_it -> f32 {
      scf.yield %sin : f32
    } else {
      %cos = math.cos %mul_iter : f32
      scf.yield %cos : f32
    }
    %mul_next = arith.mulf %mul_iter, %ifres : f32
    scf.yield %mul_next : f32
  }

  return %sum : f32
}

func.func @drecompute_if(%cond: memref<?xi1>, %arg0: f32, %dsum: f32) -> f32 {
  %darg0 = enzyme.autodiff @recompute_if(%cond, %arg0, %dsum) {
    activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (memref<?xi1>, f32, f32) -> f32
  return %darg0 : f32
}

// CHECK-LABEL:   func.func private @differecompute_if(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xi1>,
// CHECK-SAME:      %[[ARG1:.*]]: f32,
// CHECK-SAME:      %[[ARG2:.*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 9 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0.000000e+00 : f32
                    // Only one value should be cached
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<10xf32>
// CHECK-NOT:       memref.alloc
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[ARG1]]) -> (f32) {
// CHECK:             memref.store %[[VAL_1]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<10xf32>
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[VAL_0]]] {enzyme.readonly} : memref<?xi1>
// CHECK:             %[[SIN_0:.*]] = math.sin %[[VAL_1]] : f32
// CHECK:             %[[IF_0:.*]] = scf.if %[[LOAD_0]] -> (f32) {
// CHECK:               scf.yield %[[SIN_0]] : f32
// CHECK:             } else {
// CHECK:               %[[COS_0:.*]] = math.cos %[[VAL_1]] : f32
// CHECK:               scf.yield %[[COS_0]] : f32
// CHECK:             } {preserve_cache}
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[VAL_1]], %[[IF_0]] : f32
// CHECK:             scf.yield %[[MULF_0]] : f32
// CHECK:           }
// CHECK:           %[[FOR_1:.*]] = scf.for %[[VAL_2:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] iter_args(%[[VAL_3:.*]] = %[[ARG2]]) -> (f32) {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_0]], %[[VAL_2]] : index
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[SUBI_0]]] : memref<10xf32>
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ARG0]]{{\[}}%[[SUBI_0]]] {enzyme.readonly} : memref<?xi1>
// CHECK:             %[[SIN_1:.*]] = math.sin %[[LOAD_1]] : f32
                      // The if should be moved from the forward pass to reverse
// CHECK:             %[[IF_1:.*]] = scf.if %[[LOAD_2]] -> (f32) {
// CHECK:               scf.yield %[[SIN_1]] : f32
// CHECK:             } else {
// CHECK:               %[[COS_1:.*]] = math.cos %[[LOAD_1]] : f32
// CHECK:               scf.yield %[[COS_1]] : f32
// CHECK:             } {preserve_cache}
// CHECK:             %[[MULF_1:.*]] = arith.mulf %[[VAL_3]], %[[IF_1]] fastmath<fast> : f32
// CHECK:             %[[MULF_2:.*]] = arith.mulf %[[VAL_3]], %[[LOAD_1]] fastmath<fast> : f32
// CHECK:             %[[IF_2:.*]]:2 = scf.if %[[LOAD_2]] -> (f32, f32) {
// CHECK:               scf.yield %[[MULF_2]], %[[MULF_1]] : f32, f32
// CHECK:             } else {
// CHECK:               %[[SIN_2:.*]] = math.sin %[[LOAD_1]] fastmath<fast> : f32
// CHECK:               %[[NEGF_0:.*]] = arith.negf %[[SIN_2]] fastmath<fast> : f32
// CHECK:               %[[MULF_3:.*]] = arith.mulf %[[MULF_2]], %[[NEGF_0]] fastmath<fast> : f32
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[MULF_1]], %[[MULF_3]] fastmath<fast> : f32
// CHECK:               scf.yield %[[CONSTANT_4]], %[[ADDF_0]] : f32, f32
// CHECK:             }
// CHECK:             %[[COS_2:.*]] = math.cos %[[LOAD_1]] fastmath<fast> : f32
// CHECK:             %[[MULF_4:.*]] = arith.mulf %[[VAL_4:.*]]#0, %[[COS_2]] fastmath<fast> : f32
// CHECK:             %[[ADDF_1:.*]] = arith.addf %[[VAL_4]]#1, %[[MULF_4]] fastmath<fast> : f32
// CHECK:             scf.yield %[[ADDF_1]] : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<10xf32>
// CHECK:           return %[[FOR_1]] : f32
// CHECK:         }
