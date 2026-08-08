// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: f32) -> (f32) {
    %sum = affine.for %iv = 0 to 9
        iter_args(%sum_iter = %arg0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %sum_iter : f32
      %cos_next = math.cos %sum_next : f32

      %iv64 = arith.index_cast %iv : index to i64
      %ge = arith.uitofp %iv64 : i64 to f32
      %chaos = arith.mulf %cos_next, %ge : f32
      affine.yield %chaos : f32
    } {enzyme.enable_checkpointing = true}

    return %sum : f32
  }
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (-d0 + d1 + 6)>
// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %[[v0:.+]] = affine.for %arg2 = 0 to 9 step 3 iter_args(%arg3 = %arg0) -> (f32) {
// CHECK-NEXT:      %[[idx:.+]] = arith.divui %arg2, %c3 : index
// CHECK-NEXT:      memref.store %arg3, %alloc[%[[idx]]] : memref<3xf32>
// CHECK-NEXT:      %[[v3:.+]] = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (f32) {
// CHECK-NEXT:        %[[v4:.+]] = affine.apply #[[MAP]](%arg2, %arg4)
// CHECK-NEXT:        %[[v5:.+]] = arith.mulf %arg5, %arg5 : f32
// CHECK-NEXT:        %[[v6:.+]] = math.cos %[[v5]] : f32
// CHECK-NEXT:        %[[v7:.+]] = arith.index_cast %[[v4]] : index to i64
// CHECK-NEXT:        %[[v8:.+]] = arith.uitofp %[[v7]] : i64 to f32
// CHECK-NEXT:        %[[v9:.+]] = arith.mulf %[[v6]], %[[v8]] : f32
// CHECK-NEXT:        affine.yield %[[v9]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.yield %[[v3]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v1:.+]] = affine.for %arg2 = 0 to 9 step 3 iter_args(%arg3 = %arg1) -> (f32) {
// CHECK-NEXT:      %[[idx2:.+]] = arith.divui %arg2, %c3 : index
// CHECK-NEXT:      %[[ridx:.+]] = arith.subi %c2, %[[idx2]] : index
// CHECK-NEXT:      %[[v4:.+]] = memref.load %alloc[%[[ridx]]] : memref<3xf32>
// CHECK-NEXT:      %alloc_0 = memref.alloc() : memref<3xf32>
// CHECK-NEXT:      %[[v5:.+]] = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %[[v4]]) -> (f32) {
// CHECK-NEXT:        memref.store %arg5, %alloc_0[%arg4] : memref<3xf32>
// CHECK-NEXT:        %[[v7:.+]] = affine.apply #[[MAP1]](%arg2, %arg4)
// CHECK-NEXT:        %[[v8:.+]] = arith.mulf %arg5, %arg5 : f32
// CHECK-NEXT:        %[[v9:.+]] = math.cos %[[v8]] : f32
// CHECK-NEXT:        %[[v10:.+]] = arith.index_cast %[[v7]] : index to i64
// CHECK-NEXT:        %[[v11:.+]] = arith.uitofp %[[v10]] : i64 to f32
// CHECK-NEXT:        %[[v12:.+]] = arith.mulf %[[v9]], %[[v11]] : f32
// CHECK-NEXT:        affine.yield %[[v12]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[v6:.+]] = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (f32) {
// CHECK-NEXT:        %[[r7:.+]] = arith.subi %c2, %arg4 : index
// CHECK-NEXT:        %[[v8:.+]] = memref.load %alloc_0[%[[r7]]] : memref<3xf32>
// CHECK-NEXT:        %[[v9:.+]] = affine.apply #[[MAP1]](%arg2, %[[r7]])
// CHECK-NEXT:        %[[v10:.+]] = arith.mulf %[[v8]], %[[v8]] : f32
// CHECK-NEXT:        %[[v11:.+]] = arith.index_cast %[[v9]] : index to i64
// CHECK-NEXT:        %[[v12:.+]] = arith.uitofp %[[v11]] : i64 to f32
// CHECK-NEXT:        %[[v13:.+]] = arith.mulf %arg5, %[[v12]] fastmath<fast> : f32
// CHECK-NEXT:        %[[v14:.+]] = math.sin %[[v10]] fastmath<fast> : f32
// CHECK-NEXT:        %[[v15:.+]] = arith.negf %[[v14]] fastmath<fast> : f32
// CHECK-NEXT:        %[[v16:.+]] = arith.mulf %[[v13]], %[[v15]] fastmath<fast> : f32
// CHECK-NEXT:        %[[v17:.+]] = arith.mulf %[[v16]], %[[v8]] fastmath<fast> : f32
// CHECK-NEXT:        %[[v18:.+]] = arith.mulf %[[v16]], %[[v8]] fastmath<fast> : f32
// CHECK-NEXT:        %[[v19:.+]] = arith.addf %[[v17]], %[[v18]] fastmath<fast> : f32
// CHECK-NEXT:        affine.yield %[[v19]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %alloc_0 : memref<3xf32>
// CHECK-NEXT:      affine.yield %[[v6]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<3xf32>
// CHECK-NEXT:    return %[[v1]] : f32
// CHECK-NEXT:  }
