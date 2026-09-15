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

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 3 + d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 * -3 + d1 + 6)>
// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %0 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %arg0) -> (f32) {
// CHECK-NEXT:      memref.store %arg3, %alloc[%arg2] : memref<3xf32>
// CHECK-NEXT:      %2 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (f32) {
// CHECK-NEXT:        %3 = affine.apply #[[MAP]](%arg2, %arg4)
// CHECK-NEXT:        %4 = arith.mulf %arg5, %arg5 : f32
// CHECK-NEXT:        %5 = math.cos %4 : f32
// CHECK-NEXT:        %6 = arith.index_cast %3 : index to i64
// CHECK-NEXT:        %7 = arith.uitofp %6 : i64 to f32
// CHECK-NEXT:        %8 = arith.mulf %5, %7 : f32
// CHECK-NEXT:        affine.yield %8 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.yield %2 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %arg1) -> (f32) {
// CHECK-NEXT:      %2 = arith.subi %c2, %arg2 : index
// CHECK-NEXT:      %3 = memref.load %alloc[%2] : memref<3xf32>
// CHECK-NEXT:      %alloc_0 = memref.alloc() : memref<3xf32>
// CHECK-NEXT:      %4 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %3) -> (f32) {
// CHECK-NEXT:        memref.store %arg5, %alloc_0[%arg4] : memref<3xf32>
// CHECK-NEXT:        %6 = affine.apply #[[MAP1]](%arg2, %arg4)
// CHECK-NEXT:        %7 = arith.mulf %arg5, %arg5 : f32
// CHECK-NEXT:        %8 = math.cos %7 : f32
// CHECK-NEXT:        %9 = arith.index_cast %6 : index to i64
// CHECK-NEXT:        %10 = arith.uitofp %9 : i64 to f32
// CHECK-NEXT:        %11 = arith.mulf %8, %10 : f32
// CHECK-NEXT:        affine.yield %11 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (f32) {
// CHECK-NEXT:        %6 = arith.subi %c2, %arg4 : index
// CHECK-NEXT:        %7 = memref.load %alloc_0[%6] : memref<3xf32>
// CHECK-NEXT:        %8 = affine.apply #[[MAP1]](%arg2, %6)
// CHECK-NEXT:        %9 = arith.mulf %7, %7 : f32
// CHECK-NEXT:        %10 = arith.index_cast %8 : index to i64
// CHECK-NEXT:        %11 = arith.uitofp %10 : i64 to f32
// CHECK-NEXT:        %12 = arith.mulf %arg5, %11 fastmath<fast> : f32
// CHECK-NEXT:        %13 = math.sin %9 fastmath<fast> : f32
// CHECK-NEXT:        %14 = arith.negf %13 fastmath<fast> : f32
// CHECK-NEXT:        %15 = arith.mulf %12, %14 fastmath<fast> : f32
// CHECK-NEXT:        %16 = arith.mulf %15, %7 fastmath<fast> : f32
// CHECK-NEXT:        %17 = arith.mulf %15, %7 fastmath<fast> : f32
// CHECK-NEXT:        %18 = arith.addf %16, %17 fastmath<fast> : f32
// CHECK-NEXT:        affine.yield %18 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %alloc_0 : memref<3xf32>
// CHECK-NEXT:      affine.yield %5 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<3xf32>
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }
