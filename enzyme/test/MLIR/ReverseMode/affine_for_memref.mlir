// RUN: %eopt %s --enzyme-wrap="infn=square_ip outfn= argTys=enzyme_dup,enzyme_const retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize --cse | FileCheck %s

func.func @square_ip(%arg0: memref<?xf32>, %ub: index) {
  affine.for %iv = 0 to %ub {
    %v = affine.load %arg0[2 * %iv] : memref<?xf32>
    %sq = arith.mulf %v, %v : f32
    affine.store %sq, %arg0[2 * %iv + 1] : memref<?xf32>
    affine.yield
  }
  return
}

// CHECK: #map = affine_map<(d0) -> (d0 * 2 + 1)>
// CHECK: #map1 = affine_map<(d0) -> (d0 * 2)>
// CHECK:   func.func @square_ip(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: index) {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %alloc = memref.alloc(%arg2) : memref<?xindex>
// CHECK-NEXT:     %alloc_0 = memref.alloc(%arg2) : memref<?xf32>
// CHECK-NEXT:     %0 = affine.for %arg3 = 0 to %arg2 iter_args(%arg4 = %c0) -> (index) {
// CHECK-NEXT:       %2 = affine.load %arg0[%arg3 * 2] : memref<?xf32>
// CHECK-NEXT:       memref.store %2, %alloc_0[%arg4] : memref<?xf32>
// CHECK-NEXT:       %3 = arith.mulf %2, %2 : f32
// CHECK-NEXT:       memref.store %arg3, %alloc[%arg4] : memref<?xindex>
// CHECK-NEXT:       affine.store %3, %arg0[%arg3 * 2 + 1] : memref<?xf32>
// CHECK-NEXT:       %4 = arith.addi %arg4, %c1 : index
// CHECK-NEXT:       affine.yield %4 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     %1 = affine.for %arg3 = 0 to %arg2 iter_args(%arg4 = %arg2) -> (index) {
// CHECK-NEXT:       %2 = memref.load %alloc[%arg4] : memref<?xindex>
// CHECK-NEXT:       %3 = affine.apply #map(%2)
// CHECK-NEXT:       %4 = memref.load %arg1[%3] : memref<?xf32>
// CHECK-NEXT:       memref.store %cst, %arg1[%3] : memref<?xf32>
// CHECK-NEXT:       %5 = memref.load %alloc_0[%arg4] : memref<?xf32>
// CHECK-NEXT:       %6 = arith.mulf %4, %5 : f32
// CHECK-NEXT:       %7 = arith.addf %6, %6 : f32
// CHECK-NEXT:       %8 = affine.apply #map1(%2)
// CHECK-NEXT:       %9 = memref.load %arg1[%8] : memref<?xf32>
// CHECK-NEXT:       %10 = arith.addf %9, %7 : f32
// CHECK-NEXT:       memref.store %10, %arg1[%8] : memref<?xf32>
// CHECK-NEXT:       %11 = arith.subi %arg4, %c1 : index
// CHECK-NEXT:       affine.yield %11 : index
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %alloc_0 : memref<?xf32>
// CHECK-NEXT:     memref.dealloc %alloc : memref<?xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

