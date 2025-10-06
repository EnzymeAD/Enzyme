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
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %alloc = memref.alloc(%arg2) : memref<?xindex>
// CHECK-NEXT:     %alloc_0 = memref.alloc(%arg2) : memref<?xf32>
// CHECK-NEXT:     affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:       %[[f0:.+]] = affine.load %arg0[%arg3 * 2] : memref<?xf32>
// CHECK-NEXT:       memref.store %[[f0]], %alloc_0[%arg3] : memref<?xf32>
// CHECK-NEXT:       %[[f3:.+]] = arith.mulf %[[f0]], %[[f0]] : f32
// CHECK-NEXT:       memref.store %arg3, %alloc[%arg3] : memref<?xindex>
// CHECK-NEXT:       affine.store %[[f3]], %arg0[%arg3 * 2 + 1] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:       %[[nm1:.+]] = arith.subi %arg2, %c1 : index
// CHECK-NEXT:       %[[ridx:.+]] = arith.subi %[[nm1]], %arg3 : index
// CHECK-NEXT:       %[[a2:.+]] = memref.load %alloc[%[[ridx]]] : memref<?xindex>
// CHECK-NEXT:       %[[a3:.+]] = affine.apply #map(%[[a2]])
// CHECK-NEXT:       %[[a4:.+]] = memref.load %arg1[%[[a3]]] : memref<?xf32>
// CHECK-NEXT:       memref.store %cst, %arg1[%3] : memref<?xf32>
// CHECK-NEXT:       %[[a5:.+]] = memref.load %alloc_0[%[[ridx]]] : memref<?xf32>
// CHECK-NEXT:       %[[a6:.+]] = arith.mulf %[[a4]], %[[a5]] : f32
// CHECK-NEXT:       %[[a7:.+]] = arith.addf %[[a6]], %[[a6]] : f32
// CHECK-NEXT:       %[[a8:.+]] = affine.apply #map1(%[[a2]])
// CHECK-NEXT:       %[[a9:.+]] = memref.load %arg1[%[[a8]]] : memref<?xf32>
// CHECK-NEXT:       %[[a10:.+]] = arith.addf %[[a9]], %[[a7]] : f32
// CHECK-NEXT:       memref.store %[[a10]], %arg1[%[[a8]]] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %alloc_0 : memref<?xf32>
// CHECK-NEXT:     memref.dealloc %alloc : memref<?xindex>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

