// RUN: %eopt %s --enzyme-wrap="infn=square_ip outfn= argTys=enzyme_dup,enzyme_const retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --flatten-enzyme-caches --canonicalize --enzyme-simplify-math --canonicalize --cse | FileCheck %s

func.func @square_ip(%arg0: memref<?xf32>, %ub: index) {
  affine.for %iv = 0 to %ub {
    %v = affine.load %arg0[2 * %iv] : memref<?xf32>
    %sq = arith.mulf %v, %v : f32
    affine.store %sq, %arg0[2 * %iv + 1] : memref<?xf32>
    affine.yield
  }
  return
}

// CHECK-DAG: #[[REVERSE_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
// CHECK-DAG: #[[REVERSE_STORE_MAP:.+]] = affine_map<(d0)[s0] -> (d0 * -2 + s0 * 2 - 1)>
// CHECK-DAG: #[[REVERSE_LOAD_MAP:.+]] = affine_map<(d0)[s0] -> (d0 * -2 + s0 * 2 - 2)>
// CHECK:   func.func @square_ip(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: index) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %[[alloc0:.+]] = memref.alloc(%arg2) : memref<?xf32>
// CHECK-NEXT:     affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:       %[[f0:.+]] = affine.load %arg0[%arg3 * 2] : memref<?xf32>
// CHECK-NEXT:       memref.store %[[f0]], %[[alloc0]][%arg3] : memref<?xf32>
// CHECK-NEXT:       %[[f3:.+]] = arith.mulf %[[f0]], %[[f0]] : f32
// CHECK-NEXT:       affine.store %[[f3]], %arg0[%arg3 * 2 + 1] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg3 = 0 to %arg2 {
// CHECK-NEXT:       %[[ridx:.+]] = affine.apply #[[REVERSE_MAP]](%arg3)[%arg2]
// CHECK-NEXT:       %[[a5:.+]] = memref.load %[[alloc0]][%[[ridx]]] : memref<?xf32>
// CHECK-NEXT:       %[[a3:.+]] = affine.apply #[[REVERSE_STORE_MAP]](%arg3)[%arg2]
// CHECK-NEXT:       %[[a4:.+]] = memref.load %arg1[%[[a3]]] : memref<?xf32>
// CHECK-NEXT:       memref.store %cst, %arg1[%[[a3]]] : memref<?xf32>
// CHECK-NEXT:       %[[a6:.+]] = arith.mulf %[[a4]], %[[a5]] fastmath<fast> : f32
// CHECK-NEXT:       %[[a7:.+]] = arith.addf %[[a6]], %[[a6]] fastmath<fast> : f32
// CHECK-NEXT:       %[[a8:.+]] = affine.apply #[[REVERSE_LOAD_MAP]](%arg3)[%arg2]
// CHECK-NEXT:       %[[a9:.+]] = memref.load %arg1[%[[a8]]] : memref<?xf32>
// CHECK-NEXT:       %[[a10:.+]] = arith.addf %[[a9]], %[[a7]] fastmath<fast> : f32
// CHECK-NEXT:       memref.store %[[a10]], %arg1[%[[a8]]] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %[[alloc0]] : memref<?xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
