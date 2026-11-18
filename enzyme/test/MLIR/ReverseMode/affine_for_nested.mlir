// RUN: %eopt %s --enzyme-wrap="infn=nested_affine outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --flatten-enzyme-caches --canonicalize --enzyme-simplify-math | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @nested_affine(%x: memref<10xf32>) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = affine.for %i = 0 to 10 iter_args(%outer = %cst) -> (f32) {
      %1 = affine.for %j = #map(%i) to 10 iter_args(%inner = %outer) -> (f32) {
        %v = affine.load %x[%j] : memref<10xf32>
        %mul = arith.mulf %inner, %v : f32
        affine.yield %mul : f32
      }
      affine.yield %1 : f32
    }
    return %0 : f32
  }
}

// CHECK-DAG: #[[ID_MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[REV_MAP:.+]] = affine_map<(d0) -> (-d0 + 9)>
// CHECK: func.func @nested_affine(%arg0: memref<10xf32>, %arg1: memref<10xf32>, %arg2: f32) {
// CHECK-DAG:   %c9 = arith.constant 9 : index
// CHECK-DAG:   %c1 = arith.constant 1 : index
// CHECK-DAG:   %c10 = arith.constant 10 : index
// CHECK-DAG:   %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %[[ALLOC0:.+]] = memref.alloc() : memref<10xmemref<?xf32>>
// CHECK-NEXT:   %[[ALLOC1:.+]] = memref.alloc() : memref<10xmemref<?xf32>>
// CHECK-NEXT:   %{{.+}} = affine.for %[[I:.+]] = 0 to 10 iter_args(%[[OUTER:.+]] = %cst) -> (f32) {
// CHECK-NEXT:     %[[LEN:.+]] = arith.subi %c10, %[[I]] : index
// CHECK-NEXT:     %[[INNER_ALLOC0:.+]] = memref.alloc(%[[LEN]]) : memref<?xf32>
// CHECK-NEXT:     memref.store %[[INNER_ALLOC0]], %[[ALLOC1]][%[[I]]] : memref<10xmemref<?xf32>>
// CHECK-NEXT:     %[[INNER_ALLOC1:.+]] = memref.alloc(%[[LEN]]) : memref<?xf32>
// CHECK-NEXT:     memref.store %[[INNER_ALLOC1]], %[[ALLOC0]][%[[I]]] : memref<10xmemref<?xf32>>
// CHECK-NEXT:     %[[INNER_RES:.+]] = affine.for %[[J:.+]] = #[[ID_MAP]](%[[I]]) to 10 iter_args(%[[INNER:.+]] = %[[OUTER]]) -> (f32) {
// CHECK-NEXT:       %[[J_OFF:.+]] = arith.subi %[[J]], %[[I]] : index
// CHECK-NEXT:       memref.store %[[INNER]], %[[INNER_ALLOC0]][%[[J_OFF]]] : memref<?xf32>
// CHECK-NEXT:       %[[X_VAL:.+]] = affine.load %arg0[%[[J]]] : memref<10xf32>
// CHECK-NEXT:       memref.store %[[X_VAL]], %[[INNER_ALLOC1]][%[[J_OFF]]] : memref<?xf32>
// CHECK-NEXT:       %[[PROD:.+]] = arith.mulf %[[INNER]], %[[X_VAL]] : f32
// CHECK-NEXT:       affine.yield %[[PROD]] : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %[[INNER_RES]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   %{{.+}} = affine.for %[[RI:.+]] = 0 to 10 iter_args(%[[DOUTER:.+]] = %arg2) -> (f32) {
// CHECK-NEXT:     %[[ORIG_I:.+]] = affine.apply #[[REV_MAP]](%[[RI]])
// CHECK-NEXT:     %[[CACHED_X:.+]] = memref.load %[[ALLOC0]][%[[ORIG_I]]] : memref<10xmemref<?xf32>>
// CHECK-NEXT:     %[[CACHED_INNER:.+]] = memref.load %[[ALLOC1]][%[[ORIG_I]]] : memref<10xmemref<?xf32>>
// CHECK-NEXT:     %[[DINNER_RES:.+]] = scf.for %[[RJ:.+]] = %[[ORIG_I]] to %c10 step %c1 iter_args(%[[DINNER:.+]] = %[[DOUTER]]) -> (f32) {
// CHECK-NEXT:       %[[RJ_OFF:.+]] = arith.subi %[[RJ]], %[[ORIG_I]] : index
// CHECK-NEXT:       %[[MAX_OFF:.+]] = arith.subi %c9, %[[ORIG_I]] : index
// CHECK-NEXT:       %[[REV_OFF:.+]] = arith.subi %[[MAX_OFF]], %[[RJ_OFF]] : index
// CHECK-NEXT:       %[[ORIG_J:.+]] = arith.addi %[[ORIG_I]], %[[REV_OFF]] : index
// CHECK-NEXT:       %[[POP_INNER:.+]] = memref.load %[[CACHED_INNER]][%[[REV_OFF]]] : memref<?xf32>
// CHECK-NEXT:       %[[POP_X:.+]] = memref.load %[[CACHED_X]][%[[REV_OFF]]] : memref<?xf32>
// CHECK-NEXT:       %[[NEXT_DINNER:.+]] = arith.mulf %[[DINNER]], %[[POP_X]] fastmath<fast> : f32
// CHECK-NEXT:       %[[DX_CONTRIB:.+]] = arith.mulf %[[DINNER]], %[[POP_INNER]] fastmath<fast> : f32
// CHECK-NEXT:       %[[OLD_DX:.+]] = memref.load %arg1[%[[ORIG_J]]] : memref<10xf32>
// CHECK-NEXT:       %[[NEW_DX:.+]] = arith.addf %[[OLD_DX]], %[[DX_CONTRIB]] fastmath<fast> : f32
// CHECK-NEXT:       memref.store %[[NEW_DX]], %arg1[%[[ORIG_J]]] : memref<10xf32>
// CHECK-NEXT:       scf.yield %[[NEXT_DINNER]] : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.dealloc %[[CACHED_X]] : memref<?xf32>
// CHECK-NEXT:     memref.dealloc %[[CACHED_INNER]] : memref<?xf32>
// CHECK-NEXT:     affine.yield %[[DINNER_RES]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.dealloc %[[ALLOC1]] : memref<10xmemref<?xf32>>
// CHECK-NEXT:   memref.dealloc %[[ALLOC0]] : memref<10xmemref<?xf32>>
// CHECK-NEXT:   return
// CHECK-NEXT: }
