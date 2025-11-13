// RUN: %eopt %s --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active,enzyme_const,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

func.func @reduce(%x: f32, %ub_outer: index, %ub_inner: index) -> (f32) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index

  // Initial sum set to 0.
  %sum_0 = arith.constant 1.0 : f32
  // iter_args binds initial values to the loop's region arguments.
  %sum2 = scf.for %iv2 = %lb to %ub_outer step %step
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

// CHECK:  func.func @reduce(%[[arg0:.+]]: f32, %[[ub_outer:.+]]: index, %[[ub_inner:.+]]: index, %[[dres:.+]]: f32) -> f32 {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%[[ub_outer]]) : memref<?xmemref<?xf32>>
// CHECK-NEXT:    %[[v0:.+]] = scf.for %[[it:.+]] = %c0 to %[[ub_outer]] step %c1 iter_args(%[[iv4:.+]] = %cst) -> (f32) {
// CHECK-NEXT:      %[[INNER_CACHE:.+]] = memref.alloc(%[[ub_inner]]) : memref<?xf32>
// CHECK-NEXT:      memref.store %[[INNER_CACHE]], %alloc[%[[it]]] : memref<?xmemref<?xf32>>
// CHECK-NEXT:      %[[v2:.+]] = scf.for %[[iv_inner:.+]] = %c0 to %[[ub_inner]] step %c1 iter_args(%[[iv6:.+]] = %[[iv4]]) -> (f32) {
// CHECK-NEXT:        memref.store %[[iv6]], %[[INNER_CACHE]][%[[iv_inner]]] : memref<?xf32>
// CHECK-NEXT:        %[[v3:.+]] = arith.mulf %[[iv6]], %[[arg0]] : f32
// CHECK-NEXT:        scf.yield %[[v3]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v2]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v1:.+]]:2 = scf.for %[[iv:.+]] = %c0 to %[[ub_outer]] step %c1 iter_args(%[[iv4:.+]] = %[[dres]], %[[iv5:.+]] = %cst_0) -> (f32, f32) {

// CHECK-NEXT:      %[[nm1:.+]] = arith.subi %[[ub_outer]], %c1 : index 
// CHECK-NEXT:      %[[idx1:.+]] = arith.subi %[[nm1]], %[[iv]] : index 


// CHECK-NEXT:      %[[INNER_CACHE:.+]] = memref.load %alloc[%[[idx1]]] : memref<?xmemref<?xf32>>
// CHECK-NEXT:      %[[v2:.+]]:2 = scf.for %[[arg7:.+]] = %c0 to %[[ub_inner]] step %c1 iter_args(%[[arg8:.+]] = %[[iv4]], %[[arg9:.+]] = %[[iv5]]) -> (f32, f32) {

// CHECK-NEXT:        %[[idx2:.+]] = arith.subi %[[ub_inner]], %c1 : index 
// CHECK-NEXT:        %[[idx3:.+]] = arith.subi %[[idx2]], %[[arg7]] : index

// CHECK-NEXT:        %[[v4:.+]] = memref.load %[[INNER_CACHE]][%[[idx3]]] : memref<?xf32>
// CHECK-NEXT:        %[[v5:.+]] = arith.mulf %[[arg8]], %[[arg0]] : f32
// CHECK-NEXT:        %[[v6:.+]] = arith.mulf %[[arg8]], %[[v4]] : f32
// CHECK-NEXT:        %[[v7:.+]] = arith.addf %[[arg9]], %[[v6]] : f32
// CHECK-NEXT:        scf.yield %[[v5]], %[[v7]] : f32, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[INNER_CACHE]] : memref<?xf32>
// CHECK-NEXT:      scf.yield %[[v2]]#0, %[[v2]]#1 : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?xmemref<?xf32>>
// CHECK-NEXT:    return %[[v1]]#1 : f32
// CHECK-NEXT:  }
