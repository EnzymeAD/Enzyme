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

// CHECK:  func.func @reduce(%[[varg0:.+]]: f32, %[[varg1:.+]]: index, %[[varg2:.+]]: index, %[[varg3:.+]]: f32) -> f32 {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc(%[[varg1]], %[[varg2]]) : memref<?x?xf32>
// CHECK-NEXT:    %[[v0:.+]] = scf.for %[[varg4:.+]] = %c0 to %[[varg1]] step %c1 iter_args(%[[varg5:.+]] = %cst) -> (f32) {
// CHECK-NEXT:      %subview = memref.subview %alloc[%[[varg4]], 0] [1, %[[varg2]]] [1, 1] : memref<?x?xf32> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v2:.+]] = scf.for %[[varg6:.+]] = %c0 to %[[varg2]] step %c1 iter_args(%[[varg7:.+]] = %[[varg5]]) -> (f32) {
// CHECK-NEXT:        memref.store %[[varg7]], %subview[%[[varg6]]] : memref<?xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v3:.+]] = arith.mulf %[[varg7]], %[[varg0]] : f32
// CHECK-NEXT:        scf.yield %[[v3]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v2]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v1:.+]]:2 = scf.for %[[varg4:.+]] = %c0 to %[[varg1]] step %c1 iter_args(%[[varg5:.+]] = %[[varg3:.+]], %[[varg6:.+]] = %cst_0) -> (f32, f32) {
// CHECK-NEXT:      %[[v2:.+]] = arith.subi %[[varg1]], %c1 : index
// CHECK-NEXT:      %[[v3:.+]] = arith.subi %[[v2]], %[[varg4]] : index
// CHECK-NEXT:      %subview = memref.subview %alloc[%[[v3]], 0] [1, %[[varg2]]] [1, 1] : memref<?x?xf32> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-NEXT:      %[[v4:.+]]:2 = scf.for %[[varg7:.+]] = %c0 to %[[varg2]] step %c1 iter_args(%[[varg8:.+]] = %[[varg5]], %[[varg9:.+]] = %[[varg6]]) -> (f32, f32) {
// CHECK-NEXT:        %[[v5:.+]] = arith.subi %[[varg2]], %c1 : index
// CHECK-NEXT:        %[[v6:.+]] = arith.subi %[[v5]], %[[varg7]] : index
// CHECK-NEXT:        %[[v7:.+]] = memref.load %subview[%[[v6]]] : memref<?xf32, strided<[1], offset: ?>>
// CHECK-NEXT:        %[[v8:.+]] = arith.mulf %[[varg8]], %[[varg0]] : f32
// CHECK-NEXT:        %[[v9:.+]] = arith.mulf %[[varg8]], %[[v7]] : f32
// CHECK-NEXT:        %[[v10:.+]] = arith.addf %[[varg9]], %[[v9]] : f32
// CHECK-NEXT:        scf.yield %[[v8]], %[[v10]] : f32, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v4]]#0, %[[v4]]#1 : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<?x?xf32>
// CHECK-NEXT:    return %[[v1]]#1 : f32
// CHECK-NEXT:  }
