// RUN: %eopt %s --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

func.func @reduce(%x: f32, %ub: index) -> (f32) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index

  // Initial sum set to 0.
  %sum_0 = arith.constant 1.0 : f32
  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %sum_next = arith.mulf %sum_iter, %x : f32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %sum_next : f32
  } {enzyme.disable_mincut, enzyme.cache_use_tensor}
  return %sum : f32
}

// CHECK:  func.func @reduce(%arg0: f32, %arg1: index, %[[a2:.+]]: f32) -> f32 {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %0 = tensor.empty(%arg1) : tensor<?xf32>
// CHECK-NEXT:    %1 = tensor.empty(%arg1) : tensor<?xf32>
// CHECK-NEXT:    %[[a1:.+]]:3 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %cst, %arg5 = %0, %arg6 = %1) -> (f32, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:      %inserted = tensor.insert %arg4 into %arg5[%arg3] : tensor<?xf32>
// CHECK-NEXT:      %inserted_1 = tensor.insert %arg0 into %arg6[%arg3] : tensor<?xf32>
// CHECK-NEXT:      %[[f4:.+]] = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      scf.yield %[[f4]], %inserted, %inserted_1 : f32, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a3:.+]]:2 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %[[a2]], %arg5 = %cst_0) -> (f32, f32) { 
// CHECK-NEXT:      %[[nm1:.+]] = arith.subi %arg1, %c1 : index 
// CHECK-NEXT:      %[[ridx:.+]] = arith.subi %[[nm1]], %arg3 : index 

// CHECK-NEXT:      %extracted = tensor.extract %[[a1]]#1[%[[ridx]]] : tensor<?xf32>
// CHECK-NEXT:      %extracted_1 = tensor.extract %[[a1]]#2[%[[ridx]]] : tensor<?xf32>
// CHECK-NEXT:      %[[a4:.+]] = arith.mulf %arg4, %extracted_1 : f32
// CHECK-NEXT:      %[[a6:.+]] = arith.mulf %arg4, %extracted : f32
// CHECK-NEXT:      %[[a7:.+]] = arith.addf %arg5, %[[a6]] : f32
// CHECK-NEXT:      scf.yield %[[a4]], %[[a7]] : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a3]]#1 : f32
// CHECK-NEXT:  }
