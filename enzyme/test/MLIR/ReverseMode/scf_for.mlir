// RUN: %eopt %s --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

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
  } {enzyme.cache_use_tensor}
  return %sum : f32
}

// CHECK:  func.func @reduce(%arg0: f32, %arg1: index, %arg2: f32) -> f32 {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %0 = tensor.empty(%arg1) : tensor<?xf32>
// CHECK-NEXT:    %1:2 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %cst, %arg5 = %0) -> (f32, tensor<?xf32>) {
// CHECK-NEXT:      %inserted = tensor.insert %arg4 into %arg5[%arg3] : tensor<?xf32>
// CHECK-NEXT:      %4 = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      scf.yield %4, %inserted : f32, tensor<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = arith.addf %arg2, %cst_0 : f32
// CHECK-NEXT:    %3:4 = scf.for %arg3 = %c0 to %arg1 step %c1 iter_args(%arg4 = %2, %arg5 = %cst_0, %arg6 = %arg1, %arg7 = %cst_0) -> (f32, f32, index, f32) {
// CHECK-NEXT:      %extracted = tensor.extract %1#1[%arg6] : tensor<?xf32>
// CHECK-NEXT:      %4 = arith.mulf %arg4, %arg0 : f32
// CHECK-NEXT:      %5 = arith.addf %4, %cst_0 : f32
// CHECK-NEXT:      %6 = arith.mulf %arg4, %extracted : f32
// CHECK-NEXT:      %7 = arith.addf %arg5, %6 : f32
// CHECK-NEXT:      %8 = arith.subi %arg6, %c1 : index
// CHECK-NEXT:      scf.yield %5, %7, %8, %7 : f32, f32, index, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %3#1 : f32
// CHECK-NEXT:  }
