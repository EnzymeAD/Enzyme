// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: f32) -> (f32) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index

    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %arg0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %sum_iter : f32
      %cos_next = math.cos %sum_next : f32
      scf.yield %cos_next : f32
    } {enzyme.enable_mincut}

    return %sum : f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = tensor.empty() : tensor<10xf32>
// CHECK-NEXT:    %1:2 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %arg0, %arg4 = %0) -> (f32, tensor<10xf32>) {
// CHECK-NEXT:      %inserted = tensor.insert %arg3 into %arg4[%arg2] : tensor<10xf32>
// CHECK-NEXT:      %3 = arith.mulf %arg3, %arg3 : f32
// CHECK-NEXT:      %4 = math.cos %3 : f32
// CHECK-NEXT:      scf.yield %4, %inserted : f32, tensor<10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2:2 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %arg1, %arg4 = %c9) -> (f32, index) {
// CHECK-NEXT:      %extracted = tensor.extract %1#1[%arg4] : tensor<10xf32>
// CHECK-NEXT:      %3 = arith.mulf %extracted, %extracted : f32
// CHECK-NEXT:      %4 = math.sin %3 : f32
// CHECK-NEXT:      %5 = arith.negf %4 : f32
// CHECK-NEXT:      %6 = arith.mulf %arg3, %5 : f32
// CHECK-NEXT:      %7 = arith.mulf %6, %extracted : f32
// CHECK-NEXT:      %8 = arith.mulf %6, %extracted : f32
// CHECK-NEXT:      %9 = arith.addf %7, %8 : f32
// CHECK-NEXT:      %10 = arith.subi %arg4, %c1 : index
// CHECK-NEXT:      scf.yield %9, %10 : f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %2#0 : f32
// CHECK-NEXT:  }
