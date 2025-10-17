// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: f32) -> (f32) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index

    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %arg0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %arg0 : f32
      %sq_next = arith.addf %sum_iter, %arg0 : f32
      %cos_next = math.cos %sum_next : f32
      %z_next = arith.addf %cos_next, %arg0 : f32
      %y_next = arith.mulf %z_next, %sq_next : f32
      scf.yield %y_next : f32
    }

    return %sum : f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<10xf32>
// CHECK-NEXT:    %0 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %arg0) -> (f32) {
// CHECK-NEXT:      memref.store %arg3, %alloc[%arg2] : memref<10xf32>
// CHECK-NEXT:      %3 = arith.mulf %arg3, %arg0 : f32
// CHECK-NEXT:      %4 = arith.addf %arg3, %arg0 : f32
// CHECK-NEXT:      %5 = math.cos %3 : f32
// CHECK-NEXT:      %6 = arith.addf %5, %arg0 : f32
// CHECK-NEXT:      %7 = arith.mulf %6, %4 : f32
// CHECK-NEXT:      scf.yield %7 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:2 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %arg1, %arg4 = %cst) -> (f32, f32) {
// CHECK-NEXT:      %3 = arith.subi %c9, %arg2 : index
// CHECK-NEXT:      %4 = memref.load %alloc[%3] : memref<10xf32>
// CHECK-NEXT:      %5 = arith.mulf %4, %arg0 : f32
// CHECK-NEXT:      %6 = arith.addf %4, %arg0 : f32
// CHECK-NEXT:      %7 = math.cos %5 : f32
// CHECK-NEXT:      %8 = arith.addf %7, %arg0 : f32
// CHECK-NEXT:      %9 = arith.mulf %arg3, %6 : f32
// CHECK-NEXT:      %10 = arith.mulf %arg3, %8 : f32
// CHECK-NEXT:      %11 = arith.addf %arg4, %9 : f32
// CHECK-NEXT:      %12 = math.sin %5 : f32
// CHECK-NEXT:      %13 = arith.negf %12 : f32
// CHECK-NEXT:      %14 = arith.mulf %9, %13 : f32
// CHECK-NEXT:      %15 = arith.addf %11, %10 : f32
// CHECK-NEXT:      %16 = arith.mulf %14, %arg0 : f32
// CHECK-NEXT:      %17 = arith.addf %10, %16 : f32
// CHECK-NEXT:      %18 = arith.mulf %14, %4 : f32
// CHECK-NEXT:      %19 = arith.addf %15, %18 : f32
// CHECK-NEXT:      scf.yield %17, %19 : f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc : memref<10xf32>
// CHECK-NEXT:    %2 = arith.addf %1#1, %1#0 : f32
// CHECK-NEXT:    return %2 : f32
// CHECK-NEXT:  }
