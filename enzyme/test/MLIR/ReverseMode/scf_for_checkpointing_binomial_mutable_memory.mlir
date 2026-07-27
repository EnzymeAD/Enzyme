// RUN: %eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func @reduce_sum(%buf: memref<10xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = arith.constant 0.0 : f64

    %sum = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> (f64) {
      %val = memref.load %buf[%i] : memref<10xf64>
      %new_acc = arith.addf %acc, %val : f64
      memref.store %new_acc, %buf[%c0] : memref<10xf64>
      scf.yield %new_acc : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period=4,
       enzyme.disable_mincut=true}

    return %sum : f64
  }
}


// CHECK:  func.func @reduce_sum(%arg0: memref<10xf64>, %arg1: memref<10xf64>, %arg2: f64) {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf64>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<4xindex>
// CHECK-NEXT:    %alloc_1 = memref.alloc() : memref<4x10xf64>
// CHECK-NEXT:    %0:2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %c0, %arg5 = %cst) -> (index, f64) {
// CHECK-NEXT:      memref.store %arg5, %alloc[%arg3] : memref<4xf64>
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%arg3] : memref<4xindex>
// CHECK-NEXT:      %3 = arith.subi %c10, %arg4 : index
// CHECK-NEXT:      %4 = arith.subi %c4, %arg3 : index
// CHECK-NEXT:      %5 = arith.minui %4, %3 : index
// CHECK-NEXT:      %6 = enzyme.binomial_progress %3, %5 : index
// CHECK-NEXT:      %subview = memref.subview %alloc_1[%arg3, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %subview : memref<10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %7 = scf.for %arg6 = %c0 to %6 step %c1 iter_args(%arg7 = %arg5) -> (f64) {
// CHECK-NEXT:        %9 = arith.addi %arg4, %arg6 : index
// CHECK-NEXT:        %10 = memref.load %arg0[%9] : memref<10xf64>
// CHECK-NEXT:        %11 = arith.addf %arg7, %10 : f64
// CHECK-NEXT:        memref.store %11, %arg0[%c0] : memref<10xf64>
// CHECK-NEXT:        scf.yield %11 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      %8 = arith.addi %arg4, %6 : index
// CHECK-NEXT:      scf.yield %8, %7 : index, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.addf %arg2, %cst : f64
// CHECK-NEXT:    %2:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %c4, %arg5 = %1) -> (index, f64) {
// CHECK-NEXT:      %3 = arith.subi %c9, %arg3 : index
// CHECK-NEXT:      %subview = memref.subview %alloc_1[%3, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %4 = arith.subi %arg4, %c1 : index
// CHECK-NEXT:      %5 = arith.subi %c10, %arg3 : index
// CHECK-NEXT:      %6 = memref.load %alloc[%4] : memref<4xf64>
// CHECK-NEXT:      %7 = memref.load %alloc_0[%4] : memref<4xindex>
// CHECK-NEXT:      %8:3 = scf.while (%arg6 = %7, %arg7 = %4, %arg8 = %6) : (index, index, f64) -> (index, index, f64) {
// CHECK-NEXT:        %19 = arith.addi %arg6, %c1 : index
// CHECK-NEXT:        %20 = arith.cmpi slt, %19, %5 : index
// CHECK-NEXT:        scf.condition(%20) %arg6, %arg7, %arg8 : index, index, f64
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg6: index, %arg7: index, %arg8: f64):
// CHECK-NEXT:        %19 = arith.subi %5, %arg6 : index
// CHECK-NEXT:        %20 = arith.subi %c4, %arg7 : index
// CHECK-NEXT:        %21 = arith.minui %20, %19 : index
// CHECK-NEXT:        %22 = enzyme.binomial_progress %19, %21 : index
// CHECK-NEXT:        memref.store %arg8, %alloc[%arg7] : memref<4xf64>
// CHECK-NEXT:        memref.store %arg6, %alloc_0[%arg7] : memref<4xindex>
// CHECK-NEXT:        %23 = arith.addi %arg6, %22 : index
// CHECK-NEXT:        %24 = arith.cmpi eq, %23, %5 : index
// CHECK-NEXT:        %25 = arith.subi %23, %c1 : index
// CHECK-NEXT:        %26 = arith.select %24, %25, %23 : index
// CHECK-NEXT:        %27 = scf.for %arg9 = %arg6 to %26 step %c1 iter_args(%arg10 = %arg8) -> (f64) {
// CHECK-NEXT:          %29 = memref.load %subview[%arg9] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:          %30 = arith.addf %arg10, %29 : f64
// CHECK-NEXT:          memref.store %30, %subview[%c0] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:          scf.yield %30 : f64
// CHECK-NEXT:        } {enzyme.disable_mincut = true}
// CHECK-NEXT:        %28 = arith.addi %arg7, %c1 : index
// CHECK-NEXT:        scf.yield %23, %28, %27 : index, index, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = arith.subi %c9, %arg3 : index
// CHECK-NEXT:      %10 = memref.load %subview[%9] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %11 = arith.addf %8#2, %10 : f64
// CHECK-NEXT:      memref.store %11, %subview[%c0] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %12 = arith.addf %arg5, %cst : f64
// CHECK-NEXT:      %13 = memref.load %arg1[%c0] : memref<10xf64>
// CHECK-NEXT:      %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:      memref.store %cst, %arg1[%c0] : memref<10xf64>
// CHECK-NEXT:      %15 = arith.addf %14, %cst : f64
// CHECK-NEXT:      %16 = arith.addf %14, %cst : f64
// CHECK-NEXT:      %17 = memref.load %arg1[%9] : memref<10xf64>
// CHECK-NEXT:      %18 = arith.addf %17, %16 : f64
// CHECK-NEXT:      memref.store %18, %arg1[%9] : memref<10xf64>
// CHECK-NEXT:      scf.yield %8#1, %15 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    memref.dealloc %alloc_1 : memref<4x10xf64>
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf64>
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<4xindex>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
