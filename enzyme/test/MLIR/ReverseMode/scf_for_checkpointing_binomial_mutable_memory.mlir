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
// CHECK-NEXT:      %2 = arith.subi %c10, %arg4 : index
// CHECK-NEXT:      %3 = arith.subi %c4, %arg3 : index
// CHECK-NEXT:      %4 = arith.minui %3, %2 : index
// CHECK-NEXT:      %5 = enzyme.binomial_progress %2, %4 : index
// CHECK-NEXT:      %subview = memref.subview %alloc_1[%arg3, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %subview : memref<10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %6 = scf.for %arg6 = %c0 to %5 step %c1 iter_args(%arg7 = %arg5) -> (f64) {
// CHECK-NEXT:        %8 = arith.addi %arg4, %arg6 : index
// CHECK-NEXT:        %9 = memref.load %arg0[%8] : memref<10xf64>
// CHECK-NEXT:        %10 = arith.addf %arg7, %9 : f64
// CHECK-NEXT:        memref.store %10, %arg0[%c0] : memref<10xf64>
// CHECK-NEXT:        scf.yield %10 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      %7 = arith.addi %arg4, %5 : index
// CHECK-NEXT:      scf.yield %7, %6 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    %alloc_2 = memref.alloc() : memref<10xf64>
// CHECK-NEXT:    memref.copy %arg0, %alloc_2 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:    %1:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %c4, %arg5 = %arg2) -> (index, f64) {
// CHECK-NEXT:      %2 = arith.subi %arg4, %c1 : index
// CHECK-NEXT:      %3 = arith.subi %c10, %arg3 : index
// CHECK-NEXT:      %4 = memref.load %alloc[%2] : memref<4xf64>
// CHECK-NEXT:      %5 = memref.load %alloc_0[%2] : memref<4xindex>
// CHECK-NEXT:      %subview = memref.subview %alloc_1[%2, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      memref.copy %subview, %alloc_2 : memref<10xf64, strided<[1], offset: ?>> to memref<10xf64>
// CHECK-NEXT:      %6:3 = scf.while (%arg6 = %5, %arg7 = %2, %arg8 = %4) : (index, index, f64) -> (index, index, f64) {
// CHECK-NEXT:        %14 = arith.addi %arg6, %c1 : index
// CHECK-NEXT:        %15 = arith.cmpi slt, %14, %3 : index
// CHECK-NEXT:        scf.condition(%15) %arg6, %arg7, %arg8 : index, index, f64
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg6: index, %arg7: index, %arg8: f64):
// CHECK-NEXT:        %14 = arith.subi %3, %arg6 : index
// CHECK-NEXT:        %15 = arith.subi %c4, %arg7 : index
// CHECK-NEXT:        %16 = arith.minui %15, %14 : index
// CHECK-NEXT:        %17 = enzyme.binomial_progress %14, %16 : index
// CHECK-NEXT:        memref.store %arg8, %alloc[%arg7] : memref<4xf64>
// CHECK-NEXT:        memref.store %arg6, %alloc_0[%arg7] : memref<4xindex>
// CHECK-NEXT:        %subview_3 = memref.subview %alloc_1[%arg7, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:        memref.copy %alloc_2, %subview_3 : memref<10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:        %18 = arith.addi %arg6, %17 : index
// CHECK-NEXT:        %19 = arith.cmpi eq, %18, %3 : index
// CHECK-NEXT:        %20 = arith.subi %18, %c1 : index
// CHECK-NEXT:        %21 = arith.select %19, %20, %18 : index
// CHECK-NEXT:        %22 = scf.for %arg9 = %arg6 to %21 step %c1 iter_args(%arg10 = %arg8) -> (f64) {
// CHECK-NEXT:          %24 = memref.load %alloc_2[%arg9] : memref<10xf64>
// CHECK-NEXT:          %25 = arith.addf %arg10, %24 : f64
// CHECK-NEXT:          memref.store %25, %alloc_2[%c0] : memref<10xf64>
// CHECK-NEXT:          scf.yield %25 : f64
// CHECK-NEXT:        } {enzyme.disable_mincut = true}
// CHECK-NEXT:        %23 = arith.addi %arg7, %c1 : index
// CHECK-NEXT:        scf.yield %18, %23, %22 : index, index, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %7 = arith.subi %c9, %arg3 : index
// CHECK-NEXT:      %8 = memref.load %alloc_2[%7] : memref<10xf64>
// CHECK-NEXT:      %9 = arith.addf %6#2, %8 : f64
// CHECK-NEXT:      memref.store %9, %alloc_2[%c0] : memref<10xf64>
// CHECK-NEXT:      %10 = memref.load %arg1[%c0] : memref<10xf64>
// CHECK-NEXT:      %11 = arith.addf %arg5, %10 fastmath<fast> : f64
// CHECK-NEXT:      memref.store %cst, %arg1[%c0] : memref<10xf64>
// CHECK-NEXT:      %12 = memref.load %arg1[%7] : memref<10xf64>
// CHECK-NEXT:      %13 = arith.addf %12, %11 fastmath<fast> : f64
// CHECK-NEXT:      memref.store %13, %arg1[%7] : memref<10xf64>
// CHECK-NEXT:      scf.yield %6#1, %11 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf64>
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<4xindex>
// CHECK-NEXT:    memref.dealloc %alloc_2 : memref<10xf64>
// CHECK-NEXT:    memref.dealloc %alloc_1 : memref<4x10xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
