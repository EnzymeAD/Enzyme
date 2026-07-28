// RUN: %eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops | FileCheck %s

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
     enzyme.checkpoint_period=4,
     enzyme.disable_mincut=true}

  return %sum : f64
}

// CHECK:  func.func @reduce_sum(%arg0: memref<10xf64>, %arg1: memref<10xf64>, %arg2: f64) {
// CHECK-DAG:    %c4 = arith.constant 4 : index
// CHECK-DAG:    %c9 = arith.constant 9 : index
// CHECK-DAG:    %c12 = arith.constant 12 : index
// CHECK-DAG:    %c3 = arith.constant 3 : index
// CHECK-DAG:    %c1 = arith.constant 1 : index
// CHECK-DAG:    %c0 = arith.constant 0 : index
// CHECK-DAG:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-DAG:    %alloc = memref.alloc() : memref<4x10xf64>
// CHECK-DAG:    %alloc_0 = memref.alloc() : memref<4xf64>
// CHECK:    %0 = scf.for %arg3 = %c0 to %c12 step %c3 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:      %3 = arith.divui %arg3, %c3 : index
// CHECK-NEXT:      %4 = arith.cmpi eq, %arg3, %c9 : index
// CHECK-NEXT:      %5 = arith.select %4, %c1, %c3 : index
// CHECK-NEXT:      %subview = memref.subview %alloc[%3, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %subview : memref<10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %6 = scf.for %arg5 = %c0 to %5 step %c1 iter_args(%arg6 = %arg4) -> (f64) {
// CHECK-NEXT:        %7 = arith.addi %arg3, %arg5 : index
// CHECK-NEXT:        %8 = memref.load %arg0[%7] : memref<10xf64>
// CHECK-NEXT:        %9 = arith.addf %arg6, %8 : f64
// CHECK-NEXT:        memref.store %9, %arg0[%c0] : memref<10xf64>
// CHECK-NEXT:        scf.yield %9 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%3] : memref<4xf64>
// CHECK-NEXT:      scf.yield %6 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %1 = arith.addf %arg2, %cst : f64
// CHECK-NEXT:    %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (f64) {
// CHECK-NEXT:      %3 = arith.subi %c3, %arg3 : index
// CHECK-NEXT:      %subview = memref.subview %alloc[%3, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %4 = arith.subi %c3, %arg3 : index
// CHECK-NEXT:      %5 = memref.load %alloc_0[%3] : memref<4xf64>
// CHECK-NEXT:      %6 = arith.cmpi eq, %arg3, %c0 : index
// CHECK-NEXT:      %7 = arith.select %6, %c1, %c3 : index
// CHECK-NEXT:      %8 = arith.muli %4, %c3 : index
// CHECK-NEXT:      %alloc_1 = memref.alloc(%7) : memref<?xmemref<10xf64>>
// CHECK-NEXT:      %alloc_2 = memref.alloc(%7) : memref<?xindex>
// CHECK-NEXT:      %alloc_3 = memref.alloc(%7) : memref<?xindex>
// CHECK-NEXT:      %9 = scf.for %arg5 = %c0 to %7 step %c1 iter_args(%arg6 = %5) -> (f64) {
// CHECK-NEXT:        %11 = arith.addi %8, %arg5 : index
// CHECK-NEXT:        enzyme.store %11, %alloc_2[%arg5] ([%7]) : memref<?xindex>
// CHECK-NEXT:        %12 = memref.load %subview[%11] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:        %13 = arith.addf %arg6, %12 : f64
// CHECK-NEXT:        enzyme.store %arg1, %alloc_1[%arg5] ([%7]) : memref<?xmemref<10xf64>>
// CHECK-NEXT:        enzyme.store %c0, %alloc_3[%arg5] ([%7]) : memref<?xindex>
// CHECK-NEXT:        memref.store %13, %subview[%c0] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:        scf.yield %13 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %10 = scf.for %arg5 = %c0 to %7 step %c1 iter_args(%arg6 = %arg4) -> (f64) {
// CHECK-NEXT:        %11 = arith.subi %7, %c1 : index
// CHECK-NEXT:        %12 = arith.subi %11, %arg5 : index
// CHECK-NEXT:        %13 = arith.addf %arg6, %cst : f64
// CHECK-NEXT:        %14 = enzyme.load %alloc_1[%12] ([%7]) : memref<?xmemref<10xf64>>
// CHECK-NEXT:        %15 = enzyme.load %alloc_3[%12] ([%7]) : memref<?xindex>
// CHECK-NEXT:        %16 = memref.load %14[%15] : memref<10xf64>
// CHECK-NEXT:        %17 = arith.addf %13, %16 : f64
// CHECK-NEXT:        memref.store %cst, %14[%15] : memref<10xf64>
// CHECK-NEXT:        %18 = arith.addf %17, %cst : f64
// CHECK-NEXT:        %19 = arith.addf %17, %cst : f64
// CHECK-NEXT:        %20 = enzyme.load %alloc_2[%12] ([%7]) : memref<?xindex>
// CHECK-NEXT:        %21 = memref.load %14[%20] : memref<10xf64>
// CHECK-NEXT:        %22 = arith.addf %21, %19 : f64
// CHECK-NEXT:        memref.store %22, %14[%20] : memref<10xf64>
// CHECK-NEXT:        scf.yield %18 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      memref.dealloc %alloc_3 : memref<?xindex>
// CHECK-NEXT:      memref.dealloc %alloc_2 : memref<?xindex>
// CHECK-NEXT:      memref.dealloc %alloc_1 : memref<?xmemref<10xf64>>
// CHECK-NEXT:      scf.yield %10 : f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<4xf64>
// CHECK-NEXT:    memref.dealloc %alloc : memref<4x10xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
