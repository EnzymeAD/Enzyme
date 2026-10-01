// RUN: %eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops | FileCheck %s

// The period is a budget on the number of checkpoints, so 10 iterations with a
// period of 4 are cut into 4 segments of ceil(10/4) = 3 (the last one short, at
// 1) rather than into ceil(10/4) = 3 segments of 4. Both the checkpoint buffer
// and the clone buffer beside it are therefore memref<4x...>.

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
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4x10xf64>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<4xf64>
// CHECK-NEXT:    %0 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %cst) -> (f64) {
// CHECK-NEXT:      %2 = arith.muli %arg3, %c3 : index
// CHECK-NEXT:      %3 = arith.cmpi eq, %2, %c9 : index
// CHECK-NEXT:      %4 = arith.select %3, %c1, %c3 : index
// CHECK-NEXT:      %subview = memref.subview %alloc[%arg3, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %subview : memref<10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %5 = scf.for %arg5 = %c0 to %4 step %c1 iter_args(%arg6 = %arg4) -> (f64) {
// CHECK-NEXT:        %6 = arith.addi %2, %arg5 : index
// CHECK-NEXT:        %7 = memref.load %arg0[%6] : memref<10xf64>
// CHECK-NEXT:        %8 = arith.addf %arg6, %7 : f64
// CHECK-NEXT:        memref.store %8, %arg0[%c0] : memref<10xf64>
// CHECK-NEXT:        scf.yield %8 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%arg3] : memref<4xf64>
// CHECK-NEXT:      scf.yield %5 : f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg2) -> (f64) {
// CHECK-NEXT:      %2 = arith.subi %c3, %arg3 : index
// CHECK-NEXT:      %subview = memref.subview %alloc[%2, 0] [1, 10] [1, 1] : memref<4x10xf64> to memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:      %3 = arith.subi %c3, %arg3 : index
// CHECK-NEXT:      %4 = arith.muli %3, %c3 : index
// CHECK-NEXT:      %5 = arith.cmpi eq, %4, %c9 : index
// CHECK-NEXT:      %6 = arith.select %5, %c1, %c3 : index
// CHECK-NEXT:      %7 = memref.load %alloc_0[%2] : memref<4xf64>
// CHECK-NEXT:      %alloc_1 = memref.alloc(%6) : memref<?xmemref<10xf64>>
// CHECK-NEXT:      %alloc_2 = memref.alloc(%6) : memref<?xindex>
// CHECK-NEXT:      %alloc_3 = memref.alloc(%6) : memref<?xindex>
// CHECK-NEXT:      %8 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %7) -> (f64) {
// CHECK-NEXT:        %10 = arith.addi %4, %arg5 : index
// CHECK-NEXT:        enzyme.store %10, %alloc_2[%arg5] ([%6]) : memref<?xindex>
// CHECK-NEXT:        %11 = memref.load %subview[%10] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:        %12 = arith.addf %arg6, %11 : f64
// CHECK-NEXT:        enzyme.store %arg1, %alloc_1[%arg5] ([%6]) : memref<?xmemref<10xf64>>
// CHECK-NEXT:        enzyme.store %c0, %alloc_3[%arg5] ([%6]) : memref<?xindex>
// CHECK-NEXT:        memref.store %12, %subview[%c0] : memref<10xf64, strided<[1], offset: ?>>
// CHECK-NEXT:        scf.yield %12 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      %9 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %arg4) -> (f64) {
// CHECK-NEXT:        %10 = arith.subi %6, %c1 : index
// CHECK-NEXT:        %11 = arith.subi %10, %arg5 : index
// CHECK-NEXT:        %12 = enzyme.load %alloc_1[%11] ([%6]) : memref<?xmemref<10xf64>>
// CHECK-NEXT:        %13 = enzyme.load %alloc_3[%11] ([%6]) : memref<?xindex>
// CHECK-NEXT:        %14 = memref.load %12[%13] : memref<10xf64>
// CHECK-NEXT:        %15 = arith.addf %arg6, %14 fastmath<fast> : f64
// CHECK-NEXT:        memref.store %cst, %12[%13] : memref<10xf64>
// CHECK-NEXT:        %16 = enzyme.load %alloc_2[%11] ([%6]) : memref<?xindex>
// CHECK-NEXT:        %17 = memref.load %12[%16] : memref<10xf64>
// CHECK-NEXT:        %18 = arith.addf %17, %15 fastmath<fast> : f64
// CHECK-NEXT:        memref.store %18, %12[%16] : memref<10xf64>
// CHECK-NEXT:        scf.yield %15 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      memref.dealloc %alloc_3 : memref<?xindex>
// CHECK-NEXT:      memref.dealloc %alloc_2 : memref<?xindex>
// CHECK-NEXT:      memref.dealloc %alloc_1 : memref<?xmemref<10xf64>>
// CHECK-NEXT:      scf.yield %9 : f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<4xf64>
// CHECK-NEXT:    memref.dealloc %alloc : memref<4x10xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
