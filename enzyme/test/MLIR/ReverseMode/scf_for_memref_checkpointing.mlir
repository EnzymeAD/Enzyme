// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {

  func.func @main(%m: memref<f32>) -> f32 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %one = arith.constant 1 : index

    %init = arith.constant 0.0 : f32
    %0 = scf.for %i = %lb to %ub step %one iter_args(%it = %init) -> (f32) {
      %val = memref.load %m[] : memref<f32>
      %mul = arith.mulf %val, %it : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true}

    return %0 : f32
  }

}

// CHECK:  func.func @main(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: f32) {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %alloc = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %0 = scf.for %arg3 = %c0 to %c9 step %c3 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:      %2 = arith.divui %arg3, %c3 : index
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%2] : memref<3xf32>
// CHECK-NEXT:      %subview = memref.subview %alloc[%2] [1] [1] : memref<3xf32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:      memref.copy %arg0, %subview : memref<f32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:      %3 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg4) -> (f32) {
// CHECK-NEXT:        %4 = memref.load %arg0[] : memref<f32>
// CHECK-NEXT:        %5 = arith.mulf %4, %arg6 : f32
// CHECK-NEXT:        scf.yield %5 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %3 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:4 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %arg2, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32, f32) {
// CHECK-NEXT:      %2 = arith.subi %c2, %arg3 : index
// CHECK-NEXT:      %subview = memref.subview %alloc[%2] [1] [1] : memref<3xf32> to memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:      %3 = memref.load %alloc_0[%2] : memref<3xf32>
// CHECK-NEXT:      %alloc_1 = memref.alloc() : memref<3xf32>
// CHECK-NEXT:      %alloc_2 = memref.alloc() : memref<3xf32>
// CHECK-NEXT:      %4 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %3) -> (f32) {
// CHECK-NEXT:        memref.store %arg9, %alloc_2[%arg8] : memref<3xf32>
// CHECK-NEXT:        %6 = memref.load %subview[] : memref<f32, strided<[], offset: ?>>
// CHECK-NEXT:        memref.store %6, %alloc_1[%arg8] : memref<3xf32>
// CHECK-NEXT:        %7 = arith.mulf %6, %arg9 : f32
// CHECK-NEXT:        scf.yield %7 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %5:4 = scf.for %arg8 = %c0 to %c3 step %c1 iter_args(%arg9 = %arg4, %arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7) -> (f32, f32, f32, f32) {
// CHECK-NEXT:        %6 = arith.subi %c2, %arg8 : index
// CHECK-NEXT:        %7 = memref.load %alloc_1[%6] : memref<3xf32>
// CHECK-NEXT:        %8 = memref.load %alloc_2[%6] : memref<3xf32>
// CHECK-NEXT:        %9 = arith.addf %arg10, %arg9 : f32
// CHECK-NEXT:        %10 = arith.mulf %9, %8 : f32
// CHECK-NEXT:        %11 = arith.addf %arg11, %10 : f32
// CHECK-NEXT:        %12 = arith.mulf %9, %7 : f32
// CHECK-NEXT:        %13 = arith.addf %arg12, %12 : f32
// CHECK-NEXT:        %14 = memref.load %arg1[] : memref<f32>
// CHECK-NEXT:        %15 = arith.addf %14, %11 : f32
// CHECK-NEXT:        memref.store %15, %arg1[] : memref<f32>
// CHECK-NEXT:        scf.yield %13, %cst, %11, %cst : f32, f32, f32, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %alloc_2 : memref<3xf32>
// CHECK-NEXT:      memref.dealloc %alloc_1 : memref<3xf32>
// CHECK-NEXT:      scf.yield %5#0, %5#1, %5#2, %5#3 : f32, f32, f32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<3xf32>
// CHECK-NEXT:    memref.dealloc %alloc : memref<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
