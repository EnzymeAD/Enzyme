// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %c1_i64 = arith.constant 1 : i64
    %tmp = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr<f64>
    %y = arith.mulf %x, %x : f64
    llvm.store %y, %tmp : !llvm.ptr<f64>
    %r = llvm.load %tmp : !llvm.ptr<f64>
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x f64 : (i64) -> !llvm.ptr<f64>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x f64 : (i64) -> !llvm.ptr<f64>
// CHECK-NEXT:     %2 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:     %3 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:     %4 = arith.addf %2, %3 : f64
// CHECK-NEXT:     %5 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:     llvm.store %4, %0 : !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %5, %1 : !llvm.ptr<f64>
// CHECK-NEXT:     %6 = llvm.load %0 : !llvm.ptr<f64>
// CHECK-NEXT:     %7 = llvm.load %1 : !llvm.ptr<f64>
// CHECK-NEXT:     return %6 : f64
// CHECK-NEXT:   }
