// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %c0 = arith.constant 0 : index
    %tmp = memref.alloc() : memref<1xf64>
    %y = arith.mulf %x, %x : f64
    memref.store %y, %tmp[%c0] : memref<1xf64>
    %r = memref.load %tmp[%c0] : memref<1xf64>
    return %r : f64
  }
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = memref.alloc() : memref<1xf64>
// CHECK-NEXT:     %1 = memref.alloc() : memref<1xf64>
// CHECK-NEXT:     %2 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:     %3 = arith.mulf %arg1, %arg0 : f64
// CHECK-NEXT:     %4 = arith.addf %2, %3 : f64
// CHECK-NEXT:     %5 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:     memref.store %4, %0[%c0] : memref<1xf64>
// CHECK-NEXT:     memref.store %5, %1[%c0] : memref<1xf64>
// CHECK-NEXT:     %6 = memref.load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %7 = memref.load %1[%c0] : memref<1xf64>
// CHECK-NEXT:     return %6 : f64
// CHECK-NEXT:   }
