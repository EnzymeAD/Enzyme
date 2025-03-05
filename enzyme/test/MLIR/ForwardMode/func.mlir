// RUN: %eopt --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" %s | FileCheck %s

module {
  func.func @square(%arg0: f32) -> f32 {
    %0 = arith.mulf %arg0, %arg0 : f32
    return %0 : f32
  }

  func.func @main(%arg0: f32) -> f32 {
    %0 = func.call @square(%arg0) : (f32) -> f32
    return %0 : f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> (f32, f32) {
// CHECK-NEXT:    %0:2 = call @fwddiffesquare(%arg0, %arg1) : (f32, f32) -> (f32, f32)
// CHECK-NEXT:    return %0#0, %0#1 : f32, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @fwddiffesquare(%arg0: f32, %arg1: f32) -> (f32, f32) {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg0 : f32
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg0 : f32
// CHECK-NEXT:    %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:    %3 = arith.mulf %arg0, %arg0 : f32
// CHECK-NEXT:    return %3, %2 : f32, f32
// CHECK-NEXT:  }
