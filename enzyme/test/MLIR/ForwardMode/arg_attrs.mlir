// RUN: %eopt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" | FileCheck %s

module {
  func.func @f(%b: f32, %a: memref<f32> {tt.divisibility = 16 : i32, another.attribute}) -> f32 {
    %0 = memref.load %a[] : memref<f32>
    return %0 : f32
  }
}

// CHECK:  func.func @f(%arg0: f32, %arg1: f32, %arg2: memref<f32> {another.attribute, tt.divisibility = 16 : i32}, %arg3: memref<f32> {tt.divisibility = 16 : i32}) -> (f32, f32) {
// CHECK-NEXT:    %0 = memref.load %arg3[] : memref<f32>
// CHECK-NEXT:    %1 = memref.load %arg2[] : memref<f32>
// CHECK-NEXT:    return %1, %0 : f32, f32
// CHECK-NEXT:  }
