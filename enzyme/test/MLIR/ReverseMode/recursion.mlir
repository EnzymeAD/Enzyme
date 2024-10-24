// RUN: %eopt --enzyme-wrap="infn=f outfn=f_rev retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

module {
  func.func @f(%arg0: f32) -> f32 {
    %0 = call @g(%arg0) : (f32) -> f32
    return %0 : f32
  }

  func.func @g(%arg0: f32) -> f32 {
    %0 = call @f(%arg0) : (f32) -> f32
    return %0 : f32
  }
}

// CHECK:  func.func private @f_rev(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @g(%arg0) : (f32) -> f32
// CHECK-NEXT:    %1 = call @diffeg(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffeg(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @f(%arg0) : (f32) -> f32
// CHECK-NEXT:    %1 = call @diffef_0(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffef_0(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @g(%arg0) : (f32) -> f32
// CHECK-NEXT:    %1 = call @diffeg(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }
