// RUN: %eopt %s -mem2reg | FileCheck %s

module {
  func.func @main(%arg0: f32) -> f32 {
    %0 = "enzyme.init"() : () -> !enzyme.Gradient<f32>
    "enzyme.set"(%0, %arg0) : (!enzyme.Gradient<f32>, f32) -> ()
    %2 = "enzyme.get"(%0) : (!enzyme.Gradient<f32>) -> f32
    return %2 : f32
  }
}

// CHECK:  func.func @main(%arg0: f32) -> f32 {
// CHECK-NEXT:    return %arg0 : f32
// CHECK-NEXT:  }
