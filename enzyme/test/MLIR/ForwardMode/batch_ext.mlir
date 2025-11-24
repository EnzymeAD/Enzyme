// RUN: %eopt --enzyme %s | FileCheck %s

module {

  func.func @ext(%x : f32) -> f64 {
    %res = arith.extf %x : f32 to f64
    return %res : f64
  }
  func.func @dif_ext(%x : f32, %dx : tensor<2xf32>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @ext(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>], width=2 } : (f32, tensor<2xf32>) -> (tensor<2xf64>)
    return %r : tensor<2xf64>
  }

  // CHECK: func.func private @fwddiffe2ext(%arg0: f32, %arg1: tensor<2xf32>) -> tensor<2xf64> {
  // CHECK-NEXT:   %0 = enzyme.extract %arg1[0] : (tensor<2xf32>) -> f32
  // CHECK-NEXT:   %1 = arith.extf %0 : f32 to f64
  // CHECK-NEXT:   %2 = enzyme.extract %arg1[1] : (tensor<2xf32>) -> f32
  // CHECK-NEXT:   %3 = arith.extf %2 : f32 to f64
  // CHECK-NEXT:   %4 = enzyme.concat(%1, %3) : (f64, f64) -> tensor<2xf64>
  // CHECK-NEXT:   %5 = arith.extf %arg0 : f32 to f64
  // CHECK-NEXT:   return %4 : tensor<2xf64>
  // CHECK-NEXT: }
}
