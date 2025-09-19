// RUN: %eopt --enzyme-diff-batch %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : f64, %dx1 : f64, %dx2 : f64) -> (f64,f64) {
    %r1 = enzyme.fwddiff @square(%x, %dx1) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>]} : (f64, f64) -> (f64)
    %r2 = enzyme.fwddiff @square(%x, %dx2) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r1,%r2 : f64, f64
  }

  // CHECK: func.func @dsq(%arg0: f64, %arg1: f64, %arg2: f64) -> (f64, f64) {
  // CHECK-NEXT:     %from_elements = tensor.from_elements %arg1, %arg2 : tensor<2xf64>
  // CHECK-NEXT:     %0 = enzyme.fwddiff @square(%arg0, %from_elements) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>], width = 2 : i64} : (f64, tensor<2xf64>) -> tensor<2xf64>
  // CHECK-NEXT:     %c0 = arith.constant 0 : index
  // CHECK-NEXT:     %extracted = tensor.extract %0[%c0] : tensor<2xf64>
  // CHECK-NEXT:     %c1 = arith.constant 1 : index
  // CHECK-NEXT:     %extracted_0 = tensor.extract %0[%c1] : tensor<2xf64>
  // CHECK-NEXT:     return %extracted, %extracted_0 : f64, f64
  // CHECK-NEXT:   }
}

