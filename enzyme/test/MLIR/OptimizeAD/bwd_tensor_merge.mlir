// RUN: %eopt --enzyme-diff-batch %s | FileCheck %s

module {
  func.func @square(%x : tensor<10xf64>) -> tensor<10xf64>{
    %y = arith.mulf %x, %x : tensor<10xf64>
    return %y : tensor<10xf64>
  }
  func.func @dsq(%x : tensor<10xf64>, %dr1 : tensor<10xf64>, %dr2 : tensor<10xf64>) -> (tensor<10xf64>,tensor<10xf64>) {
    %r, %dx1 = enzyme.autodiff @square(%x, %dr1) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    %r2, %dx2 = enzyme.autodiff @square(%x, %dr2) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    return %dx1,%dx2 : tensor<10xf64>,tensor<10xf64>
  }


  //CHECK: func.func @dsq(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>, %arg2: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>) {
  //CHECK-NEXT:   %concat = tensor.concat dim(0) %arg1, %arg2 : (tensor<10xf64>, tensor<10xf64>) -> tensor<20xf64>
  //CHECK-NEXT:   %0:2 = enzyme.autodiff @square(%arg0, %concat) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>], width = 2 : i64} : (tensor<10xf64>, tensor<20xf64>) -> (tensor<10xf64>, tensor<2x10xf64>)
  //CHECK-NEXT:   %extracted_slice = tensor.extract_slice %0#1[0, 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
  //CHECK-NEXT:   %extracted_slice_0 = tensor.extract_slice %0#1[1, 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
  //CHECK-NEXT:   return %extracted_slice, %extracted_slice_0 : tensor<10xf64>, tensor<10xf64>
  //CHECK-NEXT: }
}

