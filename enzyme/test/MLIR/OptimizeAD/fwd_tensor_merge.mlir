// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : tensor<10xf64>) -> tensor<10xf64>{
    %y = arith.mulf %x, %x : tensor<10xf64>
    return %y : tensor<10xf64>
  }
  func.func @dsq(%x : tensor<10xf64>, %dx : tensor<10xf64>, %dx2 : tensor<10xf64>) -> (tensor<10xf64>,tensor<10xf64>) {
    %cst = arith.constant 54 : i64
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>)
    %r2 = enzyme.fwddiff @square(%x, %dx2) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>)
    return %r,%r2 : tensor<10xf64>,tensor<10xf64>
  }

  
  //CHECK: func.func @dsq(%arg0: tensor<10xf64>, %arg1: tensor<10xf64>, %arg2: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>) {
  //CHECK-NEXT:   %c54_i64 = arith.constant 54 : i64
  //CHECK-NEXT:   %concat = tensor.concat dim(0) %arg1, %arg2 : (tensor<10xf64>, tensor<10xf64>) -> tensor<20xf64>
  //CHECK-NEXT:   %0 = enzyme.fwddiff @square(%arg0, %concat) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>], width = 2 : i64} : (tensor<10xf64>, tensor<20xf64>) -> tensor<2x10xf64>
  //CHECK-NEXT:   %c0 = arith.constant 0 : index
  //CHECK-NEXT:   %extracted_slice = tensor.extract_slice %0[0, 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
  //CHECK-NEXT:   %c1 = arith.constant 1 : index
  //CHECK-NEXT:   %extracted_slice_0 = tensor.extract_slice %0[1, 0] [1, 10] [1, 1] : tensor<2x10xf64> to tensor<10xf64>
  //CHECK-NEXT:   return %extracted_slice, %extracted_slice_0 : tensor<10xf64>, tensor<10xf64>
}

