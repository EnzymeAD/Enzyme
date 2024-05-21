// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : tensor<2xf64>) -> tensor<2xf64> {
    %y = math.sin %x : tensor<2xf64>
    return %y : tensor<2xf64>
  }
  func.func @dsq(%x : tensor<2xf64>, %dx : tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>)
    return %r : tensor<2xf64>
  }
}

// CHECK:   func.func private @fwddiffesquare(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[a0:.+]] = math.cos %arg0 : tensor<2xf64>
// CHECK-NEXT:     %[[a1:.+]] = arith.mulf %arg1, %[[a0]] : tensor<2xf64>
// CHECK-NEXT:     %[[a2:.+]] = math.sin %arg0 : tensor<2xf64>
// CHECK-NEXT:     return %[[a1]] : tensor<2xf64>
// CHECK-NEXT:   }
