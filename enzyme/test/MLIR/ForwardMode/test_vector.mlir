// RUN: %eopt --enzyme %s | FileCheck %s

// TODO: actually test

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : tensor<2xf64>, %dx : tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>], width=2 } : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>)
    return %r : tensor<2xf64>
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: tensor<2xf64>, %[[arg1:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[i0:.+]] = call @fwddiffe2square(%[[arg0]], %[[arg1]]) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:     return %[[i0]] : tensor<2xf64>
// CHECK-NEXT:   }
// CHECK:   func.func private @fwddiffe2square(%[[arg0:.+]]: tensor<2xf64>, %[[arg1:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[i0:.+]] = arith.mulf %[[arg1]], %[[arg0]] : tensor<2xf64>
// CHECK-NEXT:     %[[i1:.+]] = arith.mulf %[[arg1]], %[[arg0]] : tensor<2xf64>
// CHECK-NEXT:     %[[i2:.+]] = arith.addf %[[i0]], %[[i1]] : tensor<2xf64>
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg0]], %[[arg0]] : tensor<2xf64>
// CHECK-NEXT:     return %[[i2]] : tensor<2xf64>
// CHECK-NEXT:   }