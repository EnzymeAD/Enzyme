// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : tensor<10xf64>) -> tensor<10xf64>{
    %y = arith.mulf %x, %x : tensor<10xf64>
    return %y : tensor<10xf64>
  }
  func.func @dsq(%x : tensor<10xf64>, %dx : tensor<2x10xf64>) -> tensor<2x10xf64> {
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>], width=2 } : (tensor<10xf64>, tensor<2x10xf64>) -> (tensor<2x10xf64>)
    return %r : tensor<2x10xf64>
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: tensor<10xf64>, %[[arg1:.+]]: tensor<2x10xf64>) -> tensor<2x10xf64> {
// CHECK-NEXT:     %[[i0:.+]] = call @fwddiffe2square(%[[arg0]], %[[arg1]]) : (tensor<10xf64>, tensor<2x10xf64>) -> tensor<2x10xf64>
// CHECK-NEXT:     return %[[i0]] : tensor<2x10xf64>
// CHECK-NEXT:   }
// CHECK:   func.func private @fwddiffe2square(%[[arg0:.+]]: tensor<10xf64>, %[[arg1:.+]]: tensor<2x10xf64>) -> tensor<2x10xf64> {
// CHECK-NEXT:     %[[s0:.+]] = "enzyme.broadcast"(%%[[arg0]]) <{width = 2 : i64}> : (tensor<10xf64>) -> tensor<2x10xf64>
// CHECK-NEXT:     %[[i0:.+]] = arith.mulf %[[arg1]], %[[s0]] : tensor<2x10xf64>
// CHECK-NEXT:     %[[s1:.+]] = "enzyme.broadcast"(%%[[arg0]]) <{width = 2 : i64}> : (tensor<10xf64>) -> tensor<2x10xf64>
// CHECK-NEXT:     %[[i1:.+]] = arith.mulf %[[arg1]], %[[s1]] : tensor<2x10xf64>
// CHECK-NEXT:     %[[i2:.+]] = arith.addf %[[i0]], %[[i1]] : tensor<2x10xf64>
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg0]], %[[arg0]] : tensor<10xf64>
// CHECK-NEXT:     return %[[i2]] : tensor<2x10xf64>
// CHECK-NEXT:   }