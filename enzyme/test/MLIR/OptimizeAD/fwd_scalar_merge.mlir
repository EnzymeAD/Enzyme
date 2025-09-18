// RUN: %eopt --enzyme-diff-batch %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @dsq(%x : f64, %dx1 : f64, %dx2 : f64) -> (f64,f64) {
    %r1 = enzyme.fwddiff @square(%x, %dx1) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>]} : (f64, f64) -> (f64)
    %r2 = enzyme.fwddiff @square(%x, %dx2) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r1,%r2 : (f64,f64)
  }
}

// CHECK:   func.func @dsq(%[[arg0:.+]]: f64, %[[arg1:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[i0:.+]] = call @fwddiffe2square(%[[arg0]], %[[arg1]]) : (f64, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:     return %[[i0]] : tensor<2xf64>
// CHECK-NEXT:   }
// CHECK:   func.func private @fwddiffe2square(%[[arg0:.+]]: f64, %[[arg1:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[s0:.+]] = "enzyme.broadcast"(%[[arg0]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:     %[[i0:.+]] = arith.mulf %[[arg1]], %[[s0]] : tensor<2xf64>
// CHECK-NEXT:     %[[s1:.+]] = "enzyme.broadcast"(%[[arg0]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:     %[[i1:.+]] = arith.mulf %[[arg1]], %[[s1]] : tensor<2xf64>
// CHECK-NEXT:     %[[i2:.+]] = arith.addf %[[i0]], %[[i1]] : tensor<2xf64>
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg0]], %[[arg0]] 
// CHECK-NEXT:     return %[[i2]] : tensor<2xf64>
// CHECK-NEXT:   }
