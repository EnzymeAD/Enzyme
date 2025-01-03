// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math %s | FileCheck %s

module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>], width=2 } : (f64, tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}

// CHECK:  func.func @dsquare(%arg0: f64, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %0 = call @diffesquare(%arg0, %arg1) : (f64, tensor<2xf64>) -> tensor<2xf64>
// CHECK-NEXT:    return %0 : tensor<2xf64>
// CHECK-NEXT:  }

// CHECK:  func.func private @diffe2square(%arg0: f64, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %0 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %1 = arith.mulf %arg1, %0 : tensor<2xf64>
// CHECK-NEXT:    %2 = "enzyme.broadcast"(%arg0) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %3 = arith.mulf %arg1, %2 : tensor<2xf64>
// CHECK-NEXT:    %4 = arith.addf %1, %3 : tensor<2xf64>
// CHECK-NEXT:    return %4 : tensor<2xf64>
// CHECK-NEXT:  }
