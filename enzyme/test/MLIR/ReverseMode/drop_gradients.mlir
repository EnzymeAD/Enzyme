// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    %dropped = enzyme.ignore_derivatives %next : f64 -> f64
    return %dropped : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:   func.func @square(%arg0: f64) -> f64 {
// CHECK-NEXT:     %0 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:     return %0 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @dsquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:     %0 = call @diffesquare(%arg0, %arg1) : (f64, f64) -> f64
// CHECK-NEXT:     return %0 : f64
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     return %cst : f64
// CHECK-NEXT:   }

// -----

module {
  func.func @main(%arg0: tensor<2xf64> {tf.aliasing_output = 1 : i32}) -> (tensor<2xf64>) {
    %0 = enzyme.ignore_derivatives %arg0 : tensor<2xf64> -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}

// CHECK: func.func @main(%arg0: tensor<2xf64> {tf.aliasing_output = 1 : i32}) -> tensor<2xf64> {
// CHECK-NEXT:   return %arg0 : tensor<2xf64>
// CHECK-NEXT: }
