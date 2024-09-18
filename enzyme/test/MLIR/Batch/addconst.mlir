// RUN: %eopt --enzyme-batch %s | FileCheck %s

module {
  func.func @square(%x : tensor<3xf64>) -> tensor<3xf64> {
    %cst = arith.constant dense<2.1> : tensor<3xf64>
    %y = arith.addf %x, %cst : tensor<3xf64>
    return %y : tensor<3xf64>
  }
  func.func @dsq(%x : tensor<10x2x3xf64>) -> tensor<10x2x3xf64> {
    %r = enzyme.batch @square(%x) { batch_shape=array<i64: 10, 2> } : (tensor<10x2x3xf64>) -> (tensor<10x2x3xf64>)
    return %r : tensor<10x2x3xf64>
  }
}

// CHECK:  func.func private @batched_square(%arg0: tensor<10x2x3xf64>) -> tensor<10x2x3xf64>
// CHECK-NEXT:    %cst = arith.constant dense<2.100000e+00> : tensor<10x2x3xf64>
// CHECK-NEXT:    %0 = arith.addf %arg0, %cst : tensor<10x2x3xf64>
// CHECK-NEXT:    return %0 : tensor<10x2x3xf64>
// CHECK-NEXT:  }
