// RUN: %eopt --split-input-file --enzyme-diff-batch %s | FileCheck %s

//1. Scalar test
module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @test1(%x : f64, %dr1 : f64, %dr2 : f64) -> (f64,f64) {
    %r, %dx1 = enzyme.autodiff @square(%x, %dr1) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>] } : (f64, f64) -> (f64,f64) 
    %r2, %dx2 = enzyme.autodiff @square(%x, %dr2) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>] } : (f64, f64) -> (f64,f64) 
    return %dx1,%dx2 : f64, f64
  }
}

// CHECK-LABEL: func.func @test1
// CHECK-SAME: (%[[PRIMAL:.*]]: f64, %[[DIFF1:.*]]: f64, %[[DIFF2:.*]]: f64) -> (f64, f64)
// CHECK:         %[[CONCAT:.*]] = enzyme.concat(%[[DIFF1]], %[[DIFF2]]) : (f64, f64) -> tensor<2xf64>
// CHECK:         %[[BATCHED_RES_BASE:.*]]:2 = enzyme.autodiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (f64, tensor<2xf64>) -> (f64, tensor<2xf64>)
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[RES0:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[%[[C0]]] : (tensor<2xf64>) -> f64
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[RES1:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[%[[C1]]] : (tensor<2xf64>) -> f64
// CHECK-NEXT:    return %[[RES0]], %[[RES1]]

// -----

//2. Tensor test
module {
  func.func @square(%x : tensor<10xf64>) -> tensor<10xf64>{
    %y = arith.mulf %x, %x : tensor<10xf64>
    return %y : tensor<10xf64>
  }
  func.func @test2(%x : tensor<10xf64>, %dr1 : tensor<10xf64>, %dr2 : tensor<10xf64>) -> (tensor<10xf64>,tensor<10xf64>) {
    %r, %dx1 = enzyme.autodiff @square(%x, %dr1) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    %r2, %dx2 = enzyme.autodiff @square(%x, %dr2) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    return %dx1,%dx2 : tensor<10xf64>,tensor<10xf64>
  }
}


// CHECK-LABEL: func.func @test2
// CHECK-SAME: (%[[PRIMAL:.*]]: tensor<10xf64>, %[[DIFF1:.*]]: tensor<10xf64>, %[[DIFF2:.*]]: tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
// CHECK:         %[[CONCAT:.*]] = enzyme.concat(%[[DIFF1]], %[[DIFF2]]) : (tensor<10xf64>, tensor<10xf64>) -> tensor<2x10xf64>
// CHECK:         %[[BATCHED_RES_BASE:.*]]:2 = enzyme.autodiff @square(%[[PRIMAL]], %[[CONCAT]]) {{.*}} width = 2 {{.*}} : (tensor<10xf64>, tensor<2x10xf64>) -> (tensor<10xf64>, tensor<2x10xf64>)
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[RES0:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[%[[C0]]] : (tensor<2x10xf64>) -> tensor<10xf64>
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[RES1:.*]] = enzyme.extract %[[BATCHED_RES_BASE]]#1[%[[C1]]] : (tensor<2x10xf64>) -> tensor<10xf64>
// CHECK-NEXT:    return %[[RES0]], %[[RES1]]

