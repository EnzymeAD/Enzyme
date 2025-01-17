// RUN: %eopt -enzyme-batch %s | FileCheck %s

module {
  func.func private @g(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    return %arg0 : tensor<16xf32>
  }
  func.func private @f(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %1 = func.call @g(%arg0) : (tensor<16xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  func.func @main(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) {
    %2 = enzyme.batch @f(%arg0, %arg1) {batch_shape = array<i64: 4>} : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    return
  }
}

// CHECK:   func.func @main(%[[arg0:.+]]: tensor<4x16xf32>, %[[arg1:.+]]: tensor<4x16xf32>) {
// CHECK-NEXT:     %[[v0:.+]] = call @batched_f(%[[arg0]], %[[arg1]]) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func private @batched_f(%[[arg0:.+]]: tensor<4x16xf32>, %[[arg1:.+]]: tensor<4x16xf32>) -> tensor<4x16xf32> {
// CHECK-NEXT:     %[[v0:.+]] = call @batched_g(%[[arg0]]) : (tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:     return %[[v0]] : tensor<4x16xf32>
// CHECK-NEXT:   }
// CHECK:   func.func private @batched_g(%[[arg0:.+]]: tensor<4x16xf32>) -> tensor<4x16xf32> {
// CHECK-NEXT:     return %[[arg0]] : tensor<4x16xf32>
// CHECK-NEXT:   }

