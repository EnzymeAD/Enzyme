// RUN: %eopt -enzyme-batch %s | FileCheck %s

module {
  func.func private @f(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = func.call @f(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
  func.func @main(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) {
    %0 = enzyme.batch @f(%arg0, %arg1) {batch_shape = array<i64: 4>} : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    return
  }
}

// CHECK:   func.func @main(%[[arg0:.+]]: tensor<4x16xf32>, %[[arg1:.+]]: tensor<4x16xf32>) {
// CHECK-NEXT:     %[[v0:.+]] = call @batched_f(%[[arg0]], %[[arg1]]) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func private @batched_f(%[[arg0:.+]]: tensor<4x16xf32>, %[[arg1:.+]]: tensor<4x16xf32>) -> tensor<4x16xf32> {
// CHECK-NEXT:     %[[v0:.+]] = call @batched_f(%[[arg0]], %[[arg1]]) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:     return %[[v0]] : tensor<4x16xf32>
// CHECK-NEXT:   }