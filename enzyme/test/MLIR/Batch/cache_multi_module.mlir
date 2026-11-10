// RUN: %eopt -mlir-disable-threading --pass-pipeline="builtin.module(builtin.module(enzyme-batch))" %s | FileCheck %s
// RUN: %eopt --pass-pipeline="builtin.module(builtin.module(enzyme-batch))" %s | FileCheck %s

module {
  module @first {
    func.func @f(%x: tensor<3xf64>) -> tensor<3xf64> {
      %y = math.sin %x : tensor<3xf64>
      return %y : tensor<3xf64>
    }
    func.func @caller(%x: tensor<4x3xf64>) -> tensor<4x3xf64> {
      %0 = enzyme.batch @f(%x) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
      return %0 : tensor<4x3xf64>
    }
  }

  module @second {
    func.func @f(%x: tensor<5xf32>) -> tensor<5xf32> {
      %y = arith.mulf %x, %x : tensor<5xf32>
      return %y : tensor<5xf32>
    }
    func.func @caller(%x: tensor<4x5xf32>, %z: tensor<4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>) {
      %0 = enzyme.batch @f(%x) <{batch_shape = array<i64: 4>}> : (tensor<4x5xf32>) -> tensor<4x5xf32>
      %1 = enzyme.batch @f(%z) <{batch_shape = array<i64: 4>}> : (tensor<4x5xf32>) -> tensor<4x5xf32>
      return %0, %1 : tensor<4x5xf32>, tensor<4x5xf32>
    }
  }
}

// CHECK-LABEL: module @first {
// CHECK:         func.func @caller(%[[ARG0:.+]]: tensor<4x3xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:      %[[CALL0:.+]] = call @batched_f(%[[ARG0]]) : (tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:      return %[[CALL0]] : tensor<4x3xf64>
// CHECK-NEXT:    }
// CHECK:         func.func private @batched_f(%[[BF0:.+]]: tensor<4x3xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:      %[[SIN:.+]] = math.sin %[[BF0]] : tensor<4x3xf64>
// CHECK-NEXT:      return %[[SIN]] : tensor<4x3xf64>
// CHECK-NEXT:    }
// CHECK-NOT:     @batched_f_0

// CHECK-LABEL: module @second {
// CHECK:         func.func @caller(%[[SARG0:.+]]: tensor<4x5xf32>, %[[SARG1:.+]]: tensor<4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>) {
// CHECK-NEXT:      %[[SCALL0:.+]] = call @batched_f(%[[SARG0]]) : (tensor<4x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:      %[[SCALL1:.+]] = call @batched_f(%[[SARG1]]) : (tensor<4x5xf32>) -> tensor<4x5xf32>
// CHECK-NEXT:      return %[[SCALL0]], %[[SCALL1]] : tensor<4x5xf32>, tensor<4x5xf32>
// CHECK-NEXT:    }
// CHECK:         func.func private @batched_f(%[[BF1:.+]]: tensor<4x5xf32>) -> tensor<4x5xf32> {
// CHECK-NEXT:      %[[MUL:.+]] = arith.mulf %[[BF1]], %[[BF1]] : tensor<4x5xf32>
// CHECK-NEXT:      return %[[MUL]] : tensor<4x5xf32>
// CHECK-NEXT:    }
// CHECK-NOT:     @batched_f_0
