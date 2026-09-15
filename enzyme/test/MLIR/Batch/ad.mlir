// RUN: %eopt %s --enzyme-batch | FileCheck %s

module {
  func.func private @square(%arg0: tensor<12xf64>) -> tensor<12xf64> {
    %0 = arith.mulf %arg0, %arg0 : tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func private @inner(%x: tensor<12xf64>, %dy: tensor<12xf64>) -> tensor<12xf64> {
    %0 = enzyme.autodiff @square(%x, %dy) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (tensor<12xf64>, tensor<12xf64>) -> tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func @main(%arg0: tensor<4x12xf64>, %arg1: tensor<4x12xf64>) -> tensor<4x12xf64> {
    %0 = enzyme.batch @inner(%arg0, %arg1) {
      batch_shape = array<i64: 4>
    } : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
    return %0 : tensor<4x12xf64>
  }
}

// CHECK:    func.func @main(%arg0: tensor<4x12xf64>, %arg1: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:      %0 = call @batched_inner(%arg0, %arg1) : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
// CHECK-NEXT:      return %0 : tensor<4x12xf64>
// CHECK-NEXT:    }
// CHECK:    func.func private @batched_inner(%arg0: tensor<4x12xf64>, %arg1: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:      %0 = enzyme.autodiff @batched_square(%arg0, %arg1) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
// CHECK-NEXT:      return %0 : tensor<4x12xf64>
// CHECK-NEXT:    }
// CHECK:    func.func private @batched_square(%arg0: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:      %0 = arith.mulf %arg0, %arg0 : tensor<4x12xf64>
// CHECK-NEXT:      return %0 : tensor<4x12xf64>
// CHECK-NEXT:    }

module {
  func.func private @square(%arg0: tensor<12xf64>) -> tensor<12xf64> {
    %0 = arith.mulf %arg0, %arg0 : tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func private @inner(%x: tensor<12xf64>, %dx: tensor<12xf64>) -> tensor<12xf64> {
    %0 = enzyme.fwddiff @square(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (tensor<12xf64>, tensor<12xf64>) -> tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func @main(%arg0: tensor<4x12xf64>, %arg1: tensor<4x12xf64>) -> tensor<4x12xf64> {
    %0 = enzyme.batch @inner(%arg0, %arg1) {
      batch_shape = array<i64: 4>
    } : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
    return %0 : tensor<4x12xf64>
  }
}
// CHECK:    func.func @main(%arg0: tensor<4x12xf64>, %arg1: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:      %0 = call @batched_inner(%arg0, %arg1) : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
// CHECK-NEXT:      return %0 : tensor<4x12xf64>
// CHECK-NEXT:    }
// CHECK:    func.func private @batched_inner(%arg0: tensor<4x12xf64>, %arg1: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:      %0 = enzyme.fwddiff @batched_square(%arg0, %arg1) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
// CHECK-NEXT:      return %0 : tensor<4x12xf64>
// CHECK-NEXT:    }
// CHECK:    func.func private @batched_square(%arg0: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:      %0 = arith.mulf %arg0, %arg0 : tensor<4x12xf64>
// CHECK-NEXT:      return %0 : tensor<4x12xf64>
// CHECK-NEXT:    }
