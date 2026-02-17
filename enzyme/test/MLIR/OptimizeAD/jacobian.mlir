// RUN: %eopt %s | FileCheck %s

module {
  func.func @square(%x: tensor<4xf64>) -> tensor<4xf64> {
    %y = arith.mulf %x, %x : tensor<4xf64>
    return %y : tensor<4xf64>
  }

  func.func @test_jacobian(%x: tensor<4xf64>) -> tensor<4x4xf64> {
    %j = enzyme.jacobian @square(%x) {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<4xf64>) -> tensor<4x4xf64>
    return %j : tensor<4x4xf64>
  }
}

// CHECK-LABEL: func.func @test_jacobian
// CHECK: %[[JAC:.*]] = enzyme.jacobian @square(%{{.*}}) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>]}
// CHECK: return %[[JAC]] : tensor<4x4xf64>
