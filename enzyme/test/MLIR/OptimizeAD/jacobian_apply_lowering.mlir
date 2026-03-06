// RUN: %eopt --lower-jacobian-apply %s | FileCheck %s

module {
  func.func @square(%x: tensor<4xf64>) -> tensor<4xf64> {
    %y = arith.mulf %x, %x : tensor<4xf64>
    return %y : tensor<4xf64>
  }

  func.func @test_jvp_apply(%x: tensor<4xf64>, %dx: tensor<4xf64>) -> tensor<4xf64> {
    %j = enzyme.jacobian @square(%x) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<4xf64>) -> tensor<4x4xf64>
    %out = enzyme.jvp_apply (%j, %dx) : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %out : tensor<4xf64>
  }

  func.func @test_vjp_apply(%x: tensor<4xf64>, %dout: tensor<4xf64>) -> tensor<4xf64> {
    %j = enzyme.jacobian @square(%x) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<4xf64>) -> tensor<4x4xf64>
    %dx = enzyme.vjp_apply (%j, %dout) : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %dx : tensor<4xf64>
  }
}

// CHECK-LABEL: func.func @test_jvp_apply
// CHECK-NOT: enzyme.jacobian
// CHECK-NOT: enzyme.jvp_apply
// CHECK: %[[OUT:.*]] = enzyme.fwddiff @square(%{{.*}}, %{{.*}}) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK: return %[[OUT]] : tensor<4xf64>

// CHECK-LABEL: func.func @test_vjp_apply
// CHECK-NOT: enzyme.jacobian
// CHECK-NOT: enzyme.vjp_apply
// CHECK: %[[DX:.*]] = enzyme.autodiff @square(%{{.*}}, %{{.*}}) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
// CHECK: return %[[DX]] : tensor<4xf64>
