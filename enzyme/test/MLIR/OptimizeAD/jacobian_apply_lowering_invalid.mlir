// RUN: %eopt --split-input-file --lower-jacobian-apply --verify-diagnostics %s

module {
  func.func @bad_source(%x: tensor<4xf64>, %dx: tensor<4xf64>) -> tensor<4xf64> {
    %fake = arith.constant dense<0.0> : tensor<4x4xf64>
    // expected-error @below {{expects first operand to come from enzyme.jacobian}}
    %out = enzyme.jvp_apply (%fake, %dx) : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %out : tensor<4xf64>
  }
}

// -----

module {
  func.func @two_arg(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
    %out = arith.addf %x, %y : tensor<4xf64>
    return %out : tensor<4xf64>
  }

  func.func @bad_fn_arity(%x: tensor<4xf64>, %dout: tensor<4xf64>) -> tensor<4xf64> {
    %j = enzyme.jacobian @two_arg(%x) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<4xf64>) -> tensor<4x4xf64>
    // expected-error @below {{requires referenced function to have exactly one argument and one result}}
    %dx = enzyme.vjp_apply (%j, %dout) : (tensor<4x4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %dx : tensor<4xf64>
  }
}
