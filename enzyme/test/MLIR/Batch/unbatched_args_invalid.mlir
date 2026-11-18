// RUN: %eopt --enzyme-batch --split-input-file --verify-diagnostics %s

module {
  func.func @add(%x: tensor<3xf64>, %y: tensor<3xf64>) -> tensor<3xf64> {
    %0 = arith.addf %x, %y : tensor<3xf64>
    return %0 : tensor<3xf64>
  }

  func.func @bad_operand_type(%x: tensor<5x3xf64>, %y: tensor<3xf64>) -> tensor<4x3xf64> {
    // expected-error @+1 {{'enzyme.batch' op operand #0 type 'tensor<5x3xf64>' does not match unbatched or batched callee argument type 'tensor<3xf64>'}}
    %0 = enzyme.batch @add(%x, %y) <{batch_shape = array<i64: 4>}> : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
    return %0 : tensor<4x3xf64>
  }
}

// -----

module {
  func.func @add(%x: tensor<3xf64>, %y: tensor<3xf64>) -> tensor<3xf64> {
    %0 = arith.addf %x, %y : tensor<3xf64>
    return %0 : tensor<3xf64>
  }

  func.func @bad_result_type(%x: tensor<4x3xf64>, %y: tensor<3xf64>) -> tensor<5x3xf64> {
    // expected-error @+1 {{'enzyme.batch' op result #0 type 'tensor<5x3xf64>' does not match unbatched or batched callee result type 'tensor<3xf64>'}}
    %0 = enzyme.batch @add(%x, %y) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<5x3xf64>
    return %0 : tensor<5x3xf64>
  }
}

// -----

module {
  // expected-error @+1 {{return value #0 depends on batched arguments and must have batched result type 'tensor<4x3xf64>'}}
  func.func @add(%x: tensor<3xf64>, %y: tensor<3xf64>) -> tensor<3xf64> {
    %0 = arith.addf %x, %y : tensor<3xf64>
    return %0 : tensor<3xf64>
  }

  func.func @bad_unbatched_result(%x: tensor<4x3xf64>, %y: tensor<3xf64>) -> tensor<3xf64> {
    %0 = enzyme.batch @add(%x, %y) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
