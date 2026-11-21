// RUN: %eopt --enzyme-batch --split-input-file --verify-diagnostics %s

func.func @wrong_arg_count(%x: tensor<4x3xf64>) -> tensor<4x3xf64> {
  // expected-error@below {{'enzyme.batch_region' op expected 1 block arguments, got 0}}
  %r = enzyme.batch_region(%x) {
    %cst = arith.constant dense<1.0> : tensor<3xf64>
    enzyme.yield %cst : tensor<3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}

// -----

func.func @wrong_yield_count(%x: tensor<4x3xf64>) -> tensor<4x3xf64> {
  // expected-error@below {{'enzyme.batch_region' op expected 1 yielded values, got 2}}
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    enzyme.yield %arg0, %arg0 : tensor<3xf64>, tensor<3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}

// -----

func.func @unsupported_free_var(%x: tensor<4x3xf64>, %m: memref<3xf64>) -> tensor<4x3xf64> {
  // expected-error@below {{cannot broadcast captured value of unsupported type 'memref<3xf64>'}}
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %c0 = arith.constant 0 : index
    %v = memref.load %m[%c0] : memref<3xf64>
    %splat = tensor.splat %v : tensor<3xf64>
    %y = arith.addf %arg0, %splat : tensor<3xf64>
    enzyme.yield %y : tensor<3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}
