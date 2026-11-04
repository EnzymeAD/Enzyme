// RUN: %eopt %s --split-input-file --verify-diagnostics

func.func @if_true_branch_type_mismatch(%pred : tensor<i1>, %a : tensor<2xf64>, %b : tensor<1x2xf64>) -> tensor<1x2xf64> {
  // expected-error @below {{'impulse.if' op true branch yields ('tensor<2xf64>') but results are ('tensor<1x2xf64>')}}
  %r = impulse.if(%pred) ({
    impulse.yield %a : tensor<2xf64>
  }, {
    impulse.yield %b : tensor<1x2xf64>
  }) : (tensor<i1>) -> tensor<1x2xf64>
  return %r : tensor<1x2xf64>
}

// -----

func.func @if_false_branch_type_mismatch(%pred : tensor<i1>, %a : tensor<2xf64>, %b : tensor<1x2xf64>) -> tensor<2xf64> {
  // expected-error @below {{'impulse.if' op false branch yields ('tensor<1x2xf64>') but results are ('tensor<2xf64>')}}
  %r = impulse.if(%pred) ({
    impulse.yield %a : tensor<2xf64>
  }, {
    impulse.yield %b : tensor<1x2xf64>
  }) : (tensor<i1>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}

// -----

func.func @if_yield_count_mismatch(%pred : tensor<i1>, %a : tensor<f64>) -> tensor<f64> {
  // expected-error @below {{'impulse.if' op true branch yields ('tensor<f64>', 'tensor<f64>') but results are ('tensor<f64>')}}
  %r = impulse.if(%pred) ({
    impulse.yield %a, %a : tensor<f64>, tensor<f64>
  }, {
    impulse.yield %a : tensor<f64>
  }) : (tensor<i1>) -> tensor<f64>
  return %r : tensor<f64>
}
