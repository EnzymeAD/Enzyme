// RUN: %eopt --split-input-file --verify-diagnostics %s

module {
  func.func private @s(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %m : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // expected-error @+1 {{explicit support contradicts the support derived from its dist attribute}}
    %o:2 = impulse.sample @s(%r, %m, %g) {
      symbol = #impulse.symbol<1>,
      support = #impulse.support<REAL>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<EXP>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

// Accepted: STD_NORMAL + AFFINE derives REAL.
module {
  func.func private @s(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %m : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %m: tensor<f64>, %g: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @s(%r, %m, %g) {
      symbol = #impulse.symbol<1>,
      support = #impulse.support<REAL>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}
