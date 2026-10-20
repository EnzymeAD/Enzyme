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

// -----

module {
  func.func private @s(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %a : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // expected-error @+1 {{a transform chain is only supported on the STD_NORMAL and STD_UNIFORM bases}}
    %o:2 = impulse.sample @s(%r, %a, %b) {
      symbol = #impulse.symbol<1>,
      dist = #impulse.dist<GAMMA, [#impulse.xform<AFFINE>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// -----

// Accepted: STD_UNIFORM + AFFINE derives INTERVAL.
module {
  func.func private @s(%r: tensor<2xui64>, %lo: tensor<f64>, %sc: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %lo : tensor<2xui64>, tensor<f64>
  }
  func.func @f(%r: tensor<2xui64>, %lo: tensor<f64>, %sc: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @s(%r, %lo, %sc) {
      symbol = #impulse.symbol<1>,
      support = #impulse.support<INTERVAL>,
      dist = #impulse.dist<STD_UNIFORM, [#impulse.xform<AFFINE>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}
