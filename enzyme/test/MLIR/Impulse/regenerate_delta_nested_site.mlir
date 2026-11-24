// RUN: %eopt --expand-impulse --verify-diagnostics %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %c : tensor<f64>, %pred : i1) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @normal(%rng, %c, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name = "a" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = scf.if %pred -> (tensor<2xui64>, tensor<f64>) {
      %s:2 = impulse.sample @normal(%a#0, %a#1, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<2>, name = "b" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      scf.yield %s#0, %s#1 : tensor<2xui64>, tensor<f64>
    } else {
      scf.yield %a#0, %a#1 : tensor<2xui64>, tensor<f64>
    }
    return %b#0, %b#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %c : tensor<f64>, %pred : i1) -> (tensor<1x4xf64>, tensor<f64>) {
    %prev = arith.constant dense<[[0.1, 0.2, -1.1, -1.2]]> : tensor<1x4xf64>
    // expected-error @below {{Impulse: regenerate over a score-tail trace requires a straight-line body of top-level distribution sites}}
    %res:4 = impulse.regenerate @model(%rng, %c, %pred) given %prev
        <{ selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
          regenerate_addresses = [[#impulse.symbol<1>]] }>
        : (tensor<2xui64>, tensor<f64>, i1, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1 : tensor<1x4xf64>, tensor<f64>
  }
}
