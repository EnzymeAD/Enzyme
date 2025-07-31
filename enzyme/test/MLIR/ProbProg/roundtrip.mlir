// RUN: %eopt %s | FileCheck %s

module {
  func.func private @normal(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  // CHECK:   func.func @sample(%[[seed:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[res:.+]]:2 = enzyme.sample @normal(%[[seed]], %[[mean]], %[[stddev]]) {logpdf = @logpdf, name = "r", symbol = #enzyme.symbol<3>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[res]]#0, %[[res]]#1 : tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @sample(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %r:2 = enzyme.sample @normal(%seed, %mean, %stddev) { logpdf = @logpdf, name="r", symbol = #enzyme.symbol<3> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %r#0, %r#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK:   func.func @simulate(%[[seed:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[trace:.+]], %[[weight:.+]], %[[outputs:.+]]:2 = enzyme.simulate @sample(%[[seed]], %[[mean]], %[[stddev]]) {name = "test"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[trace]], %[[weight]], %[[outputs]]#0, %[[outputs]]#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @simulate(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:4 = enzyme.simulate @sample(%seed, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  // CHECK:   func.func @generate(%[[seed:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[cst:.+]] = arith.constant dense<42> : tensor<ui64>
  // CHECK-NEXT:    %[[constraint:.+]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<ui64> to !enzyme.Constraint
  // CHECK-NEXT:    %[[trace:.+]], %[[weight:.+]], %[[outputs:.+]]:2 = enzyme.generate @sample(%[[seed]], %[[mean]], %[[stddev]]) given %[[constraint]] {constrained_addresses = {{\[}}[#enzyme.symbol<2>], [#enzyme.symbol<3>]{{\]}}, name = "test"} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[trace]], %[[weight]], %[[outputs]]#0, %[[outputs]]#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @generate(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %cst = arith.constant dense<42> : tensor<ui64>
    %0 = builtin.unrealized_conversion_cast %cst : tensor<ui64> to !enzyme.Constraint
    %res:4 = enzyme.generate @sample(%seed, %mean, %stddev) given %0 { constrained_addresses = [[#enzyme.symbol<2>], [#enzyme.symbol<3>]], name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}
