// RUN: %eopt %s | FileCheck %s

module {
  func.func private @normal(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  // CHECK:   func.func @sample(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[RES:.+]]:2 = enzyme.sample @normal(%[[SEED]], %[[MEAN]], %[[STDDEV]]) {logpdf = @logpdf, name = "r", symbol = #enzyme.symbol<3>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[RES]]#0, %[[RES]]#1 : tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @sample(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %r:2 = enzyme.sample @normal(%seed, %mean, %stddev) { logpdf = @logpdf, name="r", symbol = #enzyme.symbol<3> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %r#0, %r#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK:   func.func @simulate(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[TRACE:.+]], %[[WEIGHT:.+]], %[[OUTPUTS:.+]]:2 = enzyme.simulate @sample(%[[SEED]], %[[MEAN]], %[[STDDEV]]) {name = "test", selection = {{\[}}[#enzyme.symbol<3>]]} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[TRACE]], %[[WEIGHT]], %[[OUTPUTS]]#0, %[[OUTPUTS]]#1 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @simulate(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:4 = enzyme.simulate @sample(%seed, %mean, %stddev) { selection = [[#enzyme.symbol<3>]], name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  // CHECK:   func.func @generate(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
  // CHECK-NEXT:    %[[CONSTRAINT:.+]] = arith.constant dense<{{.*}}1.5{{.*}}> : tensor<1x1xf64>
  // CHECK-NEXT:    %[[TRACE:.+]], %[[WEIGHT:.+]], %[[OUTPUTS:.+]]:2 = enzyme.generate @sample(%[[SEED]], %[[MEAN]], %[[STDDEV]]) given %[[CONSTRAINT]] {constrained_addresses = {{\[}}[#enzyme.symbol<3>]], name = "test", selection = {{\[}}[#enzyme.symbol<3>]]} : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
  // CHECK-NEXT:    return %[[TRACE]], %[[WEIGHT]], %[[OUTPUTS]]#0, %[[OUTPUTS]]#1 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  // CHECK-NEXT:   }
  func.func @generate(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %constraint = arith.constant dense<1.5> : tensor<1x1xf64>
    %res:4 = enzyme.generate @sample(%seed, %mean, %stddev) given %constraint { selection = [[#enzyme.symbol<3>]], constrained_addresses = [[#enzyme.symbol<3>]], name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}
