// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @joint(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
  func.func private @joint_logpdf(%x1 : tensor<f64>, %x2 : tensor<2xf64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %s:3 = enzyme.sample @joint(%rng, %mean, %stddev) {
      symbol = 5 : ui64,
      name = "s",
      logpdf = @joint_logpdf,
      traced_input_indices = array<i64: 1, 2>,
      traced_output_indices = array<i64: 1, 2>,
      alias_map = array<i64: 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %s#0, %s#1, %s#2 : tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }

  func.func @foo(%rng : tensor<2xui64>, %x : tensor<f64>, %y : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %res:4 = enzyme.generate @test(%rng, %x, %y) {
      trace = 99 : ui64,
      constraints = [ #enzyme.constraint<symbol = 5, values = [ dense<8.800000e+00> : tensor<f64>, dense<[7.700000e+00, 9.900000e+00]> : tensor<2xf64> ]> ],
      name = "res"
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }
}

// CHECK: func.func @test.generate(%[[rng:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
// CHECK-NEXT: %[[zero:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[c1:.+]] = arith.constant dense<8.800000e+00> : tensor<f64>
// CHECK-NEXT: %[[c2:.+]] = arith.constant dense<[7.700000e+00, 9.900000e+00]> : tensor<2xf64>
// CHECK-NEXT: %[[logpdf:.+]] = call @joint_logpdf(%[[c1]], %[[c2]], %[[mean]], %[[stddev]]) : (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[weight:.+]] = arith.addf %[[zero]], %[[logpdf]] : tensor<f64>
// CHECK: return %[[weight]], {{.*}}, %[[c1]], %[[c2]] : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
