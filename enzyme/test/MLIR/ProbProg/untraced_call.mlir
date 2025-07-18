// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  func.func @test(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%seed, %mean, %stddev) { name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %s#1, %stddev) { name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @main(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // CHECK: %0:2 = call @test.call(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %o:2 = enzyme.untracedCall @test(%seed, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// CHECK:  func.func @test.call(%[[seed:.+]]: tensor<2xui64>, %[[mean:.+]]: tensor<f64>, %[[stddev:.+]]: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
// CHECK-NEXT:    %[[s:.+]]:2 = call @normal(%[[seed]], %[[mean]], %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[t:.+]]:2 = call @normal(%[[s]]#0, %[[s]]#1, %[[stddev]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    return %[[t]]#0, %[[t]]#1 : tensor<2xui64>, tensor<f64>
// CHECK-NEXT:  }
