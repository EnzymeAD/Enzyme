// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @normal(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)

  func.func @test(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%seed, %mean, %stddev) <{ name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %s#1, %stddev) <{ name="t" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  // CHECK-LABEL: func.func @main
  // CHECK-SAME: (%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>)
  func.func @main(%seed : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    // CHECK-NEXT: %{{.+}}:2 = call @test.call(%[[SEED]], %[[MEAN]], %[[STDDEV]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %o:2 = impulse.untraced_call @test(%seed, %mean, %stddev) <{ name = "test" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
}

// CHECK:  func.func @test.call(%[[SEED:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
// CHECK-NEXT:    %[[S:.+]]:2 = call @normal(%[[SEED]], %[[MEAN]], %[[STDDEV]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %[[T:.+]]:2 = call @normal(%[[S]]#0, %[[S]]#1, %[[STDDEV]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    return %[[T]]#0, %[[T]]#1 : tensor<2xui64>, tensor<f64>
// CHECK-NEXT:  }
