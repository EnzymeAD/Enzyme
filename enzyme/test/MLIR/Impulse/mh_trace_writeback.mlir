// RUN: %eopt --expand-impulse %s | FileCheck %s

// Unwritten columns stay the trace's own, so a sweep loop sees them invariant.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>
  func.func private @vnormal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
  func.func private @vlogpdf(%x : tensor<3xf64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %y:2 = impulse.sample @vnormal(%s#0, %s#1, %stddev) <{ logpdf = @vlogpdf, symbol = #impulse.symbol<2>, name="y" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<3xf64>)
    %t:2 = impulse.sample @normal(%y#0, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<3>, name="t" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x8xf64>, %weight : tensor<f64>) -> (tensor<1x8xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>) {
    %step:4 = impulse.mh @test(%rng, %mean, %stddev) given %trace weight %weight
        <{ name = "mh", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], regenerate_addresses = [[#impulse.symbol<1>]] }>
        : (tensor<1x8xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x8xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
    return %step#0, %step#1, %step#2, %step#3 : tensor<1x8xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK-SAME: %[[TR:[^:]+]]: tensor<1x8xf64>, %[[W:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[I5:.+]] = arith.constant dense<5> : tensor<i64>
// CHECK-DAG: %[[I0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK: %[[REGEN:.+]]:4 = call @test.regenerate(%[[TR]],
// CHECK: %[[ACC:.+]] = arith.cmpf olt, %{{.+}}, %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[SP:.+]] = impulse.slice %[[REGEN]]#0 <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x8xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[SC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x8xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[SU:.+]] = impulse.select %[[ACC]], %[[SP]], %[[SC]] : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[T1:.+]] = impulse.dynamic_update_slice %[[TR]], %[[SU]], %[[I0]], %[[I0]] : (tensor<1x8xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[TP:.+]] = impulse.slice %[[REGEN]]#0 <start_indices = [0, 5], limit_indices = [1, 8], strides = [1, 1]> : (tensor<1x8xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[TC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 5], limit_indices = [1, 8], strides = [1, 1]> : (tensor<1x8xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[TU:.+]] = impulse.select %[[ACC]], %[[TP]], %[[TC]] : (tensor<i1>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[TU]], %[[I0]], %[[I5]] : (tensor<1x8xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x8xf64>
// CHECK-NEXT: %[[SELWT:.+]] = arith.select %[[ACC]], %{{.+}}, %[[W]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: return %[[T2]], %[[SELWT]], %[[ACC]], %{{.+}} : tensor<1x8xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>
