// RUN: %eopt --expand-impulse %s | FileCheck %s

// Regenerating root s: log alpha = delta - correction = lp_t(s') - score_t.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %s#1, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<2>, name="t" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x4xf64>, %weight : tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>) {
    %step:4 = impulse.mh @test(%rng, %mean, %stddev) given %trace weight %weight
        <{ name = "mh", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], regenerate_addresses = [[#impulse.symbol<1>]] }>
        : (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
    return %step#0, %step#1, %step#2, %step#3 : tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK: %[[REGEN:.+]]:4 = call @test.regenerate(
// CHECK-NEXT: %[[DSL:.+]] = impulse.slice %[[REGEN]]#1 <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[DELTA:.+]] = impulse.reshape %[[DSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CSL:.+]] = impulse.slice %[[REGEN]]#1 <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[CORR:.+]] = impulse.reshape %[[CSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[DELTA]], %[[CORR]] : tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[DELTA]] : tensor<f64>

// CHECK-LABEL: func.func @test.regenerate
// CHECK-SAME: (%[[TR:[^:]+]]: tensor<1x4xf64>, %[[R:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>)
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[S:.+]]:2 = call @normal(%[[R]], %[[M]], %[[SD]])
// CHECK-NEXT: %[[SCSSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 2], limit_indices = [1, 3], strides = [1, 1]> : (tensor<1x4xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[SCS:.+]] = impulse.reshape %[[SCSSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[LPS:.+]] = call @logpdf(%[[S]]#1, %[[M]], %[[SD]])
// CHECK-NEXT: %[[DS:.+]] = arith.subf %[[LPS]], %[[SCS]] : tensor<f64>
// CHECK-NEXT: %[[D1:.+]] = arith.addf %[[DS]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[CORR:.+]] = arith.addf %[[DS]], %[[ZERO]] : tensor<f64>
// CHECK: %[[LPT:.+]] = call @logpdf(%{{.+}}, %[[S]]#1, %[[SD]])
// CHECK-NEXT: %[[DT:.+]] = arith.subf %[[LPT]], %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[D2:.+]] = arith.addf %[[D1]], %[[DT]] : tensor<f64>
// CHECK-NOT: %[[CORR]]
// CHECK: %[[DFL:.+]] = impulse.reshape %[[D2]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[P0:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[DFL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[CFL:.+]] = impulse.reshape %[[CORR]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %{{.+}} = impulse.dynamic_update_slice %[[P0]], %[[CFL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
