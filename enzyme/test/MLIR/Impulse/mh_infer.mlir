// RUN: %eopt --expand-impulse %s | FileCheck %s

// log alpha = new joint - old joint - (the site's new - old density).

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x2xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>) {
    %init_trace = arith.constant dense<[[0.5, 1.0]]> : tensor<1x2xf64>
    %init_weight = arith.constant dense<-2.0> : tensor<f64>
    %res:5 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace state = [%init_weight]
        { mh_config = #impulse.mh_config,
          selection = [[#impulse.symbol<2>]],
          all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]],
          state_schema = ["weight"],
          num_samples = 1 : i64 }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<i1>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#3, %res#4 : tensor<1x2xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK-SAME: (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STDDEV:.+]]: tensor<f64>)
// CHECK-DAG: %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[I0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[I1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<{{.*}}> : tensor<1x2xf64>
// CHECK-DAG: %[[INIT_WEIGHT:.+]] = arith.constant dense<-2.000000e+00> : tensor<f64>
// CHECK: %[[REGEN:.+]]:4 = call @test.regenerate(%[[INIT_TRACE]], %[[RNG]], %[[MEAN]], %[[STDDEV]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<2xf64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[WSL:.+]] = impulse.slice %[[REGEN]]#1 {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[W:.+]] = impulse.reshape %[[WSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CSL:.+]] = impulse.slice %[[REGEN]]#1 {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[CORR:.+]] = impulse.reshape %[[CSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[WDIFF:.+]] = arith.subf %[[W]], %[[INIT_WEIGHT]] : tensor<f64>
// CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[WDIFF]], %[[CORR]] : tensor<f64>
// CHECK-NEXT: %[[NEWRNG:.+]], %[[U:.+]] = impulse.random %[[REGEN]]#2, %[[ZERO]], %[[ONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
// CHECK-NEXT: %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[ALPHA]] : tensor<f64>
// CHECK-NEXT: %[[WP:.+]] = impulse.slice %[[REGEN]]#0 {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[WC:.+]] = impulse.slice %[[INIT_TRACE]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[WU:.+]] = impulse.select %[[ACC]], %[[WP]], %[[WC]] : (tensor<i1>, tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[SELTR:.+]] = impulse.dynamic_update_slice %[[INIT_TRACE]], %[[WU]], %[[I0]], %[[I1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SELWT:.+]] = arith.select %[[ACC]], %[[W]], %[[INIT_WEIGHT]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: return %[[SELTR]], %[[ACC]], %[[NEWRNG]], %[[SELWT]] : tensor<1x2xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @test.regenerate
// CHECK-SAME: (%[[TR:.+]]: tensor<1x2xf64>, %[[R:.+]]: tensor<2xui64>, %[[M:.+]]: tensor<f64>, %[[SD:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<2xf64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[Z:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[SSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[S:.+]] = impulse.reshape %[[SSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK: %[[T:.+]]:2 = call @normal(%[[R]], %[[S]], %[[SD]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[LPT:.+]] = call @logpdf(%[[T]]#1, %[[S]], %[[SD]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.addf %{{.+}}, %[[LPT]] : tensor<f64>
// CHECK-NEXT: %[[CN:.+]] = arith.addf %[[LPT]], %[[Z]] : tensor<f64>
// CHECK-NEXT: %[[TOSL:.+]] = impulse.slice %[[TR]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>, strides = array<i64: 1, 1>} : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[TO:.+]] = impulse.reshape %[[TOSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[LPTO:.+]] = call @logpdf(%[[TO]], %[[S]], %[[SD]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %{{.+}} = arith.subf %[[CN]], %[[LPTO]] : tensor<f64>
