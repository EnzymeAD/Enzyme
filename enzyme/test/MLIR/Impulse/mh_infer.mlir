// RUN: %eopt --expand-impulse %s | FileCheck %s

// engine) in DELTA-WEIGHT mode: the regenerate result is the accept-ratio
// delta over the dirty density terms only, the accept compares log(u) against
// that delta directly, and the carried weight advances by the delta on accept.

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
// CHECK-DAG: %[[INIT_TRACE:.+]] = arith.constant dense<{{.*}}> : tensor<1x2xf64>
// CHECK-DAG: %[[INIT_WEIGHT:.+]] = arith.constant dense<-2.000000e+00> : tensor<f64>
// CHECK: %[[REGEN:.+]]:4 = call @test.regenerate(%[[INIT_TRACE]], %[[RNG]], %[[MEAN]], %[[STDDEV]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[WACC:.+]] = arith.addf %[[REGEN]]#1, %[[INIT_WEIGHT]] : tensor<f64>
// CHECK-NEXT: %[[NEWRNG:.+]], %[[U:.+]] = impulse.random %[[REGEN]]#2, %[[ZERO]], %[[ONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
// CHECK-NEXT: %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[REGEN]]#1 : tensor<f64>
// CHECK-NEXT: %[[SELTR:.+]] = impulse.select %[[ACC]], %[[REGEN]]#0, %[[INIT_TRACE]] : (tensor<i1>, tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<1x2xf64>
// CHECK-NEXT: %[[SELWT:.+]] = arith.select %[[ACC]], %[[WACC]], %[[INIT_WEIGHT]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: return %[[SELTR]], %[[ACC]], %[[NEWRNG]], %[[SELWT]] : tensor<1x2xf64>, tensor<i1>, tensor<2xui64>, tensor<f64>

// CHECK-LABEL: func.func @test.regenerate
