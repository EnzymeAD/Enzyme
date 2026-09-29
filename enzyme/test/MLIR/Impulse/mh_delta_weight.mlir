// RUN: %eopt --expand-impulse %s | FileCheck %s

// Score-tail trace: accept on delta - correction (the site's own delta).

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %s#1, %stddev) { logpdf = @logpdf, symbol = #impulse.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.5, 1.0, -0.9, -1.1]]> : tensor<1x4xf64>
    %init_weight = arith.constant dense<-2.0> : tensor<f64>
    %step:4 = impulse.mh @test(%rng, %mean, %stddev) given %init_trace weight %init_weight
        { name = "mh_1", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], regenerate_addresses = [[#impulse.symbol<2>]] }
        : (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
    return %step#0, %step#1, %step#3 : tensor<1x4xf64>, tensor<f64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK-SAME:      (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STD:.+]]: tensor<f64>)
// CHECK-DAG:     %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[ZEROF:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[TR0:.+]] = arith.constant dense<{{.*}}> : tensor<1x4xf64>
// CHECK-DAG:     %[[W0:.+]] = arith.constant dense<-2.000000e+00> : tensor<f64>
// CHECK:         %[[REGEN:.+]]:4 = call @test.regenerate(%[[TR0]], %[[RNG]], %[[MEAN]], %[[STD]])
// CHECK-NEXT:    %[[DSL:.+]] = impulse.slice %[[REGEN]]#1 {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[DELTA:.+]] = impulse.reshape %[[DSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[CSL:.+]] = impulse.slice %[[REGEN]]#1 {limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[CORR:.+]] = impulse.reshape %[[CSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[ALPHA:.+]] = arith.subf %[[DELTA]], %[[CORR]] : tensor<f64>
// CHECK-NEXT:    %[[WACC:.+]] = arith.addf %[[DELTA]], %[[W0]] : tensor<f64>
// CHECK-NEXT:    %[[NRNG:.+]], %[[U:.+]] = impulse.random %[[REGEN]]#2, %[[ZEROF]], %[[ONE]]
// CHECK-NEXT:    %[[LOGU:.+]] = math.log %[[U]] : tensor<f64>
// CHECK-NEXT:    %[[ACC:.+]] = arith.cmpf olt, %[[LOGU]], %[[ALPHA]] : tensor<f64>
// CHECK-NEXT:    %[[SELTR:.+]] = impulse.select %[[ACC]], %[[REGEN]]#0, %[[TR0]]
// CHECK-NEXT:    %[[SELWT:.+]] = arith.select %[[ACC]], %[[WACC]], %[[W0]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:    return %[[SELTR]], %[[SELWT]], %[[NRNG]]

// Site 2 is an unobserved leaf: delta equals correction, so always accepted.
// CHECK-LABEL: func.func @test.regenerate
// CHECK-SAME:      (%[[TRACE:.+]]: tensor<1x4xf64>, %[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STD:.+]]: tensor<f64>)
// CHECK-DAG:     %[[PACK0:.+]] = arith.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-DAG:     %[[COL3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG:     %[[COL2:.+]] = arith.constant dense<2> : tensor<i64>
// CHECK-DAG:     %[[ZFULL:.+]] = arith.constant dense<0.000000e+00> : tensor<1x4xf64>
// CHECK-DAG:     %[[COL1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG:     %[[COL0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[VALS:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 0>
// CHECK-NEXT:    %[[S1SL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>
// CHECK-NEXT:    %[[S1:.+]] = impulse.reshape %[[S1SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[SC1SL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>
// CHECK-NEXT:    %[[SC1:.+]] = impulse.reshape %[[SC1SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[DRAW:.+]]:2 = call @normal(%[[RNG]], %[[S1]], %[[STD]])
// CHECK-NEXT:    %[[SC2SL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>
// CHECK-NEXT:    %[[SC2:.+]] = impulse.reshape %[[SC2SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LP2N:.+]] = call @logpdf(%[[DRAW]]#1, %[[S1]], %[[STD]])
// CHECK-NEXT:    %[[D2:.+]] = arith.subf %[[LP2N]], %[[SC2]] : tensor<f64>
// CHECK-NEXT:    %[[DELTA:.+]] = arith.addf %[[D2]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[CORR:.+]] = arith.addf %[[D2]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[DRAWW:.+]] = impulse.reshape %[[DRAW]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[VALS2:.+]] = impulse.dynamic_update_slice %[[VALS]], %[[DRAWW]], %[[COL0]], %[[COL1]]
// CHECK-NEXT:    %[[A0:.+]] = impulse.dynamic_update_slice %[[ZFULL]], %[[VALS2]], %[[COL0]], %[[COL0]]
// CHECK-NEXT:    %[[SC1W:.+]] = impulse.reshape %[[SC1]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A1:.+]] = impulse.dynamic_update_slice %[[A0]], %[[SC1W]], %[[COL0]], %[[COL2]]
// CHECK-NEXT:    %[[LP2W:.+]] = impulse.reshape %[[LP2N]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A2:.+]] = impulse.dynamic_update_slice %[[A1]], %[[LP2W]], %[[COL0]], %[[COL3]]
// CHECK-NEXT:    %[[DFLAT:.+]] = impulse.reshape %[[DELTA]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[P0:.+]] = impulse.dynamic_update_slice %[[PACK0]], %[[DFLAT]], %[[COL0]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:    %[[CFLAT:.+]] = impulse.reshape %[[CORR]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[P1:.+]] = impulse.dynamic_update_slice %[[P0]], %[[CFLAT]], %[[COL1]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT:    return %[[A2]], %[[P1]], %[[DRAW]]#0, %[[DRAW]]#1
