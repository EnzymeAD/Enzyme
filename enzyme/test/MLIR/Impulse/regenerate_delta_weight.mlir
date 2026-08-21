// RUN: %eopt --expand-impulse %s | FileCheck %s

// Delta-weight regenerate: clean density terms (site 1: not regenerated;
// site 3: parameters depend only on the non-regenerated site 1) cancel in
// the MH ratio and are never emitted. The regenerated site 2 and the
// dependent site 4 emit logp(new) - logp(old) pairs, with site 4's old
// parameters re-materialized from the old trace value of site 2. The
// weight result is the delta; the full new trace is still produced.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %c : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @normal(%rng, %c, %c) { logpdf = @logpdf, symbol = #impulse.symbol<1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = impulse.sample @normal(%a#0, %c, %c) { logpdf = @logpdf, symbol = #impulse.symbol<2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %m3 = arith.addf %a#1, %c : tensor<f64>
    %s3:2 = impulse.sample @normal(%b#0, %m3, %c) { logpdf = @logpdf, symbol = #impulse.symbol<3> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %m4 = arith.mulf %b#1, %c : tensor<f64>
    %s4:2 = impulse.sample @normal(%s3#0, %m4, %c) { logpdf = @logpdf, symbol = #impulse.symbol<4> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s4#0, %s4#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %c : tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>) {
    %prev = arith.constant dense<[[0.1, 0.2, 0.3, 0.4]]> : tensor<1x4xf64>
    %res:4 = impulse.regenerate @model(%rng, %c) given %prev
        { selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          regenerate_addresses = [[#impulse.symbol<2>]], delta_weight }
        : (tensor<2xui64>, tensor<f64>, tensor<1x4xf64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1 : tensor<1x4xf64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME:      (%[[TRACE:.+]]: tensor<1x4xf64>, %[[RNG:.+]]: tensor<2xui64>, %[[C:.+]]: tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[ZTRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x4xf64>
// Site 1 (clean): trace read + write, NO logpdf.
// CHECK:         %[[S1SL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 1>, start_indices = array<i64: 0, 0>
// CHECK-NEXT:    %[[S1:.+]] = impulse.reshape %[[S1SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[S1W:.+]] = impulse.reshape %[[S1]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[T1:.+]] = impulse.dynamic_update_slice %[[ZTRACE]], %[[S1W]]
// Site 2 (regenerated): fresh draw, old value from the trace, logpdf pair.
// CHECK-NEXT:    %[[DRAW:.+]]:2 = call @normal(%[[RNG]], %[[C]], %[[C]])
// CHECK-NEXT:    %[[S2OSL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 1>
// CHECK-NEXT:    %[[S2O:.+]] = impulse.reshape %[[S2OSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LP2N:.+]] = call @logpdf(%[[DRAW]]#1, %[[C]], %[[C]])
// CHECK-NEXT:    %[[LP2O:.+]] = call @logpdf(%[[S2O]], %[[C]], %[[C]])
// CHECK-NEXT:    %[[D2:.+]] = arith.subf %[[LP2N]], %[[LP2O]] : tensor<f64>
// CHECK-NEXT:    %[[W2:.+]] = arith.addf %[[D2]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[DRAWW:.+]] = impulse.reshape %[[DRAW]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[T2:.+]] = impulse.dynamic_update_slice %[[T1]], %[[DRAWW]]
// Site 3 (clean; depends only on non-regenerated site 1): NO logpdf.
// CHECK-NEXT:    %[[S3SL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 3>, start_indices = array<i64: 0, 2>
// CHECK-NEXT:    %[[S3:.+]] = impulse.reshape %[[S3SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[S3W:.+]] = impulse.reshape %[[S3]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[T3:.+]] = impulse.dynamic_update_slice %[[T2]], %[[S3W]]
// Site 4 (dirty through site 2): new and old parameter slices, logpdf pair.
// CHECK-NEXT:    %[[M4N:.+]] = arith.mulf %[[DRAW]]#1, %[[C]] : tensor<f64>
// CHECK-NEXT:    %[[S4SL:.+]] = impulse.slice %[[TRACE]] {limit_indices = array<i64: 1, 4>, start_indices = array<i64: 0, 3>
// CHECK-NEXT:    %[[S4:.+]] = impulse.reshape %[[S4SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LP4N:.+]] = call @logpdf(%[[S4]], %[[M4N]], %[[C]])
// CHECK-NEXT:    %[[M4O:.+]] = arith.mulf %[[S2O]], %[[C]] : tensor<f64>
// CHECK-NEXT:    %[[LP4O:.+]] = call @logpdf(%[[S4]], %[[M4O]], %[[C]])
// CHECK-NEXT:    %[[D4:.+]] = arith.subf %[[LP4N]], %[[LP4O]] : tensor<f64>
// CHECK-NEXT:    %[[DELTA:.+]] = arith.addf %[[W2]], %[[D4]] : tensor<f64>
// CHECK-NEXT:    %[[S4W:.+]] = impulse.reshape %[[S4]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[T4:.+]] = impulse.dynamic_update_slice %[[T3]], %[[S4W]]
// CHECK-NEXT:    return %[[T4]], %[[DELTA]], %[[DRAW]]#0, %[[S4]] : tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
