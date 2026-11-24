// RUN: %eopt --expand-impulse %s | FileCheck %s

// A trace wider than the values holds one memoized score per site after them.
module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %s#1, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<2>, name="t" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @sim(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>) {
    %tr, %w, %r, %v = impulse.simulate @model(%rng, %mean, %stddev) <{ selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]] }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %w : tensor<1x4xf64>, tensor<f64>
  }

  func.func @gen(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %obs : tensor<1x1xf64>) -> (tensor<1x4xf64>, tensor<f64>) {
    %tr, %w, %r, %v = impulse.generate @model(%rng, %mean, %stddev) given %obs <{ selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]], constrained_addresses = [[#impulse.symbol<2>]] }> : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %tr, %w : tensor<1x4xf64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @model.generate
// CHECK-SAME:      (%[[OBS:.+]]: tensor<1x1xf64>, %[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STD:.+]]: tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG:     %[[GCOL3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG:     %[[GCOL2:.+]] = arith.constant dense<2> : tensor<i64>
// CHECK-DAG:     %[[GZFULL:.+]] = arith.constant dense<0.000000e+00> : tensor<1x4xf64>
// CHECK-DAG:     %[[GCOL1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG:     %[[GCOL0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[GZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[GZVALS:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK:         %[[G1:.+]]:2 = call @normal(%[[RNG]], %[[MEAN]], %[[STD]])
// CHECK-NEXT:    %[[GLP1:.+]] = call @logpdf(%[[G1]]#1, %[[MEAN]], %[[STD]])
// CHECK-NEXT:    %[[GW1:.+]] = arith.addf %[[GLP1]], %[[GZERO]] : tensor<f64>
// CHECK-NEXT:    %[[G1W:.+]] = impulse.reshape %[[G1]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[GV1:.+]] = impulse.dynamic_update_slice %[[GZVALS]], %[[G1W]], %[[GCOL0]], %[[GCOL0]]
// CHECK-NEXT:    %[[OSL:.+]] = impulse.slice %[[OBS]] <start_indices = [0, 0], limit_indices = [1, 1]
// CHECK-NEXT:    %[[OV:.+]] = impulse.reshape %[[OSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[GLP2:.+]] = call @logpdf(%[[OV]], %[[G1]]#1, %[[STD]])
// CHECK-NEXT:    %[[GW2:.+]] = arith.addf %[[GW1]], %[[GLP2]] : tensor<f64>
// CHECK-NEXT:    %[[OW:.+]] = impulse.reshape %[[OV]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[GV2:.+]] = impulse.dynamic_update_slice %[[GV1]], %[[OW]], %[[GCOL0]], %[[GCOL1]]
// CHECK-NEXT:    %[[GA0:.+]] = impulse.dynamic_update_slice %[[GZFULL]], %[[GV2]], %[[GCOL0]], %[[GCOL0]]
// CHECK-NEXT:    %[[GLP1W:.+]] = impulse.reshape %[[GLP1]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[GA1:.+]] = impulse.dynamic_update_slice %[[GA0]], %[[GLP1W]], %[[GCOL0]], %[[GCOL2]]
// CHECK-NEXT:    %[[GLP2W:.+]] = impulse.reshape %[[GLP2]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[GA2:.+]] = impulse.dynamic_update_slice %[[GA1]], %[[GLP2W]], %[[GCOL0]], %[[GCOL3]]
// CHECK-NEXT:    return %[[GA2]], %[[GW2]], %[[G1]]#0, %[[OV]]

// CHECK-LABEL: func.func @model.simulate
// CHECK-SAME:      (%[[RNG:.+]]: tensor<2xui64>, %[[MEAN:.+]]: tensor<f64>, %[[STD:.+]]: tensor<f64>) -> (tensor<1x4xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG:     %[[COL3:.+]] = arith.constant dense<3> : tensor<i64>
// CHECK-DAG:     %[[COL2:.+]] = arith.constant dense<2> : tensor<i64>
// CHECK-DAG:     %[[ZFULL:.+]] = arith.constant dense<0.000000e+00> : tensor<1x4xf64>
// CHECK-DAG:     %[[COL1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG:     %[[COL0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[ZVALS:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK:         %[[S1:.+]]:2 = call @normal(%[[RNG]], %[[MEAN]], %[[STD]])
// CHECK-NEXT:    %[[LP1:.+]] = call @logpdf(%[[S1]]#1, %[[MEAN]], %[[STD]])
// CHECK-NEXT:    %[[W1:.+]] = arith.addf %[[LP1]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[S1W:.+]] = impulse.reshape %[[S1]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[V1:.+]] = impulse.dynamic_update_slice %[[ZVALS]], %[[S1W]], %[[COL0]], %[[COL0]]
// CHECK-NEXT:    %[[S2:.+]]:2 = call @normal(%[[S1]]#0, %[[S1]]#1, %[[STD]])
// CHECK-NEXT:    %[[LP2:.+]] = call @logpdf(%[[S2]]#1, %[[S1]]#1, %[[STD]])
// CHECK-NEXT:    %[[W2:.+]] = arith.addf %[[W1]], %[[LP2]] : tensor<f64>
// CHECK-NEXT:    %[[S2W:.+]] = impulse.reshape %[[S2]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[V2:.+]] = impulse.dynamic_update_slice %[[V1]], %[[S2W]], %[[COL0]], %[[COL1]]
// CHECK-NEXT:    %[[A0:.+]] = impulse.dynamic_update_slice %[[ZFULL]], %[[V2]], %[[COL0]], %[[COL0]]
// CHECK-NEXT:    %[[LP1W:.+]] = impulse.reshape %[[LP1]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A1:.+]] = impulse.dynamic_update_slice %[[A0]], %[[LP1W]], %[[COL0]], %[[COL2]]
// CHECK-NEXT:    %[[LP2W:.+]] = impulse.reshape %[[LP2]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A2:.+]] = impulse.dynamic_update_slice %[[A1]], %[[LP2W]], %[[COL0]], %[[COL3]]
// CHECK-NEXT:    return %[[A2]], %[[W2]], %[[S2]]#0, %[[S2]]#1
