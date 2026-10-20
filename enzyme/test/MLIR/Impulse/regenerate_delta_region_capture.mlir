// RUN: %eopt --expand-impulse %s | FileCheck %s

// Site 3 is dirty via a region reading site 1; site 4's regions read site 2.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %c : tensor<f64>, %idx : index, %pred : i1) -> (tensor<2xui64>, tensor<f64>) {
    %a:2 = impulse.sample @normal(%rng, %c, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name = "a" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %b:2 = impulse.sample @normal(%a#0, %c, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<2>, name = "b" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %mu = scf.index_switch %idx -> tensor<f64>
    case 1 {
      %m1 = arith.mulf %a#1, %c : tensor<f64>
      scf.yield %m1 : tensor<f64>
    }
    default {
      scf.yield %b#1 : tensor<f64>
    }
    %y:2 = impulse.sample @normal(%b#0, %mu, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<3>, name = "y" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %nu = scf.if %pred -> (tensor<f64>) {
      %n1 = arith.addf %b#1, %c : tensor<f64>
      scf.yield %n1 : tensor<f64>
    } else {
      scf.yield %c : tensor<f64>
    }
    %z:2 = impulse.sample @normal(%y#0, %nu, %c) <{ logpdf = @logpdf, symbol = #impulse.symbol<4>, name = "z" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %z#0, %z#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %c : tensor<f64>, %idx : index, %pred : i1) -> (tensor<1x8xf64>, tensor<f64>) {
    %prev = arith.constant dense<[[0.1, 0.2, 0.3, 0.4, -1.1, -1.2, -1.3, -1.4]]> : tensor<1x8xf64>
    %res:4 = impulse.regenerate @model(%rng, %c, %idx, %pred) given %prev
        <{ selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>], [#impulse.symbol<4>]],
          regenerate_addresses = [[#impulse.symbol<1>]] }>
        : (tensor<2xui64>, tensor<f64>, index, i1, tensor<1x8xf64>) -> (tensor<1x8xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1 : tensor<1x8xf64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME:      (%[[TRACE:.+]]: tensor<1x8xf64>, %[[RNG:.+]]: tensor<2xui64>, %[[C:.+]]: tensor<f64>, %[[IDX:.+]]: index, %[[PRED:.+]]: i1) -> (tensor<1x8xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG:     %[[C7:.+]] = arith.constant dense<7> : tensor<i64>
// CHECK-DAG:     %[[C6:.+]] = arith.constant dense<6> : tensor<i64>
// CHECK-DAG:     %[[C5:.+]] = arith.constant dense<5> : tensor<i64>
// CHECK-DAG:     %[[C4:.+]] = arith.constant dense<4> : tensor<i64>
// CHECK-DAG:     %[[ZFULL:.+]] = arith.constant dense<0.000000e+00> : tensor<1x8xf64>
// CHECK-DAG:     %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK:         %[[VALS:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 0], limit_indices = [1, 4]
// CHECK-NEXT:    %[[DRAW:.+]]:2 = call @normal(%[[RNG]], %[[C]], %[[C]])
// CHECK-NEXT:    %[[SC1SL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 4], limit_indices = [1, 5]
// CHECK-NEXT:    %[[SC1:.+]] = impulse.reshape %[[SC1SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LP1:.+]] = call @logpdf(%[[DRAW]]#1, %[[C]], %[[C]])
// CHECK-NEXT:    %[[D1:.+]] = arith.subf %[[LP1]], %[[SC1]] : tensor<f64>
// CHECK-NEXT:    %[[W1:.+]] = arith.addf %[[D1]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[DRAWW:.+]] = impulse.reshape %[[DRAW]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[VALS1:.+]] = impulse.dynamic_update_slice %[[VALS]], %[[DRAWW]], %[[C0]], %[[C0]]
// CHECK-NEXT:    %[[BSL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 1], limit_indices = [1, 2]
// CHECK-NEXT:    %[[B:.+]] = impulse.reshape %[[BSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[SC2SL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 5], limit_indices = [1, 6]
// CHECK-NEXT:    %[[SC2:.+]] = impulse.reshape %[[SC2SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[MU:.+]] = scf.index_switch %[[IDX]] -> tensor<f64>
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      %[[M1:.+]] = arith.mulf %[[DRAW]]#1, %[[C]] : tensor<f64>
// CHECK-NEXT:      scf.yield %[[M1]] : tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      scf.yield %[[B]] : tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[YSL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 2], limit_indices = [1, 3]
// CHECK-NEXT:    %[[Y:.+]] = impulse.reshape %[[YSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[SC3SL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 6], limit_indices = [1, 7]
// CHECK-NEXT:    %[[SC3:.+]] = impulse.reshape %[[SC3SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LP3:.+]] = call @logpdf(%[[Y]], %[[MU]], %[[C]])
// CHECK-NEXT:    %[[D3:.+]] = arith.subf %[[LP3]], %[[SC3]] : tensor<f64>
// CHECK-NEXT:    %[[DELTA:.+]] = arith.addf %[[W1]], %[[D3]] : tensor<f64>
// CHECK-NEXT:    %[[ZSL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 3], limit_indices = [1, 4]
// CHECK-NEXT:    %[[Z:.+]] = impulse.reshape %[[ZSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[SC4SL:.+]] = impulse.slice %[[TRACE]] <start_indices = [0, 7], limit_indices = [1, 8]
// CHECK-NEXT:    %[[SC4:.+]] = impulse.reshape %[[SC4SL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[A0:.+]] = impulse.dynamic_update_slice %[[ZFULL]], %[[VALS1]], %[[C0]], %[[C0]]
// CHECK-NEXT:    %[[LP1W:.+]] = impulse.reshape %[[LP1]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A1:.+]] = impulse.dynamic_update_slice %[[A0]], %[[LP1W]], %[[C0]], %[[C4]]
// CHECK-NEXT:    %[[SC2W:.+]] = impulse.reshape %[[SC2]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A2:.+]] = impulse.dynamic_update_slice %[[A1]], %[[SC2W]], %[[C0]], %[[C5]]
// CHECK-NEXT:    %[[LP3W:.+]] = impulse.reshape %[[LP3]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A3:.+]] = impulse.dynamic_update_slice %[[A2]], %[[LP3W]], %[[C0]], %[[C6]]
// CHECK-NEXT:    %[[SC4W:.+]] = impulse.reshape %[[SC4]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[A4:.+]] = impulse.dynamic_update_slice %[[A3]], %[[SC4W]], %[[C0]], %[[C7]]
// CHECK-NEXT:    return %[[A4]], %[[DELTA]], %[[DRAW]]#0, %[[Z]] : tensor<1x8xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
