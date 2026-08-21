// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @gam_s(%rng : tensor<2xui64>, %a : tensor<f64>, %b : tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)

  func.func @model(%rng : tensor<2xui64>, %a : tensor<f64>, %b : tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>) {
    %s:2 = impulse.sample @gam_s(%rng, %a, %b) { symbol = #impulse.symbol<1>, dist = #impulse.dist<GAMMA>, support = #impulse.support<POSITIVE> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<1xf64>
  }

  func.func @test(%rng : tensor<2xui64>, %a : tensor<f64>, %b : tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1xf64>) {
    %prev = arith.constant dense<[[0.5]]> : tensor<1x1xf64>
    %res:4 = impulse.regenerate @model(%rng, %a, %b) given %prev
        { selection = [[#impulse.symbol<1>]], regenerate_addresses = [[#impulse.symbol<1>]] }
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1xf64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1xf64>
  }
}

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME:      (%[[TRACE:.+]]: tensor<1x1xf64>, %[[RNG:.+]]: tensor<2xui64>, %[[A:.+]]: tensor<f64>, %[[B:.+]]: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1xf64>)
// CHECK-DAG:     %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<1xf64>
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:     %[[ZTRACE:.+]] = arith.constant dense<0.000000e+00> : tensor<1x1xf64>
// CHECK:         %[[DRAW:.+]]:2 = call @gam_s(%[[RNG]], %[[A]], %[[B]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<1xf64>)
// CHECK-NEXT:    %[[AB:.+]] = "enzyme.broadcast"(%[[A]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[BB:.+]] = "enzyme.broadcast"(%[[B]]) <{shape = array<i64: 1>}> : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[AM1:.+]] = arith.subf %[[AB]], %[[ONE]] : tensor<1xf64>
// CHECK-NEXT:    %[[LOGX:.+]] = math.log %[[DRAW]]#1 : tensor<1xf64>
// CHECK-NEXT:    %[[T1:.+]] = arith.mulf %[[AM1]], %[[LOGX]] : tensor<1xf64>
// CHECK-NEXT:    %[[T2:.+]] = arith.mulf %[[BB]], %[[DRAW]]#1 : tensor<1xf64>
// CHECK-NEXT:    %[[KERN:.+]] = arith.subf %[[T1]], %[[T2]] : tensor<1xf64>
// CHECK-NEXT:    %[[LG:.+]] = impulse.lgamma %[[AB]] : (tensor<1xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[LOGB:.+]] = math.log %[[BB]] : tensor<1xf64>
// CHECK-NEXT:    %[[ALB:.+]] = arith.mulf %[[AB]], %[[LOGB]] : tensor<1xf64>
// CHECK-NEXT:    %[[NORM:.+]] = arith.subf %[[ALB]], %[[LG]] : tensor<1xf64>
// CHECK-NEXT:    %[[KS:.+]] = impulse.reshape %[[KERN]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[NS:.+]] = impulse.reshape %[[NORM]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[LP:.+]] = arith.addf %[[KS]], %[[NS]] : tensor<f64>
// CHECK-NEXT:    %[[W:.+]] = arith.addf %[[LP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:    %[[VAL2D:.+]] = impulse.reshape %[[DRAW]]#1 : (tensor<1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:    %[[NEWTRACE:.+]] = impulse.dynamic_update_slice %[[ZTRACE]], %[[VAL2D]], %[[C0]], %[[C0]] : (tensor<1x1xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
// CHECK-NEXT:    return %[[NEWTRACE]], %[[W]], %[[DRAW]]#0, %[[DRAW]]#1 : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<1xf64>
