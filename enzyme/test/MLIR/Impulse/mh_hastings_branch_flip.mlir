// RUN: %eopt --expand-impulse %s | FileCheck %s

// A latent made live by a flip is drawn fresh; its prior density is removed.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %z:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name = "z" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %pred = arith.cmpf ogt, %z#1, %mean : tensor<f64>
    %if:2 = impulse.if (%pred) ({
      %a:2 = impulse.sample @normal(%z#0, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<2>, name = "a" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %a#0, %a#1 : tensor<2xui64>, tensor<f64>
    }, {
      %b:2 = impulse.sample @normal(%z#0, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<3>, name = "b" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %b#0, %b#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @mh(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x3xf64>, %weight : tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>) {
    %step:4 = impulse.mh @model(%rng, %mean, %stddev) given %trace weight %weight
        <{ name = "mh", selection = [[#impulse.symbol<1>], [#impulse.symbol<2>], [#impulse.symbol<3>]], regenerate_addresses = [[#impulse.symbol<1>]] }>
        : (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>)
    return %step#0, %step#1, %step#2, %step#3 : tensor<1x3xf64>, tensor<f64>, tensor<i1>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @mh
// CHECK-SAME: %[[TR:[^:]+]]: tensor<1x3xf64>, %{{[^:]+}}: tensor<f64>)
// CHECK-DAG: %[[I0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK: %[[REGEN:.+]]:4 = call @model.regenerate(
// CHECK-NEXT: %[[WSL:.+]] = impulse.slice %[[REGEN]]#1 <start_indices = [0], limit_indices = [1], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[W:.+]] = impulse.reshape %[[WSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CSL:.+]] = impulse.slice %[[REGEN]]#1 <start_indices = [1], limit_indices = [2], strides = [1]> : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[CORR:.+]] = impulse.reshape %[[CSL]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[WDIFF:.+]] = arith.subf %[[W]], %{{.+}} : tensor<f64>
// CHECK-NEXT: %[[ALPHA:.+]] = arith.subf %[[WDIFF]], %[[CORR]] : tensor<f64>
// CHECK: %[[ACC:.+]] = arith.cmpf olt, %{{.+}}, %[[ALPHA]] : tensor<f64>
// CHECK-NEXT: %[[WP:.+]] = impulse.slice %[[REGEN]]#0 <start_indices = [0, 0], limit_indices = [1, 3], strides = [1, 1]> : (tensor<1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[WC:.+]] = impulse.slice %[[TR]] <start_indices = [0, 0], limit_indices = [1, 3], strides = [1, 1]> : (tensor<1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %[[WU:.+]] = impulse.select %[[ACC]], %[[WP]], %[[WC]] : (tensor<i1>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT: %{{.+}} = impulse.dynamic_update_slice %[[TR]], %[[WU]], %[[I0]], %[[I0]] : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>

// CHECK-LABEL: func.func @model.regenerate(
// CHECK-SAME: %[[TR:[^:]+]]: tensor<1x3xf64>, %[[R:[^:]+]]: tensor<2xui64>, %[[M:[^:]+]]: tensor<f64>, %[[SD:[^:]+]]: tensor<f64>) -> (tensor<1x3xf64>, tensor<2xf64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK: %[[OLD:.+]]:4 = call @model.regenerate_0(%[[TR]], %[[R]], %[[M]], %[[SD]]) : (tensor<1x3xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x3xf64>, tensor<2x2xf64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[Z:.+]]:2 = call @normal(%[[R]], %[[M]], %[[SD]])
// CHECK-NEXT: %[[LPZ:.+]] = call @logpdf(%[[Z]]#1, %[[M]], %[[SD]])
// CHECK-NEXT: %[[W1:.+]] = arith.addf %[[LPZ]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[C1:.+]] = arith.addf %[[LPZ]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[ZOSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]>
// CHECK-NEXT: %[[ZO:.+]] = impulse.reshape %[[ZOSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[LPZO:.+]] = call @logpdf(%[[ZO]], %[[M]], %[[SD]])
// CHECK-NEXT: %[[C2:.+]] = arith.subf %[[C1]], %[[LPZO]] : tensor<f64>
// CHECK: %[[PRED:.+]] = arith.cmpf ogt, %[[Z]]#1, %[[M]] : tensor<f64>
// CHECK-NEXT: %[[IF:.+]]:5 = impulse.if(%[[PRED]]) ({
// CHECK-NEXT: %[[AOSL:.+]] = impulse.slice %[[TR]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]>
// CHECK-NEXT: %[[AO:.+]] = impulse.reshape %[[AOSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[ALIVESL:.+]] = impulse.slice %[[OLD]]#1 <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[ALIVEF:.+]] = impulse.reshape %[[ALIVESL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[ALIVE:.+]] = arith.cmpf ogt, %[[ALIVEF]], %[[HALF]] : tensor<f64>
// CHECK-NEXT: %[[AF:.+]]:2 = func.call @normal(%[[Z]]#0, %[[M]], %[[SD]])
// CHECK-NEXT: %[[A:.+]] = impulse.select %[[ALIVE]], %[[AO]], %[[AF]]#1 : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[LPA:.+]] = func.call @logpdf(%[[A]], %[[M]], %[[SD]])
// CHECK-NEXT: %[[WA:.+]] = arith.addf %[[W1]], %[[LPA]] : tensor<f64>
// CHECK-NEXT: %[[AOLDSL:.+]] = impulse.slice %[[OLD]]#1 <start_indices = [1, 0], limit_indices = [2, 1], strides = [1, 1]> : (tensor<2x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[AOLD:.+]] = impulse.reshape %[[AOLDSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[ATERM:.+]] = arith.select %[[ALIVE]], %[[AOLD]], %[[LPA]] : tensor<i1>, tensor<f64>
// CHECK-NEXT: %[[CA:.+]] = arith.addf %[[C2]], %[[ATERM]] : tensor<f64>
// CHECK: impulse.yield %[[AF]]#0, %[[A]], %{{.+}}, %[[WA]], %[[CA]] : tensor<2xui64>, tensor<f64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<1x3xf64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT: %[[OASL:.+]] = impulse.slice %[[OLD]]#1 <start_indices = [1, 0], limit_indices = [2, 1], strides = [1, 1]>
// CHECK-NEXT: %[[OA:.+]] = impulse.reshape %[[OASL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CX:.+]] = arith.subf %[[IF]]#4, %[[OA]] : tensor<f64>
// CHECK-NEXT: %[[OBSL:.+]] = impulse.slice %[[OLD]]#1 <start_indices = [1, 1], limit_indices = [2, 2], strides = [1, 1]>
// CHECK-NEXT: %[[OB:.+]] = impulse.reshape %[[OBSL]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[CY:.+]] = arith.subf %[[CX]], %[[OB]] : tensor<f64>
// CHECK-NEXT: %[[WFL:.+]] = impulse.reshape %[[IF]]#3 : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[P0:.+]] = impulse.dynamic_update_slice %{{.+}}, %[[WFL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[CFL:.+]] = impulse.reshape %[[CY]] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[P1:.+]] = impulse.dynamic_update_slice %[[P0]], %[[CFL]], %{{.+}} : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: return %[[IF]]#2, %[[P1]], %[[IF]]#0, %[[IF]]#1

// Scoring pass weight: row 0 liveness, row 1 liveness-masked density.
// CHECK-LABEL: func.func @model.regenerate_0(
// CHECK-SAME: -> (tensor<1x3xf64>, tensor<2x2xf64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[S_C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[S_ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<1x1xf64>
// CHECK-DAG: %[[S_C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[S_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<2x2xf64>
// CHECK: %[[S_IF:.+]]:4 = impulse.if(
// CHECK: %[[S_LPA:.+]] = func.call @logpdf(
// CHECK-NEXT: %[[S_LPAR:.+]] = impulse.reshape %[[S_LPA]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[S_A0:.+]] = impulse.dynamic_update_slice %[[S_INIT]], %[[S_ONE]], %[[S_C0]], %[[S_C0]] : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT: %[[S_A1:.+]] = impulse.dynamic_update_slice %[[S_A0]], %[[S_LPAR]], %[[S_C1]], %[[S_C0]] : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK: impulse.yield %{{.+}}, %{{.+}}, %{{.+}}, %[[S_A1]] : tensor<2xui64>, tensor<f64>, tensor<1x3xf64>, tensor<2x2xf64>
// CHECK: %[[S_LPB:.+]] = func.call @logpdf(
// CHECK-NEXT: %[[S_LPBR:.+]] = impulse.reshape %[[S_LPB]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[S_B0:.+]] = impulse.dynamic_update_slice %[[S_INIT]], %[[S_ONE]], %[[S_C0]], %[[S_C1]] : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK-NEXT: %[[S_B1:.+]] = impulse.dynamic_update_slice %[[S_B0]], %[[S_LPBR]], %[[S_C1]], %[[S_C1]] : (tensor<2x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2x2xf64>
// CHECK: impulse.yield %{{.+}}, %{{.+}}, %{{.+}}, %[[S_B1]] : tensor<2xui64>, tensor<f64>, tensor<1x3xf64>, tensor<2x2xf64>
// CHECK: return %{{.+}}, %[[S_IF]]#3, %{{.+}}, %{{.+}} : tensor<1x3xf64>, tensor<2x2xf64>, tensor<2xui64>, tensor<f64>
