// RUN: %eopt --expand-impulse --split-input-file %s | FileCheck %s --check-prefixes=BASE,HIER,CHECK

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %s#1, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<2> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %t#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test_base(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %prev_trace = arith.constant dense<[[0.5, 1.0]]> : tensor<1x2xf64>
    %res:4 = impulse.regenerate @model(%rng, %mean, %stddev) given %prev_trace
        <{ all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], selection = [[#impulse.symbol<2>]] }>
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// BASE-LABEL: func.func @test_base
// BASE-SAME: (%[[BASE_RNG:.+]]: tensor<2xui64>, %[[BASE_MEAN:.+]]: tensor<f64>, %[[BASE_STDDEV:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// BASE-NEXT: %[[BASE_CST:.+]] = arith.constant dense<{{.*}}> : tensor<1x2xf64>
// BASE-NEXT: %[[BASE_CALL:.+]]:4 = call @model.regenerate(%[[BASE_CST]], %[[BASE_RNG]], %[[BASE_MEAN]], %[[BASE_STDDEV]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// BASE-NEXT: return %[[BASE_CALL]]#0, %[[BASE_CALL]]#1, %[[BASE_CALL]]#2, %[[BASE_CALL]]#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>

// BASE-LABEL: func.func @model.regenerate
// BASE-SAME: (%[[ARG0:.+]]: tensor<1x2xf64>, %[[ARG1:.+]]: tensor<2xui64>, %[[ARG2:.+]]: tensor<f64>, %[[ARG3:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// BASE-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// BASE-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// BASE-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// BASE-DAG: %[[TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// BASE: %[[KEPT_SLICED:.+]] = impulse.slice %[[ARG0]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// BASE-NEXT: %[[KEPT:.+]] = impulse.reshape %[[KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// BASE-NEXT: %[[LP1:.+]] = call @logpdf(%[[KEPT]], %[[ARG2]], %[[ARG3]])
// BASE-NEXT: %[[W1:.+]] = arith.addf %[[LP1]], %[[ZERO]] : tensor<f64>
// BASE-NEXT: %[[RS1:.+]] = impulse.reshape %[[KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// BASE-NEXT: %[[TR1:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[RS1]], %[[C0]], %[[C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// BASE-NEXT: %[[REGEN:.+]]:2 = call @normal(%[[ARG1]], %[[KEPT]], %[[ARG3]])
// BASE-NEXT: %[[LP2:.+]] = call @logpdf(%[[REGEN]]#1, %[[KEPT]], %[[ARG3]])
// BASE-NEXT: %[[W2:.+]] = arith.addf %[[W1]], %[[LP2]] : tensor<f64>
// BASE-NEXT: %[[RS2:.+]] = impulse.reshape %[[REGEN]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// BASE-NEXT: %[[TR2:.+]] = impulse.dynamic_update_slice %[[TR1]], %[[RS2]], %[[C0]], %[[C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// BASE-NEXT: return %[[TR2]], %[[W2]], %[[REGEN]]#0, %[[REGEN]]#1

// -----

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @inner(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<3> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = impulse.sample @normal(%s#0, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<4> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %s#1, %t#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @outer(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:3 = impulse.sample @inner(%s#0, %s#1, %stddev) <{ symbol = #impulse.symbol<2> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t#0, %t#1, %t#2 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test_hier(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %prev_trace = arith.constant dense<[[0.5, 1.0, 2.0]]> : tensor<1x3xf64>
    %res:5 = impulse.regenerate @outer(%rng, %mean, %stddev) given %prev_trace
        <{ all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>, #impulse.symbol<3>], [#impulse.symbol<2>, #impulse.symbol<4>]],
          selection = [[#impulse.symbol<2>, #impulse.symbol<4>]] }>
        : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3, %res#4 : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// HIER-LABEL: func.func @test_hier
// HIER-SAME: (%[[HIER_RNG:.+]]: tensor<2xui64>, %[[HIER_MEAN:.+]]: tensor<f64>, %[[HIER_STDDEV:.+]]: tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-NEXT: %[[HIER_CST:.+]] = arith.constant dense<{{.*}}> : tensor<1x3xf64>
// HIER-NEXT: %[[HIER_CALL:.+]]:5 = call @outer.regenerate(%[[HIER_CST]], %[[HIER_RNG]], %[[HIER_MEAN]], %[[HIER_STDDEV]]) : (tensor<1x3xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-NEXT: return %[[HIER_CALL]]#0, %[[HIER_CALL]]#1, %[[HIER_CALL]]#2, %[[HIER_CALL]]#3, %[[HIER_CALL]]#4 : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>

// HIER-LABEL: func.func @outer.regenerate
// HIER-SAME: (%[[O_ARG0:.+]]: tensor<1x3xf64>, %[[O_ARG1:.+]]: tensor<2xui64>, %[[O_ARG2:.+]]: tensor<f64>, %[[O_ARG3:.+]]: tensor<f64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-DAG: %[[O_C1:.+]] = arith.constant dense<1> : tensor<i64>
// HIER-DAG: %[[O_C0:.+]] = arith.constant dense<0> : tensor<i64>
// HIER-DAG: %[[O_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// HIER-DAG: %[[O_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x3xf64>
// HIER: %[[O_KEPT_SLICED:.+]] = impulse.slice %[[O_ARG0]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x3xf64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[O_KEPT:.+]] = impulse.reshape %[[O_KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// HIER-NEXT: %[[O_LP1:.+]] = call @logpdf(%[[O_KEPT]], %[[O_ARG2]], %[[O_ARG3]])
// HIER-NEXT: %[[O_W1:.+]] = arith.addf %[[O_LP1]], %[[O_ZERO]] : tensor<f64>
// HIER-NEXT: %[[O_RS1:.+]] = impulse.reshape %[[O_KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[O_TR1:.+]] = impulse.dynamic_update_slice %[[O_TRACE_INIT]], %[[O_RS1]], %[[O_C0]], %[[O_C0]] : (tensor<1x3xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// HIER-NEXT: %[[O_SUB_TRACE:.+]] = impulse.slice %[[O_ARG0]] <start_indices = [0, 1], limit_indices = [1, 3], strides = [1, 1]> : (tensor<1x3xf64>) -> tensor<1x2xf64>
// HIER-NEXT: %[[O_NESTED:.+]]:5 = call @inner.regenerate(%[[O_SUB_TRACE]], %[[O_ARG1]], %[[O_KEPT]], %[[O_ARG3]]) : (tensor<1x2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-NEXT: %[[O_W2:.+]] = arith.addf %[[O_W1]], %[[O_NESTED]]#1 : tensor<f64>
// HIER-NEXT: %[[O_TR2:.+]] = impulse.dynamic_update_slice %[[O_TR1]], %[[O_NESTED]]#0, %[[O_C0]], %[[O_C1]] : (tensor<1x3xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// HIER-NEXT: return %[[O_TR2]], %[[O_W2]], %[[O_NESTED]]#2, %[[O_NESTED]]#3, %[[O_NESTED]]#4

// HIER-LABEL: func.func @inner.regenerate
// HIER-SAME: (%[[I_ARG0:.+]]: tensor<1x2xf64>, %[[I_ARG1:.+]]: tensor<2xui64>, %[[I_ARG2:.+]]: tensor<f64>, %[[I_ARG3:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// HIER-DAG: %[[I_C1:.+]] = arith.constant dense<1> : tensor<i64>
// HIER-DAG: %[[I_C0:.+]] = arith.constant dense<0> : tensor<i64>
// HIER-DAG: %[[I_ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// HIER-DAG: %[[I_TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// HIER: %[[I_KEPT_SLICED:.+]] = impulse.slice %[[I_ARG0]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[I_KEPT:.+]] = impulse.reshape %[[I_KEPT_SLICED]] : (tensor<1x1xf64>) -> tensor<f64>
// HIER-NEXT: %[[I_LP1:.+]] = call @logpdf(%[[I_KEPT]], %[[I_ARG2]], %[[I_ARG3]])
// HIER-NEXT: %[[I_W1:.+]] = arith.addf %[[I_LP1]], %[[I_ZERO]] : tensor<f64>
// HIER-NEXT: %[[I_RS1:.+]] = impulse.reshape %[[I_KEPT]] : (tensor<f64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[I_TR1:.+]] = impulse.dynamic_update_slice %[[I_TRACE_INIT]], %[[I_RS1]], %[[I_C0]], %[[I_C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// HIER-NEXT: %[[I_REGEN:.+]]:2 = call @normal(%[[I_ARG1]], %[[I_ARG2]], %[[I_ARG3]])
// HIER-NEXT: %[[I_LP2:.+]] = call @logpdf(%[[I_REGEN]]#1, %[[I_ARG2]], %[[I_ARG3]])
// HIER-NEXT: %[[I_W2:.+]] = arith.addf %[[I_W1]], %[[I_LP2]] : tensor<f64>
// HIER-NEXT: %[[I_RS2:.+]] = impulse.reshape %[[I_REGEN]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// HIER-NEXT: %[[I_TR2:.+]] = impulse.dynamic_update_slice %[[I_TR1]], %[[I_RS2]], %[[I_C0]], %[[I_C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// HIER-NEXT: return %[[I_TR2]], %[[I_W2]], %[[I_REGEN]]#0, %[[I_KEPT]], %[[I_REGEN]]#1

// -----

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %pred : tensor<i1>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %if:2 = impulse.if (%pred) ({
      %e:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<1> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %e#0, %e#1 : tensor<2xui64>, tensor<f64>
    }, {
      %l:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf_fn = @logpdf, symbol = #impulse.symbol<2> }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %l#0, %l#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %pred : tensor<i1>, %mean : tensor<f64>, %stddev : tensor<f64>, %trace : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:4 = impulse.regenerate @model(%rng, %pred, %mean, %stddev) given %trace
        <{ all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], selection = [[#impulse.symbol<1>]] }>
        : (tensor<2xui64>, tensor<i1>, tensor<f64>, tensor<f64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x2xf64>, %[[ARG1:.+]]: tensor<2xui64>, %[[ARG2:.+]]: tensor<i1>, %[[ARG3:.+]]: tensor<f64>, %[[ARG4:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK: %[[IF:.+]]:4 = impulse.if(%[[ARG2]]) ({
// CHECK-NEXT: %[[TS:.+]]:2 = func.call @normal(%[[ARG1]], %[[ARG3]], %[[ARG4]]) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: %[[TLP:.+]] = func.call @logpdf(%[[TS]]#1, %[[ARG3]], %[[ARG4]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[TW:.+]] = arith.addf %[[TLP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[TRS:.+]] = impulse.reshape %[[TS]]#1 : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[TTR:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[TRS]], %[[C0]], %[[C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: impulse.yield %[[TS]]#0, %[[TS]]#1, %[[TTR]], %[[TW]] : tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[SLICE:.+]] = impulse.slice %[[ARG0]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[FVAL:.+]] = impulse.reshape %[[SLICE]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[FLP:.+]] = func.call @logpdf(%[[FVAL]], %[[ARG3]], %[[ARG4]]) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[FW:.+]] = arith.addf %[[FLP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[FRS:.+]] = impulse.reshape %[[FVAL]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[FTR:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[FRS]], %[[C0]], %[[C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: impulse.yield %[[ARG1]], %[[FVAL]], %[[FTR]], %[[FW]] : tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
// CHECK-NEXT: return %[[IF]]#2, %[[IF]]#3, %[[IF]]#0, %[[IF]]#1 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>

// -----

// Branch sites read the constraint: the old trace slot is stale after a flip.

module {
  func.func private @pois(%rng : tensor<2xui64>, %rate : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @poislp(%y : tensor<f64>, %rate : tensor<f64>) -> tensor<f64>

  func.func @model(%rng : tensor<2xui64>, %pred : tensor<i1>, %l1 : tensor<f64>, %l2 : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %if:2 = impulse.if (%pred) ({
      %e:2 = impulse.sample @pois(%rng, %l1) <{ logpdf_fn = @poislp, symbol = #impulse.symbol<1> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %e#0, %e#1 : tensor<2xui64>, tensor<f64>
    }, {
      %l:2 = impulse.sample @pois(%rng, %l2) <{ logpdf_fn = @poislp, symbol = #impulse.symbol<2> }> : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
      impulse.yield %l#0, %l#1 : tensor<2xui64>, tensor<f64>
    }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>)
    return %if#0, %if#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %pred : tensor<i1>, %l1 : tensor<f64>, %l2 : tensor<f64>, %trace : tensor<1x2xf64>, %constraint : tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %res:4 = impulse.regenerate @model(%rng, %pred, %l1, %l2) given %trace constrained_by %constraint
        <{ all_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]], selection = [], constrained_addresses = [[#impulse.symbol<1>], [#impulse.symbol<2>]] }>
        : (tensor<2xui64>, tensor<i1>, tensor<f64>, tensor<f64>, tensor<1x2xf64>, tensor<1x2xf64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @test
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2xui64>, %[[ARG1:.+]]: tensor<i1>, %[[ARG2:.+]]: tensor<f64>, %[[ARG3:.+]]: tensor<f64>, %[[TRACE:.+]]: tensor<1x2xf64>, %[[CONSTRAINT:.+]]: tensor<1x2xf64>)
// CHECK-NEXT: %[[CALL:.+]]:4 = call @model.regenerate(%[[TRACE]], %[[CONSTRAINT]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]) : (tensor<1x2xf64>, tensor<1x2xf64>, tensor<2xui64>, tensor<i1>, tensor<f64>, tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-NEXT: return %[[CALL]]#0, %[[CALL]]#1, %[[CALL]]#2, %[[CALL]]#3

// CHECK-LABEL: func.func @model.regenerate
// CHECK-SAME: (%[[R_TRACE:.+]]: tensor<1x2xf64>, %[[R_CONSTRAINT:.+]]: tensor<1x2xf64>, %[[R_RNG:.+]]: tensor<2xui64>, %[[R_PRED:.+]]: tensor<i1>, %[[R_L1:.+]]: tensor<f64>, %[[R_L2:.+]]: tensor<f64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG: %[[TRACE_INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<1x2xf64>
// CHECK: %[[IF:.+]]:4 = impulse.if(%[[R_PRED]]) ({
// CHECK-NEXT: %[[TSLICE:.+]] = impulse.slice %[[R_CONSTRAINT]] <start_indices = [0, 0], limit_indices = [1, 1], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[TVAL:.+]] = impulse.reshape %[[TSLICE]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[TLP:.+]] = func.call @poislp(%[[TVAL]], %[[R_L1]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[TW:.+]] = arith.addf %[[TLP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[TRS:.+]] = impulse.reshape %[[TVAL]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[TTR:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[TRS]], %[[C0]], %[[C0]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: impulse.yield %[[R_RNG]], %[[TVAL]], %[[TTR]], %[[TW]] : tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>
// CHECK-NEXT: }, {
// CHECK-NEXT: %[[FSLICE:.+]] = impulse.slice %[[R_CONSTRAINT]] <start_indices = [0, 1], limit_indices = [1, 2], strides = [1, 1]> : (tensor<1x2xf64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[FVAL:.+]] = impulse.reshape %[[FSLICE]] : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[FLP:.+]] = func.call @poislp(%[[FVAL]], %[[R_L2]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT: %[[FW:.+]] = arith.addf %[[FLP]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT: %[[FRS:.+]] = impulse.reshape %[[FVAL]] : (tensor<f64>) -> tensor<1x1xf64>
// CHECK-NEXT: %[[FTR:.+]] = impulse.dynamic_update_slice %[[TRACE_INIT]], %[[FRS]], %[[C0]], %[[C1]] : (tensor<1x2xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK-NEXT: impulse.yield %[[R_RNG]], %[[FVAL]], %[[FTR]], %[[FW]] : tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>
// CHECK-NEXT: }) : (tensor<i1>) -> (tensor<2xui64>, tensor<f64>, tensor<1x2xf64>, tensor<f64>)
// CHECK-NEXT: return %[[IF]]#2, %[[IF]]#3, %[[IF]]#0, %[[IF]]#1 : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
