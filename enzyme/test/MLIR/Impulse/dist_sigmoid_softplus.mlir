// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func @model(%r: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %p:2 = impulse.sample(%r) <{
      symbol = #impulse.symbol<1>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<SIGMOID>]>
    }> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
    %s:2 = impulse.sample(%p#0) <{
      symbol = #impulse.symbol<2>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<SOFTPLUS>]>
    }> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %p#1, %s#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }
  func.func @top(%r: tensor<2xui64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>) {
    %t, %w, %rng, %p, %s = impulse.simulate @model(%r) <{selection = [[#impulse.symbol<1>], [#impulse.symbol<2>]]}>
      : (tensor<2xui64>) -> (tensor<1x2xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t, %w, %rng : tensor<1x2xf64>, tensor<f64>, tensor<2xui64>
  }
}

// y = sigmoid(x): x = logit(y), log-det log sigmoid(x) + log sigmoid(-x),
// each log sigmoid(v) = -log_add_exp(-v, 0).
// CHECK-LABEL: func.func @model.simulate
// CHECK-SAME:  (%[[KEY:.+]]: tensor<2xui64>)
// CHECK-DAG:   %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
// CHECK-DAG:   %[[LOG2PI:.+]] = arith.constant dense<1.8378770664093453> : tensor<f64>
// CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[RNG:.+]], %[[Z:.+]] = impulse.random %[[KEY]], %[[ZERO]], %[[ONE]] <rng_distribution = NORMAL>
// CHECK-NEXT:  %[[P:.+]] = impulse.logistic %[[Z]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[ONEMP:.+]] = arith.subf %[[ONE]], %[[P]] : tensor<f64>
// CHECK-NEXT:  %[[LOGP:.+]] = math.log %[[P]] : tensor<f64>
// CHECK-NEXT:  %[[LOG1MP:.+]] = math.log %[[ONEMP]] : tensor<f64>
// CHECK-NEXT:  %[[X:.+]] = arith.subf %[[LOGP]], %[[LOG1MP]] : tensor<f64>
// CHECK-NEXT:  %[[NEGX:.+]] = arith.negf %[[X]] : tensor<f64>
// CHECK-NEXT:  %[[MAX1:.+]] = arith.maximumf %[[NEGX]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[NAN1:.+]] = arith.cmpf une, %[[NEGX]], %[[NEGX]] : tensor<f64>
// CHECK-NEXT:  %[[SUM1:.+]] = arith.addf %[[NEGX]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[ABS1:.+]] = math.absf %[[NEGX]] : tensor<f64>
// CHECK-NEXT:  %[[NABS1:.+]] = arith.negf %[[ABS1]] : tensor<f64>
// CHECK-NEXT:  %[[EXP1:.+]] = math.exp %[[NABS1]] : tensor<f64>
// CHECK-NEXT:  %[[L1P1:.+]] = math.log1p %[[EXP1]] : tensor<f64>
// CHECK-NEXT:  %[[LSE1:.+]] = arith.addf %[[MAX1]], %[[L1P1]] : tensor<f64>
// CHECK-NEXT:  %[[SEL1:.+]] = arith.select %[[NAN1]], %[[SUM1]], %[[LSE1]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:  %[[LSX:.+]] = arith.negf %[[SEL1]] : tensor<f64>
// CHECK-NEXT:  %[[MAX2:.+]] = arith.maximumf %[[X]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[NAN2:.+]] = arith.cmpf une, %[[X]], %[[X]] : tensor<f64>
// CHECK-NEXT:  %[[SUM2:.+]] = arith.addf %[[X]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[ABS2:.+]] = math.absf %[[X]] : tensor<f64>
// CHECK-NEXT:  %[[NABS2:.+]] = arith.negf %[[ABS2]] : tensor<f64>
// CHECK-NEXT:  %[[EXP2:.+]] = math.exp %[[NABS2]] : tensor<f64>
// CHECK-NEXT:  %[[L1P2:.+]] = math.log1p %[[EXP2]] : tensor<f64>
// CHECK-NEXT:  %[[LSE2:.+]] = arith.addf %[[MAX2]], %[[L1P2]] : tensor<f64>
// CHECK-NEXT:  %[[SEL2:.+]] = arith.select %[[NAN2]], %[[SUM2]], %[[LSE2]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:  %[[LSNX:.+]] = arith.negf %[[SEL2]] : tensor<f64>
// CHECK-NEXT:  %[[LD:.+]] = arith.addf %[[LSX]], %[[LSNX]] : tensor<f64>
// CHECK-NEXT:  %[[LDR:.+]] = impulse.reshape %[[LD]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LDT:.+]] = arith.addf %[[LDR]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[X2:.+]] = arith.mulf %[[X]], %[[X]] : tensor<f64>
// CHECK-NEXT:  %[[X2P:.+]] = arith.addf %[[X2]], %[[LOG2PI]] : tensor<f64>
// CHECK-NEXT:  %[[KERNEL:.+]] = arith.mulf %[[X2P]], %[[NEGHALF]] : tensor<f64>
// CHECK-NEXT:  %[[BASE:.+]] = impulse.reshape %[[KERNEL]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LP:.+]] = arith.subf %[[BASE]], %[[LDT]] : tensor<f64>
// CHECK-NEXT:  %[[W1:.+]] = arith.addf %[[LP]], %[[ZERO]] : tensor<f64>

// y = softplus(x): x = log(exp(y) - 1), log-det log sigmoid(x).
// CHECK:       %[[RNG2:.+]], %[[Z2:.+]] = impulse.random %[[RNG]], %[[ZERO]], %[[ONE]] <rng_distribution = NORMAL>
// CHECK-NEXT:  %[[EZ:.+]] = math.exp %[[Z2]] : tensor<f64>
// CHECK-NEXT:  %[[S:.+]] = math.log1p %[[EZ]] : tensor<f64>
// CHECK-NEXT:  %[[ES:.+]] = math.exp %[[S]] : tensor<f64>
// CHECK-NEXT:  %[[ESM1:.+]] = arith.subf %[[ES]], %[[ONE]] : tensor<f64>
// CHECK-NEXT:  %[[XS:.+]] = math.log %[[ESM1]] : tensor<f64>
// CHECK-NEXT:  %[[NEGXS:.+]] = arith.negf %[[XS]] : tensor<f64>
// CHECK-NEXT:  %[[MAX3:.+]] = arith.maximumf %[[NEGXS]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[NAN3:.+]] = arith.cmpf une, %[[NEGXS]], %[[NEGXS]] : tensor<f64>
// CHECK-NEXT:  %[[SUM3:.+]] = arith.addf %[[NEGXS]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[ABS3:.+]] = math.absf %[[NEGXS]] : tensor<f64>
// CHECK-NEXT:  %[[NABS3:.+]] = arith.negf %[[ABS3]] : tensor<f64>
// CHECK-NEXT:  %[[EXP3:.+]] = math.exp %[[NABS3]] : tensor<f64>
// CHECK-NEXT:  %[[L1P3:.+]] = math.log1p %[[EXP3]] : tensor<f64>
// CHECK-NEXT:  %[[LSE3:.+]] = arith.addf %[[MAX3]], %[[L1P3]] : tensor<f64>
// CHECK-NEXT:  %[[SEL3:.+]] = arith.select %[[NAN3]], %[[SUM3]], %[[LSE3]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:  %[[LSXS:.+]] = arith.negf %[[SEL3]] : tensor<f64>
// CHECK-NEXT:  %[[LDSR:.+]] = impulse.reshape %[[LSXS]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LDST:.+]] = arith.addf %[[LDSR]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[XS2:.+]] = arith.mulf %[[XS]], %[[XS]] : tensor<f64>
