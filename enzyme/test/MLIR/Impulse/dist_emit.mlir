// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @normal_sampler(%r: tensor<2xui64>, %mu: tensor<f64>, %sig: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %mu : tensor<2xui64>, tensor<f64>
  }
  func.func @model(%r: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
    %mu = arith.constant dense<0.5> : tensor<f64>
    %sig = arith.constant dense<2.0> : tensor<f64>
    %o:2 = impulse.sample @normal_sampler(%r, %mu, %sig) {
      symbol = #impulse.symbol<1>,
      dist = #impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @top(%r: tensor<2xui64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.simulate @model(%r) {selection = [[#impulse.symbol<1>]]}
      : (tensor<2xui64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func @model.simulate
// CHECK-DAG:   %[[LOG2PI:.+]] = arith.constant dense<1.8378770664093453> : tensor<f64>
// CHECK-DAG:   %[[NEGHALF:.+]] = arith.constant dense<-5.000000e-01> : tensor<f64>
// CHECK-DAG:   %[[LOGSIG:.+]] = arith.constant dense<0.69314718055994529> : tensor<f64>
// CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[MU:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG:   %[[SIG:.+]] = arith.constant dense<2.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[RNG:.+]], %[[Z:.+]] = impulse.random %arg0, %[[ZERO]], %[[ONE]] {rng_distribution = #impulse.rng_distribution<NORMAL>}
// CHECK-NEXT:  %[[SCALED:.+]] = arith.mulf %[[Z]], %[[SIG]] : tensor<f64>
// CHECK-NEXT:  %[[Y:.+]] = arith.addf %[[SCALED]], %[[MU]] : tensor<f64>
// CHECK-NEXT:  %[[CENT:.+]] = arith.subf %[[Y]], %[[MU]] : tensor<f64>
// CHECK-NEXT:  %[[ZREC:.+]] = arith.divf %[[CENT]], %[[SIG]] : tensor<f64>
// CHECK-NEXT:  %[[LOGDET:.+]] = impulse.reshape %[[LOGSIG]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LOGDET0:.+]] = arith.addf %[[LOGDET]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[Z2:.+]] = arith.mulf %[[ZREC]], %[[ZREC]] : tensor<f64>
// CHECK-NEXT:  %[[SUM:.+]] = arith.addf %[[Z2]], %[[LOG2PI]] : tensor<f64>
// CHECK-NEXT:  %[[KERNEL:.+]] = arith.mulf %[[SUM]], %[[NEGHALF]] : tensor<f64>
// CHECK-NEXT:  %[[BASE:.+]] = impulse.reshape %[[KERNEL]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LP:.+]] = arith.subf %[[BASE]], %[[LOGDET0]] : tensor<f64>
// CHECK-NEXT:  %[[W:.+]] = arith.addf %[[LP]], %[[ZERO]] : tensor<f64>
