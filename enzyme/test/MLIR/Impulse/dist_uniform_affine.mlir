// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @unif_sampler(%r: tensor<2xui64>, %lo: tensor<f64>, %sc: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %lo : tensor<2xui64>, tensor<f64>
  }
  func.func @model(%r: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
    %lo = arith.constant dense<-2.0> : tensor<f64>
    %sc = arith.constant dense<4.0> : tensor<f64>
    %o:2 = impulse.sample @unif_sampler(%r, %lo, %sc) {
      symbol = #impulse.symbol<1>,
      dist = #impulse.dist<STD_UNIFORM, [#impulse.xform<AFFINE>]>
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
// CHECK-DAG:   %[[NEGINF:.+]] = arith.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-DAG:   %[[LOG4:.+]] = arith.constant dense<1.3862943611198906> : tensor<f64>
// CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[LO:.+]] = arith.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[SC:.+]] = arith.constant dense<4.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[RNG:.+]], %[[U:.+]] = impulse.random %arg0, %[[ZERO]], %[[ONE]] {rng_distribution = #impulse.rng_distribution<UNIFORM>}
// CHECK-NEXT:  %[[SCALED:.+]] = arith.mulf %[[U]], %[[SC]] : tensor<f64>
// CHECK-NEXT:  %[[Y:.+]] = arith.addf %[[SCALED]], %[[LO]] : tensor<f64>
// CHECK-NEXT:  %[[CENT:.+]] = arith.subf %[[Y]], %[[LO]] : tensor<f64>
// CHECK-NEXT:  %[[Z:.+]] = arith.divf %[[CENT]], %[[SC]] : tensor<f64>
// CHECK-NEXT:  %[[LOGDET:.+]] = impulse.reshape %[[LOG4]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LOGDET0:.+]] = arith.addf %[[LOGDET]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[GE:.+]] = arith.cmpf oge, %[[Z]], %[[ZERO]] : tensor<f64>
// CHECK-NEXT:  %[[LE:.+]] = arith.cmpf ole, %[[Z]], %[[ONE]] : tensor<f64>
// CHECK-NEXT:  %[[IN:.+]] = arith.andi %[[GE]], %[[LE]] : tensor<i1>
// CHECK-NEXT:  %[[TERM:.+]] = arith.select %[[IN]], %[[ZERO]], %[[NEGINF]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:  %[[BASE:.+]] = impulse.reshape %[[TERM]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LP:.+]] = arith.subf %[[BASE]], %[[LOGDET0]] : tensor<f64>
// CHECK-NEXT:  %[[W:.+]] = arith.addf %[[LP]], %[[ZERO]] : tensor<f64>
// CHECK:       return %{{.+}}, %[[W]], %[[RNG]], %[[Y]]
