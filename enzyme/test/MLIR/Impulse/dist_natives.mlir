// RUN: %eopt --expand-impulse %s | FileCheck %s

module {
  func.func private @exp_s(%r: tensor<2xui64>, %lam: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %lam : tensor<2xui64>, tensor<f64>
  }
  func.func @exp_model(%r: tensor<2xui64>, %lam: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @exp_s(%r, %lam) {symbol = #impulse.symbol<1>, dist = #impulse.dist<EXPONENTIAL>}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @exp_top(%r: tensor<2xui64>, %lam: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.simulate @exp_model(%r, %lam) {selection = [[#impulse.symbol<1>]]}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  func.func private @bern_s(%r: tensor<2xui64>, %lg: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %lg : tensor<2xui64>, tensor<f64>
  }
  func.func @bern_model(%r: tensor<2xui64>, %lg: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @bern_s(%r, %lg) {symbol = #impulse.symbol<2>, dist = #impulse.dist<BERNOULLI>}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @bern_top(%r: tensor<2xui64>, %lg: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.simulate @bern_model(%r, %lg) {selection = [[#impulse.symbol<2>]]}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  func.func private @du_s(%r: tensor<2xui64>, %lo: tensor<f64>, %hi: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %lo : tensor<2xui64>, tensor<f64>
  }
  func.func @du_model(%r: tensor<2xui64>, %lo: tensor<f64>, %hi: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @du_s(%r, %lo, %hi) {symbol = #impulse.symbol<3>, dist = #impulse.dist<DISCRETE_UNIFORM>}
      : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @du_top(%r: tensor<2xui64>, %lo: tensor<f64>, %hi: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.simulate @du_model(%r, %lo, %hi) {selection = [[#impulse.symbol<3>]]}
      : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  func.func private @gam_s(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %a : tensor<2xui64>, tensor<f64>
  }
  func.func @gam_model(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @gam_s(%r, %a, %b) {symbol = #impulse.symbol<4>, dist = #impulse.dist<GAMMA>}
      : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @gam_top(%r: tensor<2xui64>, %a: tensor<f64>, %b: tensor<f64>, %c: tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.generate @gam_model(%r, %a, %b) given %c {selection = [[#impulse.symbol<4>]], constrained_addresses = [[#impulse.symbol<4>]]}
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  func.func private @hn_s(%r: tensor<2xui64>, %s: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %s : tensor<2xui64>, tensor<f64>
  }
  func.func @hn_model(%r: tensor<2xui64>, %s: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @hn_s(%r, %s) {symbol = #impulse.symbol<5>, dist = #impulse.dist<HALF_NORMAL>}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @hn_top(%r: tensor<2xui64>, %s: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.simulate @hn_model(%r, %s) {selection = [[#impulse.symbol<5>]]}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  func.func private @hc_s(%r: tensor<2xui64>, %s: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %r, %s : tensor<2xui64>, tensor<f64>
  }
  func.func @hc_model(%r: tensor<2xui64>, %s: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %o:2 = impulse.sample @hc_s(%r, %s) {symbol = #impulse.symbol<6>, dist = #impulse.dist<HALF_CAUCHY>}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<f64>
  }
  func.func @hc_top(%r: tensor<2xui64>, %s: tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>) {
    %t, %w, %rng, %v = impulse.simulate @hc_model(%r, %s) {selection = [[#impulse.symbol<6>]]}
      : (tensor<2xui64>, tensor<f64>) -> (tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>)
    return %t, %w, %rng, %v : tensor<1x1xf64>, tensor<f64>, tensor<2xui64>, tensor<f64>
  }

  // lgamma(sum alpha) reduces per block (last axis); other terms reduce fully.
  func.func private @dir_s(%r: tensor<2xui64>, %a: tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    return %r, %a : tensor<2xui64>, tensor<3xf64>
  }
  func.func @dir_model(%r: tensor<2xui64>, %a: tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %o:2 = impulse.sample @dir_s(%r, %a) {symbol = #impulse.symbol<7>, dist = #impulse.dist<DIRICHLET>, support = #impulse.support<SIMPLEX event_size = 3 : i64>}
      : (tensor<2xui64>, tensor<3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %o#0, %o#1 : tensor<2xui64>, tensor<3xf64>
  }
  func.func @dir_top(%r: tensor<2xui64>, %a: tensor<3xf64>, %c: tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<3xf64>) {
    %t, %w, %rng, %v = impulse.generate @dir_model(%r, %a) given %c {selection = [[#impulse.symbol<7>]], constrained_addresses = [[#impulse.symbol<7>]]}
      : (tensor<2xui64>, tensor<3xf64>, tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<3xf64>)
    return %t, %w, %rng, %v : tensor<1x3xf64>, tensor<f64>, tensor<2xui64>, tensor<3xf64>
  }
}

// CHECK-LABEL: func.func @dir_model.generate
// CHECK-NOT:   impulse.random
// CHECK:       %[[A2:.+]] = impulse.reshape %arg2 : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %[[X2:.+]] = impulse.reshape %{{.+}} : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %[[AM1:.+]] = arith.subf %[[A2]], %{{.+}} : tensor<1x3xf64>
// CHECK-NEXT:  %[[LOGX:.+]] = math.log %[[X2]] : tensor<1x3xf64>
// CHECK-NEXT:  %[[KER:.+]] = arith.mulf %[[LOGX]], %[[AM1]] : tensor<1x3xf64>
// CHECK-NEXT:  %[[LGA:.+]] = impulse.lgamma %[[A2]] : (tensor<1x3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:  %[[LGAF:.+]] = impulse.reshape %[[LGA]] : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:  %[[TERMA:.+]] = impulse.dot %[[LGAF]], %{{.+}} {{.*}} : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
// CHECK-NEXT:  %[[SUMK:.+]] = impulse.dot %[[A2]], %{{.+}} {{.*}} : (tensor<1x3xf64>, tensor<3xf64>) -> tensor<1xf64>
// CHECK-NEXT:  %[[LGSK:.+]] = impulse.lgamma %[[SUMK]] : (tensor<1xf64>) -> tensor<1xf64>

// CHECK-LABEL: func.func @hc_model.simulate
// CHECK-DAG:   %[[HCLOG2PI:.+]] = arith.constant dense<-0.45158270528945482> : tensor<f64>
// CHECK-DAG:   %[[HCPI:.+]] = arith.constant dense<3.1415926535897931> : tensor<f64>
// CHECK-DAG:   %[[HCHALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG:   %[[HCEPS:.+]] = arith.constant dense<2.2204460492503131E-16> : tensor<f64>
// CHECK-DAG:   %[[HC1ME:.+]] = arith.constant dense<0.99999999999999978> : tensor<f64>
// CHECK-DAG:   %[[HCZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[HCONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[URNG:.+]], %[[U:.+]] = impulse.random %arg0, %[[HCZERO]], %[[HCONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>}
// CHECK-NEXT:  %[[UM:.+]] = arith.mulf %[[U]], %[[HC1ME]] : tensor<f64>
// CHECK-NEXT:  %[[UC:.+]] = arith.addf %[[UM]], %[[HCEPS]] : tensor<f64>
// CHECK-NEXT:  %[[SUBH:.+]] = arith.subf %[[UC]], %[[HCHALF]] : tensor<f64>
// CHECK-NEXT:  %[[ARG:.+]] = arith.mulf %[[SUBH]], %[[HCPI]] : tensor<f64>
// CHECK-NEXT:  %[[TAN:.+]] = math.tan %[[ARG]] : tensor<f64>
// CHECK-NEXT:  %[[ATAN:.+]] = math.absf %[[TAN]] : tensor<f64>
// CHECK-NEXT:  %[[YC:.+]] = arith.mulf %arg1, %[[ATAN]] : tensor<f64>
// CHECK-NEXT:  %[[XOS:.+]] = arith.divf %[[YC]], %arg1 : tensor<f64>
// CHECK-NEXT:  %[[SQ:.+]] = arith.mulf %[[XOS]], %[[XOS]] : tensor<f64>
// CHECK-NEXT:  %[[L1P:.+]] = math.log1p %[[SQ]] : tensor<f64>
// CHECK-NEXT:  %[[KER:.+]] = arith.negf %[[L1P]] : tensor<f64>
// CHECK-NEXT:  %[[LOGS:.+]] = math.log %arg1 : tensor<f64>
// CHECK-NEXT:  %[[NORM:.+]] = arith.subf %[[HCLOG2PI]], %[[LOGS]] : tensor<f64>
// CHECK-NEXT:  %[[KR:.+]] = impulse.reshape %[[KER]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[NR:.+]] = impulse.reshape %[[NORM]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LP:.+]] = arith.addf %[[KR]], %[[NR]] : tensor<f64>

// CHECK-LABEL: func.func @hn_model.simulate
// CHECK-DAG:   %[[HNLOG2PI:.+]] = arith.constant dense<0.91893853320467266> : tensor<f64>
// CHECK-DAG:   %[[HNLOG2:.+]] = arith.constant dense<0.69314718055994529> : tensor<f64>
// CHECK-DAG:   %[[HNTWO:.+]] = arith.constant dense<2.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[HNZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[HNONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[HNRNG:.+]], %[[Z:.+]] = impulse.random %arg0, %[[HNZERO]], %[[HNONE]] {rng_distribution = #impulse<rng_distribution NORMAL>}
// CHECK-NEXT:  %[[SCL:.+]] = arith.mulf %arg1, %[[Z]] : tensor<f64>
// CHECK-NEXT:  %[[Y:.+]] = math.absf %[[SCL]] : tensor<f64>
// CHECK-NEXT:  %[[SIG2:.+]] = arith.mulf %arg1, %arg1 : tensor<f64>
// CHECK-NEXT:  %[[DEN:.+]] = arith.mulf %[[SIG2]], %[[HNTWO]] : tensor<f64>
// CHECK-NEXT:  %[[X2:.+]] = arith.mulf %[[Y]], %[[Y]] : tensor<f64>
// CHECK-NEXT:  %[[DIV:.+]] = arith.divf %[[X2]], %[[DEN]] : tensor<f64>
// CHECK-NEXT:  %[[KERN:.+]] = arith.negf %[[DIV]] : tensor<f64>
// CHECK-NEXT:  %[[LOGSIG:.+]] = math.log %arg1 : tensor<f64>
// CHECK-NEXT:  %[[N1:.+]] = arith.subf %[[HNLOG2]], %[[LOGSIG]] : tensor<f64>
// CHECK-NEXT:  %[[NORM2:.+]] = arith.subf %[[N1]], %[[HNLOG2PI]] : tensor<f64>

// CHECK-LABEL: func.func @gam_model.generate
// CHECK-NOT:   impulse.random
// CHECK:       %[[GX:.+]] = impulse.reshape %{{.+}} : (tensor<1x1xf64>) -> tensor<f64>
// CHECK-NEXT:  %[[AM1:.+]] = arith.subf %arg2, %{{.+}} : tensor<f64>
// CHECK-NEXT:  %[[LOGX:.+]] = math.log %[[GX]] : tensor<f64>
// CHECK-NEXT:  %[[T1:.+]] = arith.mulf %[[AM1]], %[[LOGX]] : tensor<f64>
// CHECK-NEXT:  %[[BX:.+]] = arith.mulf %arg3, %[[GX]] : tensor<f64>
// CHECK-NEXT:  %[[KER:.+]] = arith.subf %[[T1]], %[[BX]] : tensor<f64>
// CHECK-NEXT:  %[[LG:.+]] = impulse.lgamma %arg2 : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LOGB:.+]] = math.log %arg3 : tensor<f64>
// CHECK-NEXT:  %[[ALB:.+]] = arith.mulf %arg2, %[[LOGB]] : tensor<f64>
// CHECK-NEXT:  %[[NORM:.+]] = arith.subf %[[ALB]], %[[LG]] : tensor<f64>
// CHECK-NEXT:  %[[KERR:.+]] = impulse.reshape %[[KER]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[NORMR:.+]] = impulse.reshape %[[NORM]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LPG:.+]] = arith.addf %[[KERR]], %[[NORMR]] : tensor<f64>

// CHECK-LABEL: func.func @du_model.simulate
// CHECK-DAG:   %[[DZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[DONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[DRNG:.+]], %[[DU:.+]] = impulse.random %arg0, %[[DZERO]], %[[DONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>}
// CHECK-NEXT:  %[[HML:.+]] = arith.subf %arg2, %arg1 : tensor<f64>
// CHECK-NEXT:  %[[N:.+]] = arith.addf %[[HML]], %[[DONE]] : tensor<f64>
// CHECK-NEXT:  %[[SC:.+]] = arith.mulf %[[DU]], %[[N]] : tensor<f64>
// CHECK-NEXT:  %[[FL:.+]] = math.floor %[[SC]] : tensor<f64>
// CHECK-NEXT:  %[[YD:.+]] = arith.addf %arg1, %[[FL]] : tensor<f64>
// CHECK-NEXT:  %[[HP1:.+]] = arith.addf %arg2, %[[DONE]] : tensor<f64>
// CHECK-NEXT:  %[[RANGE:.+]] = arith.subf %[[HP1]], %arg1 : tensor<f64>
// CHECK-NEXT:  %[[LOGR:.+]] = math.log %[[RANGE]] : tensor<f64>
// CHECK-NEXT:  %[[NLOG:.+]] = arith.negf %[[LOGR]] : tensor<f64>
// CHECK-NEXT:  %[[LNRD:.+]] = impulse.reshape %[[NLOG]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LPD:.+]] = arith.addf %[[LNRD]], %[[DZERO]] : tensor<f64>

// CHECK-LABEL: func.func @bern_model.simulate
// CHECK-DAG:   %[[BZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[BONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[BRNG:.+]], %[[BU:.+]] = impulse.random %arg0, %[[BZERO]], %[[BONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>}
// CHECK-NEXT:  %[[NLG:.+]] = arith.negf %arg1 : tensor<f64>
// CHECK-NEXT:  %[[E:.+]] = math.exp %[[NLG]] : tensor<f64>
// CHECK-NEXT:  %[[DEN:.+]] = arith.addf %[[E]], %[[BONE]] : tensor<f64>
// CHECK-NEXT:  %[[P:.+]] = arith.divf %[[BONE]], %[[DEN]] : tensor<f64>
// CHECK-NEXT:  %[[LT:.+]] = arith.cmpf olt, %[[BU]], %[[P]] : tensor<f64>
// CHECK-NEXT:  %[[YB:.+]] = arith.select %[[LT]], %[[BONE]], %[[BZERO]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:  %[[YL:.+]] = arith.mulf %[[YB]], %arg1 : tensor<f64>
// CHECK-NEXT:  %[[UNRB:.+]] = impulse.reshape %[[YL]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[MAXL:.+]] = arith.maximumf %arg1, %[[BZERO]] : tensor<f64>
// CHECK-NEXT:  %[[ABSL:.+]] = math.absf %arg1 : tensor<f64>
// CHECK-NEXT:  %[[NABS:.+]] = arith.negf %[[ABSL]] : tensor<f64>
// CHECK-NEXT:  %[[SPE:.+]] = math.exp %[[NABS]] : tensor<f64>
// CHECK-NEXT:  %[[SP:.+]] = math.log1p %[[SPE]] : tensor<f64>
// CHECK-NEXT:  %[[SUMN:.+]] = arith.addf %[[MAXL]], %[[SP]] : tensor<f64>
// CHECK-NEXT:  %[[NEGN:.+]] = arith.negf %[[SUMN]] : tensor<f64>
// CHECK-NEXT:  %[[LNRB:.+]] = impulse.reshape %[[NEGN]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LPB:.+]] = arith.addf %[[UNRB]], %[[LNRB]] : tensor<f64>

// CHECK-LABEL: func.func @exp_model.simulate
// CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-DAG:   %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:       %[[RNG:.+]], %[[U:.+]] = impulse.random %arg0, %[[ZERO]], %[[ONE]] {rng_distribution = #impulse<rng_distribution UNIFORM>}
// CHECK-NEXT:  %[[NU:.+]] = arith.negf %[[U]] : tensor<f64>
// CHECK-NEXT:  %[[L1P:.+]] = math.log1p %[[NU]] : tensor<f64>
// CHECK-NEXT:  %[[Z:.+]] = arith.negf %[[L1P]] : tensor<f64>
// CHECK-NEXT:  %[[Y:.+]] = arith.divf %[[Z]], %arg1 : tensor<f64>
// CHECK-NEXT:  %[[LX:.+]] = arith.mulf %arg1, %[[Y]] : tensor<f64>
// CHECK-NEXT:  %[[UN:.+]] = arith.negf %[[LX]] : tensor<f64>
// CHECK-NEXT:  %[[UNR:.+]] = impulse.reshape %[[UN]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LOGLAM:.+]] = math.log %arg1 : tensor<f64>
// CHECK-NEXT:  %[[LNR:.+]] = impulse.reshape %[[LOGLAM]] : (tensor<f64>) -> tensor<f64>
// CHECK-NEXT:  %[[LP:.+]] = arith.addf %[[UNR]], %[[LNR]] : tensor<f64>
// CHECK-NEXT:  %[[W:.+]] = arith.addf %[[LP]], %[[ZERO]] : tensor<f64>
