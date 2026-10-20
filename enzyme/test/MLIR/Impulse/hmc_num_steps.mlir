// RUN: %eopt --expand-impulse %s | FileCheck %s

// num_steps overrides trajectory_length, which would give 3 steps of 0.25 / 3.

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %s:2 = impulse.sample @normal(%rng, %mean, %stddev) <{ logpdf = @logpdf, symbol = #impulse.symbol<1>, name="s" }> : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %s#0, %s#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @hmc(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<10x1xf64>, tensor<10x2xi1>, tensor<10xf64>, tensor<2xui64>) {
    %init_trace = arith.constant dense<[[0.0]]> : tensor<1x1xf64>
    %step_size = arith.constant dense<0.1> : tensor<f64>
    %res:4 = impulse.infer @test(%rng, %mean, %stddev) given %init_trace
      init_step_size = %step_size
      <{ hmc_config = #impulse.hmc_config<trajectory_length = 0.25, num_steps = 7>,
        name = "hmc", selection = [[#impulse.symbol<1>]], all_addresses = [[#impulse.symbol<1>]], num_warmup = 0, num_samples = 10 }>
      : (tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<1x1xf64>, tensor<f64>) -> (tensor<10x1xf64>, tensor<10x2xi1>, tensor<10xf64>, tensor<2xui64>)
    return %res#0, %res#1, %res#2, %res#3 : tensor<10x1xf64>, tensor<10x2xi1>, tensor<10xf64>, tensor<2xui64>
  }
}

// CHECK-LABEL: func.func @hmc
// CHECK-NOT: math.ceil
// CHECK-DAG: %[[NEG_EPS:.+]] = arith.constant dense<-1.000000e-01> : tensor<f64>
// CHECK-DAG: %[[HALF:.+]] = arith.constant dense<5.000000e-01> : tensor<f64>
// CHECK-DAG: %[[C7:.+]] = arith.constant dense<7> : tensor<i64>
// CHECK-DAG: %[[C10:.+]] = arith.constant dense<10> : tensor<i64>
// CHECK-DAG: %[[C1:.+]] = arith.constant dense<1> : tensor<i64>
// CHECK-DAG: %[[C0:.+]] = arith.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[EPS:.+]] = arith.constant dense<1.000000e-01> : tensor<f64>
// CHECK: impulse.for(%[[C0]] : tensor<i64>) to(%[[C10]] : tensor<i64>) step(%[[C1]] : tensor<i64>)
// CHECK-NOT: math.ceil
// CHECK: impulse.for(%[[C0]] : tensor<i64>) to(%[[C7]] : tensor<i64>) step(%[[C1]] : tensor<i64>)
