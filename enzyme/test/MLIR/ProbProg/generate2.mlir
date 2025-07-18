// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @joint(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
  func.func private @joint_logpdf(%x1 : tensor<f64>, %x2 : tensor<2xf64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %s:3 = enzyme.sample @joint(%rng, %mean, %stddev) {
      symbol = #enzyme.symbol<5>,
      name = "s",
      logpdf = @joint_logpdf,
      traced_input_indices = array<i64: 1, 2>,
      traced_output_indices = array<i64: 1, 2>,
      alias_map = array<i64: 0, 0>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %s#0, %s#1, %s#2 : tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }

  func.func @foo(%rng : tensor<2xui64>, %x : tensor<f64>, %y : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
    %constraint = enzyme.initConstraint : !enzyme.Constraint
    %res:5 = enzyme.generate @test(%rng, %x, %y) given %constraint {
      constrained_symbols = [#enzyme.symbol<2>, #enzyme.symbol<5>],
      traced_output_indices = array<i64: 1, 2>
    } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>)
    return %res#1, %res#2, %res#3, %res#4 : tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
  }
}

// CHECK:  func.func @test.generate(%arg0: !enzyme.Constraint, %arg1: tensor<2xui64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1:2 = enzyme.getSampleFromConstraint %arg0 {symbol = #enzyme.symbol<5>} : tensor<f64>, tensor<2xf64>
// CHECK-NEXT:    %2 = call @joint_logpdf(%1#0, %1#1, %arg2, %arg3) : (tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:    %4 = enzyme.addSampleToTrace(%1#0, %1#1 : tensor<f64>, tensor<2xf64>) into %0 {symbol = #enzyme.symbol<5>}
// CHECK-NEXT:    %5 = enzyme.addWeightToTrace(%3 : tensor<f64>) into %4
// CHECK-NEXT:    %6 = enzyme.addRetvalToTrace(%1#0, %1#1 : tensor<f64>, tensor<2xf64>) into %5
// CHECK-NEXT:    return %6, %3, %arg1, %1#0, %1#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<2xf64>
// CHECK-NEXT:  }
