// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @two_normals(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:4 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<3>, name="s", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %t:4 = enzyme.sample @normal(%s#1, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<4>, name="t", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %s#0, %t#0, %t#1, %t#2, %t#3 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:4 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s", traced_output_indices = array<i64: 0>, traced_input_indices = array<i64: 1, 2> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %t:5 = enzyme.sample @two_normals(%s#1, %s#0, %stddev) { symbol = #enzyme.symbol<2>, name="t", traced_output_indices = array<i64: 0, 1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t#0, %t#1, %t#2, %t#3, %t#4 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @simulate(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %res:7 = enzyme.simulate @test(%rng, %mean, %stddev) { name = "test", traced_output_indices = array<i64: 0, 1> } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#2, %res#3, %res#4, %res#5, %res#6 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CHECK:   func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:     %4 = enzyme.addSampleToTrace(%1#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:     %5:7 = call @two_normals.simulate(%1#1, %1#0, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %6 = enzyme.addSubtrace %5#0 into %4 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:     %7 = arith.addf %3, %5#1 : tensor<f64>
// CHECK-NEXT:     %8 = enzyme.addSampleToTrace(%5#2, %5#3 : tensor<f64>, tensor<f64>) into %6 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:     %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:     %10 = enzyme.addRetvalToTrace(%5#2, %5#3 : tensor<f64>, tensor<f64>) into %9
// CHECK-NEXT:     return %10, %7, %5#2, %5#3, %5#4, %5#5, %5#6 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }

// CHECK:   func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:     %4 = enzyme.addSampleToTrace(%1#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<3>}
// CHECK-NEXT:     %5:4 = call @normal(%1#1, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %6 = call @logpdf(%5#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %7 = arith.addf %3, %6 : tensor<f64>
// CHECK-NEXT:     %8 = enzyme.addSampleToTrace(%5#0 : tensor<f64>) into %4 {symbol = #enzyme.symbol<4>}
// CHECK-NEXT:     %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:     %10 = enzyme.addRetvalToTrace(%1#0, %5#0 : tensor<f64>, tensor<f64>) into %9
// CHECK-NEXT:     return %10, %7, %1#0, %5#0, %5#1, %5#2, %5#3 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
