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
    %res:7 = enzyme.simulate @test(%rng, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#2, %res#3, %res#4, %res#5, %res#6 : tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}

// CHECK:   func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:     enzyme.addSampleToTrace(%1#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:     %4:7 = call @two_normals.simulate(%1#1, %1#0, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     enzyme.addSubtrace %4#0 into %0 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:     %5 = arith.addf %3, %4#1 : tensor<f64>
// CHECK-NEXT:     enzyme.addSampleToTrace(%4#2, %4#3 : tensor<f64>, tensor<f64>) into %0 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:     return %0, %5, %4#2, %4#3, %4#4, %4#5, %4#6 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }

// CHECK:   func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:     %1:4 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %2 = call @logpdf(%1#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:     enzyme.addSampleToTrace(%1#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<3>}
// CHECK-NEXT:     %4:4 = call @normal(%1#1, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:     %5 = call @logpdf(%4#0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:     %6 = arith.addf %3, %5 : tensor<f64>
// CHECK-NEXT:     enzyme.addSampleToTrace(%4#0 : tensor<f64>) into %0 {symbol = #enzyme.symbol<4>}
// CHECK-NEXT:     return %0, %6, %1#0, %4#0, %4#1, %4#2, %4#3 : !enzyme.Trace, tensor<f64>, tensor<f64>, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:   }
