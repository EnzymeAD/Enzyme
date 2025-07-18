// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
  func.func private @logpdf(%x : tensor<f64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> tensor<f64>

  func.func @two_normals(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<3>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:2 = enzyme.sample @normal(%s#0, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<4>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %t#0, %s#1, %t#1 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @test(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>) {
    %s:2 = enzyme.sample @normal(%rng, %mean, %stddev) { logpdf = @logpdf, symbol = #enzyme.symbol<1>, name="s" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %t:3 = enzyme.sample @two_normals(%s#0, %s#1, %stddev) { symbol = #enzyme.symbol<2>, name="t" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %t#0, %t#1, %t#2 : tensor<2xui64>, tensor<f64>, tensor<f64>
  }

  func.func @simulate(%rng : tensor<2xui64>, %mean : tensor<f64>, %stddev : tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
    // CHECK: %0:5 = call @test.simulate(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
    %res:5 = enzyme.simulate @test(%rng, %mean, %stddev) { name = "test" } : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>,tensor<2xui64>, tensor<f64>, tensor<f64>)
    return %res#0, %res#1, %res#2, %res#3, %res#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
  }
}


// CHECK:  func.func @test.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1:2 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %2 = call @logpdf(%1#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:    %4 = enzyme.addSampleToTrace(%1#1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %5:5 = call @two_normals.simulate(%1#0, %1#1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>)
// CHECK-NEXT:    %6 = enzyme.addSubtrace %5#0 into %4 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %7 = arith.addf %3, %5#1 : tensor<f64>
// CHECK-NEXT:    %8 = enzyme.addSampleToTrace(%5#3, %5#4 : tensor<f64>, tensor<f64>) into %6 {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:    %10 = enzyme.addRetvalToTrace(%5#3, %5#4 : tensor<f64>, tensor<f64>) into %9
// CHECK-NEXT:    return %10, %7, %5#2, %5#3, %5#4 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:  }

// CHECK:  func.func @two_normals.simulate(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %1:2 = call @normal(%arg0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %2 = call @logpdf(%1#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %3 = arith.addf %2, %cst : tensor<f64>
// CHECK-NEXT:    %4 = enzyme.addSampleToTrace(%1#1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<3>}
// CHECK-NEXT:    %5:2 = call @normal(%1#0, %arg1, %arg2) : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
// CHECK-NEXT:    %6 = call @logpdf(%5#1, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %7 = arith.addf %3, %6 : tensor<f64>
// CHECK-NEXT:    %8 = enzyme.addSampleToTrace(%5#1 : tensor<f64>) into %4 {symbol = #enzyme.symbol<4>}
// CHECK-NEXT:    %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
// CHECK-NEXT:    %10 = enzyme.addRetvalToTrace(%1#1, %5#1 : tensor<f64>, tensor<f64>) into %9
// CHECK-NEXT:    return %10, %7, %5#0, %1#1, %5#1 : !enzyme.Trace, tensor<f64>, tensor<2xui64>, tensor<f64>, tensor<f64>
// CHECK-NEXT:  }
