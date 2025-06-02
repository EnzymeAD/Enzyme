// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %s_sym = llvm.mlir.constant(1 : i64) : i64
    %t_sym = llvm.mlir.constant(2 : i64) : i64
    %s = enzyme.sample [%s_sym : i64] @normal(%mean, %stddev) { name="s" } : (f64, f64) -> f64
    %t = enzyme.sample [%t_sym : i64] @normal(%s, %stddev) { name="t" } : (f64, f64) -> f64
    return %t : f64
  }

  func.func @simulate(%mean : f64, %stddev : f64) -> tensor<1xui64> {
    %trace = enzyme.simulate @test(%mean, %stddev) { name = "test" } : (f64, f64) -> tensor<1xui64>
    return %trace : tensor<1xui64>
  }
}

// CHECK:   func.func @simulate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> tensor<1xui64> {
// CHECK-NEXT:    %0 = call @test.simulate(%[[mean]], %[[stddev]]) : (f64, f64) -> !enzyme.Trace
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : !enzyme.Trace to tensor<1xui64>
// CHECK-NEXT:    return %1 : tensor<1xui64>
// CHECK-NEXT:   }

// CHECK:   func.func @test.simulate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> !enzyme.Trace {
// CHECK-NEXT:    %[[trace:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[s_sym:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[t_sym:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    enzyme.addSampleToTrace[%[[s_sym]] : i64] %[[trace]] : !enzyme.Trace, %[[s]] : f64
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    enzyme.addSampleToTrace[%[[t_sym]] : i64] %[[trace]] : !enzyme.Trace, %[[t]] : f64
// CHECK-NEXT:    return %[[trace]] : !enzyme.Trace
// CHECK-NEXT:   }
