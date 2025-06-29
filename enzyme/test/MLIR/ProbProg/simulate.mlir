// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %s = enzyme.sample @normal(%mean, %stddev) { symbol = #enzyme.symbol<1>, name="s", traced_output_indices = array<i64: 0> } : (f64, f64) -> f64
    %t = enzyme.sample @normal(%s, %stddev) { symbol = #enzyme.symbol<2>, name="t", traced_output_indices = array<i64: 0> } : (f64, f64) -> f64
    return %t : f64
  }

  func.func @simulate(%mean : f64, %stddev : f64) -> f64 {
    %trace, %result = enzyme.simulate @test(%mean, %stddev) { name = "test" } : (f64, f64) -> (!enzyme.Trace, f64)
    return %result : f64
  }
}

// CHECK:   func.func @test.simulate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> (!enzyme.Trace, f64) {
// CHECK-NEXT:    %[[trace:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    enzyme.addSampleToTrace(%[[s]] : f64) into %[[trace]] {symbol = #enzyme.symbol<1>}
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    enzyme.addSampleToTrace(%[[t]] : f64) into %[[trace]] {symbol = #enzyme.symbol<2>}
// CHECK-NEXT:    return %[[trace]], %[[t]] : !enzyme.Trace, f64
// CHECK-NEXT:   }
