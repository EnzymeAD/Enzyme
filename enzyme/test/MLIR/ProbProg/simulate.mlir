// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %s = enzyme.sample @normal(%mean, %stddev) { symbol = 1 : ui64, name="s", traced_output_indices = array<i64: 0> } : (f64, f64) -> f64
    %t = enzyme.sample @normal(%s, %stddev) { symbol = 2 : ui64, name="t", traced_output_indices = array<i64: 0> } : (f64, f64) -> f64
    return %t : f64
  }

  func.func @simulate(%mean : f64, %stddev : f64) -> f64 {
    %trace = enzyme.simulate @test(%mean, %stddev) { name = "test", trace = 42 : ui64 } : (f64, f64) -> f64
    return %trace : f64
  }
}

// CHECK:   func.func @test.simulate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    enzyme.addSampleToTrace %[[s]] {name = "s", symbol = 1 : ui64, trace = 42 : ui64} : (f64) -> ()
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    enzyme.addSampleToTrace %[[t]] {name = "t", symbol = 2 : ui64, trace = 42 : ui64} : (f64) -> ()
// CHECK-NEXT:    return %[[t]] : f64
// CHECK-NEXT:   }
