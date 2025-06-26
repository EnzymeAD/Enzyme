// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %s = enzyme.sample @normal(%mean, %stddev) { name="s" } : (f64, f64) -> f64
    %t = enzyme.sample @normal(%s, %stddev) { name="t" } : (f64, f64) -> f64
    return %t : f64
  }

  func.func @main(%mean : f64, %stddev : f64) -> f64 {
    %trace = enzyme.untracedCall @test(%mean, %stddev) { name = "test" } : (f64, f64) -> f64
    return %trace : f64
  }
}

// CHECK:   func.func @test.call(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    return %[[t]] : f64
// CHECK-NEXT:   }
