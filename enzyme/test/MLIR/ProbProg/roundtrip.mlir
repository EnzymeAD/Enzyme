// RUN: %eopt %s | FileCheck %s

module {
  func.func private @norm(%sampler : i64, %mean : f64, %stddev : f64) -> f64

  func.func @dsq(%mean : f64, %stddev : f64) -> f64 {
    %r = enzyme.sample @norm(%mean, %stddev) { name="m" } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK:   func.func @dsq(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[res:.+]] = enzyme.sample @norm(%[[mean]], %[[stddev]]) {name = "m"} : (f64, f64) -> f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:   }