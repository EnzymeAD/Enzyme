// RUN: %eopt %s | FileCheck %s

module {
  func.func private @normal(%seed : i64, %mean : f64, %stddev : f64) -> f64

  // CHECK:   func.func @sample_no_symbol(%[[seed:.+]]: i64, %[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
  // CHECK-NEXT:    %[[res:.+]] = enzyme.sample @normal(%[[seed]], %[[mean]], %[[stddev]]) {name = "m"} : (i64, f64, f64) -> f64
  // CHECK-NEXT:    return %[[res]] : f64
  // CHECK-NEXT:   }
  func.func @sample_no_symbol(%seed : i64, %mean : f64, %stddev : f64) -> f64 {
    %r = enzyme.sample @normal(%seed, %mean, %stddev) { name="m" } : (i64, f64, f64) -> (f64)
    return %r : f64
  }

  // CHECK:   func.func @sample(%[[seed:.+]]: i64, %[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
  // CHECK-NEXT:    %[[res:.+]] = enzyme.sample @normal(%[[seed]], %[[mean]], %[[stddev]]) {name = "m", symbol = 1 : ui64} : (i64, f64, f64) -> f64
  // CHECK-NEXT:    return %[[res]] : f64
  // CHECK-NEXT:   }
  func.func @sample(%seed : i64, %mean : f64, %stddev : f64) -> f64 {
    %r = enzyme.sample @normal(%seed, %mean, %stddev) { name="m", symbol = 1 : ui64 } : (i64, f64, f64) -> (f64)
    return %r : f64
  }

  // CHECK:   func.func @generate(%[[seed:.+]]: i64, %[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
  // CHECK-NEXT:    %[[res0:.+]] = enzyme.generate @sample(%[[seed]], %[[mean]], %[[stddev]]) {name = "test"} : (i64, f64, f64) -> f64
  // CHECK-NEXT:    return %[[res0]] : f64
  // CHECK-NEXT:   }
  func.func @generate(%seed : i64, %mean : f64, %stddev : f64) -> f64 {
    %res = enzyme.generate @sample(%seed, %mean, %stddev) { name = "test" } : (i64, f64, f64) -> f64
    return %res : f64
  }

  // CHECK:   func.func @simulate(%[[seed:.+]]: i64, %[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
  // CHECK-NEXT:    %[[res0:.+]] = enzyme.simulate @sample(%[[seed]], %[[mean]], %[[stddev]]) {name = "test", trace = 42 : ui64} : (i64, f64, f64) -> f64
  // CHECK-NEXT:    return %[[res0]] : f64
  // CHECK-NEXT:   }
  func.func @simulate(%seed : i64, %mean : f64, %stddev : f64) -> f64 {
    %res = enzyme.simulate @sample(%seed, %mean, %stddev) { name = "test", trace = 42 : ui64 } : (i64, f64, f64) -> f64
    return %res : f64
  }
}
