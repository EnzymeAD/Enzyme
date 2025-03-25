// RUN: %eopt %s | FileCheck %s

module {
  func.func private @norm(%mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %r = enzyme.sample @norm(%mean, %stddev) { name="m" } : (f64, f64) -> (f64)
    return %r : f64
  }

  func.func @generate(%mean : f64, %stddev : f64) -> f64 {
    %trace0, %res0 = enzyme.trace @test(%mean, %stddev) [] [] { name = "test" } : (f64, f64) -> (!enzyme.Trace<f64>, f64)
    %trace1, %res1 = enzyme.trace @test(%res0, %stddev) [] [%trace0 : !enzyme.Trace<f64>] { name = "test" } : (f64, f64) -> (!enzyme.Trace<f64>, f64)
    return %res1 : f64
  }
}

// CHECK:   func.func @test(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[res:.+]] = enzyme.sample @norm(%[[mean]], %[[stddev]]) {name = "m"} : (f64, f64) -> f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:   }

// CHECK:   func.func @generate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[trace0:.+]], %[[res0:.+]] = enzyme.trace @test(%[[mean]], %[[stddev]]) [] [] {name = "test"} : (f64, f64) -> (!enzyme.Trace<f64>, f64)
// CHECK-NEXT:    %[[trace1:.+]], %[[res1:.+]] = enzyme.trace @test(%[[res0]], %[[stddev]]) [] [%[[trace0]] : !enzyme.Trace<f64>] {name = "test"} : (f64, f64) -> (!enzyme.Trace<f64>, f64)
// CHECK-NEXT:    return %[[res1]] : f64
// CHECK-NEXT:   }