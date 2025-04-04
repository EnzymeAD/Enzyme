// RUN: %eopt %s | FileCheck %s

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64
  func.func private @normal_logpdf(%value : f64, %mean : f64, %stddev : f64) -> f64

  // CHECK:   func.func @test(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
  // CHECK-NEXT:    %[[res:.+]] = enzyme.sample @normal(%[[mean]], %[[stddev]]) @normal_logpdf {name = "m"} : (f64, f64) -> f64
  // CHECK-NEXT:    return %[[res]] : f64
  // CHECK-NEXT:   }
  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %r = enzyme.sample @normal(%mean, %stddev) @normal_logpdf { name="m" } : (f64, f64) -> (f64)
    return %r : f64
  }

  // CHECK:   func.func @generate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> !enzyme.Trace<f64> {
  // CHECK-NEXT:    %[[trace0:.+]], %[[res0:.+]] = enzyme.trace @test(%[[mean]], %[[stddev]]) [] [] {name = "test"} : (f64, f64) -> (!enzyme.Trace<f64>, f64)
  // CHECK-NEXT:    %[[trace1:.+]] = enzyme.trace @test(%[[res0]], %[[stddev]]) [] [%[[trace0]] : !enzyme.Trace<f64>] {name = "test"} : (f64, f64) -> !enzyme.Trace<f64>
  // CHECK-NEXT:    return %[[trace1]] : !enzyme.Trace<f64>
  // CHECK-NEXT:   }
  func.func @generate(%mean : f64, %stddev : f64) -> !enzyme.Trace<f64> {
    %trace0, %res0 = enzyme.trace @test(%mean, %stddev) [] [] { name = "test" } : (f64, f64) -> (!enzyme.Trace<f64>, f64)
    %trace1 = enzyme.trace @test(%res0, %stddev) [] [%trace0 : !enzyme.Trace<f64>] { name = "test" } : (f64, f64) -> !enzyme.Trace<f64>
    return %trace1 : !enzyme.Trace<f64>
  }
}
