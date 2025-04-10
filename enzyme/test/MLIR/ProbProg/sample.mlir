// RUN: %eopt --probprog %s | FileCheck %s
// XFAIL: *

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64
  func.func private @normal_logpdf(%value : f64, %mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %s = enzyme.sample @normal(%mean, %stddev) @normal_logpdf { name="s" } : (f64, f64) -> (f64)
    %t = enzyme.sample @normal(%s, %stddev) @normal_logpdf { name="t" } : (f64, f64) -> (f64)
    return %t : f64
  }

  func.func @generate(%mean : f64, %stddev : f64) -> (!enzyme.Trace<f64>, f64) {
    %trace, %res = enzyme.trace @test(%mean, %stddev) [] [] { name = "test" } : (f64, f64) -> (!enzyme.Trace<f64>, f64)
    return %trace, %res : !enzyme.Trace<f64>, f64
  }
}

// CHECK:   func.func @test.trace(%[[trace:.+]]: !enzyme.Trace<f64>, %[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> (!enzyme.Trace<f64>, f64) {
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    %[[ls:.+]] = call @normal_logpdf(%[[s]], %[[mean]], %[[stddev]]) : (f64, f64, f64) -> f64
// CHECK-NEXT:    %[[trace_s:.+]] = enzyme.insert_choice %[[trace]], %[[s]], %[[ls]] {name = "s"} : (!enzyme.Trace<f64>, f64, f64) -> !enzyme.Trace<f64>
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) {name = "t"} : (f64, f64) -> f64
// CHECK-NEXT:    %[[lt:.+]] = call @normal_logpdf(%[[t]], %[[s]], %[[stddev]]) : (f64, f64, f64) -> f64
// CHECK-NEXT:    %[[trace_t:.+]] = enzyme.insert_choice %[[trace_s]], %[[t]], %[[lt]] {name = "t"} : (!enzyme.Trace<f64>, f64, f64) -> !enzyme.Trace<f64>
// CHECK-NEXT:    return %[[trace_t]], %[[lt]] : !enzyme.Trace<f64>, f64
// CHECK-NEXT:   }
