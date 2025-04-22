// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64

  func.func @test(%mean : f64, %stddev : f64) -> f64 {
    %s = enzyme.sample @normal(%mean, %stddev) { name="s" } : (f64, f64) -> f64
    %t = enzyme.sample @normal(%s, %stddev) { name="t" } : (f64, f64) -> f64
    return %t : f64
  }

  func.func @generate(%mean : f64, %stddev : f64) -> !enzyme.Trace<f64> {
    %trace = enzyme.simulate @test(%mean, %stddev) { name = "test" } : (f64, f64) -> !enzyme.Trace<f64>
    return %trace : !enzyme.Trace<f64>
  }
}

// CHECK:   func.func @test.simulate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> !enzyme.Trace<f64> {
// CHECK-NEXT:    %[[trace:.+]] = "enzyme.init"() : () -> !enzyme.Trace<f64>
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    "enzyme.addSampleToTrace"(%[[trace]], %[[s]]) <{name = "s"}> : (!enzyme.Trace<f64>, f64) -> ()
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    "enzyme.addSampleToTrace"(%[[trace]], %[[t]]) <{name = "t"}> : (!enzyme.Trace<f64>, f64) -> ()
// CHECK-NEXT:    return %[[trace]] : !enzyme.Trace<f64>
// CHECK-NEXT:   }
