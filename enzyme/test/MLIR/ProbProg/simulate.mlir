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

  func.func @simulate(%mean : f64, %stddev : f64) -> !enzyme.Trace {
    %trace = enzyme.simulate @test(%mean, %stddev) { name = "test" } : (f64, f64) -> !enzyme.Trace
    return %trace : !enzyme.Trace
  }
}

// CHECK:   func.func @test.simulate(%[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> !enzyme.Trace {
// CHECK-NEXT:    %[[trace:.+]] = enzyme.initTrace : !enzyme.Trace
// CHECK-NEXT:    %[[s_sym:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[t_sym:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[s:.+]] = call @normal(%[[mean]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    "enzyme.addSampleToTrace"(%[[trace]], %[[s_sym]], %[[s]]) <{name = "s"}> : (!enzyme.Trace, i64, f64) -> ()
// CHECK-NEXT:    %[[t:.+]] = call @normal(%[[s]], %[[stddev]]) : (f64, f64) -> f64
// CHECK-NEXT:    "enzyme.addSampleToTrace"(%[[trace]], %[[t_sym]], %[[t]]) <{name = "t"}> : (!enzyme.Trace, i64, f64) -> ()
// CHECK-NEXT:    return %[[trace]] : !enzyme.Trace
// CHECK-NEXT:   }
