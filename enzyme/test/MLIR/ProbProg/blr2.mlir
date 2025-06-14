// RUN: %eopt --probprog %s | FileCheck %s

// Bayesian linear regression: https://www.gen.dev/tutorials/intro-to-modeling/tutorial
module {
  func.func private @normal(%mean : f64, %stddev : f64) -> f64

  func.func @line_model(%xs : memref<?xf64>, %n : index) -> f64 {
    // Sample the slope and intercept
    %pnan = arith.constant 0x7FF0000001000000 : f64
    %c0 = arith.constant 0.0 : f64
    %c0_1 = arith.constant 0.1 : f64
    %c1 = arith.constant 1.0 : f64
    %c2 = arith.constant 2.0 : f64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %slope = enzyme.sample @normal(%c0, %c1) { name = "slope", symbol = 42 : ui64, traced_output_indices = array<i64: 0> } : (f64, f64) -> f64
    %intercept = enzyme.sample @normal(%c0, %c2) { name = "intercept", symbol = 43 : ui64, traced_output_indices = array<i64: 0> } : (f64, f64) -> f64

    %final = scf.for %i = %zero to %n step %one
             iter_args(%prev_y = %pnan) -> (f64) {
      %x = memref.load %xs[%i] : memref<?xf64>
      %prod = arith.mulf %slope, %x : f64
      %y_mean = arith.addf %prod, %intercept : f64
      %y_sample = enzyme.sample @normal(%y_mean, %c0_1) { name = "y", symbol = 44 : ui64, traced_output_indices = array<i64: 0> } : (f64, f64) -> f64
      scf.yield %y_sample : f64
    }
    return %final : f64
  }

  func.func @generate(%xs : memref<?xf64>, %n : index) -> f64 {
    %trace = enzyme.simulate @line_model(%xs, %n) { name = "line_model", trace = 42 : ui64 } : (memref<?xf64>, index) -> f64
    return %trace : f64
  }
}

// CHECK: func.func @line_model.simulate(%[[xs:.+]]: memref<?xf64>, %[[n:.+]]: index) -> f64 {
// CHECK-NEXT:   %[[pnan:.+]] = arith.constant 0x7FF0000001000000 : f64
// CHECK-NEXT:   %[[c0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[c0_1:.+]] = arith.constant 1.000000e-01 : f64
// CHECK-NEXT:   %[[c1:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[c2:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[zero:.+]] = arith.constant 0 : index
// CHECK-NEXT:   %[[one:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[slope:.+]] = call @normal(%[[c0]], %[[c1]]) : (f64, f64) -> f64
// CHECK-NEXT:   enzyme.addSampleToTrace %[[slope]] {name = "slope", symbol = 42 : ui64, trace = 42 : ui64} : (f64) -> ()
// CHECK-NEXT:   %[[intercept:.+]] = call @normal(%[[c0]], %[[c2]]) : (f64, f64) -> f64
// CHECK-NEXT:   enzyme.addSampleToTrace %[[intercept]] {name = "intercept", symbol = 43 : ui64, trace = 42 : ui64} : (f64) -> ()
// CHECK-NEXT:   %[[final:.+]] = scf.for %[[i:.+]] = %[[zero]] to %[[n]] step %[[one]] iter_args(%[[prev_y:.+]] = %[[pnan]]) -> (f64) {
// CHECK-NEXT:       %[[x:.+]] = memref.load %[[xs]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:       %[[prod:.+]] = arith.mulf %[[slope]], %[[x]] : f64
// CHECK-NEXT:       %[[y_mean:.+]] = arith.addf %[[prod]], %[[intercept]] : f64
// CHECK-NEXT:       %[[y_sample:.+]] = func.call @normal(%[[y_mean]], %[[c0_1]]) : (f64, f64) -> f64
// CHECK-NEXT:       enzyme.addSampleToTrace %[[y_sample]] {name = "y", symbol = 44 : ui64, trace = 42 : ui64} : (f64) -> ()
// CHECK-NEXT:       scf.yield %[[y_sample]] : f64
// CHECK-NEXT:     }
// CHECK-NEXT:   return %[[final]] : f64
// CHECK-NEXT: }
