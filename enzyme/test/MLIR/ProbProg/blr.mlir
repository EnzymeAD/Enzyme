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
    %i = arith.constant 0 : index
    %slope_sym = llvm.mlir.constant(1 : i64) : i64
    %intercept_sym = llvm.mlir.constant(2 : i64) : i64
    %slope = enzyme.sample [%slope_sym : i64] @normal(%c0, %c1) { name = "slope" } : (f64, f64) -> f64
    %intercept = enzyme.sample [%intercept_sym : i64] @normal(%c0, %c2) { name = "intercept" } : (f64, f64) -> f64
    cf.br ^loop(%i, %pnan : index, f64)

  ^loop(%i1 : index, %prev_y : f64):
    %cond = arith.cmpi slt, %i1, %n : index
    cf.cond_br %cond, ^body(%i1 : index), ^exit(%prev_y : f64)

  ^body(%i2 : index):
    %x = memref.load %xs[%i2] : memref<?xf64>

    // Compute y_mean = slope * x + intercept.
    %prod = arith.mulf %slope, %x : f64
    %y_mean = arith.addf %prod, %intercept : f64

    // Sample an observation for y
    %y_sym = llvm.mlir.constant(3 : i64) : i64
    %y_sample = enzyme.sample [%y_sym : i64] @normal(%y_mean, %c0_1) { name = "y" } : (f64, f64) -> f64

    %one = arith.constant 1 : index
    %next_i = arith.addi %i2, %one : index
    cf.br ^loop(%next_i, %y_sample : index, f64)

  ^exit(%prev_y1 : f64):
    return %prev_y1 : f64
  }

  func.func @generate(%xs : memref<?xf64>, %n : index) -> !enzyme.Trace {
    %trace = enzyme.simulate @line_model(%xs, %n) { name = "line_model" } : (memref<?xf64>, index) -> !enzyme.Trace
    return %trace : !enzyme.Trace
  }
}

// CHECK: func.func @line_model.simulate(%[[xs:.+]]: memref<?xf64>, %[[n:.+]]: index) -> !enzyme.Trace {
// CHECK-NEXT:   %[[trace:.+]] = "enzyme.initTrace"() : () -> !enzyme.Trace
// CHECK-NEXT:   %[[pnan:.+]] = arith.constant 0x7FF0000001000000 : f64
// CHECK-NEXT:   %[[c0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[c0_1:.+]] = arith.constant 1.000000e-01 : f64
// CHECK-NEXT:   %[[c1:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:   %[[c2:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[i:.+]] = arith.constant 0 : index
// CHECK-NEXT:   %[[slope_sym:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:   %[[intercept_sym:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:   %[[slope:.+]] = call @normal(%[[c0]], %[[c1]]) : (f64, f64) -> f64
// CHECK-NEXT:   "enzyme.addSampleToTrace"(%[[trace]], %[[slope_sym]], %[[slope]]) <{name = "slope"}> : (!enzyme.Trace, i64, f64) -> ()
// CHECK-NEXT:   %[[intercept:.+]] = call @normal(%[[c0]], %[[c2]]) : (f64, f64) -> f64
// CHECK-NEXT:   "enzyme.addSampleToTrace"(%[[trace]], %[[intercept_sym]], %[[intercept]]) <{name = "intercept"}> : (!enzyme.Trace, i64, f64) -> ()
// CHECK-NEXT:   cf.br ^[[loop:.+]](%[[i]], %[[pnan]] : index, f64)

// CHECK-NEXT: ^[[loop]](%[[i1:.+]]: index, %[[prev_y:.+]]: f64):
// CHECK-NEXT:   %[[cond:.+]] = arith.cmpi slt, %[[i1]], %[[n]] : index
// CHECK-NEXT:   cf.cond_br %[[cond]], ^[[body:.+]](%[[i1]] : index), ^[[exit:.+]](%[[prev_y]] : f64)

// CHECK: ^[[body]](%[[i2:.+]]: index):
// CHECK-NEXT:   %[[x:.+]] = memref.load %[[xs]][%[[i2]]] : memref<?xf64>
// CHECK-NEXT:   %[[prod:.+]] = arith.mulf %[[slope]], %[[x]] : f64
// CHECK-NEXT:   %[[y_mean:.+]] = arith.addf %[[prod]], %[[intercept]] : f64
// CHECK-NEXT:   %[[y_sym:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:   %[[y_sample:.+]] = call @normal(%[[y_mean]], %[[c0_1]]) : (f64, f64) -> f64
// CHECK-NEXT:   "enzyme.addSampleToTrace"(%[[trace]], %[[y_sym]], %[[y_sample]]) <{name = "y"}> : (!enzyme.Trace, i64, f64) -> ()
// CHECK-NEXT:   %[[one:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[i_next:.+]] = arith.addi %[[i2]], %[[one]] : index
// CHECK-NEXT:   cf.br ^[[loop]](%[[i_next]], %[[y_sample]] : index, f64)

// CHECK: ^[[exit]](%[[prev_y_exit:.+]]: f64):
// CHECK-NEXT:   return %[[trace]] : !enzyme.Trace
// CHECK-NEXT: }
