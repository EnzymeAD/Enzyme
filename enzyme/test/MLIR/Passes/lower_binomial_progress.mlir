// RUN: %eopt -lower-enzyme-binomial-progress %s | FileCheck %s

// Constant operands fold to a plain constant: binomial_progress(9, 3) == 2.
func.func @cst() -> index {
  %n = arith.constant 9 : index
  %s = arith.constant 3 : index
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// CHECK-LABEL: func.func @cst() -> index {
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         return %[[C2]] : index
// CHECK-NOT:     enzyme.binomial_progress

// Dynamic operands lower to the Revolve computation on scf/arith.
func.func @dyn(%n: index, %s: index) -> index {
  %r = enzyme.binomial_progress %n, %s : index
  return %r : index
}

// CHECK-LABEL: func.func @dyn(
// CHECK:         %[[ONE:.+]] = arith.constant 1 : index
// CHECK:         scf.if
// CHECK:           scf.while
// CHECK:             arith.cmpi slt
// CHECK:             scf.condition
// CHECK:             arith.muli
// CHECK:             arith.divui
// CHECK:           arith.select
// CHECK-NOT:     enzyme.binomial_progress

// Dynamic i64 operands lower the same way, using i64 constants.
func.func @dyn_i64(%n: i64, %s: i64) -> i64 {
  %r = enzyme.binomial_progress %n, %s : i64
  return %r : i64
}

// CHECK-LABEL: func.func @dyn_i64(
// CHECK:         %[[ONE:.+]] = arith.constant 1 : i64
// CHECK:         scf.if
// CHECK:           scf.while
// CHECK:             arith.cmpi slt
// CHECK:             scf.condition
// CHECK:             arith.muli
// CHECK:             arith.divui
// CHECK:           arith.select
// CHECK-NOT:     enzyme.binomial_progress

// Unranked tensor operands are left untouched by this pass; tensor lowering
// is handled elsewhere.
func.func @tensor(%n: tensor<*xi64>, %s: tensor<*xi64>) -> tensor<*xi64> {
  %r = enzyme.binomial_progress %n, %s : tensor<*xi64>
  return %r : tensor<*xi64>
}

// CHECK-LABEL: func.func @tensor(
// CHECK:         enzyme.binomial_progress
