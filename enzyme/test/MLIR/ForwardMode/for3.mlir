// RUN: %eopt --enzyme --split-input-file %s | FileCheck %s

module {
  func.func @carry_mismatch_scf(%x : f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (f64) {
      scf.yield %x : f64
    }
    return %r : f64
  }

  func.func @dcarry_mismatch_scf(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @carry_mismatch_scf(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffecarry_mismatch_scf(
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C10:.+]] = arith.constant 10 : index
// CHECK: %[[LOOP:.+]]:2 = scf.for %[[IV:.+]] = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%[[ACC:.+]] = %{{.+}}, %[[DACC:.+]] = %{{.+}}) -> (f64, f64) {
// CHECK-NEXT:   scf.yield %[[ARG0:.+]], %[[ARG1:.+]] : f64, f64
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LOOP]]#1 : f64

// -----

module {
  func.func @carry_mismatch_affine(%x : f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %r = affine.for %i = 0 to 10 iter_args(%acc = %zero) -> (f64) {
      affine.yield %x : f64
    }
    return %r : f64
  }

  func.func @dcarry_mismatch_affine(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @carry_mismatch_affine(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffecarry_mismatch_affine(
// CHECK: %[[ALOOP:.+]]:2 = affine.for %[[AIV:.+]] = 0 to 10 iter_args(%[[AACC:.+]] = %{{.+}}, %[[ADACC:.+]] = %{{.+}}) -> (f64, f64) {
// CHECK-NEXT:   affine.yield %[[ARG0:.+]], %[[ARG1:.+]] : f64, f64
// CHECK-NEXT: }
// CHECK-NEXT: return %[[ALOOP]]#1 : f64
