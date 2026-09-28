// RUN: %eopt %s | FileCheck %s

// Checks that enzymemlir-opt registers the CIR dialect and come back without problem.

module {
  cir.func @f(%arg0: !cir.double, %arg1: !cir.double) -> !cir.double {
    %0 = cir.sin %arg0 : !cir.double
    %1 = cir.const #cir.fp<2.000000e+00> : !cir.double
    %2 = cir.pow %arg0, %1 : !cir.double
    %3 = cir.fmuladd %0, %arg1, %2 : !cir.double
    cir.return %3 : !cir.double
  }
}

// CHECK-LABEL: cir.func @f(
// CHECK-SAME:  %[[X:.+]]: !cir.double, %[[Y:.+]]: !cir.double) -> !cir.double
// CHECK-NEXT:    %[[S:.+]] = cir.sin %[[X]] : !cir.double
// CHECK-NEXT:    %[[C:.+]] = cir.const #cir.fp<2.000000e+00> : !cir.double
// CHECK-NEXT:    %[[P:.+]] = cir.pow %[[X]], %[[C]] : !cir.double
// CHECK-NEXT:    %[[R:.+]] = cir.fmuladd %[[S]], %[[Y]], %[[P]] : !cir.double
// CHECK-NEXT:    cir.return %[[R]] : !cir.double
