// Tier-1 forward-mode rule for hlfir.matmul:  d(A*B) = dA*B + A*dB.
// The two products are hlfir.matmul; the sum is an elementwise hlfir.elemental
// built by the !hlfir.expr AutoDiffTypeInterface.
//
// REQUIRES: fir_enzyme_opt
// RUN: %fireopt --enzyme %s | FileCheck %s

module {
  func.func @mm(%a: !hlfir.expr<2x2xf32>, %b: !hlfir.expr<2x2xf32>)
      -> !hlfir.expr<2x2xf32> {
    %c = hlfir.matmul %a %b
      : (!hlfir.expr<2x2xf32>, !hlfir.expr<2x2xf32>) -> !hlfir.expr<2x2xf32>
    return %c : !hlfir.expr<2x2xf32>
  }

  func.func @dmm(%a: !hlfir.expr<2x2xf32>, %da: !hlfir.expr<2x2xf32>,
                 %b: !hlfir.expr<2x2xf32>, %db: !hlfir.expr<2x2xf32>)
      -> !hlfir.expr<2x2xf32> {
    %r = enzyme.fwddiff @mm(%a, %da, %b, %db) {
      activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (!hlfir.expr<2x2xf32>, !hlfir.expr<2x2xf32>,
         !hlfir.expr<2x2xf32>, !hlfir.expr<2x2xf32>) -> (!hlfir.expr<2x2xf32>)
    return %r : !hlfir.expr<2x2xf32>
  }

  // CHECK-LABEL: func.func private @fwddiffemm
  // dA*B and A*dB
  // CHECK-DAG: hlfir.matmul %[[da:.+]] %[[b:.+]]
  // CHECK-DAG: hlfir.matmul %[[a:.+]] %[[db:.+]]
  // sum of the two tangent contributions
  // CHECK: hlfir.elemental
  // CHECK: arith.addf
}
