// Tier-1 reverse-mode rule for hlfir.matmul.  For C = A*B with result adjoint G:
//   Ā += matmul(G, transpose(B))
//   B̄ += matmul_transpose(A, G)   ( == Aᵀ·G )
// The primal A and B are cached in the forward sweep and popped in reverse.
//
// REQUIRES: fir_enzyme_opt
// RUN: %fireopt %s --enzyme-wrap="infn=mm outfn= argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func @mm(%a: !hlfir.expr<2x2xf32>, %b: !hlfir.expr<2x2xf32>)
      -> !hlfir.expr<2x2xf32> {
    %c = hlfir.matmul %a %b
      : (!hlfir.expr<2x2xf32>, !hlfir.expr<2x2xf32>) -> !hlfir.expr<2x2xf32>
    return %c : !hlfir.expr<2x2xf32>
  }

  // CHECK-LABEL: func.func @mm
  // B̄ += Aᵀ·G
  // CHECK-DAG: hlfir.matmul_transpose %[[a:.+]] %[[g:.+]]
  // Ā += G·Bᵀ
  // CHECK-DAG: hlfir.transpose %[[b:.+]]
  // CHECK-DAG: hlfir.matmul %[[g2:.+]] %[[bt:.+]]
}
