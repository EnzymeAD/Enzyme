// REQUIRES: poseidon-mlir
// RUN: %poseidon_mlir_opt --poseidon-rewrite %s | FileCheck %s

// The pass does nothing yet: the function comes back unchanged.

// CHECK-LABEL: func.func @axpy
// CHECK-NEXT:    %[[M:.*]] = arith.mulf %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:    %[[A:.*]] = arith.addf %[[M]], %arg2 : tensor<4xf64>
// CHECK-NEXT:    return %[[A]] : tensor<4xf64>
func.func @axpy(%a: tensor<4xf64>, %x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %0 = arith.mulf %a, %x : tensor<4xf64>
  %1 = arith.addf %0, %y : tensor<4xf64>
  return %1 : tensor<4xf64>
}
