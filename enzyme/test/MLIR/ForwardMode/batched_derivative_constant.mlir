// RUN: %eopt --enzyme %s | FileCheck %s

// Regression for a width-forward derivative with an operand-less floating
// constant. The constant participates in the tangent expression and must use
// the width-stacked shadow type.
module {
  func.func @atan(%x: tensor<4xf32>) -> tensor<4xf32> {
    %0 = math.atan %x : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @main(%x: tensor<4xf32>, %dx: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = enzyme.fwddiff @atan(%x, %dx) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>], width = 2 : i64} : (tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func private @fwddiffe2atan
// CHECK: arith.constant dense<1.000000e+00> : tensor<2x4xf32>
// CHECK: arith.divf
