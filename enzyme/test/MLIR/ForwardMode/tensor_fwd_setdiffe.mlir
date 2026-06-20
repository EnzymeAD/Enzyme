// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @sincos(%x : tensor<2xf64>) -> tensor<2xf64> {
    %y = math.sin %x : tensor<2xf64>
    %z = math.cos %y : tensor<2xf64>
    return %z : tensor<2xf64>
  }
  func.func @dsincos(%x : tensor<2xf64>, %dx : tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @sincos(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>)
    return %r : tensor<2xf64>
  }
}

// CHECK:   func.func private @fwddiffesincos(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:     %[[cos0:.+]] = math.cos %arg0 : tensor<2xf64>
// CHECK-NEXT:     %[[dsin:.+]] = arith.mulf %arg1, %[[cos0]] : tensor<2xf64>
// CHECK-NEXT:     %[[sin0:.+]] = math.sin %arg0 : tensor<2xf64>
// CHECK-NEXT:     %[[sin1:.+]] = math.sin %[[sin0]] : tensor<2xf64>
// CHECK-NEXT:     %[[negsin:.+]] = arith.negf %[[sin1]] : tensor<2xf64>
// CHECK-NEXT:     %[[dcos:.+]] = arith.mulf %[[dsin]], %[[negsin]] : tensor<2xf64>
// CHECK-NEXT:     %[[cos1:.+]] = math.cos %[[sin0]] : tensor<2xf64>
// CHECK-NEXT:     return %[[dcos]] : tensor<2xf64>
// CHECK-NEXT:   }
