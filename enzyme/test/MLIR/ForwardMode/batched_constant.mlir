// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// A rule whose pattern contains a literal constant (d tanh = 1 - tanh^2) at
// vector width > 1: the primal-only part of the rule is computed once at
// primal shape and broadcast only where it meets the batched tangent. Before
// the constant kept its primal shape while everything else was batched, and
// building subf(tensor<3xf64>, tensor<2x3xf64>) failed.

module {
  func.func @tanh(%x : tensor<3xf64>) -> tensor<3xf64> {
    %y = math.tanh %x : tensor<3xf64>
    return %y : tensor<3xf64>
  }
  func.func @dtanh(%x : tensor<3xf64>, %dx : tensor<2x3xf64>) -> tensor<2x3xf64> {
    %r = enzyme.fwddiff @tanh(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], width=2 }> : (tensor<3xf64>, tensor<2x3xf64>) -> (tensor<2x3xf64>)
    return %r : tensor<2x3xf64>
  }

  func.func @stanh(%x : f64) -> f64 {
    %y = math.tanh %x : f64
    return %y : f64
  }
  func.func @dstanh(%x : f64, %dx : tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.fwddiff @stanh(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], width=2 }> : (f64, tensor<2xf64>) -> (tensor<2xf64>)
    return %r : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func private @fwddiffe2tanh(
// CHECK-SAME:      %[[X:.+]]: tensor<3xf64>, %[[DX:.+]]: tensor<2x3xf64>)
// CHECK:         %[[ONE:.+]] = arith.constant dense<1.000000e+00> : tensor<3xf64>
// CHECK:         %[[T:.+]] = math.tanh %[[X]] {{.*}} : tensor<3xf64>
// CHECK:         %[[T2:.+]] = arith.mulf %[[T]], %[[T]] {{.*}} : tensor<3xf64>
// CHECK:         %[[D:.+]] = arith.subf %[[ONE]], %[[T2]] {{.*}} : tensor<3xf64>
// CHECK:         %[[B:.+]] = "enzyme.broadcast"(%[[D]]) <{shape = array<i64: 2>}> : (tensor<3xf64>) -> tensor<2x3xf64>
// CHECK:         %[[R:.+]] = arith.mulf %[[DX]], %[[B]] {{.*}} : tensor<2x3xf64>
// CHECK:         return %[[R]]

// CHECK-LABEL: func.func private @fwddiffe2stanh(
// CHECK-SAME:      %[[X:.+]]: f64, %[[DX:.+]]: tensor<2xf64>)
// CHECK:         %[[ONE:.+]] = arith.constant 1.000000e+00 : f64
// CHECK:         %[[T:.+]] = math.tanh %[[X]] {{.*}} : f64
// CHECK:         %[[T2:.+]] = arith.mulf %[[T]], %[[T]] {{.*}} : f64
// CHECK:         %[[D:.+]] = arith.subf %[[ONE]], %[[T2]] {{.*}} : f64
// CHECK:         %[[B:.+]] = "enzyme.broadcast"(%[[D]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK:         %[[R:.+]] = arith.mulf %[[DX]], %[[B]] {{.*}} : tensor<2xf64>
// CHECK:         return %[[R]]
