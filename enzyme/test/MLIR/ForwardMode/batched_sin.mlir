// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

module @reactant_jac attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"Const{typeof(f)}(Main.f)_autodiff"(%arg0: tensor<10xf64>) -> (tensor<10xf64>) {
    %0 = math.sin %arg0 : tensor<10xf64>
    return %0 : tensor<10xf64>
  }
  func.func @main(%arg0: tensor<10xf64>) -> tensor<10x10xf64> {
    %cst = arith.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<10x10xf64>
    %0 = enzyme.fwddiff @"Const{typeof(f)}(Main.f)_autodiff"(%arg0, %cst) {
      activity = [#enzyme<activity enzyme_dup>],
      enzymexla.guaranteed_symmetric = false,
      ret_activity = [#enzyme<activity enzyme_dupnoneed>],
      width = 10 : i64} : (tensor<10xf64>, tensor<10x10xf64>) -> tensor<10x10xf64>
    return %0 : tensor<10x10xf64>
  }
}
// CHECK:  func.func private @"fwddiffe10Const{typeof(f)}(Main.f)_autodiff"(%arg0: tensor<10xf64>, %arg1: tensor<10x10xf64>) -> tensor<10x10xf64> {
// CHECK-NEXT:    %[[v0:.+]] = "enzyme.broadcast"(%arg0) <{shape = array<i64: 10>}> : (tensor<10xf64>) -> tensor<10x10xf64>
// CHECK-NEXT:    %[[v1:.+]] = math.cos %[[v0]] : tensor<10x10xf64>
// CHECK-NEXT:    %[[v2:.+]] = arith.mulf %arg1, %[[v1]] : tensor<10x10xf64>
// CHECK-NEXT:    return %[[v2]] : tensor<10x10xf64>
// CHECK-NEXT:  }
