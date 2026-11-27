// RUN: %eopt --allow-unregistered-dialect --enzyme-batch %s | FileCheck %s

module {
  func.func @attrs(%x: tensor<3x4xf64>) -> tensor<3x4xf64> {
    %0 = "test.dims_dense64"(%x) {dimensions = array<i64: 0, 1>} : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %1 = "test.dims_dense32"(%0) {broadcast_dimensions = array<i32: 1>} : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %2 = "test.dims_array"(%1) {my_dimensions = [0 : i64, 1 : i64]} : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %3 = "test.dims_str"(%2) {dimensions = ["x"]} : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %4 = "test.dim_i64"(%3) {dimension = 0 : i64} : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %5 = "test.dim_i32"(%4) {concat_dimension = 1 : i32, other = 5 : i64} : (tensor<3x4xf64>) -> tensor<3x4xf64>
    return %5 : tensor<3x4xf64>
  }
  func.func @batch_attrs(%x: tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64> {
    %r = enzyme.batch @attrs(%x) <{batch_shape = array<i64: 10, 2>}> : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
    return %r : tensor<10x2x3x4xf64>
  }

  func.func @nullary(%x: tensor<3xf64>) -> (tensor<3xf64>, f64) {
    %iota = "test.iota"() {iota_dimension = 0 : i64} : () -> tensor<3xf64>
    %m:2 = "test.multi"() : () -> (tensor<3xf64>, f64)
    %r = "test.region_nullary"() ({
      %s = math.sin %x : tensor<3xf64>
      "test.yield"(%s) : (tensor<3xf64>) -> ()
    }) : () -> tensor<3xf64>
    %empty = tensor.empty() : tensor<3xf64>
    %ns = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
    %sc = arith.constant 2.5 : f64
    %s1 = arith.addf %iota, %m#0 : tensor<3xf64>
    %s2 = arith.addf %s1, %r : tensor<3xf64>
    %s3 = arith.addf %s2, %empty : tensor<3xf64>
    %s4 = arith.addf %s3, %ns : tensor<3xf64>
    %s5 = arith.addf %m#1, %sc : f64
    return %s4, %s5 : tensor<3xf64>, f64
  }
  func.func @batch_nullary(%x: tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<1xf64>) {
    %r:2 = enzyme.batch @nullary(%x) <{batch_shape = array<i64: 1>}> : (tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<1xf64>)
    return %r#0, %r#1 : tensor<1x3xf64>, tensor<1xf64>
  }
}

// CHECK-LABEL: func.func private @batched_attrs(%arg0: tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    %[[V0:.+]] = "test.dims_dense64"(%arg0) {dimensions = array<i64: 2, 3>} : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    %[[V1:.+]] = "test.dims_dense32"(%[[V0]]) {broadcast_dimensions = array<i32: 3>} : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    %[[V2:.+]] = "test.dims_array"(%[[V1]]) {my_dimensions = [2, 3]} : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    %[[V3:.+]] = "test.dims_str"(%[[V2]]) {dimensions = ["x"]} : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    %[[V4:.+]] = "test.dim_i64"(%[[V3]]) {dimension = 2 : i64} : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    %[[V5:.+]] = "test.dim_i32"(%[[V4]]) {concat_dimension = 3 : i32, other = 5 : i64} : (tensor<10x2x3x4xf64>) -> tensor<10x2x3x4xf64>
// CHECK-NEXT:    return %[[V5]] : tensor<10x2x3x4xf64>

// CHECK-LABEL: func.func private @batched_nullary(%arg0: tensor<1x3xf64>) -> (tensor<1x3xf64>, tensor<1xf64>)
// CHECK-NEXT:    %[[IOTA:.+]] = "test.iota"() {iota_dimension = 0 : i64} : () -> tensor<3xf64>
// CHECK-NEXT:    %[[IOTA_B:.+]] = "enzyme.broadcast"(%[[IOTA]]) <{shape = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[M:.+]]:2 = "test.multi"() : () -> (tensor<3xf64>, f64)
// CHECK-NEXT:    %[[M0_B:.+]] = "enzyme.broadcast"(%[[M]]#0) <{shape = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[M1_B:.+]] = "enzyme.broadcast"(%[[M]]#1) <{shape = array<i64: 1>}> : (f64) -> tensor<1xf64>
// CHECK-NEXT:    %[[REG:.+]] = "test.region_nullary"() ({
// CHECK-NEXT:      %[[SIN:.+]] = math.sin %arg0 : tensor<1x3xf64>
// CHECK-NEXT:      "test.yield"(%[[SIN]]) : (tensor<1x3xf64>) -> ()
// CHECK-NEXT:    }) : () -> tensor<3xf64>
// CHECK-NEXT:    %[[REG_B:.+]] = "enzyme.broadcast"(%[[REG]]) <{shape = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[EMPTY:.+]] = tensor.empty() : tensor<3xf64>
// CHECK-NEXT:    %[[EMPTY_B:.+]] = "enzyme.broadcast"(%[[EMPTY]]) <{shape = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[NS:.+]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
// CHECK-NEXT:    %[[NS_B:.+]] = "enzyme.broadcast"(%[[NS]]) <{shape = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[SC:.+]] = arith.constant 2.500000e+00 : f64
// CHECK-NEXT:    %[[SC_B:.+]] = "enzyme.broadcast"(%[[SC]]) <{shape = array<i64: 1>}> : (f64) -> tensor<1xf64>
// CHECK-NEXT:    %[[A1:.+]] = arith.addf %[[IOTA_B]], %[[M0_B]] : tensor<1x3xf64>
// CHECK-NEXT:    %[[A2:.+]] = arith.addf %[[A1]], %[[REG_B]] : tensor<1x3xf64>
// CHECK-NEXT:    %[[A3:.+]] = arith.addf %[[A2]], %[[EMPTY_B]] : tensor<1x3xf64>
// CHECK-NEXT:    %[[A4:.+]] = arith.addf %[[A3]], %[[NS_B]] : tensor<1x3xf64>
// CHECK-NEXT:    %[[A5:.+]] = arith.addf %[[M1_B]], %[[SC_B]] : tensor<1xf64>
// CHECK-NEXT:    return %[[A4]], %[[A5]] : tensor<1x3xf64>, tensor<1xf64>
