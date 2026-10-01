// RUN: %eopt %s --split-multi-results --split-input-file | FileCheck %s

func.func @multi_if(%cond: i1, %x: f64) -> (f64, f64) {
  %res:2 = scf.if %cond -> (f64, f64) {
    %cos = math.cos %x : f64
    %tanh = math.tanh %x : f64
    scf.yield %cos, %tanh : f64, f64
  } else {
    %sin = math.sin %x : f64
    scf.yield %x, %sin : f64, f64
  }
  %mul = arith.mulf %res#0, %res#0 : f64
  %add = arith.addf %res#1, %res#1 : f64
  return %mul, %add : f64, f64
}

// CHECK-LABEL:   func.func @multi_if(
// CHECK-SAME:      %[[ARG0:.*]]: i1,
// CHECK-SAME:      %[[ARG1:.*]]: f64) -> (f64, f64) {
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG0]] -> (f64) {
// CHECK:             %[[COS_0:.*]] = math.cos %[[ARG1]] : f64
// CHECK:             scf.yield %[[COS_0]] : f64
// CHECK:           } else {
// CHECK:             scf.yield %[[ARG1]] : f64
// CHECK:           }
// CHECK:           %[[IF_1:.*]] = scf.if %[[ARG0]] -> (f64) {
// CHECK:             %[[TANH_0:.*]] = math.tanh %[[ARG1]] : f64
// CHECK:             scf.yield %[[TANH_0]] : f64
// CHECK:           } else {
// CHECK:             %[[SIN_0:.*]] = math.sin %[[ARG1]] : f64
// CHECK:             scf.yield %[[SIN_0]] : f64
// CHECK:           }
// CHECK:           %[[MULF_0:.*]] = arith.mulf %[[IF_0]], %[[IF_0]] : f64
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[IF_1]], %[[IF_1]] : f64
// CHECK:           return %[[MULF_0]], %[[ADDF_0]] : f64, f64
// CHECK:         }
