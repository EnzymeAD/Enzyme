// RUN: %eopt --pass-pipeline="builtin.module(print-activity-analysis{dataflow=false annotate=true})" %s --split-input-file 2>&1 | FileCheck %s

// A function that contains active and inactive region dataflow

func.func @region(%x: f64) -> (f64, f64) {
  %f0 = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %r0:2 = scf.for %arg12 = %c0 to %c10 step %c1 iter_args(%arg13 = %f0, %arg14 = %f0) -> (f64, f64) {
    %m = arith.addf %arg13, %x : f64
    scf.yield %m, %arg14 : f64, f64
  }
  return %r0#0, %r0#1 : f64, f64
}

// CHECK:  func.func @region(%arg0: f64) -> (f64, f64) attributes {enzyme.arg_icv0 = false, enzyme.ici = false} {
// CHECK-NEXT:    %cst = arith.constant {enzyme.ici = true, enzyme.res_icv0 = true} 0.000000e+00 : f64
// CHECK-NEXT:    %c0 = arith.constant {enzyme.ici = true, enzyme.res_icv0 = true} 0 : index
// CHECK-NEXT:    %c10 = arith.constant {enzyme.ici = true, enzyme.res_icv0 = true} 10 : index
// CHECK-NEXT:    %c1 = arith.constant {enzyme.ici = true, enzyme.res_icv0 = true} 1 : index
// CHECK-NEXT:    %0:2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst, %arg3 = %cst) -> (f64, f64) {
// CHECK-NEXT:      %1 = arith.addf %arg2, %arg0 {enzyme.ici = false, enzyme.res_icv0 = false} : f64
// CHECK-NEXT:      scf.yield {enzyme.ici = true} %1, %arg3 : f64, f64
// CHECK-NEXT:    } {enzyme.arg_icv0 = true, enzyme.arg_icv1 = false, enzyme.arg_icv2 = true, enzyme.ici = false, enzyme.res_icv0 = false, enzyme.res_icv1 = true}
// CHECK-NEXT:    return {enzyme.ici = true} %0#0, %0#1 : f64, f64
// CHECK-NEXT:  }
