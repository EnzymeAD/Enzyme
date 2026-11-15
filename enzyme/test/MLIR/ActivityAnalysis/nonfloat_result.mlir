// RUN: %eopt --pass-pipeline="builtin.module(print-activity-analysis{dataflow=false annotate=true})" %s --split-input-file 2>&1 | FileCheck %s

// An scf.for that only yields an integer but stores a float computation into
// memory must stay active: it writes memory, so the non-float result does not
// make it inactive.

func.func @store_in_loop(%x: f64, %m: memref<10xf64>) -> index {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (index) {
    %y = arith.mulf %x, %x : f64
    memref.store %y, %m[%i] : memref<10xf64>
    %n = arith.addi %acc, %c1 : index
    scf.yield %n : index
  }
  return %r : index
}

// CHECK:  func.func @store_in_loop(%arg0: f64, %arg1: memref<10xf64>) -> index attributes {enzyme.arg_icv0 = false, enzyme.arg_icv1 = false, enzyme.ici = false} {
// CHECK:    %0 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %c0) -> (index) {
// CHECK-NEXT:      %1 = arith.mulf %arg0, %arg0 {enzyme.ici = false, enzyme.res_icv0 = false} : f64
// CHECK-NEXT:      memref.store %1, %arg1[%arg2] {enzyme.ici = false} : memref<10xf64>
// CHECK-NEXT:      %2 = arith.addi %arg3, %c1 {enzyme.ici = true, enzyme.res_icv0 = true} : index
// CHECK-NEXT:      scf.yield {enzyme.ici = true} %2 : index
// CHECK-NEXT:    } {enzyme.arg_icv0 = true, enzyme.arg_icv1 = true, enzyme.ici = false, enzyme.res_icv0 = true}

// -----

// An scf.for that does not write memory and only yields an integer is
// inactive, even though its body computes on floats.

func.func @readonly_loop(%x: f64) -> index {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (index) {
    %y = arith.mulf %x, %x : f64
    %t = arith.fptosi %y : f64 to i64
    %ti = arith.index_cast %t : i64 to index
    %n = arith.addi %acc, %ti : index
    scf.yield %n : index
  }
  return %r : index
}

// CHECK:  func.func @readonly_loop(%arg0: f64) -> index attributes {enzyme.arg_icv0 = false, enzyme.ici = true} {
// CHECK:    %0 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %c0) -> (index) {
// CHECK-NEXT:      %1 = arith.mulf %arg0, %arg0 {enzyme.ici = true, enzyme.res_icv0 = true} : f64
// CHECK-NEXT:      %2 = arith.fptosi %1 {enzyme.ici = true, enzyme.res_icv0 = true} : f64 to i64
// CHECK-NEXT:      %3 = arith.index_cast %2 {enzyme.ici = true, enzyme.res_icv0 = true} : i64 to index
// CHECK-NEXT:      %4 = arith.addi %arg2, %3 {enzyme.ici = true, enzyme.res_icv0 = true} : index
// CHECK-NEXT:      scf.yield {enzyme.ici = true} %4 : index
// CHECK-NEXT:    } {enzyme.arg_icv0 = true, enzyme.arg_icv1 = true, enzyme.ici = true, enzyme.res_icv0 = true}
