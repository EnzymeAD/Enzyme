// RUN: %eopt --print-activity-analysis="dataflow=false annotate=true" %s --split-input-file 2>&1 | FileCheck %s

// A view's activity depends on its source, not the memory containing its shape.
// CHECK-LABEL: func.func @inactive_reshape(
// CHECK: memref.reshape {{.*}}enzyme.ici = true, enzyme.res_icv0 = true
// CHECK: memref.load {{.*}}enzyme.ici = true, enzyme.res_icv0 = true
func.func @inactive_reshape(%source: memref<4xf64> {enzyme.const}, %shape: memref<2xindex>) -> f64 {
  %view = memref.reshape %source(%shape) : (memref<4xf64>, memref<2xindex>) -> memref<?x?xf64>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0, %c0] : memref<?x?xf64>
  return %value : f64
}

// -----

// CHECK-LABEL: func.func @active_reshape(
// CHECK: memref.reshape {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// CHECK: memref.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
func.func @active_reshape(%source: memref<4xf64>, %shape: memref<2xindex> {enzyme.const}) -> f64 {
  %view = memref.reshape %source(%shape) : (memref<4xf64>, memref<2xindex>) -> memref<?x?xf64>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0, %c0] : memref<?x?xf64>
  return %value : f64
}

// -----

// Follow a nonzero-offset view to its allocation when checking memory activity.
// CHECK-LABEL: func.func @view_of_allocation(
// CHECK: memref.subview {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// CHECK: memref.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
func.func @view_of_allocation(%value: f64) -> f64 {
  %source = memref.alloca() : memref<4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  memref.store %value, %source[%c1] : memref<4xf64>
  %view = memref.subview %source[1] [2] [1] : memref<4xf64> to memref<2xf64, strided<[1], offset: 1>>
  %result = memref.load %view[%c0] : memref<2xf64, strided<[1], offset: 1>>
  return %result : f64
}
