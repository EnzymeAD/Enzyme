// RUN: %eopt --print-activity-analysis="dataflow=false annotate=true" --split-input-file %s 2>&1 | FileCheck %s
// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s --check-prefix=DATAFLOW

// An active value stored and loaded through affine operations remains active.
// CHECK-LABEL: func.func @affine_active_stack(
// CHECK: affine.store {{.*}}enzyme.ici = false
// CHECK: affine.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// DATAFLOW-LABEL: @affine_active_stack:
// DATAFLOW: "result": Active
func.func @affine_active_stack(%x: f64) -> f64 {
  %slot = memref.alloca() : memref<1xf64>
  affine.store %x, %slot[0] : memref<1xf64>
  %value = affine.load %slot[0] {tag = "result"} : memref<1xf64>
  return %value : f64
}

// -----

// LoadLike returns the base memref even when the access uses an affine map.
// CHECK-LABEL: func.func @affine_indexed_active(
// CHECK: affine.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// DATAFLOW-LABEL: @affine_indexed_active:
// DATAFLOW: "result": Active
func.func @affine_indexed_active(%source: memref<?xf64>, %i: index) -> f64 {
  %value = affine.load %source[2 * %i + 1] {tag = "result"} : memref<?xf64>
  return %value : f64
}

// -----

// Loading from a constant base memref must not introduce activity.
// CHECK-LABEL: func.func @affine_indexed_constant(
// CHECK: affine.load {{.*}}enzyme.ici = true, enzyme.res_icv0 = true
// DATAFLOW-LABEL: @affine_indexed_constant:
// DATAFLOW: "result": Constant
func.func @affine_indexed_constant(%source: memref<?xf64> {enzyme.const}, %i: index) -> f64 {
  %value = affine.load %source[2 * %i + 1] {tag = "result"} : memref<?xf64>
  return %value : f64
}

// -----

// Integer values remain inactive even when stored and loaded on the stack.
// CHECK-LABEL: func.func @affine_integer_stack(
// CHECK: affine.store {{.*}}enzyme.ici = true
// CHECK: affine.load {{.*}}enzyme.ici = true, enzyme.res_icv0 = true
// DATAFLOW-LABEL: @affine_integer_stack:
// DATAFLOW: "result": Constant
func.func @affine_integer_stack(%x: i64) -> i64 {
  %slot = memref.alloca() : memref<1xi64>
  affine.store %x, %slot[0] : memref<1xi64>
  %value = affine.load %slot[0] {tag = "result"} : memref<1xi64>
  return %value : i64
}
