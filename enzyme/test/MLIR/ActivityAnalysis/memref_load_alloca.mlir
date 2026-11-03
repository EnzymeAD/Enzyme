// RUN: %eopt --print-activity-analysis="dataflow=false annotate=true" --split-input-file %s 2>&1 | FileCheck %s
// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s --check-prefix=DATAFLOW

// A stack slot carrying an active value must remain active through a load.
// CHECK-LABEL: func.func @active_stack(
// CHECK: memref.alloca() {enzyme.ici = true, enzyme.res_icv0 = false}
// CHECK: memref.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// DATAFLOW-LABEL: @active_stack:
// DATAFLOW: "result": Active
func.func @active_stack(%x: f64) -> f64 {
  %slot = memref.alloca() : memref<f64>
  memref.store %x, %slot[] : memref<f64>
  %value = memref.load %slot[] {tag = "result"} : memref<f64>
  return %value : f64
}

// -----

// Loading from a constant argument must not introduce activity.
// CHECK-LABEL: func.func @constant_argument(
// CHECK: memref.load {{.*}}enzyme.ici = true, enzyme.res_icv0 = true
// DATAFLOW-LABEL: @constant_argument:
// DATAFLOW: "result": Constant
func.func @constant_argument(%source: memref<f64> {enzyme.const}) -> f64 {
  %value = memref.load %source[] {tag = "result"} : memref<f64>
  return %value : f64
}

// -----

// An active argument's loaded floating-point value remains active.
// CHECK-LABEL: func.func @active_argument(
// CHECK: memref.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// DATAFLOW-LABEL: @active_argument:
// DATAFLOW: "result": Active
func.func @active_argument(%source: memref<f64>) -> f64 {
  %value = memref.load %source[] {tag = "result"} : memref<f64>
  return %value : f64
}

// -----

// Integer-only stack memory and its loaded value need no derivative.
// CHECK-LABEL: func.func @integer_stack(
// CHECK: memref.alloca() {enzyme.ici = true, enzyme.res_icv0 = true}
// CHECK: memref.load {{.*}}enzyme.ici = true, enzyme.res_icv0 = true
// DATAFLOW-LABEL: @integer_stack:
// DATAFLOW: "result": Constant
func.func @integer_stack(%x: i64) -> i64 {
  %slot = memref.alloca() : memref<i64>
  memref.store %x, %slot[] : memref<i64>
  %value = memref.load %slot[] {tag = "result"} : memref<i64>
  return %value : i64
}

// -----

// Keep a constant store inactive alongside an active store. The legacy activity
// analysis conservatively keeps both floating-point slots and loads active.
// CHECK-LABEL: func.func @independent_stack_slots(
// CHECK: memref.alloca() {enzyme.ici = true, enzyme.res_icv0 = false}
// CHECK: memref.alloca() {enzyme.ici = true, enzyme.res_icv0 = false}
// CHECK: memref.store {{.*}}enzyme.ici = false
// CHECK: memref.store {{.*}}enzyme.ici = true
// CHECK: memref.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// CHECK: memref.load {{.*}}enzyme.ici = false, enzyme.res_icv0 = false
// DATAFLOW-LABEL: @independent_stack_slots:
// DATAFLOW: "active_load": Active
// DATAFLOW: "constant_load": Constant
func.func @independent_stack_slots(%x: f64) -> f64 {
  %active = memref.alloca() : memref<f64>
  %constant = memref.alloca() : memref<f64>
  %one = arith.constant 1.0 : f64
  memref.store %x, %active[] : memref<f64>
  memref.store %one, %constant[] : memref<f64>
  %a = memref.load %active[] {tag = "active_load"} : memref<f64>
  %b = memref.load %constant[] {tag = "constant_load"} : memref<f64>
  %result = arith.addf %a, %b : f64
  return %result : f64
}
