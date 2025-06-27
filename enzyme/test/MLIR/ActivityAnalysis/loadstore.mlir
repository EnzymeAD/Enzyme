// RUN: %eopt --print-activity-analysis='funcs=loadstore annotate' %s --split-input-file 2>&1 | FileCheck %s

// A function that contains active dataflow via side effects, then operands to results,
// then side effects again.

// CHECK-LABEL: @loadstore:
// CHECK: "x": Active
// CHECK: "squared": Active
// CHECK: "cos": Active
// CHECK: "ret": Active
func.func @loadstore(%x: f64 {enzyme.tag = "x"}) -> f64 {
  %c1 = llvm.mlir.constant (1) : i64
  %m1 = llvm.alloca %c1 x f64 : (i64) -> !llvm.ptr
  %m2 = llvm.alloca %c1 x f64 : (i64) -> !llvm.ptr

  llvm.store %x, %m1 : f64, !llvm.ptr
  %load1 = llvm.load %m1 : !llvm.ptr -> f64

  %squared = arith.mulf %load1, %load1 {tag = "squared"}: f64
  %cos = math.cos %squared {tag = "cos"} : f64

  llvm.store %cos, %m2 : f64, !llvm.ptr
  %load2 = llvm.load %m2 {tag = "ret"} : !llvm.ptr -> f64
  return %load2 : f64
}
