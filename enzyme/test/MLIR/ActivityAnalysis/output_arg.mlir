// RUN: %eopt --print-activity-analysis='annotate' --split-input-file %s 2>&1 | FileCheck %s

// This moves active data from x to y
// CHECK-LABEL: @output_arg
// CHECK:         "x": Active
// CHECK:         "y": Active
func.func @output_arg(%x: !llvm.ptr {enzyme.tag = "x", llvm.noalias}, %y: !llvm.ptr {enzyme.tag = "y", llvm.noalias}) {
  %c8 = llvm.mlir.constant (8) : i64
  %0 = llvm.load %x : !llvm.ptr -> f64
  %m = llvm.alloca %c8 x f64 {tag = "m"} : (i64) -> !llvm.ptr
  llvm.store %0, %m : f64, !llvm.ptr
  %casted = llvm.bitcast %y : !llvm.ptr to !llvm.ptr
  %1 = llvm.load %m : !llvm.ptr -> f64
  llvm.store %1, %casted : f64, !llvm.ptr
  return
}
