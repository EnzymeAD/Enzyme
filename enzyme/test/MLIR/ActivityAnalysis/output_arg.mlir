// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s

module {
  // This moves active data from x to y
  // CHECK-LABEL: @output_arg
  // CHECK:         "x": Active
  // CHECK:         "y": Active
  func.func @output_arg(%x: !llvm.ptr {enzyme.tag = "x", llvm.noalias}, %y: !llvm.ptr {enzyme.tag = "y", llvm.noalias}) {
    %0 = llvm.load %x : !llvm.ptr -> f64
    %casted = llvm.bitcast %y : !llvm.ptr to !llvm.ptr
    llvm.store %0, %casted : f64, !llvm.ptr
    return
  }
}
