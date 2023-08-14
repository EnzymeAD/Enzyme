// RUN: %eopt --print-activity-analysis=funcs=dps %s | FileCheck %s

module {
  // This moves active data from x to y
  func.func @output_arg(%x: !llvm.ptr, %y: !llvm.ptr) {
    %0 = llvm.load %x : !llvm.ptr -> f64
    %casted = llvm.bitcast %y : !llvm.ptr to !llvm.ptr
    llvm.store %0, %casted : f64, !llvm.ptr
    return
  }
}
