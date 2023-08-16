// RUN: %eopt --print-activity-analysis %s 2>&1 | FileCheck %s

module {
  func.func private @printF64(f64)
  // CHECK-LABEL: @print_const:
  // CHECK          "loaded": Constant
  func.func @print_const(%x: f64) {
    %c1 = llvm.mlir.constant (1) : i64 
    %0 = llvm.alloca %c1 x f64 : (i64) -> !llvm.ptr
    llvm.store %x, %0 : f64, !llvm.ptr
    %1 = llvm.load %0 {tag = "loaded"} : !llvm.ptr -> f64
    call @printF64(%1) : (f64) -> ()
    return
  }
}
