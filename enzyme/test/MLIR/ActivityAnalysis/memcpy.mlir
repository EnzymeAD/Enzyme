// RUN: %eopt --print-activity-analysis %s --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: @memcpy_active
// CHECK:         "z": Active
func.func @memcpy_active(%x: !llvm.ptr {llvm.noalias}, %y: !llvm.ptr {llvm.noalias}) {
    %size = llvm.mlir.constant (4) : i64
    %z = llvm.alloca %size x f64 {tag = "z"} : (i64) -> !llvm.ptr
    "llvm.intr.memcpy"(%z, %x, %size) {isVolatile = false} : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memmove"(%y, %z, %size) {isVolatile = false} : (!llvm.ptr, !llvm.ptr, i64) -> ()
    return
}
