// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

// CHECK: "x" and "z": NoAlias
func.func @activereturn(%x: memref<f64> {enzyme.tag = "x"}) -> memref<f64> {
    %z = memref.alloc() {tag = "z"} : memref<f64>
    return %z : memref<f64>
}

// -----

// Entry arguments might alias in the absence of attributes specifying they do not alias
// CHECK: "a" and "b": MayAlias
func.func @aliased_args(%x: !llvm.ptr {enzyme.tag = "a"}, %y: !llvm.ptr {enzyme.tag = "b"}) -> f64 {
    %one = llvm.mlir.constant (1.0) : f64
    %two = llvm.mlir.constant (2.0) : f64
    llvm.store %one, %x : f64, !llvm.ptr
    llvm.store %two, %y : f64, !llvm.ptr
    %res = llvm.load %x : !llvm.ptr -> f64
    return %res : f64
}

// -----

// CHECK: "a" and "b": NoAlias
func.func @nonaliased_args(%x: !llvm.ptr {llvm.noalias, enzyme.tag = "a"}, %y: !llvm.ptr {enzyme.tag = "b"}) {
    return
}
