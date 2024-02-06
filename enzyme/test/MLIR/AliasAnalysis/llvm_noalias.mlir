// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

// The dataflow framework never sees what is stored into %x, but we know its load result does not alias
// with %x itself because of the `llvm.noalias`

// CHECK: "x" and "x_load": NoAlias
// CHECK: "y" and "x_load": MayAlias
llvm.func @f(%x: !llvm.ptr {enzyme.tag = "x", llvm.noalias}, %y: !llvm.ptr {enzyme.tag = "y"}) {
    %x_load = llvm.load %x {tag = "x_load"} : !llvm.ptr -> !llvm.ptr
    llvm.return
}
