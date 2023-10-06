// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

// CHECK: "readptr" and "writeptr": NoAlias
func.func private @read(%ptr: !llvm.ptr {enzyme.tag = "readptr"}) -> f64 {
    %res = llvm.load %ptr : !llvm.ptr -> f64
    return %res : f64
}

func.func private @write(%arg0: f64, %ptr: !llvm.ptr {enzyme.tag = "writeptr"}) {
    llvm.store %arg0, %ptr : f64, !llvm.ptr
    return
}

func.func @read_write_separate(%arg0: f64) -> f64 {
    %c1 = llvm.mlir.constant (1) : i64
    %ptr1 = llvm.alloca %c1 x f64 : (i64) -> !llvm.ptr
    %ptr2 = llvm.alloca %c1 x f64 : (i64) -> !llvm.ptr
    call @write(%arg0, %ptr1) : (f64, !llvm.ptr) -> ()
    %val = call @read(%ptr2) : (!llvm.ptr) -> f64
    return %val : f64
}
