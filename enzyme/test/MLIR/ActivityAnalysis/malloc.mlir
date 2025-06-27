// RUN: %eopt --print-activity-analysis='annotate' %s --split-input-file 2>&1 | FileCheck %s

func.func private @malloc(%size: i64) -> !llvm.ptr
func.func private @free(%ptr: !llvm.ptr)

// Check that the forward state is being propagated over calls to `malloc` (i.e. that the state is not reset)
// CHECK-LABEL: @loadstore_malloc
// CHECK:         "y": Active
// CHECK:         "z": Active
func.func @loadstore_malloc(%x: f64) -> f64 {
    %cst = llvm.mlir.constant (8) : i64
    %m1 = call @malloc(%cst) : (i64) -> !llvm.ptr
    llvm.store %x, %m1 : f64, !llvm.ptr
    %m2 = call @malloc(%cst) : (i64) -> !llvm.ptr
    %y = llvm.load %m1 {tag = "y"} : !llvm.ptr -> f64
    call @free(%m1) : (!llvm.ptr) -> ()
    llvm.store %y, %m2 : f64, !llvm.ptr
    %z = llvm.load %m2 {tag = "z"} : !llvm.ptr -> f64
    call @free(%m2) : (!llvm.ptr) -> ()
    return %z : f64
}
