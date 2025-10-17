// RUN: %eopt %s --lower-llvm-ext | FileCheck %s

module {

  llvm.func @f(%a: i64) -> !llvm.ptr {
    %0 = llvm_ext.alloc %a : (i64) -> !llvm.ptr
    %1 = llvm_ext.ptr_size_hint %0, %a : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }

  llvm.func @noop() {
    %a = arith.constant 8 : i64
    %0 = llvm_ext.alloc %a : (i64) -> !llvm.ptr
    llvm_ext.free %0 : !llvm.ptr
    llvm.return
  }

}

// CHECK:  llvm.func @free(!llvm.ptr)

// CHECK:  llvm.func @malloc(i64) -> !llvm.ptr

// CHECK:  llvm.func @f(%arg0: i64) -> !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr
// CHECK-NEXT:  }

// CHECK:  llvm.func @noop() {
// CHECK-NEXT:    %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:    %0 = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.call @free(%0) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
