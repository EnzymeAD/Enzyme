// RUN: %eopt %s --raise-llvm-ext | FileCheck %s

module {
  llvm.func @__enzyme_ptr_size_hint(!llvm.ptr, i64)

  llvm.func @f(%a: i64) -> !llvm.ptr {
    %0 = llvm_ext.alloc %a : (i64) -> !llvm.ptr
    llvm.call @__enzyme_ptr_size_hint(%0, %a) : (!llvm.ptr, i64) -> ()
    llvm.return %0 : !llvm.ptr
  }
}

// CHECK:  llvm.func @f(%[[SIZE:.+]]: i64) -> !llvm.ptr {
// CHECK-NEXT:    %[[PTR:.+]] = llvm_ext.alloc %[[SIZE]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.ptr_size_hint %[[PTR]], %[[SIZE]] : !llvm.ptr, i64
// CHECK-NEXT:    llvm.return %[[PTR]] : !llvm.ptr
// CHECK-NEXT:  }
