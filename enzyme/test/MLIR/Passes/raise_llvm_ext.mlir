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

// -----

// The three-argument form annotates the memory space the pointer really
// addresses, for frontends whose types cannot say it -- a cudaMalloc'd buffer
// is a plain `float *`. It is taken at its word: the hint is put on an
// addrspacecast to that space, which is what makes the clone of this
// allocation a device one.

module {
  llvm.func @__enzyme_ptr_size_hint(!llvm.ptr, i64, i64)

  llvm.func @device(%p: !llvm.ptr, %n: i64) {
    %space = llvm.mlir.constant(1 : i64) : i64
    llvm.call @__enzyme_ptr_size_hint(%p, %n, %space) : (!llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK:  llvm.func @device(%[[PTR:.+]]: !llvm.ptr, %[[SIZE:.+]]: i64) {
// CHECK:    %[[CAST:.+]] = llvm.addrspacecast %[[PTR]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    llvm_ext.ptr_size_hint %[[CAST]], %[[SIZE]] : !llvm.ptr<1>, i64
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// -----

// An annotation that agrees with the pointer's type has nothing to add.

module {
  llvm.func @__enzyme_ptr_size_hint(!llvm.ptr, i64, i64)

  llvm.func @agrees(%p: !llvm.ptr, %n: i64) {
    %space = llvm.mlir.constant(0 : i64) : i64
    llvm.call @__enzyme_ptr_size_hint(%p, %n, %space) : (!llvm.ptr, i64, i64) -> ()
    llvm.return
  }
}

// CHECK:  llvm.func @agrees(%[[PTR:.+]]: !llvm.ptr, %[[SIZE:.+]]: i64) {
// CHECK-NOT:   llvm.addrspacecast
// CHECK:    llvm_ext.ptr_size_hint %[[PTR]], %[[SIZE]] : !llvm.ptr, i64
