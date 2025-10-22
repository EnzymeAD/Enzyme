// RUN: %eopt %s --lower-llvm-ext=lower-to-alloca-threshold=4 | FileCheck %s

module {

  llvm.func @g() {
    %cst = arith.constant 4 : i64
    %0 = llvm_ext.alloc %cst : (i64) -> !llvm.ptr

    %v = arith.constant 42 : i32
    llvm.store %v, %0 : i32, !llvm.ptr

    llvm_ext.free %0 : !llvm.ptr
    llvm.return
  }

  llvm.func @f(%a: i64) -> !llvm.ptr {
    %0 = llvm_ext.alloc %a : (i64) -> !llvm.ptr
    llvm_ext.ptr_size_hint %0, %a : !llvm.ptr, i64
    llvm.return %0 : !llvm.ptr
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

// CHECK:  llvm.func @g() {
// CHECK-NEXT:    %[[SIZE:.+]] = arith.constant 4 : i64
// CHECK-NEXT:    %[[PTR:.+]] = llvm.alloca %[[SIZE]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.start %[[PTR]] : !llvm.ptr
// CHECK-NEXT:    %[[V:.+]] = arith.constant 42 : i32
// CHECK-NEXT:    llvm.store %[[V]], %[[PTR]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.end %[[PTR]] : !llvm.ptr
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// CHECK:  llvm.func @f(%[[SIZE:.+]]: i64) -> !llvm.ptr {
// CHECK-NEXT:    %[[PTR:.+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.return %[[PTR]] : !llvm.ptr
// CHECK-NEXT:  }

// CHECK:  llvm.func @noop() {
// CHECK-NEXT:    %[[SIZE:.+]] = arith.constant 8 : i64
// CHECK-NEXT:    %[[PTR:.+]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.call @free(%[[PTR]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
