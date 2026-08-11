// RUN: %eopt %s --allow-unregistered-dialect --lower-llvm-ext=lower-to-alloca-threshold=4 | FileCheck %s

// The enzymexla ops the copies lower to live outside this repository, hence
// --allow-unregistered-dialect.

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

  // A host clone: alloc/memcpy/free all in memory space 0.
  llvm.func @clone_host(%src: !llvm.ptr, %n: i64) {
    %0 = llvm_ext.alloc %n : (i64) -> !llvm.ptr
    llvm_ext.memcpy %0, %src, %n : !llvm.ptr, !llvm.ptr, i64
    llvm_ext.free %0 : !llvm.ptr
    llvm.return
  }

  // The same clone in device memory. Nothing on the ops says so -- the address
  // space of the pointers does, and it is what routes the allocation and the
  // free through the GPU runtime instead of malloc/free.
  llvm.func @clone_device(%src: !llvm.ptr<1>, %n: i64) {
    %0 = llvm_ext.alloc %n : (i64) -> !llvm.ptr<1>
    llvm_ext.memcpy %0, %src, %n : !llvm.ptr<1>, !llvm.ptr<1>, i64
    llvm_ext.free %0 : !llvm.ptr<1>
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

// CHECK:  llvm.func @clone_host(%[[SRC:.+]]: !llvm.ptr, %[[N:.+]]: i64) {
// CHECK-NEXT:    %[[PTR:.+]] = llvm.call @malloc(%[[N]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[SIZE:.+]] = arith.index_cast %[[N]] : i64 to index
// CHECK-NEXT:    %[[DST:.+]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    %[[SRCBUF:.+]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    "enzymexla.memcpy"(%[[DST]], %[[SRCBUF]], %[[SIZE]]) : (memref<?xi8>, memref<?xi8>, index) -> ()
// CHECK-NEXT:    llvm.call @free(%[[PTR]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// The device clone: same ops, but the alloc/free go through the GPU runtime and
// every view carries memory space 1, all of it driven by the pointer type.
// CHECK:  llvm.func @clone_device(%[[SRC:.+]]: !llvm.ptr<1>, %[[N:.+]]: i64) {
// CHECK-NEXT:    %[[ASIZE:.+]] = arith.index_cast %[[N]] : i64 to index
// CHECK-NEXT:    %[[BUF:.+]] = gpu.alloc  (%[[ASIZE]]) : memref<?xi8, 1>
// CHECK-NEXT:    %[[PTR:.+]] = "enzymexla.memref2pointer"(%[[BUF]]) : (memref<?xi8, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[SIZE:.+]] = arith.index_cast %[[N]] : i64 to index
// CHECK-NEXT:    %[[DST:.+]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<?xi8, 1>
// CHECK-NEXT:    %[[SRCBUF:.+]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr<1>) -> memref<?xi8, 1>
// CHECK-NEXT:    "enzymexla.memcpy"(%[[DST]], %[[SRCBUF]], %[[SIZE]]) : (memref<?xi8, 1>, memref<?xi8, 1>, index) -> ()
// CHECK-NEXT:    %[[FREEBUF:.+]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<?xi8, 1>
// CHECK-NEXT:    gpu.dealloc  %[[FREEBUF]] : memref<?xi8, 1>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
