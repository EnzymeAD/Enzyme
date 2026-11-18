// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// The callee writes through its !llvm.ptr argument, which comes from an
// alloca in the caller (so its extent is known). Reverse mode caches a clone
// of %tmp before calling @scale and passes the clone to @diffescale, freeing
// it afterwards.

module {
  llvm.func @scale(%p: !llvm.ptr, %f: f64) {
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %f : f64
    llvm.store %s, %p : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @f(%x: !llvm.ptr, %out: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %tmp = llvm.alloca %c1 x f64 : (i32) -> !llvm.ptr
    %v = llvm.load %x : !llvm.ptr -> f64
    llvm.store %v, %tmp : f64, !llvm.ptr
    llvm.call @scale(%tmp, %two) : (!llvm.ptr, f64) -> ()
    %r = llvm.load %tmp : !llvm.ptr -> f64
    llvm.store %r, %out : f64, !llvm.ptr
    llvm.return
  }
  func.func @df(%x: !llvm.ptr, %dx: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
    enzyme.autodiff @f(%x, %dx, %out, %dout) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[] }> : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: llvm.func @diffef(
// CHECK:         %[[DTMP:.+]] = llvm.alloca {{.*}} x f64 : (i32) -> !llvm.ptr
// CHECK:         %[[TMP:.+]] = llvm.alloca {{.*}} x f64 : (i32) -> !llvm.ptr
// CHECK:         %[[CLONE:.+]] = llvm_ext.alloc
// CHECK:         llvm_ext.memcpy %[[CLONE]], %[[TMP]],
// CHECK:         llvm.call @scale(%[[TMP]],
// CHECK:         llvm.call @diffescale(%[[CLONE]], %[[DTMP]],
// CHECK:         llvm_ext.free %[[CLONE]]
// CHECK-LABEL: llvm.func @diffescale(
