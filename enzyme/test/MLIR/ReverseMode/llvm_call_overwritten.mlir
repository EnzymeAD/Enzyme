// RUN: %eopt --enzyme --verify-diagnostics %s

// The reverse call runs long after the forward one, and even argument memory
// may have been overwritten in between; deciding which of it was is the
// overwritten-args analysis Enzyme's LLVM side has and this side does not
// yet. Until then a callee that touches memory -- its own argument here --
// is refused.

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
    // expected-error @below {{cannot differentiate a call in reverse mode whose callee touches memory; caching of overwritten arguments is not yet implemented here: "scale"}}
    llvm.call @scale(%tmp, %two) : (!llvm.ptr, f64) -> ()
    %r = llvm.load %tmp : !llvm.ptr -> f64
    llvm.store %r, %out : f64, !llvm.ptr
    llvm.return
  }
  func.func @df(%x: !llvm.ptr, %dx: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
    enzyme.autodiff @f(%x, %dx, %out, %dout) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}
