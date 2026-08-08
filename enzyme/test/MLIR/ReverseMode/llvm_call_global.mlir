// RUN: %eopt --enzyme --verify-diagnostics %s

// The reverse call runs long after the forward one; the only state carried
// across is the cached argument values and shadows. A callee that touches
// memory beyond its own arguments -- a global here -- reads whatever is
// there by reverse time, so it is refused.

module {
  llvm.mlir.global internal @counter(0.000000e+00 : f64) : f64
  llvm.func @scale(%p: !llvm.ptr, %f: f64) {
    %g = llvm.mlir.addressof @counter : !llvm.ptr
    %c = llvm.load %g : !llvm.ptr -> f64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %f : f64
    %t = arith.addf %s, %c : f64
    llvm.store %t, %p : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @f(%x: !llvm.ptr, %out: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %tmp = llvm.alloca %c1 x f64 : (i32) -> !llvm.ptr
    %v = llvm.load %x : !llvm.ptr -> f64
    llvm.store %v, %tmp : f64, !llvm.ptr
    // expected-error @below {{cannot differentiate a call in reverse mode whose callee touches memory beyond its own arguments; no state is carried between the forward and reverse passes: "scale"}}
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
