// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// A call is the same work to differentiate whichever dialect spelled it, but
// only func.call had the models for it; an llvm.call stopped at "could not
// compute the adjoint for this operation". The models are now written over the
// call op type and attached from both dialects.
//
// The argument here is an alloca, which is what the reverse pass has to keep a
// copy of, and an alloca says how much it is: the size of the element type,
// that many times. Nothing has to have annotated it.

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
    enzyme.autodiff @f(%x, %dx, %out, %dout) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// The alloca holds one f64, so eight bytes are put by; the copy is what the
// reverse call reads the primal from.

// CHECK: llvm.func @diffef(%[[x:.+]]: !llvm.ptr, %[[dx:.+]]: !llvm.ptr, %[[out:.+]]: !llvm.ptr, %[[dout:.+]]: !llvm.ptr)
// CHECK:         %[[eight:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:         llvm.intr.memset
// CHECK:         %[[tmp:.+]] = llvm.alloca %{{.+}} x f64 : (i32) -> !llvm.ptr
// CHECK:         %[[size:.+]] = llvm.mul %{{.+}}, %[[eight]] : i64
// CHECK-NEXT:    %[[copy:.+]] = llvm_ext.alloc %[[size]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.memcpy %[[copy]], %[[tmp]], %[[size]]
// CHECK-NEXT:    llvm.call @scale(%[[tmp]], %{{.+}})
// CHECK:         llvm.call @diffescale(%[[copy]], %{{.+}}, %{{.+}})
