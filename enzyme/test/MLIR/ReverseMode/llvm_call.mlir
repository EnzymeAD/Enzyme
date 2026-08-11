// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// A call is the same work to differentiate whichever dialect spelled it, but
// only func.call had the models for it; an llvm.call stopped at "could not
// compute the adjoint for this operation". The models are now written over the
// call op type and attached from both dialects.
//
// The callee here touches no memory, which is what the reverse handler
// accepts today: nothing is carried between the passes for it to need.

module {
  llvm.func @scale(%v: f64, %f: f64) -> f64 {
    %s = arith.mulf %v, %f : f64
    llvm.return %s : f64
  }
  llvm.func @f(%x: !llvm.ptr, %out: !llvm.ptr) {
    %two = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %v = llvm.load %x : !llvm.ptr -> f64
    %r = llvm.call @scale(%v, %two) : (f64, f64) -> f64
    llvm.store %r, %out : f64, !llvm.ptr
    llvm.return
  }
  func.func @df(%x: !llvm.ptr, %dx: !llvm.ptr, %out: !llvm.ptr, %dout: !llvm.ptr) {
    enzyme.autodiff @f(%x, %dx, %out, %dout) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @diffef(%[[x:.+]]: !llvm.ptr, %[[dx:.+]]: !llvm.ptr, %[[out:.+]]: !llvm.ptr, %[[dout:.+]]: !llvm.ptr)
// CHECK:   %[[v:.+]] = llvm.load %[[x]] : !llvm.ptr -> f64
// CHECK:   %[[r:.+]] = llvm.call @scale(%[[v]], %{{.+}})
// CHECK:   llvm.store %[[r]], %[[out]]
// CHECK:   %[[seed:.+]] = llvm.load %[[dout]]
// CHECK:   %[[dv:.+]] = llvm.call @diffescale(%[[v]], %{{.+}}, %{{.+}})
// CHECK:   llvm.store %{{.+}}, %[[dx]]

// The reverse of v * f in v is the incoming adjoint times f.
// CHECK: llvm.func @diffescale(%[[pv:.+]]: f64, %[[pf:.+]]: f64, %[[dr:.+]]: f64) -> f64
// CHECK:   %[[m:.+]] = arith.mulf %{{.+}}, %[[pf]] fastmath<fast> : f64
