// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// A mutable shadow is a value of the forward pass: the shadow of this
// loop-local alloca is a fresh allocation every iteration, and the reverse
// loop runs the iterations backwards, where that SSA value is not visible.
// The augmented forward puts the shadow by beside the primal copy, and the
// reverse call reads both back at the reversed index.

module {
  llvm.func @scale(%p: !llvm.ptr, %f: f64) {
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %f : f64
    llvm.store %s, %p : f64, !llvm.ptr
    llvm.return
  }
  func.func @f(%x: !llvm.ptr, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %two = llvm.mlir.constant(2.000000e+00 : f64) : f64
    scf.for %i = %c0 to %n step %c1 {
      %tmp = llvm.alloca %c1_i32 x f64 : (i32) -> !llvm.ptr
      %v = llvm.load %x : !llvm.ptr -> f64
      llvm.store %v, %tmp : f64, !llvm.ptr
      llvm.call @scale(%tmp, %two) : (!llvm.ptr, f64) -> ()
      %r = llvm.load %tmp : !llvm.ptr -> f64
      llvm.store %r, %x : f64, !llvm.ptr
    }
    return
  }
  func.func @df(%x: !llvm.ptr, %dx: !llvm.ptr, %n: index) {
    enzyme.autodiff @f(%x, %dx, %n) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, index) -> ()
    return
  }
}

// CHECK-LABEL: func.func private @diffef
// CHECK: %[[shadows:.+]] = memref.alloc(%{{.+}}) : memref<?x!llvm.ptr>
// CHECK: %[[copies:.+]] = memref.alloc(%{{.+}}) : memref<?x!llvm.ptr>
// CHECK: scf.for %[[i:.+]] =
// CHECK:   %[[shadow:.+]] = llvm.alloca
// CHECK:   enzyme.store %[[shadow]], %[[shadows]][%[[i]]]
// CHECK:   "llvm.intr.memset"(%[[shadow]]
// CHECK:   %[[copy:.+]] = llvm_ext.alloc
// CHECK:   enzyme.store %[[copy]], %[[copies]][%[[i]]]
// CHECK:   llvm.call @scale(
// CHECK: scf.for %[[j:.+]] =
// CHECK:   %[[rshadow:.+]] = enzyme.load %[[shadows]][%[[ri:.+]]]
// CHECK:   %[[rcopy:.+]] = enzyme.load %[[copies]][%[[ri]]]
// CHECK:   llvm.call @diffescale(%[[rcopy]], %[[rshadow]], %{{.+}})
