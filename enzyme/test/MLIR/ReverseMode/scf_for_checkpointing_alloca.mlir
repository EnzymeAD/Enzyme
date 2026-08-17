// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s

// Checkpointing a loop that mutates a buffer it reached through a plain
// llvm.alloca: nothing hints the pointer's extent, but the alloca's type
// already says it -- element size times array size. The clone buffers must be
// sized 4 (one f32) with no ptr_size_hint anywhere in sight.

module {
  func.func @main(%x: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %one = llvm.mlir.constant(1 : i64) : i64
    %f1 = arith.constant 1.0 : f32
    %init = arith.constant 2.0 : f32

    %p = llvm.alloca %one x f32 : (i64) -> !llvm.ptr
    llvm.store %init, %p : f32, !llvm.ptr

    %0 = scf.for %i = %c0 to %c8 step %c1 iter_args(%it = %x) -> (f32) {
      %v = llvm.load %p : !llvm.ptr -> f32
      %m = arith.mulf %v, %it : f32
      %n = arith.addf %v, %f1 : f32
      llvm.store %n, %p : f32, !llvm.ptr
      scf.yield %m : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.checkpoint_period = 4 : i64}

    return %0 : f32
  }
}

// CHECK-LABEL: func.func @main(
// CHECK: %[[ELEM:.+]] = llvm.mlir.constant(4 : i64) : i64
// The shadow of the buffer is zeroed over the computed extent.
// CHECK: %[[ZSZ:.+]] = llvm.mul %[[ELEM]], %{{.+}} : i64
// CHECK: "llvm.intr.memset"(%{{.+}}, %{{.+}}, %[[ZSZ]])
// Each checkpoint clone is allocated and filled over that same extent.
// CHECK: %[[SZ:.+]] = llvm.mul %[[ELEM]], %{{.+}} : i64
// CHECK: %[[CLONE:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK: llvm_ext.memcpy %[[CLONE]], %{{.+}}, %[[SZ]]
