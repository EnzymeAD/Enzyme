// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --lower-llvm-ext --canonicalize %s | FileCheck %s

// An over-aligned llvm.load/llvm.store keeps its alignment in the reverse pass:
// the shadow load/store the load and store adjoints build must carry the same
// alignment as the primal, or the shadow pointer is accessed under-aligned.

module {
llvm.func @loadstore(%a: !llvm.ptr, %b: f32) -> f32 {
  %sz = arith.constant 32 : i64
  llvm_ext.ptr_size_hint %a, %sz : !llvm.ptr, i64
  llvm.store %b, %a {alignment = 16 : i64} : f32, !llvm.ptr
  %0 = llvm.load %a {alignment = 16 : i64} : !llvm.ptr -> f32
  llvm.return %0 : f32
}

func.func @dloadstore(%a: !llvm.ptr, %da: !llvm.ptr, %b: f32, %dres: f32) -> f32 {
  %res = enzyme.autodiff @loadstore(%a, %da, %b, %dres)
    {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (!llvm.ptr, !llvm.ptr, f32, f32) -> f32
  return %res : f32
}
}

// CHECK-LABEL:  llvm.func @diffeloadstore(
// The load adjoint accumulates into the shadow pointer.
// CHECK:          llvm.load %[[da:.+]] {alignment = 16 : i64} : !llvm.ptr -> f32
// CHECK:          llvm.store %{{.+}}, %[[da]] {alignment = 16 : i64} : f32, !llvm.ptr
// The store adjoint loads and zeroes the shadow pointer.
// CHECK:          llvm.load %[[da]] {alignment = 16 : i64} : !llvm.ptr -> f32
// CHECK:          llvm.store %{{.+}}, %[[da]] {alignment = 16 : i64} : f32, !llvm.ptr
